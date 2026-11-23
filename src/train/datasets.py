"""Instance-folder datasets + episodic sampler.

Expected layout (works for ORBIT, CORe50, your own recordings, Stanford Online Products):

    root/<instance_id>/<frame>.jpg

For ORBIT use  `<user>__<object>`  as the instance id (tools/prepare_orbit.py does this)
so that splits can be made by USER: the same person never appears in train and test.
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from ..imgutil import MEAN, STD

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FILL = (124, 116, 104)


def scan_instances(root: str, min_images: int = 8) -> Dict[str, List[str]]:
    out = {}
    for inst in sorted(os.listdir(root)):
        d = os.path.join(root, inst)
        if not os.path.isdir(d):
            continue
        files = sorted(os.path.join(d, f) for f in os.listdir(d)
                       if os.path.splitext(f)[1].lower() in IMG_EXT)
        if len(files) >= min_images:
            out[inst] = files
    return out


def group_of(inst: str) -> str:
    """'user12__mug' -> 'user12'. Instances without '__' are their own group."""
    return inst.split("__")[0] if "__" in inst else inst


def split_instances(insts: Sequence[str], ratios=(0.7, 0.15, 0.15), seed=0,
                    by_group=True) -> Dict[str, List[str]]:
    """Disjoint train/val/test split. With by_group, all instances of a group
    (e.g. a blind user) land in the same split -> no identity leakage."""
    groups: Dict[str, List[str]] = {}
    for i in insts:
        groups.setdefault(group_of(i) if by_group else i, []).append(i)
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    n = len(keys)
    n_tr = max(1, int(round(ratios[0] * n)))
    n_va = max(1, int(round(ratios[1] * n))) if n >= 3 else 0
    parts = {"train": keys[:n_tr], "val": keys[n_tr:n_tr + n_va], "test": keys[n_tr + n_va:]}
    if n >= 3 and not parts["test"]:                 # make sure test is never empty
        parts["test"] = [parts["val"].pop()] if parts["val"] else [parts["train"].pop()]
    return {k: sorted(i for g in v for i in groups[g]) for k, v in parts.items()}


def save_splits(splits, path):
    with open(path, "w") as f:
        json.dump(splits, f, indent=2)


def load_splits(path):
    with open(path) as f:
        return json.load(f)


def ensure_splits(root, path=None, seed=0):
    """Load <root>/splits.json, or create it (user-disjoint 70/15/15) if it does not exist yet."""
    path = path or os.path.join(root, "splits.json")
    if os.path.exists(path):
        return load_splits(path)
    insts = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    splits = split_instances(insts, seed=seed)
    save_splits(splits, path)
    return splits


# ------------------------------------------------------------- transforms
class PadSquare:
    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        s = max(w, h)
        canvas = Image.new("RGB", (s, s), FILL)
        canvas.paste(img, ((s - w) // 2, (s - h) // 2))
        return canvas


class MotionBlur:
    """Random linear motion blur (camera shake from hand-held blind-user video)."""

    def __init__(self, p=0.3, kmin=5, kmax=15):
        self.p, self.kmin, self.kmax = p, kmin, kmax

    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.rand() > self.p:
            return img
        import cv2
        k = int(np.random.randint(self.kmin, self.kmax)) | 1
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1.0 / k
        M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), np.random.uniform(0, 180), 1.0)
        kernel = cv2.warpAffine(kernel, M, (k, k))
        kernel /= max(kernel.sum(), 1e-6)
        arr = cv2.filter2D(np.asarray(img), -1, kernel)
        return Image.fromarray(arr)


def train_transform(size=224):
    return T.Compose([
        PadSquare(),
        T.RandomResizedCrop(size, scale=(0.35, 1.0), ratio=(0.75, 1.33)),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.05)], p=0.8),
        T.RandomApply([T.GaussianBlur(7, (0.1, 3.0))], p=0.3),
        MotionBlur(p=0.3),
        T.RandomRotation(20, fill=FILL),
        T.ToTensor(),
        T.Normalize(MEAN.tolist(), STD.tolist()),
        T.RandomErasing(p=0.25, scale=(0.02, 0.2)),      # partial occlusion
    ])


def eval_transform(size=224):
    return T.Compose([PadSquare(), T.Resize((size, size)), T.ToTensor(),
                      T.Normalize(MEAN.tolist(), STD.tolist())])


def load_images(paths: Sequence[str], tf, workers=8) -> torch.Tensor:
    def _one(p):
        with Image.open(p) as im:
            return tf(im.convert("RGB"))
    with ThreadPoolExecutor(workers) as ex:
        return torch.stack(list(ex.map(_one, paths)))


# ---------------------------------------------------------------- episodes
@dataclass
class Episode:
    support_paths: List[str]
    support_labels: List[int]
    query_paths: List[str]
    query_labels: List[int]
    unknown_paths: List[str]          # queries from classes NOT in the support set
    classes: List[str]
    unknown_labels: List[int] = None  # class id (0..n_unk-1) of each unknown frame


def _kind(path: str) -> str:
    b = os.path.basename(path).lower()
    return "clean" if b.startswith("clean_") else ("clutter" if b.startswith("clutter_") else "any")


class EpisodeSampler:
    """N-way K-shot episodes with an additional set of 'unknown' classes whose
    frames must be rejected (open-set episodic training)."""

    def __init__(self, data: Dict[str, List[str]], instances: Sequence[str], n_way=5,
                 k_shot=5, q_query=10, n_unknown_classes=2, q_unknown=5, seed=0,
                 contiguous_support=False, cross_video=False):
        self.data = {i: data[i] for i in instances
                     if len(data[i]) >= k_shot + q_query}
        self.insts = sorted(self.data)
        if len(self.insts) < n_way:
            raise ValueError(f"need >= {n_way} instances with >= {k_shot + q_query} images, "
                             f"got {len(self.insts)}")
        self.n_way, self.k, self.q = n_way, k_shot, q_query
        self.n_unk = min(n_unknown_classes, max(0, len(self.insts) - n_way))
        self.q_unk = q_unknown
        self.contiguous = contiguous_support
        self.cross_video = cross_video       # ORBIT protocol: support = clean_* frames, queries = clutter_* frames
        self.kinds = {i: np.array([_kind(f) for f in self.data[i]]) for i in self.insts}
        self.rng = np.random.default_rng(seed)

    def sample(self) -> Episode:
        pick = self.rng.choice(len(self.insts), self.n_way + self.n_unk, replace=False)
        known = [self.insts[i] for i in pick[:self.n_way]]
        unk = [self.insts[i] for i in pick[self.n_way:]]
        sp, sl, qp, ql = [], [], [], []
        for c, inst in enumerate(known):
            files = self.data[inst]
            clean = np.where(self.kinds[inst] == "clean")[0]
            clutter = np.where(self.kinds[inst] == "clutter")[0]
            if self.cross_video and len(clean) >= self.k and len(clutter) >= 3:
                sidx = self.rng.choice(clean, self.k, replace=False)
                qidx = self.rng.choice(clutter, min(self.q, len(clutter)), replace=False)
            elif self.contiguous:                     # support = a short video segment
                s = int(self.rng.integers(0, len(files) - self.k + 1))
                sidx = np.arange(s, s + self.k)
                rest = np.setdiff1d(np.arange(len(files)), sidx)
                qidx = self.rng.choice(rest, min(self.q, len(rest)), replace=False)
            else:
                perm = self.rng.permutation(len(files))
                sidx, qidx = perm[:self.k], perm[self.k:self.k + self.q]
            sp += [files[i] for i in sidx]; sl += [c] * len(sidx)
            qp += [files[i] for i in qidx]; ql += [c] * len(qidx)
        up, ul = [], []
        for j, inst in enumerate(unk):
            files = self.data[inst]
            pool = np.where(self.kinds[inst] == "clutter")[0] if self.cross_video else np.arange(0)
            if len(pool) >= 3:
                idx = self.rng.choice(pool, min(self.q_unk, len(pool)), replace=False)
            else:
                idx = self.rng.choice(len(files), min(self.q_unk, len(files)), replace=False)
            up += [files[i] for i in idx]
            ul += [j] * len(idx)
        return Episode(sp, sl, qp, ql, up, known, ul)
