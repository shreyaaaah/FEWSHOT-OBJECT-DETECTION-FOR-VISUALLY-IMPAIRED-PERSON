"""Shared helpers for the evaluation scripts."""
import json
import os
from typing import Dict, List, Optional

import cv2
import numpy as np

from ..config import get_device, load_config
from ..train.datasets import IMG_EXT, load_splits


def ci95(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def mean_ci(x):
    return float(np.mean(x)), ci95(x)


def fmt(x, ci=None, pct=True):
    k = 100.0 if pct else 1.0
    d = 1 if pct else 3
    return f"{k * x:.{d}f}" + (f" ± {k * ci:.{d}f}" if ci is not None else "")


def list_instance_files(root: str, instances: List[str], min_images: int = 8) -> Dict[str, List[str]]:
    out = {}
    for inst in instances:
        d = os.path.join(root, inst)
        if not os.path.isdir(d):
            continue
        files = sorted(os.path.join(d, f) for f in os.listdir(d)
                       if os.path.splitext(f)[1].lower() in IMG_EXT)
        if len(files) >= min_images:
            out[inst] = files
    return out


def embed_files(model, files: List[str], batch_size=32, crop_fn=None) -> np.ndarray:
    """Embed image files with the SAME preprocessing used at run time (OpenCV pad+resize)."""
    imgs = []
    out = []
    for i, p in enumerate(files):
        im = cv2.imread(p)
        if im is None:
            continue
        imgs.append(crop_fn(im) if crop_fn else im)
        if len(imgs) == batch_size:
            out.append(model.embed_bgr(imgs, batch_size))
            imgs = []
    if imgs:
        out.append(model.embed_bgr(imgs, batch_size))
    return np.concatenate(out, 0) if out else np.zeros((0, model.out_dim), np.float32)


def video_kind(path: str) -> str:
    b = os.path.basename(path).lower()
    return "clean" if b.startswith("clean_") else ("clutter" if b.startswith("clutter_") else "any")


class HistEmbedder:
    """Hand-crafted baseline (no deep learning): 3x3-grid HSV colour histograms, Hellinger
    normalised.  Stage-0 row of the ablation table; also lets the eval code be tested offline."""
    out_dim = 3 * 3 * 64

    def embed_bgr(self, imgs, batch_size=32):
        from ..imgutil import pad_square
        out = []
        for im in imgs:
            hsv = cv2.cvtColor(cv2.resize(pad_square(im), (96, 96)), cv2.COLOR_BGR2HSV)
            feats = []
            for gy in range(3):
                for gx in range(3):
                    c = hsv[gy * 32:(gy + 1) * 32, gx * 32:(gx + 1) * 32]
                    h = cv2.calcHist([c], [0, 1, 2], None, [4, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
                    feats.append(np.sqrt(h / max(h.sum(), 1.0)))
            v = np.concatenate(feats).astype(np.float32)
            out.append(v / (np.linalg.norm(v) + 1e-8))
        return np.array(out, np.float32).reshape(len(imgs), -1)


def load_model_for_eval(ckpt: Optional[str], backbone: Optional[str], no_pretrained=False,
                        device=None, img_size=None):
    """ckpt -> fine-tuned model; otherwise a frozen pretrained backbone (no head)."""
    if backbone == "hsv_hist":
        return HistEmbedder(), "baseline:hsv_hist"
    from ..embedder import Embedder
    cfg = load_config()
    dev = device or get_device(cfg["train"].get("device", "auto"))
    if ckpt:
        return Embedder.load(ckpt, device=dev), f"ckpt:{os.path.basename(ckpt)}"
    name = backbone or cfg["backbone"]["name"]
    m = Embedder(name, cfg["backbone"]["embed_dim"], img_size or cfg["backbone"]["img_size"],
                 mode="frozen", pretrained=not no_pretrained, use_head=False)
    return m.to(dev).eval(), f"frozen:{name}"


def write_report(path_base: str, data: dict, markdown: str):
    os.makedirs(os.path.dirname(path_base), exist_ok=True)
    with open(path_base + ".json", "w") as f:
        json.dump(data, f, indent=2, default=float)
    with open(path_base + ".md", "w") as f:
        f.write(markdown)
    print(f"\n[report] {path_base}.json / .md")
