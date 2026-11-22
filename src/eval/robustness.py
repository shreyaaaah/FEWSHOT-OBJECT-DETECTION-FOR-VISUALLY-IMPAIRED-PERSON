"""Robustness of recognition to the conditions blind users actually produce:
blur, low light, partial occlusion and small/far objects.

Protocol: enrol each object from CLEAN frames (K shots), then query with the SAME
held-out frames after applying a degradation of increasing severity.  Reports
correct-ID rate (TPR) and false-accept rate on unknown objects per severity.

    python -m src.eval.robustness --root data/orbit_instances --ckpt checkpoints/embedder_main.pt
"""
import argparse
import os

import cv2
import numpy as np

from ..augment import adjust_light, gaussian_blur, motion_blur, occlude
from ..config import ROOT
from ..memory import PersonalMemory
from ..train.datasets import load_splits
from .common import fmt, list_instance_files, load_model_for_eval, mean_ci, write_report

DEGRADATIONS = {
    "gaussian_blur": [("sigma 0", 0.0), ("sigma 1.5", 1.5), ("sigma 3", 3.0), ("sigma 5", 5.0)],
    "motion_blur": [("k 1", 1), ("k 9", 9), ("k 17", 17), ("k 29", 29)],
    "low_light": [("gain 1.0", 1.0), ("gain 0.5", 0.5), ("gain 0.25", 0.25), ("gain 0.1", 0.1)],
    "occlusion": [("0%", 0.0), ("20%", 0.2), ("40%", 0.4), ("60%", 0.6)],
    "small_object": [("100%", 1.0), ("50%", 0.5), ("25%", 0.25), ("12%", 0.12)],
}


def degrade(img, kind, level, rng):
    if kind == "gaussian_blur":
        return gaussian_blur(img, level)
    if kind == "motion_blur":
        return motion_blur(img, level, 0.0) if level > 1 else img
    if kind == "low_light":
        return adjust_light(img, gamma=1.0, gain=level)
    if kind == "occlusion":
        return occlude(img, level, rng)
    if kind == "small_object":                      # object occupies only `level` of the crop side
        h, w = img.shape[:2]
        nh, nw = max(4, int(h * level)), max(4, int(w * level))
        small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.full_like(img, (124, 116, 104))
        y, x = (h - nh) // 2, (w - nw) // 2
        canvas[y:y + nh, x:x + nw] = small
        return canvas
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--splits", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--frozen-backbone", default=None)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--k-shot", type=int, default=5)
    ap.add_argument("--n-known", type=int, default=5)
    ap.add_argument("--n-unknown", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max-queries", type=int, default=10, help="frames per object per episode")
    ap.add_argument("--kinds", nargs="+", default=list(DEGRADATIONS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    model, desc = load_model_for_eval(a.ckpt, a.frozen_backbone, a.no_pretrained, img_size=a.img_size)
    splits = load_splits(a.splits or os.path.join(a.root, "splits.json"))
    files = list_instance_files(a.root, splits[a.split], min_images=a.k_shot + 3)
    names = sorted(files)
    n_known = min(a.n_known, len(names) - 1)
    n_unk = min(a.n_unknown, len(names) - n_known)
    if n_known < 2:
        raise SystemExit("not enough instances in this split")
    rng = np.random.default_rng(a.seed)
    imgs = {n: [cv2.imread(f) for f in files[n]] for n in names}
    imgs = {n: [i for i in v if i is not None] for n, v in imgs.items()}
    print(f"[eval] model={desc} instances={len(names)}")

    res = {k: {lv[0]: {"tpr": [], "far": []} for lv in DEGRADATIONS[k]} for k in a.kinds}
    for _ in range(a.episodes):
        pick = rng.choice(len(names), n_known + n_unk, replace=False)
        known = [names[i] for i in pick[:n_known]]
        unk = [names[i] for i in pick[n_known:]]
        mem = PersonalMemory()
        queries = {}
        for c in known:
            n = len(imgs[c])
            s = np.linspace(0, n - 1, a.k_shot).astype(int)
            mem.enroll(c, model.embed_bgr([imgs[c][i] for i in s]))
            rest = np.setdiff1d(np.arange(n), s)
            queries[c] = [imgs[c][i] for i in rng.permutation(rest)[:a.max_queries]]
        uq = [im for u in unk for im in [imgs[u][i] for i in rng.permutation(len(imgs[u]))[:a.max_queries]]]
        for kind in a.kinds:
            for label, lv in DEGRADATIONS[kind]:
                ok = tot = 0
                for c in known:
                    z = model.embed_bgr([degrade(im, kind, lv, rng) for im in queries[c]])
                    ok += sum(mem.query(e).name == c for e in z)
                    tot += len(z)
                zu = model.embed_bgr([degrade(im, kind, lv, rng) for im in uq])
                fa = np.mean([mem.query(e).name is not None for e in zu]) if len(zu) else float("nan")
                res[kind][label]["tpr"].append(ok / max(tot, 1))
                res[kind][label]["far"].append(fa)

    md = [f"# Robustness evaluation\n\nmodel `{desc}` | K={a.k_shot} | {n_known} known + {n_unk} unknown | "
          f"{a.episodes} episodes (enrol clean, query degraded)\n"]
    for kind in a.kinds:
        md.append(f"\n## {kind}\n\n| severity | correct-ID % | false-accept % |\n|---|---|---|")
        for label, _ in DEGRADATIONS[kind]:
            t, f = mean_ci(res[kind][label]["tpr"]), mean_ci(res[kind][label]["far"])
            md.append(f"| {label} | {fmt(*t)} | {fmt(*f)} |")
    text = "\n".join(md) + "\n"
    print(text)
    write_report(os.path.join(ROOT, "reports", f"robustness_{a.tag or desc.replace(':', '_')}_{a.split}"),
                 {"desc": desc, "results": res}, text)


if __name__ == "__main__":
    main()
