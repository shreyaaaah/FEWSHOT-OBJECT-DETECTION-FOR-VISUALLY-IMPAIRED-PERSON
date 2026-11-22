"""Create a tiny synthetic instance dataset (coloured shapes + texture on random
backgrounds, with blur/brightness jitter).  Only for SMOKE-TESTING the training and
evaluation code without downloading ORBIT.  Do not report these numbers.

    python tools/make_synthetic.py --out data/synthetic --instances 24 --frames 20
"""
import argparse
import os

import cv2
import numpy as np


def make_instance(rng, size=128):
    col = rng.integers(40, 255, 3).tolist()
    col2 = rng.integers(0, 255, 3).tolist()
    shape = int(rng.integers(0, 3))
    stripes = int(rng.integers(0, 4))
    return col, col2, shape, stripes


def render(spec, rng, size=128):
    col, col2, shape, stripes = spec
    bg = rng.integers(60, 200, 3)
    img = np.full((size, size, 3), bg, np.uint8)
    img = np.clip(img + rng.normal(0, 10, img.shape), 0, 255).astype(np.uint8)
    cx, cy = size // 2 + int(rng.integers(-12, 13)), size // 2 + int(rng.integers(-12, 13))
    r = int(rng.integers(28, 44))
    if shape == 0:
        cv2.circle(img, (cx, cy), r, col, -1)
    elif shape == 1:
        cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), col, -1)
    else:
        pts = np.array([[cx, cy - r], [cx - r, cy + r], [cx + r, cy + r]])
        cv2.fillPoly(img, [pts], col)
    for k in range(stripes):
        y = cy - r + (k + 1) * (2 * r) // (stripes + 1)
        cv2.line(img, (cx - r // 2, y), (cx + r // 2, y), col2, 3)
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (7, 7), float(rng.uniform(0.5, 2.5)))
    return np.clip(img.astype(np.float32) * rng.uniform(0.75, 1.25), 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--instances", type=int, default=24)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--users", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    for i in range(a.instances):
        spec = make_instance(rng)
        d = os.path.join(a.out, f"user{i % a.users}__obj{i}")
        os.makedirs(d, exist_ok=True)
        for j in range(a.frames):
            cv2.imwrite(os.path.join(d, f"{j:03d}.jpg"), render(spec, rng))
    print(f"wrote {a.instances} instances x {a.frames} frames to {a.out}")


if __name__ == "__main__":
    main()
