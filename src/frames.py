"""Frame sampling + blur-aware selection for enrollment videos."""
from typing import List, Tuple

import cv2
import numpy as np

from .imgutil import laplacian_var


def read_video_frames(path: str, n_samples: int = 40, max_side: int = 960) -> List[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total <= 0:                                  # unknown length: read all
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        idx = np.linspace(0, len(frames) - 1, min(n_samples, len(frames))).astype(int)
        frames = [frames[i] for i in idx]
    else:
        for i in np.linspace(0, total - 1, min(n_samples, total)).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            if ok:
                frames.append(f)
    cap.release()
    out = []
    for f in frames:
        h, w = f.shape[:2]
        s = max(h, w)
        if s > max_side:
            k = max_side / s
            f = cv2.resize(f, (int(w * k), int(h * k)))
        out.append(f)
    return out


def select_frames(frames: List[np.ndarray], keep: int = 25,
                  min_sharp: float = 40.0) -> Tuple[List[int], List[float]]:
    """Pick the sharpest frame inside each of `keep` time bins (so views are spread
    over the whole rotation), then drop blurry ones. Returns (indices, sharpness)."""
    n = len(frames)
    sharp = [laplacian_var(f) for f in frames]
    if n == 0:
        return [], sharp
    bins = np.array_split(np.arange(n), min(keep, n))
    chosen = []
    for b in bins:
        if len(b) == 0:
            continue
        chosen.append(int(b[int(np.argmax([sharp[i] for i in b]))]))
    good = [j for j in chosen if sharp[j] >= min_sharp]
    min_keep = max(5, keep // 3)
    if len(good) < min(min_keep, len(chosen)):      # video is blurry overall: keep best
        good = sorted(sorted(chosen, key=lambda j: -sharp[j])[:max(5, keep // 2)])
    return sorted(good), sharp
