"""Distance estimation  (core novelty #4: known-size-anchored metric depth).

A single camera cannot measure metric distance directly, so we combine two cues:

  * Known-size cue:  d = f_px * real_longest_side / pixel_longest_side
    (real sizes come from a COCO table or from the size the user gave when
    enrolling a personal object).
  * Relative depth cue: a monocular model (Depth Anything V2 Small) gives a
    disparity-like map, i.e. r ~ 1 / d up to an unknown scale k.  The objects whose
    real size is known ("anchors") each give one estimate k_i = d_i * r_i; the
    median is the per-frame scale.  Then ANY object, including ones with unknown
    size, gets d = k / r.

Everything except `RelativeDepth` is pure NumPy/Python so it can be unit tested.
"""
import math
from statistics import median
from typing import List, Optional, Sequence, Tuple

import numpy as np

# longest side of a typical instance, in metres (approximate, documented assumption)
SIZE_TABLE = {
    "bottle": 0.24, "cup": 0.11, "wine glass": 0.20, "cell phone": 0.15,
    "laptop": 0.33, "remote": 0.18, "book": 0.24, "scissors": 0.18,
    "mouse": 0.11, "keyboard": 0.45, "chair": 0.90, "backpack": 0.45,
    "handbag": 0.30, "umbrella": 0.90, "clock": 0.28, "vase": 0.28,
    "bowl": 0.16, "toothbrush": 0.19, "fork": 0.19, "knife": 0.22,
    "spoon": 0.18, "banana": 0.19, "apple": 0.08, "orange": 0.08,
    "tv": 0.80, "couch": 1.90, "bed": 2.0, "dining table": 1.4,
    "suitcase": 0.65, "potted plant": 0.40, "person": 1.7,
    "stop sign": 0.75, "bicycle": 1.7, "bench": 1.5, "sports ball": 0.22,
    "refrigerator": 1.8, "microwave": 0.50, "oven": 0.60, "sink": 0.50, "toilet": 0.70,
    "teddy bear": 0.35, "traffic light": 0.90, "fire hydrant": 0.75,
}


def size_distance(box, real_m: float, f_px: float, W: Optional[int] = None,
                  H: Optional[int] = None) -> Tuple[Optional[float], bool]:
    x1, y1, x2, y2 = box
    L = max(x2 - x1, y2 - y1)
    if L <= 1 or not real_m:
        return None, False
    d = f_px * real_m / L
    reliable = True
    if W is not None and H is not None:          # box cut by the frame edge -> size is wrong
        if x1 <= 2 or y1 <= 2 or x2 >= W - 2 or y2 >= H - 2:
            reliable = False
    return float(d), reliable


def box_depth_value(depth_map: np.ndarray, box) -> Optional[float]:
    """Median disparity in the central 50% of a box."""
    H, W = depth_map.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    cx1 = int(max(0, x1 + 0.25 * bw))
    cx2 = int(min(W, x2 - 0.25 * bw))
    cy1 = int(max(0, y1 + 0.25 * bh))
    cy2 = int(min(H, y2 - 0.25 * bh))
    if cx2 - cx1 < 1 or cy2 - cy1 < 1:
        return None
    v = float(np.median(depth_map[cy1:cy2, cx1:cx2]))
    return v if v > 1e-6 else None


def fuse_distances(size_d: Sequence[Optional[float]], reliable: Sequence[bool],
                   rel_vals: Sequence[Optional[float]], min_anchors: int = 1):
    """Combine size-based distances with a relative (disparity) depth cue.

    size_d[i]   metric distance from the known-size cue (None if size unknown)
    reliable[i] False if the box touches the frame edge
    rel_vals[i] disparity-like value for the box (larger = nearer)
    Returns (distances, info)."""
    anchors = [d * r for d, ok, r in zip(size_d, reliable, rel_vals)
               if d is not None and ok and r is not None and r > 0]
    info = {"anchors": len(anchors), "used_depth": False, "scale": None}
    if len(anchors) < min_anchors:
        return list(size_d), info
    k = median(anchors)
    info.update(used_depth=True, scale=k)
    out = []
    for d, r in zip(size_d, rel_vals):
        dd = (k / r) if (r is not None and r > 0) else None
        if dd is not None and d is not None and len(anchors) >= 3:
            out.append(math.sqrt(d * dd))        # geometric mean when well anchored
        elif dd is not None:
            out.append(dd)
        else:
            out.append(d)
    return out, info


class RelativeDepth:
    """Depth Anything V2 Small via HuggingFace (downloads ~100 MB on first use)."""

    def __init__(self, model_id="depth-anything/Depth-Anything-V2-Small-hf", device=None):
        from transformers import pipeline
        dev = -1
        if device is not None and str(device).startswith("cuda"):
            dev = 0
        self.pipe = pipeline("depth-estimation", model=model_id, device=dev)

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        import cv2
        from PIL import Image
        rgb = Image.fromarray(bgr[:, :, ::-1])
        out = self.pipe(rgb)
        d = out["predicted_depth"]
        d = d.squeeze().detach().cpu().numpy().astype(np.float32)
        d = cv2.resize(d, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
        d = d - d.min()                           # make strictly positive
        return d + 1e-3 * (d.max() + 1e-6)
