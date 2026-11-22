"""Image helpers shared by training-free code (OpenCV / NumPy only)."""
import math
from typing import List, Sequence, Tuple

import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PAD_BGR = (104, 116, 124)        # ImageNet mean colour in BGR


def crop_box(img, box, pad=0.08):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1 = int(max(0, x1 - pad * bw))
    y1 = int(max(0, y1 - pad * bh))
    x2 = int(min(W, x2 + pad * bw))
    y2 = int(min(H, y2 + pad * bh))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return img.copy()
    return img[y1:y2, x1:x2].copy()


def pad_square(img, fill=PAD_BGR):
    h, w = img.shape[:2]
    s = max(h, w)
    top = (s - h) // 2
    left = (s - w) // 2
    return cv2.copyMakeBorder(img, top, s - h - top, left, s - w - left,
                              cv2.BORDER_CONSTANT, value=fill)


def prep_bgr(img, size=224):
    """BGR uint8 -> normalised float32 CHW (aspect preserved by padding)."""
    sq = pad_square(img)
    interp = cv2.INTER_AREA if sq.shape[0] > size else cv2.INTER_LINEAR
    sq = cv2.resize(sq, (size, size), interpolation=interp)
    rgb = sq[:, :, ::-1].astype(np.float32) / 255.0
    rgb = (rgb - MEAN) / STD
    return np.ascontiguousarray(rgb.transpose(2, 0, 1))


def laplacian_var(img) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def nms(boxes: Sequence, scores: Sequence[float], thr=0.4) -> List[int]:
    order = sorted(range(len(boxes)), key=lambda i: -scores[i])
    keep = []
    for i in order:
        if all(iou(boxes[i], boxes[j]) < thr for j in keep):
            keep.append(i)
    return keep


def grid_boxes(W, H, scales=(0.3, 0.5, 0.75), overlap=0.33):
    """Multi-scale square windows covering the frame (used for detector-free search)."""
    out = []
    m = min(W, H)
    for s in scales:
        side = int(m * s)
        stride = max(1, int(side * (1 - overlap)))
        nx = max(1, math.ceil((W - side) / stride) + 1)
        ny = max(1, math.ceil((H - side) / stride) + 1)
        xs = np.linspace(0, W - side, nx).astype(int)
        ys = np.linspace(0, H - side, ny).astype(int)
        for y in ys:
            for x in xs:
                out.append((int(x), int(y), int(x + side), int(y + side)))
    return out


def center_box(W, H, frac=0.6):
    s = frac * min(W, H)
    cx, cy = W / 2.0, H / 2.0
    return (cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2)
