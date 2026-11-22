"""Augmentations / degradations that imitate blind-user footage
(motion blur, bad light, partial occlusion, off-centre crops)."""
import cv2
import numpy as np


def motion_blur(img, k=9, angle=0.0):
    k = max(3, int(k) | 1)
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    s = kernel.sum()
    kernel = kernel / s if s > 0 else kernel
    return cv2.filter2D(img, -1, kernel)


def gaussian_blur(img, sigma):
    if sigma <= 0:
        return img
    k = int(sigma * 6) | 1
    return cv2.GaussianBlur(img, (max(3, k), max(3, k)), sigma)


def adjust_light(img, gamma=1.0, gain=1.0):
    x = (img.astype(np.float32) / 255.0) ** gamma * gain
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def occlude(img, frac, rng=None):
    """Black out a random rectangle covering ~frac of the image area."""
    if frac <= 0:
        return img
    rng = rng or np.random.default_rng()
    h, w = img.shape[:2]
    area = frac * h * w
    ar = rng.uniform(0.5, 2.0)
    bw = int(min(w, np.sqrt(area * ar)))
    bh = int(min(h, area / max(bw, 1)))
    x = int(rng.integers(0, max(1, w - bw + 1)))
    y = int(rng.integers(0, max(1, h - bh + 1)))
    out = img.copy()
    out[y:y + bh, x:x + bw] = 0
    return out


def augment_crop(img, rng=None):
    """One random enrollment-time augmentation of an object crop."""
    rng = rng or np.random.default_rng()
    out = img.copy()
    h, w = out.shape[:2]
    # random rescale / re-crop (simulates different framing distance)
    s = rng.uniform(0.75, 1.0)
    ch, cw = int(h * s), int(w * s)
    y0 = int(rng.integers(0, h - ch + 1))
    x0 = int(rng.integers(0, w - cw + 1))
    out = out[y0:y0 + ch, x0:x0 + cw]
    # small rotation
    ang = rng.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((out.shape[1] / 2, out.shape[0] / 2), ang, 1.0)
    out = cv2.warpAffine(out, M, (out.shape[1], out.shape[0]),
                         borderMode=cv2.BORDER_REFLECT)
    if rng.random() < 0.5:
        out = out[:, ::-1]
    out = adjust_light(out, gamma=float(rng.uniform(0.7, 1.4)),
                       gain=float(rng.uniform(0.7, 1.2)))
    r = rng.random()
    if r < 0.3:
        out = gaussian_blur(out, float(rng.uniform(0.8, 2.5)))
    elif r < 0.55:
        out = motion_blur(out, int(rng.integers(5, 13)), float(rng.uniform(0, 180)))
    return np.ascontiguousarray(out)


def random_background_crops(frame, avoid_box, n=2, rng=None, min_side=0.2, max_side=0.45):
    """Random patches that do not overlap the object box -> impostor/background samples."""
    rng = rng or np.random.default_rng()
    H, W = frame.shape[:2]
    out = []
    tries = 0
    while len(out) < n and tries < 30:
        tries += 1
        s = int(min(H, W) * rng.uniform(min_side, max_side))
        x = int(rng.integers(0, max(1, W - s)))
        y = int(rng.integers(0, max(1, H - s)))
        b = (x, y, x + s, y + s)
        ix = max(0, min(b[2], avoid_box[2]) - max(b[0], avoid_box[0]))
        iy = max(0, min(b[3], avoid_box[3]) - max(b[1], avoid_box[1]))
        if ix * iy > 0.1 * s * s:
            continue
        out.append(frame[y:y + s, x:x + s].copy())
    return out
