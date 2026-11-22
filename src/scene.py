"""General scene understanding (no teaching needed):
for every object the detector finds -> label, confidence, direction, distance.

Confidence shown here is the DETECTOR's score (0-1), not a calibrated probability.
Distance comes from a typical real-world size of the class and the camera field of
view (see depth.py), so it is an estimate; objects without a trustworthy size prior
(cars, buses, animals) get direction only.
"""
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from . import depth as depthlib
from . import spatial
from .schema import Det

HIGH, MEDIUM = 0.60, 0.40
FULL_BODY_ASPECT = 1.8     # a standing person's box is >= ~1.8x taller than wide
PLURALS = {"person": "people", "knife": "knives", "sheep": "sheep", "scissors": "pairs of scissors",
           "skis": "skis"}


@dataclass
class SceneItem:
    label: str
    conf: float
    box: tuple
    angle_deg: float          # negative = left of camera axis
    side: str                 # spoken side phrase
    clock: int
    vertical: str
    dist_m: Optional[float]
    reliable: bool            # False if the box touches the frame edge (object cut off)
    level: str                # high | medium | low (detector confidence band)


def confidence_level(conf: float) -> str:
    return "high" if conf >= HIGH else ("medium" if conf >= MEDIUM else "low")


def analyze(dets: List[Det], W: int, H: int, hfov_deg: float = 65.0) -> List[SceneItem]:
    f = spatial.focal_px(W, hfov_deg)
    items = []
    for d in dets:
        cx, cy = spatial.box_center(d.box)
        ang = spatial.horizontal_angle(cx, W, hfov_deg)
        real = depthlib.SIZE_TABLE.get(d.label)
        dist, rel = (depthlib.size_distance(d.box, real, f, W, H) if real else (None, False))
        if d.label == "person" and dist is not None:
            bw, bh = d.box[2] - d.box[0], d.box[3] - d.box[1]
            if bh / max(bw, 1.0) < FULL_BODY_ASPECT:     # cropped / sitting / lying: height cue invalid
                dist, rel = None, False
        items.append(SceneItem(
            label=d.label, conf=d.conf, box=d.box, angle_deg=ang,
            side=spatial.side_phrase(ang, hfov_deg), clock=spatial.clock_hour(ang),
            vertical=spatial.vertical_phrase(cy, H), dist_m=dist,
            reliable=rel, level=confidence_level(d.conf)))
    # nearest first; objects with unknown distance after, strongest detection first
    items.sort(key=lambda it: (it.dist_m is None, it.dist_m if it.dist_m is not None else 0, -it.conf))
    return items


def _count_phrase(label: str, n: int) -> str:
    if n == 1:
        return ("an " if label[:1] in "aeiou" else "a ") + label
    plural = PLURALS.get(label, label + "s")
    return f"{n} {plural}"


def _phrase(it: SceneItem) -> str:
    pre = "maybe " if it.level == "low" else ""
    art = "an " if it.label[:1] in "aeiou" else "a "
    t = f"{pre}{art}{it.label} {it.side}"
    if it.dist_m is not None:
        t += ", " + spatial.distance_phrase(it.dist_m, it.reliable)
    return t


def summary(items: List[SceneItem], max_items: int = 4) -> str:
    """Spoken summary: what is there, then the nearest objects with direction + distance."""
    if not items:
        return "I do not see any objects I recognise."
    if len(items) == 1:
        return "I can see " + _phrase(items[0]) + "."
    confident = [i for i in items if i.level != "low"]
    counts = {}
    for it in confident:
        counts[it.label] = counts.get(it.label, 0) + 1
    parts = [_count_phrase(l, n) for l, n in sorted(counts.items(), key=lambda t: -t[1])]
    if len(parts) > 1:
        s = "I can see " + ", ".join(parts[:-1]) + " and " + parts[-1] + "."
    elif parts:
        s = "I can see " + parts[0] + "."
    else:
        s = "I am not sure about anything in view."
    shown = items[:max_items]
    lead = "Nearest first: " if any(i.dist_m is not None for i in shown) else "Positions: "
    return s + " " + lead + "; ".join(_phrase(i) for i in shown) + "."


def table(items: List[SceneItem]) -> str:
    rows = [f"{'object':<14}{'confidence':>11}  {'direction':<26}{'angle':>7}  {'distance':>10}"]
    rows.append("-" * 72)
    for it in items:
        d = "n/a" if it.dist_m is None else (("~" if not it.reliable else "") + f"{it.dist_m:.1f} m")
        rows.append(f"{it.label:<14}{it.conf * 100:>9.0f} %  {it.side:<26}{it.angle_deg:>+6.0f}°  {d:>10}")
    return "\n".join(rows)


COLORS = {"high": (60, 180, 75), "medium": (0, 165, 255), "low": (60, 60, 230)}   # BGR


def _overlap(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def annotate(img: np.ndarray, items: List[SceneItem]) -> np.ndarray:
    """Boxes + one two-line label per object (name, confidence, distance / direction).
    Labels are kept inside the image and moved so they do not cover each other."""
    out = img.copy()
    H, W = out.shape[:2]
    th = max(2, int(round(min(H, W) / 300)))
    fs = max(0.5, min(H, W) / 1100)
    for x in (W // 3, 2 * W // 3):                       # left / centre / right zones
        for y in range(0, H, 24):
            cv2.line(out, (x, y), (x, min(H, y + 10)), (255, 255, 255), 1)
    for name, xc in (("LEFT", W // 6), ("CENTRE", W // 2), ("RIGHT", 5 * W // 6)):
        cv2.putText(out, name, (xc - int(36 * fs * 1.6), H - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    fs * 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    placed = []
    for it in sorted(items, key=lambda i: i.conf):       # strongest drawn last (on top)
        x1, y1, x2, y2 = [int(v) for v in it.box]
        cv2.rectangle(out, (x1, y1), (x2, y2), COLORS[it.level], th)
    for it in sorted(items, key=lambda i: -i.conf):      # strongest gets the best label spot
        x1, y1, x2, y2 = [int(v) for v in it.box]
        d = "" if it.dist_m is None else f" | {'~' if not it.reliable else ''}{it.dist_m:.1f} m"
        lines = [f"{it.label} {it.conf * 100:.0f}%{d}", it.side]
        sizes = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, fs, 1) for t in lines]
        lh = max(s[0][1] + s[1] for s in sizes) + 5
        bw, bh = max(s[0][0] for s in sizes) + 8, lh * len(lines) + 2
        x = int(min(max(0, x1), max(0, W - bw)))
        y = y1 - bh if y1 - bh >= 0 else y1              # above the box, else inside its top
        for _ in range(12):                              # slide down until nothing is covered
            r = (x, y, x + bw, y + bh)
            if not any(_overlap(r, p) for p in placed):
                break
            y = min(y + bh, H - bh)
        y = int(min(max(0, y), H - bh))
        placed.append((x, y, x + bw, y + bh))
        cv2.rectangle(out, (x, y), (x + bw, y + bh), COLORS[it.level], -1)
        for k, t in enumerate(lines):
            cv2.putText(out, t, (x + 4, y + (k + 1) * lh - 4), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        (0, 0, 0), 1, cv2.LINE_AA)
    return out