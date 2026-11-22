"""Turn pixel boxes into spoken spatial language (direction, clock position,
distance bins, guidance).  Pure Python, no dependencies."""
import math
from typing import List, Optional

# upper edge of each distance bin (metres) and its spoken form
DIST_BINS = [
    (0.45, "within arm's reach"),
    (0.80, "about half a metre away"),
    (1.40, "about one metre away"),
    (2.45, "about two metres away"),
    (3.50, "about three metres away"),
]
FAR_PHRASE = "more than three metres away"


def distance_bin(d: Optional[float]) -> Optional[int]:
    if d is None:
        return None
    for i, (edge, _) in enumerate(DIST_BINS):
        if d < edge:
            return i
    return len(DIST_BINS)


def distance_phrase(d: Optional[float], reliable: bool = True) -> str:
    b = distance_bin(d)
    if b is None:
        return ""
    txt = DIST_BINS[b][1] if b < len(DIST_BINS) else FAR_PHRASE
    if reliable:
        return txt
    return ("roughly " + txt[len("about "):]) if txt.startswith("about ") else "probably " + txt


def focal_px(width: int, hfov_deg: float) -> float:
    return (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


def horizontal_angle(cx: float, width: int, hfov_deg: float) -> float:
    """Degrees, negative = left of the camera axis, positive = right."""
    f = focal_px(width, hfov_deg)
    return math.degrees(math.atan((cx - width / 2.0) / f))


def clock_hour(angle_deg: float) -> int:
    h = int(round(angle_deg / 30.0)) % 12
    return 12 if h == 0 else h


def side_phrase(angle_deg: float, hfov_deg: float) -> str:
    frac = angle_deg / (hfov_deg / 2.0)
    side = "right" if frac > 0 else "left"
    a = abs(frac)
    if a < 0.15:
        return "straight ahead"
    if a < 0.5:
        return f"slightly to your {side}"
    return f"to your {side}"


def vertical_phrase(cy: float, height: int) -> str:
    r = cy / float(height)
    if r < 0.33:
        return "high up"
    if r > 0.66:
        return "low down"
    return ""


def box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def verb_for(subject: str) -> str:
    """'is' / 'are' agreement for spoken names (keys, spectacles, scissors -> are)."""
    last = subject.strip().lower().split()[-1] if subject.strip() else ""
    return "are" if (last.endswith("s") and not last.endswith(("ss", "us", "is"))) else "is"


def describe_location(subject: str, box, W: int, H: int, dist_m: Optional[float],
                      reliable: bool = True, hfov_deg: float = 65.0,
                      use_clock: bool = True) -> str:
    cx, cy = box_center(box)
    ang = horizontal_angle(cx, W, hfov_deg)
    s = f"{subject} {verb_for(subject)} {side_phrase(ang, hfov_deg)}"
    if use_clock and abs(ang) / (hfov_deg / 2.0) >= 0.15:
        s += f", around {clock_hour(ang)} o'clock"
    v = vertical_phrase(cy, H)
    if v:
        s += f", {v}"
    d = distance_phrase(dist_m, reliable)
    if d:
        s += f", {d}"
    return s + "."


def guidance(box, W: int, H: int, dist_m: Optional[float] = None,
             hfov_deg: float = 65.0) -> str:
    """Short action cue that steers the user toward the object."""
    cx, cy = box_center(box)
    ang = horizontal_angle(cx, W, hfov_deg)
    frac = ang / (hfov_deg / 2.0)
    if abs(frac) > 0.25:
        return "Turn left." if frac < 0 else "Turn right."
    r = cy / float(H)
    if r < 0.25:
        return "Tilt up."
    if r > 0.75:
        return "Tilt down."
    if dist_m is not None and dist_m < 0.45:
        return "It is right in front of you. Reach forward."
    return "It is straight ahead. Move forward."


def framing_hint(box, W: int, H: int) -> Optional[str]:
    """Hint to help a blind user frame an object for enrollment/recognition."""
    x1, y1, x2, y2 = box
    area = max(0.0, (x2 - x1) * (y2 - y1)) / float(W * H)
    cx, cy = box_center(box)
    if area < 0.04:
        return "Move the object closer to the camera."
    if area > 0.75:
        return "Hold the object a little farther from the camera."
    if abs(cx - W / 2) > 0.3 * W:
        return "Move the object toward the middle."
    return None


def scene_phrase(items, W: int, H: int, hfov_deg: float = 65.0, max_items: int = 5) -> str:
    """items: list of (display_name, box, dist_m, reliable). Nearest first."""
    if not items:
        return "I do not see any objects I know."
    items = sorted(items, key=lambda t: (t[2] is None, t[2] if t[2] is not None else 0))
    parts = []
    for name, box, d, rel in items[:max_items]:
        cx, cy = box_center(box)
        ang = horizontal_angle(cx, W, hfov_deg)
        p = f"{_with_article(name)} {side_phrase(ang, hfov_deg)}"
        dp = distance_phrase(d, rel)
        if dp:
            p += f", {dp}"
        parts.append(p)
    return "I can see " + "; ".join(parts) + "."


def _with_article(name: str) -> str:
    if name.startswith("your "):
        return name
    return ("an " if name[:1] in "aeiou" else "a ") + name
