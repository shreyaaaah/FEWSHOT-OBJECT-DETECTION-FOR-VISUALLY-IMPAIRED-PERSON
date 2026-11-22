"""Sahayak LIVE demo - one file, nothing else needed.
Shows every object the camera sees with: name, confidence %, direction, distance.

Run:   python live.py
Keys (click the video window first):  S = speak what is in view,  A = auto-speak on/off,  Q = quit
Options: --source 1 (other camera)   --hfov 65 (camera field of view)   --conf 0.3
"""
import argparse
import math
import sys
import time

import cv2
from ultralytics import YOLO
from src.tts import Voice

# typical real-world size (longest side, metres) -> used for distance
SIZES = {
    "bottle": 0.24, "cup": 0.11, "wine glass": 0.20, "cell phone": 0.15, "laptop": 0.33,
    "remote": 0.18, "book": 0.24, "scissors": 0.18, "mouse": 0.11, "keyboard": 0.45,
    "chair": 0.90, "backpack": 0.45, "handbag": 0.30, "umbrella": 0.90, "clock": 0.28,
    "vase": 0.28, "bowl": 0.16, "toothbrush": 0.19, "fork": 0.19, "knife": 0.22,
    "spoon": 0.18, "banana": 0.19, "apple": 0.08, "orange": 0.08, "tv": 0.80,
    "couch": 1.90, "bed": 2.0, "dining table": 1.4, "suitcase": 0.65, "potted plant": 0.40,
    "person": 1.7, "stop sign": 0.75, "bicycle": 1.7, "bench": 1.5, "sports ball": 0.22,
    "refrigerator": 1.8, "microwave": 0.50, "oven": 0.60, "sink": 0.50, "toilet": 0.70,
    "teddy bear": 0.35, "traffic light": 0.90, "fire hydrant": 0.75,
}
BINS = [(0.45, "within arm's reach"), (0.80, "about half a metre away"),
        (1.40, "about one metre away"), (2.45, "about two metres away"),
        (3.50, "about three metres away")]
COLORS = {"high": (60, 180, 75), "medium": (0, 165, 255), "low": (60, 60, 230)}  # BGR


def level(conf):
    return "high" if conf >= 0.6 else ("medium" if conf >= 0.4 else "low")


def dist_phrase(d, reliable):
    text = next((t for edge, t in BINS if d < edge), "more than three metres away")
    if reliable:
        return text
    return "roughly " + text[6:] if text.startswith("about ") else "probably " + text


def side_phrase(angle, hfov):
    frac = angle / (hfov / 2.0)
    side = "right" if frac > 0 else "left"
    if abs(frac) < 0.15:
        return "straight ahead"
    return f"slightly to your {side}" if abs(frac) < 0.5 else f"to your {side}"


def analyze(boxes, W, H, hfov):
    """boxes: list of (x1, y1, x2, y2, label, conf) -> list of dicts, nearest first."""
    f = (W / 2.0) / math.tan(math.radians(hfov) / 2.0)
    items = []
    for x1, y1, x2, y2, label, conf in boxes:
        cx = (x1 + x2) / 2
        ang = math.degrees(math.atan((cx - W / 2) / f))
        dist, reliable = None, False
        real = SIZES.get(label)
        if real:
            longest = max(x2 - x1, y2 - y1)
            dist = f * real / max(longest, 1)
            reliable = not (x1 <= 2 or y1 <= 2 or x2 >= W - 2 or y2 >= H - 2)
            if label == "person" and (y2 - y1) / max(x2 - x1, 1) < 1.8:
                dist, reliable = None, False          # cropped / seated person: size cue invalid
        items.append(dict(label=label, conf=conf, box=(x1, y1, x2, y2), side=side_phrase(ang, hfov),
                          dist=dist, reliable=reliable, level=level(conf)))
    items.sort(key=lambda i: (i["dist"] is None, i["dist"] or 0, -i["conf"]))
    return items


def one(it):
    art = "an " if it["label"][0] in "aeiou" else "a "
    t = ("maybe " if it["level"] == "low" else "") + art + it["label"] + " " + it["side"]
    if it["dist"] is not None:
        t += ", " + dist_phrase(it["dist"], it["reliable"])
    return t


def summary(items):
    if not items:
        return "I do not see any objects I recognise."
    if len(items) == 1:
        return "I can see " + one(items[0]) + "."
    counts = {}
    for it in items:
        if it["level"] != "low":
            counts[it["label"]] = counts.get(it["label"], 0) + 1
    parts = []
    for label, n in sorted(counts.items(), key=lambda t: -t[1]):
        if n == 1:
            parts.append(("an " if label[0] in "aeiou" else "a ") + label)
        else:
            parts.append(f"{n} " + ("people" if label == "person" else label + "s"))
    head = ("I can see " + (", ".join(parts[:-1]) + " and " + parts[-1] if len(parts) > 1 else parts[0]) + "."
            if parts else "I am not sure about anything in view.")
    shown = items[:4]
    lead = "Nearest first: " if any(i["dist"] is not None for i in shown) else "Positions: "
    return head + " " + lead + "; ".join(one(i) for i in shown) + "."


def overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def draw(img, items):
    out = img.copy()
    H, W = out.shape[:2]
    fs = max(0.5, min(H, W) / 1100)
    for x in (W // 3, 2 * W // 3):                      # left / centre / right zone lines
        for y in range(0, H, 24):
            cv2.line(out, (x, y), (x, min(H, y + 10)), (255, 255, 255), 1)
    for name, xc in (("LEFT", W // 6), ("CENTRE", W // 2), ("RIGHT", 5 * W // 6)):
        cv2.putText(out, name, (xc - 30, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    for it in items:
        x1, y1, x2, y2 = [int(v) for v in it["box"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), COLORS[it["level"]], 2)
    placed = []
    for it in sorted(items, key=lambda i: -i["conf"]):
        x1, y1, x2, y2 = [int(v) for v in it["box"]]
        d = "" if it["dist"] is None else f" | {'~' if not it['reliable'] else ''}{it['dist']:.1f} m"
        lines = [f"{it['label']} {it['conf'] * 100:.0f}%{d}", it["side"]]
        sz = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, fs, 1) for t in lines]
        lh = max(s[0][1] + s[1] for s in sz) + 5
        bw, bh = max(s[0][0] for s in sz) + 8, lh * 2 + 2
        x = int(min(max(0, x1), max(0, W - bw)))
        y = y1 - bh if y1 - bh >= 0 else y1
        for _ in range(12):                             # move label down if it covers another
            if not any(overlap((x, y, x + bw, y + bh), p) for p in placed):
                break
            y = min(y + bh, H - bh)
        y = int(min(max(0, y), H - bh))
        placed.append((x, y, x + bw, y + bh))
        cv2.rectangle(out, (x, y), (x + bw, y + bh), COLORS[it["level"]], -1)
        for k, t in enumerate(lines):
            cv2.putText(out, t, (x + 4, y + (k + 1) * lh - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="camera number (0, 1, ...) or a video file")
    ap.add_argument("--weights", default="yolo26n.pt")
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--hfov", type=float, default=65.0)
    ap.add_argument("--no-speak", action="store_true")
    ap.add_argument("--test-voice", action="store_true")
    a = ap.parse_args()

    if a.test_voice:
        voice = Voice()
        voice.wait_ready()
        done = voice.speak("Sahayak voice test")
        done.wait()
        print("Voice backend:", voice.backend)
        voice.close()
        return

    model = YOLO(a.weights)
    voice = None if a.no_speak else Voice()
    src = int(a.source) if a.source.isdigit() else a.source
    cap = (cv2.VideoCapture(src, cv2.CAP_DSHOW) if isinstance(src, int) and sys.platform.startswith("win")
           else cv2.VideoCapture(src))
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        sys.exit("Cannot open the camera. Close other apps using it, or try:  python live.py --source 1")

    fps, last_t, auto, last_auto, last_text = 0.0, time.time(), False, 0.0, ""
    print("Running. Click the video window, then press S (speak), A (auto-speak), Q (quit).")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]
        res = model.predict(frame, conf=a.conf, verbose=False, agnostic_nms=True)[0]
        boxes = []
        for b, c, k in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(),
                           res.boxes.cls.cpu().numpy().astype(int)):
            boxes.append((float(b[0]), float(b[1]), float(b[2]), float(b[3]), res.names[int(k)], float(c)))
        items = analyze(boxes, W, H, a.hfov)
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - last_t, 1e-3) if fps else 1.0 / max(now - last_t, 1e-3)
        last_t = now
        vis = draw(frame, items)
        cv2.putText(vis, f"{fps:.1f} FPS | {len(items)} objects | S=speak A=auto Q=quit"
                    + (" [AUTO]" if auto else ""), (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Sahayak live", vis)
        key = cv2.waitKey(1) & 0xFF
        text = summary(items)
        if key == ord("q"):
            break
        if key == ord("s"):
            print("SPOKEN:", text)
            voice and voice.speak(text)
        if key == ord("a"):
            auto = not auto
            print("auto-speak", "ON" if auto else "OFF")
        if auto and now - last_auto > 6 and text != last_text:
            last_auto, last_text = now, text
            print("SPOKEN:", text)
            voice and voice.speak(text)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
