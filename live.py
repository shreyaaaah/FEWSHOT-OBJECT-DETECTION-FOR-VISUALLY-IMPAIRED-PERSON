"""Sahayak LIVE demo - one file, nothing else needed.
Shows every object the camera sees with: name, confidence %, direction, distance.

Run:   python live.py
It speaks by itself: when the scene changes (and every ~20 s while objects stay in view).
An object is announced only if it was seen in 3 of the last 5 frames (no flicker).
Key (click the video window first):  Q = quit
ASK WHERE AN OBJECT IS: click the black console window, type e.g.  where is my laptop  and press Enter.
It answers like "The laptop is slightly to your right, about one metre away." If the object is not in view it
says so and keeps looking for 25 seconds. (Optional spoken questions: python live.py --mic, needs SpeechRecognition + PyAudio.)
Options: --source 1 (other camera)   --hfov 65 (camera field of view)   --conf 0.30 (lowest confidence shown)
         --speak-conf 0.40 (lowest confidence that is spoken)
"""
import argparse
import math
from collections import deque
import queue
import re
import sys
import threading
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


class Announcer:
    """Decides WHEN to speak. Temporal voting: a label counts only if it was seen in at least
    `votes` of the last `window` frames, so single-frame flicker is never announced, while a
    briefly missed object (1-2 dropped frames) stays announced."""

    def __init__(self, window=5, votes=3, min_gap=4.0, repeat=20.0):
        self.hist = deque(maxlen=window)
        self.votes, self.min_gap, self.repeat = votes, min_gap, repeat
        self.last_sig, self.last_t = None, -1e9

    def stable(self, items):
        self.hist.append(items)
        out = []
        for label in {i["label"] for f in self.hist for i in f}:
            if sum(any(i["label"] == label for i in f) for f in self.hist) >= self.votes:
                latest = next(f for f in reversed(self.hist) if any(i["label"] == label for i in f))
                out += [i for i in latest if i["label"] == label]
        out.sort(key=lambda i: (i["dist"] is None, i["dist"] or 0, -i["conf"]))
        return out

    @staticmethod
    def signature(items):
        sig = {}
        for i in items:                                   # items are nearest first
            zone = "left" if "left" in i["side"] else ("right" if "right" in i["side"] else "ahead")
            sig.setdefault(i["label"], zone)
        return frozenset(sig.items())

    def update(self, items, now, busy=False):
        """Return (text_to_speak_or_None, stable_items)."""
        st = self.stable(items)
        sig = self.signature(st)
        if busy or (not sig and not self.last_sig):
            return None, st
        changed = sig != self.last_sig
        if (changed and now - self.last_t >= self.min_gap) or (sig and now - self.last_t >= self.repeat):
            self.last_sig, self.last_t = sig, now
            return summary(st), st
        return None, st


PERSON_MIN = 0.55          # 'person' is the most common false alarm (wall cracks, shadows): demand more
SYNONYMS = {"phone": "cell phone", "mobile": "cell phone", "cellphone": "cell phone", "smartphone": "cell phone",
            "mug": "cup", "tv": "tv", "television": "tv", "sofa": "couch", "table": "dining table",
            "desk": "dining table", "bag": "backpack", "plant": "potted plant", "teddy": "teddy bear",
            "bike": "bicycle", "computer": "laptop", "notebook computer": "laptop", "glasses": "wine glass",
            "water bottle": "bottle", "man": "person", "woman": "person", "people": "person"}


def keep(label, conf, floor):
    """Is this detection trustworthy enough (>= floor; persons need PERSON_MIN)?"""
    return conf >= (max(floor, PERSON_MIN) if label == "person" else floor)


def find_target(text, names):
    """'where is my laptop' -> 'laptop'. names = every label the detector knows. None if no object is named."""
    t = " " + re.sub(r"[^a-z ]", " ", text.lower()) + " "
    options = {n.lower(): n for n in names}
    options.update({k: v for k, v in SYNONYMS.items() if v in options.values()})
    for key in sorted(options, key=len, reverse=True):       # longest match first ('cell phone' before 'phone')
        if re.search(r"\b" + re.escape(key) + r"s?\b", t):
            return options[key]
    return None


def answer(label, items):
    """Spoken answer for 'where is my <label>' from the current items (nearest first), or None if not visible."""
    cand = [i for i in items if i["label"] == label]
    if not cand:
        return None
    it = cand[0]
    text = f"The {label} is {it['side']}"
    if it["dist"] is not None:
        text += ", " + dist_phrase(it["dist"], it["reliable"])
    text += "."
    if len(cand) > 1:
        text = f"I can see {len(cand)} {label}s. The nearest one: " + text[4:]
    return text


def start_question_listeners(q, use_mic):
    def typed():
        for line in sys.stdin:
            if line.strip():
                q.put(line.strip())
    threading.Thread(target=typed, daemon=True).start()
    if not use_mic:
        return
    try:
        import speech_recognition as sr
    except ImportError:
        print("--mic needs:  pip install SpeechRecognition pyaudio   (typed questions still work)")
        return

    def listen():
        rec, mic = sr.Recognizer(), sr.Microphone()
        with mic as src:
            rec.adjust_for_ambient_noise(src, 1)
        while True:
            try:
                with mic as src:
                    audio = rec.listen(src, phrase_time_limit=5)
                q.put(rec.recognize_google(audio))
            except Exception:
                continue
    threading.Thread(target=listen, daemon=True).start()


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
    ap.add_argument("--conf", type=float, default=0.30, help="lowest confidence that is shown")
    ap.add_argument("--speak-conf", type=float, default=0.40, help="lowest confidence that is spoken")
    ap.add_argument("--mic", action="store_true", help="also listen to spoken questions (needs SpeechRecognition)")
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

    fps, last_t = 0.0, time.time()
    ann, pending = Announcer(), None
    questions, target, target_until = queue.Queue(), None, 0.0
    start_question_listeners(questions, a.mic)
    names = list(model.names.values())
    if voice:
        voice.wait_ready(10)
        print("Voice backend:", voice.backend)
        voice.speak("Sahayak is ready.")
    print("Running. It speaks automatically. Click the video window and press Q to quit.")
    print('Ask in this console, e.g.:  where is my laptop   (then Enter)')
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
        shown = analyze([b for b in boxes if keep(b[4], b[5], a.conf)], W, H, a.hfov)
        items = [i for i in shown if keep(i["label"], i["conf"], a.speak_conf)]      # only these are spoken
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - last_t, 1e-3) if fps else 1.0 / max(now - last_t, 1e-3)
        last_t = now
        vis = draw(frame, shown)
        text, _ = ann.update(items, now, busy=pending is not None and not pending.is_set())
        reply = None
        while not questions.empty():
            q = questions.get()
            found = find_target(q, names)
            print("QUESTION:", q, "->", found)
            if found is None:
                reply = "Sorry, I do not know that object. Try for example: where is my laptop."
                target = None
            else:
                reply = answer(found, items)
                if reply is None:
                    target, target_until = found, now + 25
                    reply = f"I cannot see a {found} right now. Turn slowly and I will keep looking."
                else:
                    target = None
        if reply is None and target:
            hit = answer(target, items)
            if hit:
                reply, target = "Found it. " + hit, None
            elif now > target_until:
                reply, target = f"I could not find the {target}.", None
        if reply:
            text = reply
            ann.last_t = now                      # do not talk over the answer
        cv2.putText(vis, f"{fps:.1f} FPS | {len(items)} objects | Q=quit", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Sahayak live", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if text:
            print("SPOKEN:", text)
            if voice:
                pending = voice.speak(text)
    cap.release()
    cv2.destroyAllWindows()
    if voice:
        voice.close()


if __name__ == "__main__":
    main()
