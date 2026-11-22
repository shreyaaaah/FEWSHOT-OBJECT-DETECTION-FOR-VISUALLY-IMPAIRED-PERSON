"""Live camera assistant.

    python -m app.live                       # webcam 0
    python -m app.live --source clip.mp4     # replay a video (handy for demos / testing)
    python -m app.live --depth --yoloworld   # metric-anchored depth + open-vocab detector

Keys (focus the video window):
    e  teach a new object   (asks for a name, records ~6 s while you rotate it)
    v  voice command        (microphone, or type it in the terminal)
    i  what is this?        s  what is around me?
    m  mute / unmute        q  quit
While running, the assistant continuously recognises the object you hold and announces it
when the (temporally smoothed) answer changes.
"""
import argparse
import time

import cv2

from src import spatial
from src.audio import beep
from src.factory import build_system, save_memory
from src.smoothing import VoteSmoother
from src.voice import get_command, parse_command


class LiveApp:
    def __init__(self, system, stride=6, record_seconds=6.0, use_mic=True):
        self.s = system
        self.stride = stride
        self.record_seconds = record_seconds
        self.use_mic = use_mic
        self.smoother = VoteSmoother(window=7, min_votes=3)
        self.muted = False
        self.n = 0
        self.overlay = ""

    def say(self, text, **kw):
        self.overlay = text
        if not self.muted:
            self.s.speaker.say(text, **kw)

    # ------------------------------------------------------------ continuous
    def passive(self, frame):
        """Called every frame; runs recognition every `stride` frames."""
        self.n += 1
        if self.n % self.stride or len(self.s.mem) == 0:
            return
        o = self.s.identify_held(frame)
        name = o.match.name if (o.match is not None and o.match.level in ("high", "medium")) else None
        stable, changed = self.smoother.update(name, o.match.prob if o.match else 0.1)
        if changed and stable:
            from src.phrases import identity_phrase
            self.say(identity_phrase(o.match, o.label), key=f"id:{stable}")

    # ------------------------------------------------------------- commands
    def command(self, cap, text):
        it = parse_command(text)
        if it.kind == "enroll":
            return self.enroll(cap, it.target)
        ok, frame = cap.read()
        if not ok:
            return self.say("I cannot see anything.", force=True)
        self.say(self.s.respond(frame, text), force=True, urgent=True)

    def enroll(self, cap, name=None):
        if not name:
            self.say("What is this object called?", force=True, urgent=True)
            time.sleep(1.5)
            name = get_command("Object name> ", self.use_mic).strip()
            if not name:
                return self.say("I did not catch a name.", force=True)
        self.say(f"Okay. Hold your {name} in front of the camera and turn it slowly.", force=True, urgent=True)
        time.sleep(3.5)
        beep("start")
        frames, t0, last_hint = [], time.time(), 0.0
        while time.time() - t0 < self.record_seconds:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
            if time.time() - last_hint > 2.0:                # framing guidance while recording
                last_hint = time.time()
                box, _, src = self.s._locate_held_object(f)
                H, W = f.shape[:2]
                hint = spatial.framing_hint(box, W, H) if src == "detector" else None
                if hint:
                    self.say(hint, key="hint")
            cv2.imshow("Sahayak", f)
            cv2.waitKey(1)
        beep("ok")
        try:
            rep = self.s.enroll_from_frames(name, frames)
        except ValueError as e:
            beep("err")
            return self.say("That recording was too blurry. Please try again.", force=True)
        save_memory(self.s)
        self.smoother.reset()
        self.say(rep.speech(), force=True, urgent=True)

    # ------------------------------------------------------------------ loop
    def run(self, source):
        cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
        if not cap.isOpened():
            raise SystemExit(f"cannot open video source {source}")
        self.say("Ready. Press E to teach me an object, V to ask me something.", force=True)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            self.passive(frame)
            vis = frame.copy()
            if self.overlay:
                cv2.putText(vis, self.overlay[:80], (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Sahayak", vis)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            elif k == ord("e"):
                self.enroll(cap)
            elif k == ord("v"):
                self.say("Listening.", force=True, urgent=True)
                self.command(cap, get_command("Command> ", self.use_mic))
            elif k == ord("i"):
                self.command(cap, "what is this")
            elif k == ord("s"):
                self.command(cap, "what is around me")
            elif k == ord("m"):
                self.muted = not self.muted
        cap.release()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--memory-dir", default=None)
    ap.add_argument("--depth", action="store_true")
    ap.add_argument("--no-detector", action="store_true")
    ap.add_argument("--no-speak", action="store_true")
    ap.add_argument("--no-mic", action="store_true")
    ap.add_argument("--frozen", action="store_true")
    a = ap.parse_args()
    system = build_system(ckpt=a.ckpt, memory_dir=a.memory_dir, speak=not a.no_speak,
                          use_detector=not a.no_detector, frozen_only=a.frozen,
                          use_depth=True if a.depth else None)
    LiveApp(system, use_mic=not a.no_mic).run(a.source)


if __name__ == "__main__":
    main()
