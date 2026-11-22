"""Command-line front end (works with files, so no camera is needed).

    python -m app.cli enroll --name keys --video clips/keys.mp4 --size-cm 10
    python -m app.cli enroll --name "medicine box" --video clips/med.mp4      # auto 'critical'
    python -m app.cli ask --image room.jpg --text "where is my keys"
    python -m app.cli ask --image hand.jpg --text "what is this"
    python -m app.cli list
    python -m app.cli forget --name keys
Common flags:  --ckpt PATH  --no-detector  --depth  --no-speak  --memory-dir DIR
"""
import argparse
import sys

import cv2

from src.factory import build_system, save_memory
from src.frames import read_video_frames


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["enroll", "ask", "list", "forget"])
    ap.add_argument("--name")
    ap.add_argument("--video")
    ap.add_argument("--image")
    ap.add_argument("--text")
    ap.add_argument("--size-cm", type=float, default=None, help="longest side of the object (improves distance)")
    ap.add_argument("--critical", action="store_true", help="safety-critical object: stricter threshold")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--memory-dir", default=None)
    ap.add_argument("--no-detector", action="store_true")
    ap.add_argument("--depth", action="store_true")
    ap.add_argument("--no-speak", action="store_true")
    ap.add_argument("--frozen", action="store_true", help="ignore fine-tuned checkpoint")
    a = ap.parse_args(argv)

    sysm = build_system(ckpt=a.ckpt, memory_dir=a.memory_dir, speak=not a.no_speak,
                        use_detector=not a.no_detector, frozen_only=a.frozen,
                        use_depth=True if a.depth else None)
    say = lambda t: sysm.speaker.say(t, force=True)
    if a.cmd == "enroll":
        if not (a.name and a.video):
            sys.exit("enroll needs --name and --video")
        frames = read_video_frames(a.video, sysm.n_sample)
        rep = sysm.enroll_from_frames(a.name, frames, size_cm=a.size_cm, critical=a.critical or None)
        save_memory(sysm)
        print(f"threshold={rep.threshold:.2f} sigma={rep.sigma:.3f} prototypes={rep.n_prototypes} "
              f"frames={rep.n_frames} warnings={rep.warnings}")
        say(rep.speech())
    elif a.cmd == "ask":
        if not (a.image and a.text):
            sys.exit("ask needs --image and --text")
        frame = cv2.imread(a.image)
        if frame is None:
            sys.exit(f"cannot read image {a.image}")
        say(sysm.respond(frame, a.text))
    elif a.cmd == "list":
        say(sysm.respond(None, "what do you know"))
    elif a.cmd == "forget":
        if not a.name:
            sys.exit("forget needs --name")
        say(sysm.respond(None, f"forget {a.name}"))
        save_memory(sysm)
    import time
    time.sleep(0.3)           # let the speech thread print / speak before exit


if __name__ == "__main__":
    main()
