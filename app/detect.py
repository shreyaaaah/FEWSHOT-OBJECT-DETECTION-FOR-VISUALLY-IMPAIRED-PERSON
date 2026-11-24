"""General object detection demo: label + confidence + direction + distance for every
object in a photo, video or live camera.  No teaching, no memory, no GPU needed.

    python -m app.detect --image room.jpg                      # saves room_detected.jpg
    python -m app.detect --image a.jpg b.jpg --out results     # several photos
    python -m app.detect --video clip.mp4 --save clip_out.mp4  # annotate a whole video
    python -m app.detect --live                                # webcam (S = speak summary, Q = quit)

Options: --conf 0.35 (detector threshold)  --hfov 65 (camera horizontal field of view,
degrees; phones are ~65-75)  --no-speak  --weights yolo26n.pt
Distance accuracy depends on the field of view being right and on the object being
fully visible; "~" in front of a distance means the object touches the frame edge.
"""
import argparse
import os
import sys
import time

import cv2

from src.speaker import VoiceSpeaker
from src.detector import Detector
from src.scene import analyze, annotate, summary, table


def process_image(det, path, out_dir, hfov, speaker=None, quiet=False):
    img = cv2.imread(path)
    if img is None:
        print(f"cannot read {path}")
        return None
    H, W = img.shape[:2]
    t = time.time()
    items = analyze(det.detect(img), W, H, hfov)
    ms = (time.time() - t) * 1000
    stem = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(out_dir, exist_ok=True) if out_dir else None
    out_path = os.path.join(out_dir or os.path.dirname(path) or ".", f"{stem}_detected.jpg")
    cv2.imwrite(out_path, annotate(img, items))
    if not quiet:
        print(f"\n=== {path}  ({W}x{H}, detection+analysis {ms:.0f} ms, assumed field of view {hfov:.0f} deg)")
        print(table(items) if items else "no objects found")
        print("\nSPOKEN:", summary(items))
        print("saved  :", out_path)
    if speaker:
        speaker.say(summary(items), force=True)
    return items


def process_video(det, src, dst, hfov):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W, H = int(cap.get(3)), int(cap.get(4))
    w = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    n = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        w.write(annotate(f, analyze(det.detect(f), W, H, hfov)))
        n += 1
    cap.release()
    w.release()
    print(f"annotated {n} frames -> {dst}")


def open_camera(source):
    """Webcam index (int) or video path.  On Windows the DirectShow backend is more
    reliable for laptop cameras; capture is limited to 640x480 to keep detection fast."""
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW) if sys.platform.startswith("win") \
            else cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return cap
    return cv2.VideoCapture(source)


def run_live(det, hfov, speaker, source=0, auto=0.0, max_frames=None):
    """Real-time loop.  Keys (click the video window first): S = speak what is in view,
    A = toggle auto-speak, Q = quit.  `auto` = seconds between automatic announcements."""
    cap = open_camera(source)
    if not cap.isOpened():
        raise SystemExit("cannot open the camera (is another app using it? try --source 1)")
    fps, last_t, last_auto, last_text, n = 0.0, time.time(), 0.0, "", 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        H, W = f.shape[:2]
        items = analyze(det.detect(f), W, H, hfov)
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - last_t, 1e-3) if fps else 1.0 / max(now - last_t, 1e-3)
        last_t = now
        vis = annotate(f, items)
        cv2.putText(vis, f"{fps:.1f} FPS | {len(items)} objects | S=speak A=auto Q=quit"
                    + (" [AUTO]" if auto else ""), (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Sahayak - live scene", vis)
        k = cv2.waitKey(1) & 0xFF
        text = summary(items)
        if k == ord("q"):
            break
        if k == ord("s"):
            print("SPOKEN:", text)
            if speaker:
                speaker.say(text, force=True)
        if k == ord("a"):
            auto = 0.0 if auto else 6.0
            print("auto-speak", "ON (every 6 s)" if auto else "OFF")
        if auto and now - last_auto >= auto and text != last_text:
            last_auto, last_text = now, text
            print("SPOKEN:", text)
            if speaker:
                speaker.say(text, force=True)
        n += 1
        if max_frames and n >= max_frames:
            break
    cap.release()
    cv2.destroyAllWindows()
    return fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", nargs="+")
    ap.add_argument("--video")
    ap.add_argument("--save", default="detected.mp4")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--source", default="0")
    ap.add_argument("--out", default=None, help="folder for annotated images")
    ap.add_argument("--weights", default="yolo26n.pt")
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--hfov", type=float, default=65.0)
    ap.add_argument("--no-speak", action="store_true")
    ap.add_argument("--auto", type=float, default=0.0, help="live: announce the scene every N seconds")
    a = ap.parse_args()
    det = Detector(a.weights, conf=a.conf, max_det=30)
    spk = None if a.no_speak else VoiceSpeaker(enabled=True, verbose=False)
    if a.image:
        for p in a.image:
            process_image(det, p, a.out, a.hfov, spk)
    elif a.video:
        process_video(det, a.video, a.save, a.hfov)
    elif a.live:
        run_live(det, a.hfov, spk, int(a.source) if a.source.isdigit() else a.source, auto=a.auto)
    else:
        ap.error("give --image, --video or --live")
    if spk:
        time.sleep(0.5)


if __name__ == "__main__":
    main()
