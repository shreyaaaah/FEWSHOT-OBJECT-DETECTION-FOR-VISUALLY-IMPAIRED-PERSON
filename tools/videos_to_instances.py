"""Turn your OWN recordings into the instance-folder layout.

Name videos  <object>__<environment>.mp4   e.g.  keys__kitchen.mp4, keys__bedroom.mp4
(all videos of one object end up in one instance; the environment becomes the 'video'
name so cross-environment evaluation is possible).  Use --user to tag the person.

    python tools/videos_to_instances.py --videos my_videos --out data/own --user me --frames 40
"""
import argparse
import os

import cv2

from src.frames import read_video_frames, select_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--user", default="me")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--keep", type=int, default=30)
    ap.add_argument("--min-sharp", type=float, default=30.0)
    a = ap.parse_args()
    n = 0
    for f in sorted(os.listdir(a.videos)):
        stem, ext = os.path.splitext(f)
        if ext.lower() not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
            continue
        obj, _, env = stem.partition("__")
        frames = read_video_frames(os.path.join(a.videos, f), a.frames)
        idx, _ = select_frames(frames, a.keep, a.min_sharp)
        d = os.path.join(a.out, f"{a.user}__{obj}")
        os.makedirs(d, exist_ok=True)
        for k, i in enumerate(idx):
            cv2.imwrite(os.path.join(d, f"{(env or 'v')}_{k:03d}.jpg"), frames[i])
            n += 1
        print(f"{f}: kept {len(idx)}/{len(frames)} frames -> {d}")
    print("total frames:", n)


if __name__ == "__main__":
    main()
