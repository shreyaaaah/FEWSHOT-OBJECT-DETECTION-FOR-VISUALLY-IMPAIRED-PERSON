"""Flatten an ORBIT-style frame tree into the `<user>__<object>/` layout used here.

ORBIT stores frames as (approximately)
    <root>/[train|validation|test]/<P123>/<object>/<clean|clutter>/<video>/*.jpg
This script is deliberately tolerant of the exact nesting: for every directory that
directly contains images it finds
    user   = first path component matching  P\\d+   (else the first component)
    object = the component right after the user
    kind   = 'clean' / 'clutter' if either word appears in the path, else 'any'
and writes  <out>/<user>__<object>/<kind>_<video>_<frame>.jpg  (every N-th frame).
The `clean_` / `clutter_` prefixes let src.eval.episodic --cross-video enrol from clean
videos and test on cluttered ones, which is ORBIT's own protocol.

NOTE: written from the dataset description, not tested against the real download.
Run with --dry-run first and check that users/objects look right.

    python tools/prepare_orbit.py --src /path/to/orbit --out data/orbit_instances --every 5 --max-side 320
"""
import argparse
import os
import re
import shutil

import cv2

IMG = {".jpg", ".jpeg", ".png"}
SPLIT_NAMES = {"train", "validation", "val", "test"}


def parse(rel_parts):
    parts = [p for p in rel_parts if p.lower() not in SPLIT_NAMES]
    user_i = next((i for i, p in enumerate(parts) if re.fullmatch(r"P\d+", p)), 0)
    if user_i + 1 >= len(parts):
        return None
    user, obj = parts[user_i], parts[user_i + 1]
    low = [p.lower() for p in parts]
    kind = "clean" if "clean" in low else ("clutter" if "clutter" in low else "any")
    video = parts[-1] if len(parts) > user_i + 2 else "v"
    return user, obj, kind, video


def safe(s):
    return re.sub(r"[^A-Za-z0-9\-]+", "-", s).strip("-")[:40]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--max-side", type=int, default=0, help="downscale frames (0 = keep)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    seen, n_files = {}, 0
    for dirpath, _, files in os.walk(a.src):
        imgs = sorted(f for f in files if os.path.splitext(f)[1].lower() in IMG)
        if not imgs:
            continue
        rel = os.path.relpath(dirpath, a.src).split(os.sep)
        info = parse(rel)
        if info is None:
            continue
        user, obj, kind, video = info
        inst = f"{safe(user)}__{safe(obj)}"
        seen.setdefault(inst, set()).add(kind)
        if a.dry_run:
            continue
        dst = os.path.join(a.out, inst)
        os.makedirs(dst, exist_ok=True)
        for j, f in enumerate(imgs[::max(1, a.every)]):
            name = f"{kind}_{safe(video)}_{j:04d}.jpg"
            src = os.path.join(dirpath, f)
            if a.max_side:
                im = cv2.imread(src)
                if im is None:
                    continue
                h, w = im.shape[:2]
                k = a.max_side / max(h, w)
                if k < 1:
                    im = cv2.resize(im, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(dst, name), im)
            else:
                shutil.copy(src, os.path.join(dst, os.path.splitext(name)[0] + os.path.splitext(f)[1]))
            n_files += 1
    users = {i.split("__")[0] for i in seen}
    print(f"found {len(seen)} instances from {len(users)} users; wrote {n_files} frames"
          + (" (dry run)" if a.dry_run else ""))
    for i in sorted(seen)[:8]:
        print("  ", i, sorted(seen[i]))


if __name__ == "__main__":
    main()
