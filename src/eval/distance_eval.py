"""Distance-estimation evaluation against tape-measured ground truth.

Create a CSV (one row per photo) while collecting your own data:

    image,box_x1,box_y1,box_x2,box_y2,category,true_m,real_size_cm
    img001.jpg,210,150,330,400,bottle,1.00,
    img002.jpg,100,180,170,260,keys,0.60,10
    ...
(`real_size_cm` optional; if empty the COCO size table for `category` is used.)

    python -m src.eval.distance_eval --csv data/distance/labels.csv --images data/distance \
        --hfov 65 [--depth]

Reports MAE, median abs. error, relative error, and the accuracy of the SPOKEN distance
bin ("within arm's reach", "about one metre", ...) which is what the user actually hears.
`--depth` additionally evaluates the fused (size + Depth-Anything) estimate.
"""
import argparse
import csv
import os

import cv2
import numpy as np

from ..config import ROOT
from ..depth import SIZE_TABLE, RelativeDepth, box_depth_value, fuse_distances, size_distance
from ..spatial import distance_bin, focal_px
from .common import write_report


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def stats(est, truth):
    est, truth = np.array(est, float), np.array(truth, float)
    ok = ~np.isnan(est)
    if not ok.any():
        return {}
    err = np.abs(est[ok] - truth[ok])
    bins_ok = [distance_bin(e) == distance_bin(t) for e, t in zip(est[ok], truth[ok])]
    within1 = [abs(distance_bin(e) - distance_bin(t)) <= 1 for e, t in zip(est[ok], truth[ok])]
    return {"n": int(ok.sum()), "mae_m": float(err.mean()), "median_abs_m": float(np.median(err)),
            "mean_rel_err": float(np.mean(err / truth[ok])), "bin_exact": float(np.mean(bins_ok)),
            "bin_within_1": float(np.mean(within1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--hfov", type=float, default=65.0)
    ap.add_argument("--depth", action="store_true")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "reports"))
    a = ap.parse_args()
    rows = read_rows(a.csv)
    rd = RelativeDepth() if a.depth else None
    size_est, fused_est, truth = [], [], []
    for r in rows:
        img = cv2.imread(os.path.join(a.images, r["image"]))
        if img is None:
            print("skip (unreadable):", r["image"])
            continue
        H, W = img.shape[:2]
        f = focal_px(W, a.hfov)
        box = tuple(float(r[k]) for k in ("box_x1", "box_y1", "box_x2", "box_y2"))
        real = float(r["real_size_cm"]) / 100 if r.get("real_size_cm") else SIZE_TABLE.get(r["category"])
        d, ok = size_distance(box, real, f, W, H) if real else (None, False)
        size_est.append(d if d is not None else np.nan)
        truth.append(float(r["true_m"]))
        if rd is not None:
            rv = box_depth_value(rd(img), box)
            out, _ = fuse_distances([d], [ok], [rv], min_anchors=1) if d else ([None], None)
            fused_est.append(out[0] if out[0] is not None else np.nan)
    res = {"size_only": stats(size_est, truth)}
    if rd is not None:
        res["fused"] = stats(fused_est, truth)
    md = "# Distance evaluation\n\n| method | n | MAE (m) | median abs (m) | mean rel. err | exact spoken-bin | within ±1 bin |\n|---|---|---|---|---|---|---|\n"
    for k, v in res.items():
        if v:
            md += (f"| {k} | {v['n']} | {v['mae_m']:.2f} | {v['median_abs_m']:.2f} | "
                   f"{100 * v['mean_rel_err']:.0f}% | {100 * v['bin_exact']:.0f}% | {100 * v['bin_within_1']:.0f}% |\n")
    print(md)
    write_report(os.path.join(a.out_dir, "distance"), res, md)


if __name__ == "__main__":
    main()
