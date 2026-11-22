import csv
import subprocess
import sys

import cv2
import numpy as np

from src.spatial import focal_px


def test_distance_eval_recovers_pinhole_truth(tmp_path):
    W, H, hfov = 640, 480, 65.0
    f = focal_px(W, hfov)
    rows = []
    for i, (d, real) in enumerate([(0.4, 0.24), (1.0, 0.24), (2.0, 0.24), (1.0, 0.11), (3.0, 0.9)]):
        L = f * real / d
        cx, cy = 320, 240
        box = (cx - L / 4, cy - L / 2, cx + L / 4, cy + L / 2)      # tall box, longest side = L
        name = f"im{i}.jpg"
        cv2.imwrite(str(tmp_path / name), np.full((H, W, 3), 100, np.uint8))
        cat = {0.24: "bottle", 0.11: "cup", 0.9: "umbrella"}[real]
        rows.append(dict(image=name, box_x1=box[0], box_y1=box[1], box_x2=box[2], box_y2=box[3],
                         category=cat, true_m=d, real_size_cm=""))
    p = tmp_path / "labels.csv"
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    out = subprocess.run([sys.executable, "-m", "src.eval.distance_eval", "--csv", str(p),
                          "--images", str(tmp_path), "--hfov", str(hfov),
                          "--out-dir", str(tmp_path / "rep")],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "| size_only | 5 | 0.00 |" in out.stdout and "100%" in out.stdout
