"""Print macro precision/recall/F1 of every curves_*.npz (written by lab.py) at chosen confidence thresholds.
    python tools/operating_points.py lab_out
Read-only analysis of the saved test-split curves; nothing is re-tuned."""
import glob, os, sys
import numpy as np

d = sys.argv[1] if len(sys.argv) > 1 else "lab_out"
for f in sorted(glob.glob(os.path.join(d, "curves_*.npz"))):
    z = np.load(f); idx = [i for i in range(len(z["n_gt"])) if z["n_gt"][i] > 0]
    P, R, F = (z[k][idx].mean(0) for k in ("P", "R", "F1"))
    print(os.path.basename(f))
    for c in (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5):
        k = int(round(c * 1000)); print(f"   conf {c:.2f}: P={P[k]:.3f} R={R[k]:.3f} F1={F[k]:.3f}")
