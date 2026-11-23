"""Regression test: ORBIT instance names contain upper-case letters ('P189__keys'), but PersonalMemory
stores lower-cased names. The episodic evaluation once compared them raw, so the calibrated memory's
correct-ID rate was exactly 0.0 for every model. It must be > 0 on data with upper-case names."""
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_ours_tpr_is_not_zero_with_uppercase_instance_names(tmp_path):
    root = tmp_path / "inst"
    subprocess.run([sys.executable, os.path.join(REPO, "tools", "make_synthetic.py"), "--out", str(root),
                    "--instances", "24", "--frames", "20"], check=True, cwd=REPO, capture_output=True)
    for d in sorted(os.listdir(root)):                       # rename to ORBIT style: upper-case user + object
        if os.path.isdir(root / d):
            os.rename(root / d, root / ("PUSER" + d[-1] + "__Obj" + d))
    tag = "pytest_names"
    out = subprocess.run([sys.executable, "-m", "src.eval.episodic", "--root", str(root), "--split", "train",
                          "--frozen-backbone", "hsv_hist", "--episodes", "10", "--k-shots", "5", "--tag", tag],
                         check=True, cwd=REPO, capture_output=True, text=True)
    path = os.path.join(root, "reports", f"episodic_{tag}_train.json")
    rep = path if os.path.exists(path) else os.path.join(REPO, "reports", f"episodic_{tag}_train.json")
    try:
        res = json.load(open(rep))["results"]["5"]["methods"]["ours"]["tpr"][0]
    finally:
        for ext in (".json", ".md"):
            f = rep.replace(".json", ext)
            if os.path.exists(f) and "pytest_names" in f:
                os.remove(f)
    assert res > 0.05, f"ours TPR is {res}: name comparison is broken again\n{out.stdout[-400:]}"
