"""Why does the calibrated memory accept (almost) nothing?  Read-only diagnosis, no training, no tuning on test.

For a model, runs a few episodes like src.eval.episodic and prints, per K:
  * the thresholds the memory chose and the score ranges of genuine queries (clutter frames of the SAME object)
    and of unknown objects (best score against any enrolled object),
  * TPR/FAR of the memory as it is, and of two simple alternatives:
      - impostor rule : threshold = 99th percentile of scores of the OTHER enrolled objects + 0.01
      - best global t : the single cosine threshold giving FAR <= 5% (shows the headroom that exists at all)
Use --split val (never test) when choosing anything from this output.

    python tools/diagnose_memory.py --root /kaggle/working/orbit_instances --split val --frozen-backbone vit_small_patch14_dinov2.lvd142m
    python tools/diagnose_memory.py --root ... --split val --ckpt checkpoints/embedder_C_partial_open.pt
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.eval.common import embed_files, list_instance_files, load_model_for_eval
from src.eval.episodic import choose_support
from src.memory import PersonalMemory, bank_score, l2n, norm_name
from src.train.datasets import ensure_splits


def q(x, ps=(5, 25, 50, 75, 95)):
    return " ".join(f"{np.percentile(x, p):.2f}" for p in ps) if len(x) else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--frozen-backbone", default=None)
    ap.add_argument("--k-shots", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--n-way", type=int, default=5)
    ap.add_argument("--n-unknown", type=int, default=3)
    ap.add_argument("--no-cross-video", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    model, desc = load_model_for_eval(a.ckpt, a.frozen_backbone)
    splits = ensure_splits(a.root, None)
    files = list_instance_files(a.root, splits[a.split], min_images=max(a.k_shots) + 3)
    names = sorted(files)
    print(f"[diag] model={desc} split={a.split} instances={len(names)}")
    emb = {n: embed_files(model, files[n]) for n in names}
    rng = np.random.default_rng(a.seed)
    n_way = min(a.n_way, len(names) - 1)
    n_unk = max(1, min(a.n_unknown, len(names) - n_way))

    for K in a.k_shots:
        thr, gen_own, unk_best, gen_best_other = [], [], [], []
        tpr_mem, far_mem, tpr_imp, far_imp = [], [], [], []
        for _ in range(a.episodes):
            pick = rng.choice(len(names), n_way + n_unk, replace=False)
            known, unk = [names[i] for i in pick[:n_way]], [names[i] for i in pick[n_way:]]
            mem, qs = PersonalMemory(), {}
            for c in known:
                s, qi = choose_support(files[c], len(emb[c]), K, "spread", rng, not a.no_cross_video)
                mem.enroll(c, emb[c][s])
                qs[c] = emb[c][qi]
            u_x = np.concatenate([emb[u] for u in unk])
            for c in known:
                e = mem.entries[norm_name(c)]
                thr.append(e.threshold)
                own = bank_score(e.bank, l2n(qs[c]))
                gen_own.extend(own.tolist())
                others = np.vstack([mem.entries[k].X for k in mem.entries if k != e.name])
                t_imp = float(np.percentile(bank_score(e.bank, others), 99)) + 0.01
                tpr_imp.append(np.mean(own >= t_imp)); far_imp.append(np.mean(bank_score(e.bank, l2n(u_x)) >= t_imp))
                tpr_mem.append(np.mean(own >= e.threshold)); far_mem.append(np.mean(bank_score(e.bank, l2n(u_x)) >= e.threshold))
            unk_best.extend(np.max([bank_score(mem.entries[k].bank, l2n(u_x)) for k in mem.entries], 0).tolist())
        gen_own, unk_best = np.array(gen_own), np.array(unk_best)
        ts = np.linspace(0, 1, 201)
        far = np.array([(unk_best >= t).mean() for t in ts]); tpr = np.array([(gen_own >= t).mean() for t in ts])
        ok = np.where(far <= 0.05)[0]
        best = (ts[ok[0]], tpr[ok[0]], far[ok[0]]) if len(ok) else (None, 0, 0)
        print(f"\n== K={K}  ({a.episodes} episodes)  quantiles shown: 5/25/50/75/95 %")
        print(f"  memory thresholds chosen        : {q(thr)}   (clip range 0.45-0.90)")
        print(f"  genuine query score (own object): {q(gen_own)}")
        print(f"  unknown objects, best score     : {q(unk_best)}")
        print(f"  memory as is        : TPR {np.mean(tpr_mem)*100:5.1f}%  FAR {np.mean(far_mem)*100:5.1f}%")
        print(f"  impostor-based rule : TPR {np.mean(tpr_imp)*100:5.1f}%  FAR {np.mean(far_imp)*100:5.1f}%")
        if best[0] is not None:
            print(f"  best single global t with FAR<=5%: t={best[0]:.2f} -> TPR {best[1]*100:.1f}% FAR {best[2]*100:.1f}%")
        else:
            print("  no global threshold reaches FAR<=5%")


if __name__ == "__main__":
    main()
