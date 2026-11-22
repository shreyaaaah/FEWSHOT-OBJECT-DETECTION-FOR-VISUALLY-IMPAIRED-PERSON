"""Few-shot + open-set evaluation on a held-out split.

For every K in --k-shots, runs --episodes random episodes:
   * N known objects: K support frames each are 'enrolled', the rest are queries
   * M unknown objects: never enrolled; every frame of them MUST be rejected
Compares on identical episodes:
   mean-proto@t   nearest mean prototype + fixed global cosine threshold t   (baseline)
   ours           PersonalMemory: multi-prototype bank + per-object blocked-CV
                  calibrated threshold + margin test  (the project's method)
Reports closed-set accuracy, correct-ID rate (TPR), false-accept rate (FAR) on unknown
objects, and open-set AUROC, each with 95% confidence intervals over episodes.

    python -m src.eval.episodic --root data/orbit_instances --ckpt checkpoints/embedder_main.pt
    python -m src.eval.episodic --root data/orbit_instances --frozen-backbone resnet50.a1_in1k
"""
import argparse
import os

import numpy as np

from ..config import ROOT
from ..memory import PersonalMemory, auroc, l2n
from ..train.datasets import load_splits
from .common import (embed_files, fmt, list_instance_files, load_model_for_eval, mean_ci,
                     video_kind, write_report)

FIXED_THRESHOLDS = (0.5, 0.6, 0.7, 0.8)


def choose_support(files, n_frames, k, protocol, rng, cross_video):
    """Return (support_idx, query_idx) for one object."""
    n = len(files)
    idx = np.arange(n)
    if cross_video:
        kinds = np.array([video_kind(f) for f in files])
        clean, clutter = idx[kinds == "clean"], idx[kinds == "clutter"]
        if len(clean) >= k and len(clutter) >= 3:
            pick = np.linspace(0, len(clean) - 1, k).astype(int)
            return clean[pick], clutter
    if protocol == "spread":          # k frames spread over the whole video (like enrollment)
        s = np.linspace(0, n - 1, k).astype(int)
    elif protocol == "contiguous":    # k consecutive frames (a very short clip)
        st = int(rng.integers(0, n - k + 1))
        s = np.arange(st, st + k)
    else:                             # random frames
        s = rng.choice(n, k, replace=False)
    return s, np.setdiff1d(idx, s)


def run(args):
    model, desc = load_model_for_eval(args.ckpt, args.frozen_backbone, args.no_pretrained,
                                      img_size=args.img_size)
    splits = load_splits(args.splits or os.path.join(args.root, "splits.json"))
    insts = splits[args.split]
    files = list_instance_files(args.root, insts, min_images=max(args.k_shots) + 3)
    names = sorted(files)
    print(f"[eval] model={desc} split={args.split} instances={len(names)}")
    if len(names) < args.n_way + 1:
        raise SystemExit(f"need >= {args.n_way + 1} instances in split '{args.split}', got {len(names)}")
    emb = {n: embed_files(model, files[n]) for n in names}
    print("[eval] embedded", sum(len(v) for v in emb.values()), "frames")
    n_way = min(args.n_way, len(names) - 1)
    n_unk = min(args.n_unknown, len(names) - n_way)
    rng = np.random.default_rng(args.seed)

    results = {}
    for K in args.k_shots:
        rows = {m: {"acc": [], "tpr": [], "far": []} for m in
                ["ours"] + [f"mean-proto@{t}" for t in FIXED_THRESHOLDS]}
        au_ours, au_base, closed = [], [], []
        for _ in range(args.episodes):
            pick = rng.choice(len(names), n_way + n_unk, replace=False)
            known = [names[i] for i in pick[:n_way]]
            unk = [names[i] for i in pick[n_way:]]
            mem = PersonalMemory()
            protos, q_x, q_y = {}, [], []
            for c in known:
                s_idx, q_idx = choose_support(files[c], len(emb[c]), K, args.support, rng, args.cross_video)
                mem.enroll(c, emb[c][s_idx])
                protos[c] = l2n(emb[c][s_idx].mean(0))
                q_x.append(emb[c][q_idx]); q_y += [c] * len(q_idx)
            q_x = np.concatenate(q_x)
            u_x = np.concatenate([emb[u] for u in unk])
            P = np.stack([protos[c] for c in known])
            sims_q, sims_u = q_x @ P.T, u_x @ P.T
            top_q = np.array(known)[sims_q.argmax(1)]
            correct = top_q == np.array(q_y)
            closed.append(correct.mean())
            au_base.append(auroc(sims_q.max(1), sims_u.max(1)))
            # ours
            mq = [mem.query(x) for x in q_x]
            mu = [mem.query(x) for x in u_x]
            rows["ours"]["tpr"].append(np.mean([m.name == y for m, y in zip(mq, q_y)]))
            rows["ours"]["acc"].append(np.mean([(m.best_name == y) for m, y in zip(mq, q_y)]))
            rows["ours"]["far"].append(np.mean([m.name is not None for m in mu]))
            au_ours.append(auroc([m.prob for m in mq], [m.prob for m in mu]))
            for t in FIXED_THRESHOLDS:
                r = rows[f"mean-proto@{t}"]
                r["tpr"].append(np.mean(correct & (sims_q.max(1) >= t)))
                r["acc"].append(correct.mean())
                r["far"].append(np.mean(sims_u.max(1) >= t))
        results[K] = {"closed_acc": mean_ci(closed), "auroc_maxcos": mean_ci(au_base),
                      "auroc_ours_prob": mean_ci(au_ours),
                      "methods": {m: {k: mean_ci(v) for k, v in r.items()} for m, r in rows.items()}}

    md = [f"# Episodic open-set evaluation\n",
          f"model `{desc}` | split `{args.split}` | {n_way}-way + {n_unk} unknown objects | "
          f"{args.episodes} episodes | support = `{args.support}`"
          + (" (cross-video where available)" if args.cross_video else "") + "\n"]
    for K, r in results.items():
        md.append(f"\n## K = {K} shots\n")
        md.append(f"closed-set accuracy (mean prototype): **{fmt(*r['closed_acc'])}**  |  "
                  f"open-set AUROC: max-cosine **{fmt(*r['auroc_maxcos'], pct=False)}**, "
                  f"ours (calibrated prob) **{fmt(*r['auroc_ours_prob'], pct=False)}**\n")
        md.append("| method | correct-ID rate % (TPR) | false-accept rate % on unknown objects (FAR) |")
        md.append("|---|---|---|")
        for m, v in r["methods"].items():
            md.append(f"| {m} | {fmt(*v['tpr'])} | {fmt(*v['far'])} |")
    text = "\n".join(md) + "\n"
    print(text)
    tag = args.tag or desc.replace(":", "_").replace("/", "_")
    write_report(os.path.join(ROOT, "reports", f"episodic_{tag}_{args.split}"),
                 {"desc": desc, "args": vars(args), "results": {str(k): v for k, v in results.items()}}, text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--splits", default=None)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--frozen-backbone", default=None, help="timm name for a frozen baseline")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--k-shots", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--n-way", type=int, default=5)
    ap.add_argument("--n-unknown", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--support", default="spread", choices=["spread", "contiguous", "random"])
    ap.add_argument("--cross-video", action="store_true",
                    help="support from clean_* frames, queries from clutter_* frames (ORBIT protocol)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
