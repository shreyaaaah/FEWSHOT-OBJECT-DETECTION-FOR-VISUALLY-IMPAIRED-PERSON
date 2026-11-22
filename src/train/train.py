"""Episodic transfer-learning trainer.

Examples
--------
# main method (Stage C): DINOv2 ViT-S, last 2 blocks + head, open-set prototypical loss
python -m src.train.train --root data/orbit_instances --mode partial --loss open

# ablations
python -m src.train.train --root data/orbit_instances --mode frozen --loss open --tag frozen_head
python -m src.train.train --root data/orbit_instances --mode lora   --loss open --tag lora
python -m src.train.train --root data/orbit_instances --mode partial --loss proto    --tag closed_proto
python -m src.train.train --root data/orbit_instances --mode partial --loss supcon   --tag supcon
python -m src.train.train --root data/orbit_instances --mode partial --loss triplet  --tag triplet
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from ..config import ROOT, get_device, load_config
from ..embedder import Embedder
from ..memory import auroc
from .datasets import (EpisodeSampler, eval_transform, load_images, load_splits,
                       save_splits, scan_instances, split_instances, train_transform)
from .losses import OpenProtoLoss, batch_hard_triplet, prototypes, supcon_loss


def embed_episode(model, ep, tf, device, amp):
    paths = ep.support_paths + ep.query_paths + ep.unknown_paths
    x = load_images(paths, tf).to(device)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        z = model(x)
    z = z.float()
    ns, nq = len(ep.support_paths), len(ep.query_paths)
    return z[:ns], z[ns:ns + nq], z[ns + nq:]


@torch.no_grad()
def evaluate(model, sampler, n_eps, device, tf, amp=False):
    """Closed-set accuracy + open-set AUROC (max-cosine known vs unknown)."""
    model.eval()
    accs, pos, neg = [], [], []
    for _ in range(n_eps):
        ep = sampler.sample()
        s, q, u = embed_episode(model, ep, tf, device, amp)
        sl = torch.tensor(ep.support_labels, device=device)
        ql = torch.tensor(ep.query_labels, device=device)
        P = prototypes(s, sl, len(ep.classes))
        sim_q = q @ P.t()
        accs.append((sim_q.argmax(1) == ql).float().mean().item())
        pos += sim_q.max(1).values.cpu().tolist()
        if len(u):
            neg += (u @ P.t()).max(1).values.cpu().tolist()
    model.train()
    return float(np.mean(accs)), (auroc(pos, neg) if neg else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder of <instance>/<frames>")
    ap.add_argument("--splits", default=None, help="splits json (created if missing)")
    ap.add_argument("--mode", default="partial", choices=["frozen", "partial", "lora", "full"])
    ap.add_argument("--loss", default="open", choices=["open", "proto", "supcon", "triplet"])
    ap.add_argument("--lam-supcon", type=float, default=0.2,
                    help="extra SupCon weight on top of the open prototypical loss")
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--no-pretrained", action="store_true", help="random init (tests only)")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--n-way", type=int, default=None)
    ap.add_argument("--k-shot", type=int, default=None)
    ap.add_argument("--q-query", type=int, default=None)
    ap.add_argument("--val-every", type=int, default=200)
    ap.add_argument("--val-episodes", type=int, default=40)
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--amp", action="store_true", help="fp16 autocast (CUDA only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config()
    tc = cfg["train"]
    bc = cfg["backbone"]
    episodes = args.episodes or tc["episodes"]
    n_way = args.n_way or tc["n_way"]
    k_shot = args.k_shot or tc["k_shot"]
    q_query = args.q_query or tc["q_query"]
    img = args.img_size or bc["img_size"]
    name = args.backbone or bc["name"]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device(tc.get("device", "auto"))
    amp = args.amp and device.type == "cuda"

    data = scan_instances(args.root, min_images=k_shot + q_query)
    split_path = args.splits or os.path.join(args.root, "splits.json")
    if os.path.exists(split_path):
        splits = load_splits(split_path)
    else:
        splits = split_instances(list(data), seed=args.seed)
        save_splits(splits, split_path)
        print(f"[data] wrote {split_path}")
    print({k: len(v) for k, v in splits.items()}, "instances per split")

    train_s = EpisodeSampler(data, splits["train"], n_way, k_shot, q_query, seed=args.seed)
    val_insts = splits["val"] or splits["train"]
    n_val_ok = sum(1 for i in val_insts if i in data and len(data[i]) >= k_shot + q_query)
    val_way = min(n_way, max(2, n_val_ok - 1))          # leave >=1 instance as 'unknown'
    if val_way < n_way:
        print(f"[data] only {n_val_ok} validation instances -> validating with {val_way}-way episodes")
    val_s = EpisodeSampler(data, val_insts, val_way, k_shot, q_query,
                           seed=1234, contiguous_support=True)

    model = Embedder(name, bc["embed_dim"], img, mode=args.mode,
                     unfreeze_last_blocks=bc["unfreeze_last_blocks"],
                     pretrained=not args.no_pretrained, use_head=True).to(device)
    print(f"[model] {name} mode={args.mode} trainable params: {model.n_trainable():,}")

    open_loss = OpenProtoLoss(scale=1.0 / tc["temperature"], open_set=(args.loss == "open")).to(device)
    groups = model.param_groups(tc["lr_backbone"], tc["lr_head"])
    if args.loss == "open":
        groups.append({"params": open_loss.parameters(), "lr": tc["lr_head"], "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 50) * 0.5 * (1 + math.cos(math.pi * s / max(1, episodes))))
    scaler = torch.amp.GradScaler(enabled=amp)

    tf_tr, tf_ev = train_transform(img), eval_transform(img)
    out_path = args.out or os.path.join(ROOT, "checkpoints", f"embedder_{args.tag}.pt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    best, log, t0 = -1.0, [], time.time()
    model.train()
    for ep_i in range(1, episodes + 1):
        ep = train_s.sample()
        s, q, u = embed_episode(model, ep, tf_tr, device, amp)
        sl = torch.tensor(ep.support_labels, device=device)
        ql = torch.tensor(ep.query_labels, device=device)

        if args.loss in ("open", "proto"):
            loss, st = open_loss(s, sl, q, ql, u if args.loss == "open" else None)
            if args.loss == "open" and args.lam_supcon > 0:
                z_all, y_all = torch.cat([s, q]), torch.cat([sl, ql])
                if len(u):                      # unknown classes get their own labels
                    ul = torch.tensor(ep.unknown_labels, device=device) + n_way
                    z_all, y_all = torch.cat([z_all, u]), torch.cat([y_all, ul])
                loss = loss + args.lam_supcon * supcon_loss(z_all, y_all)
        else:
            z_all = torch.cat([s, q])
            y_all = torch.cat([sl, ql])
            loss = supcon_loss(z_all, y_all) if args.loss == "supcon" else batch_hard_triplet(z_all, y_all)
            st = {}
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

        if ep_i % 20 == 0:
            print(f"ep {ep_i:5d}/{episodes} loss {loss.item():.4f} "
                  + " ".join(f"{k} {v:.3f}" for k, v in st.items()) +
                  f"  [{time.time() - t0:.0f}s]")
        if ep_i % args.val_every == 0 or ep_i == episodes:
            acc, au = evaluate(model, val_s, args.val_episodes, device, tf_ev, amp)
            score = 0.5 * acc + 0.5 * (0.0 if math.isnan(au) else au)
            log.append({"episode": ep_i, "val_acc": acc, "val_auroc": au, "score": score})
            print(f"  [val] acc {acc:.3f} auroc {au:.3f} score {score:.3f}")
            if score > best:
                best = score
                model.save(out_path, extra={"episode": ep_i, "val_acc": acc, "val_auroc": au,
                                            "loss": args.loss, "mode": args.mode})
                print(f"  [save] {out_path}")
    with open(out_path.replace(".pt", "_log.json"), "w") as f:
        json.dump({"args": vars(args), "log": log}, f, indent=2)
    print(f"done. best val score {best:.3f}  -> {out_path}")


if __name__ == "__main__":
    main()
