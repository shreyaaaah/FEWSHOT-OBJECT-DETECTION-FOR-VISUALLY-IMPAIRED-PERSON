"""Losses for episodic metric learning.

OpenProtoLoss  (core novelty #1):  prototypical loss with a LEARNED REJECT LOGIT.
    logits = scale * [cos(q, p_1) ... cos(q, p_N), b]      b = learnable threshold
    known query of class c  -> target c
    unknown query           -> target N  (reject)
  The network is therefore trained so that, in an episode, every unknown-object
  frame is less similar to all prototypes than every known frame is to its own
  prototype; i.e. the embedding space is shaped for *open-set* thresholding, not
  just for closed-set classification.

Also implemented for the ablation table: closed prototypical, SupCon, batch-hard triplet.
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def prototypes(support_emb: torch.Tensor, support_lab: torch.Tensor, n_way: int):
    protos = torch.stack([support_emb[support_lab == c].mean(0) for c in range(n_way)])
    return F.normalize(protos, dim=-1)


class OpenProtoLoss(nn.Module):
    def __init__(self, scale: float = 10.0, init_thr: float = 0.5, open_set: bool = True):
        super().__init__()
        self.scale = scale
        self.open_set = open_set
        self.thr = nn.Parameter(torch.tensor(float(init_thr)))

    def forward(self, s_emb, s_lab, q_emb, q_lab, u_emb: Optional[torch.Tensor] = None):
        n_way = int(s_lab.max().item()) + 1
        P = prototypes(s_emb, s_lab, n_way)
        lg_q = self.scale * q_emb @ P.t()
        if not self.open_set:
            loss = F.cross_entropy(lg_q, q_lab)
            acc = (lg_q.argmax(1) == q_lab).float().mean()
            return loss, {"acc": acc.item(), "rej": float("nan")}
        rej = (self.scale * self.thr).expand(lg_q.size(0), 1)
        logits = torch.cat([lg_q, rej], 1)
        targets = q_lab
        stats = {"acc": (lg_q.argmax(1) == q_lab).float().mean().item(),
                 "known_kept": (logits.argmax(1) == q_lab).float().mean().item()}
        if u_emb is not None and len(u_emb):
            lg_u = self.scale * u_emb @ P.t()
            rej_u = (self.scale * self.thr).expand(lg_u.size(0), 1)
            logits_u = torch.cat([lg_u, rej_u], 1)
            tgt_u = torch.full((lg_u.size(0),), n_way, dtype=torch.long, device=lg_u.device)
            logits = torch.cat([logits, logits_u], 0)
            targets = torch.cat([targets, tgt_u], 0)
            stats["rej"] = (logits_u.argmax(1) == n_way).float().mean().item()
        else:
            stats["rej"] = float("nan")
        stats["thr"] = self.thr.item()
        return F.cross_entropy(logits, targets), stats


def supcon_loss(z: torch.Tensor, y: torch.Tensor, temperature: float = 0.1):
    n = z.size(0)
    sim = z @ z.t() / temperature
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = (y[:, None] == y[None, :]) & ~eye
    cnt = pos.sum(1)
    valid = cnt > 0
    if not valid.any():
        return z.sum() * 0.0
    loss = -(log_prob * pos).sum(1)[valid] / cnt[valid]
    return loss.mean()


def batch_hard_triplet(z: torch.Tensor, y: torch.Tensor, margin: float = 0.2):
    d = 1.0 - z @ z.t()
    same = y[:, None] == y[None, :]
    eye = torch.eye(len(y), dtype=torch.bool, device=z.device)
    pos_mask = same & ~eye
    valid = pos_mask.any(1) & (~same).any(1)
    if not valid.any():
        return z.sum() * 0.0
    hardest_pos = d.masked_fill(~pos_mask, -1.0).max(1).values
    hardest_neg = d.masked_fill(same, 2.0).min(1).values
    return F.relu(hardest_pos - hardest_neg + margin)[valid].mean()
