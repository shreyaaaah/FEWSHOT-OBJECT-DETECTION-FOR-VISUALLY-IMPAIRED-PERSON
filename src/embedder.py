"""Embedding network = pretrained backbone (transfer learning) + small metric head.

Training modes (these are your ablation stages):
    frozen   : backbone frozen, only the head is trained       (Stage B+)
    partial  : last N backbone blocks + head are trained       (Stage C, main method)
    lora     : low-rank adapters on attention qkv + head       (Stage D)
    full     : everything trained (usually overfits - ablation only)
`use_head=False` gives the pure pretrained features (Stage A/B baselines).
"""
import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .imgutil import prep_bgr


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        self.scale = alpha / r
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale


def apply_lora(backbone: nn.Module, r: int = 8, alpha: int = 16) -> int:
    """Freeze the backbone and add LoRA adapters to every ViT attention qkv."""
    for p in backbone.parameters():
        p.requires_grad = False
    n = 0
    for blk in getattr(backbone, "blocks", []):
        blk.attn.qkv = LoRALinear(blk.attn.qkv, r, alpha)   # new params are trainable
        n += 1
    if n == 0:
        raise ValueError("LoRA mode needs a ViT-style backbone with .blocks")
    return n


class Embedder(nn.Module):
    def __init__(self, name="vit_small_patch14_dinov2.lvd142m", embed_dim=256,
                 img_size=224, mode="partial", unfreeze_last_blocks=2,
                 pretrained=True, use_head=True, lora_rank=8):
        super().__init__()
        import timm
        self.cfg = dict(name=name, embed_dim=embed_dim, img_size=img_size, mode=mode,
                        unfreeze_last_blocks=unfreeze_last_blocks, use_head=use_head,
                        lora_rank=lora_rank)
        self.mode = mode
        self.img_size = img_size
        self.use_head = use_head
        kw = {"img_size": img_size} if ("vit" in name or "deit" in name) else {}
        self.backbone = timm.create_model(name, pretrained=pretrained, num_classes=0, **kw)
        feat = self.backbone.num_features
        self.feat_dim = feat
        self.head = nn.Sequential(nn.Linear(feat, 512), nn.GELU(), nn.Linear(512, embed_dim))
        self.out_dim = embed_dim if use_head else feat
        self._set_trainable(mode, unfreeze_last_blocks, lora_rank)

    # ------------------------------------------------------------ trainability
    def _set_trainable(self, mode, n_blocks, lora_rank):
        for p in self.backbone.parameters():
            p.requires_grad = False
        if mode == "frozen":
            pass
        elif mode == "full":
            for p in self.backbone.parameters():
                p.requires_grad = True
        elif mode == "partial":
            self._unfreeze_last(n_blocks)
        elif mode == "lora":
            apply_lora(self.backbone, r=lora_rank, alpha=2 * lora_rank)
        else:
            raise ValueError(f"unknown mode {mode}")
        for p in self.head.parameters():
            p.requires_grad = self.use_head

    def _unfreeze_last(self, n):
        bb = self.backbone
        if hasattr(bb, "blocks"):                        # ViT / DINOv2
            for blk in list(bb.blocks)[-n:]:
                for p in blk.parameters():
                    p.requires_grad = True
            for nm in ("norm", "fc_norm"):
                m = getattr(bb, nm, None)
                if isinstance(m, nn.Module):
                    for p in m.parameters():
                        p.requires_grad = True
        elif hasattr(bb, "layer4"):                      # ResNet family
            layers = [bb.layer4, bb.layer3, bb.layer2][:max(1, n)]
            for l in layers:
                for p in l.parameters():
                    p.requires_grad = True
        else:
            for p in bb.parameters():
                p.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        if self.mode != "full":                          # keep frozen BN statistics fixed
            for m in self.backbone.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
        return self

    def param_groups(self, lr_backbone, lr_head, wd=1e-4):
        bb = [p for p in self.backbone.parameters() if p.requires_grad]
        hd = [p for p in self.head.parameters() if p.requires_grad]
        groups = []
        if bb:
            groups.append({"params": bb, "lr": lr_backbone, "weight_decay": wd})
        if hd:
            groups.append({"params": hd, "lr": lr_head, "weight_decay": wd})
        return groups

    def n_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ---------------------------------------------------------------- forward
    def forward(self, x):
        f = self.backbone(x)
        z = self.head(f) if self.use_head else f
        return F.normalize(z, dim=-1)

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def embed_bgr(self, imgs: List[np.ndarray], batch_size: int = 32) -> np.ndarray:
        """List of BGR uint8 images -> (N, D) float32 L2-normalised embeddings."""
        self.eval()
        outs = []
        for i in range(0, len(imgs), batch_size):
            x = np.stack([prep_bgr(im, self.img_size) for im in imgs[i:i + batch_size]])
            x = torch.from_numpy(x).to(self.device)
            outs.append(self(x).float().cpu().numpy())
        if not outs:
            return np.zeros((0, self.out_dim), np.float32)
        return np.concatenate(outs, 0)

    # ------------------------------------------------------------- checkpoint
    def save(self, path, extra: Optional[dict] = None):
        torch.save({"state": self.state_dict(), "cfg": self.cfg, "extra": extra or {}}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = dict(ck["cfg"])
        model = cls(pretrained=False, **cfg)           # weights come from the checkpoint
        model.load_state_dict(ck["state"])
        return model.to(device).eval()
