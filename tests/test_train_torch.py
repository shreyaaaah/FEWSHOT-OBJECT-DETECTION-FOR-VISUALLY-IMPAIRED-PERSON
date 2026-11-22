"""Torch-dependent tests (skipped automatically when torch/timm are missing)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from src.embedder import Embedder, LoRALinear
from src.train.losses import OpenProtoLoss, batch_hard_triplet, supcon_loss


def tiny(mode, **kw):
    return Embedder("vit_small_patch14_dinov2.lvd142m", 32, 28, mode=mode, pretrained=False, **kw)


def test_modes_trainable_counts_ordered():
    n = {m: tiny(m).n_trainable() for m in ("frozen", "lora", "partial", "full")}
    assert n["frozen"] < n["lora"] or n["lora"] < n["partial"]
    assert n["frozen"] < n["partial"] < n["full"]
    assert n["lora"] < n["partial"]


def test_lora_wraps_every_block_and_starts_as_identity():
    m = tiny("lora")
    assert all(isinstance(b.attn.qkv, LoRALinear) for b in m.backbone.blocks)
    x = torch.randn(2, 3, 28, 28)
    m.eval()
    with torch.no_grad():
        f1 = m.backbone(x)
        for b in m.backbone.blocks:           # zero the adapters: output must not change
            b.attn.qkv.B.zero_()
        f2 = m.backbone(x)
    assert torch.allclose(f1, f2, atol=1e-5)


def test_embeddings_are_unit_norm():
    m = tiny("partial", unfreeze_last_blocks=1).eval()
    z = m(torch.randn(3, 3, 28, 28))
    assert torch.allclose(z.norm(dim=-1), torch.ones(3), atol=1e-4)


def test_open_loss_decreases_when_overfitting_one_episode():
    torch.manual_seed(0)
    m = tiny("full")
    crit = OpenProtoLoss(scale=10.0)
    opt = torch.optim.AdamW(list(m.parameters()) + list(crit.parameters()), lr=3e-4)
    # 3 known classes (3 support + 3 query each) and 2 unknown classes
    protos = torch.randn(5, 3, 28, 28)
    def sample(c, n):
        return protos[c][None] + 0.15 * torch.randn(n, 3, 28, 28)
    s = torch.cat([sample(c, 3) for c in range(3)]); sl = torch.arange(3).repeat_interleave(3)
    q = torch.cat([sample(c, 3) for c in range(3)]); ql = sl.clone()
    u = torch.cat([sample(c, 3) for c in (3, 4)])
    first = last = None
    for step in range(60):
        loss, st = crit(m(s), sl, m(q), ql, m(u))
        opt.zero_grad(); loss.backward(); opt.step()
        first = loss.item() if first is None else first
        last = loss.item()
    assert last < 0.5 * first
    assert st["rej"] > 0.8 and st["acc"] > 0.8


def test_other_losses_finite_and_differentiable():
    z = torch.nn.functional.normalize(torch.randn(12, 8, requires_grad=True), dim=-1)
    y = torch.arange(4).repeat_interleave(3)
    for f in (supcon_loss, batch_hard_triplet):
        l = f(z, y)
        assert torch.isfinite(l)
    l.backward()


def test_checkpoint_roundtrip(tmp_path):
    m = tiny("lora").eval()
    p = str(tmp_path / "e.pt")
    m.save(p)
    m2 = Embedder.load(p)
    x = torch.randn(2, 3, 28, 28)
    with torch.no_grad():
        assert torch.allclose(m(x), m2(x), atol=1e-5)


def test_embed_bgr_shape():
    m = tiny("frozen")
    imgs = [np.random.randint(0, 255, (50, 80, 3), np.uint8) for _ in range(3)]
    e = m.embed_bgr(imgs)
    assert e.shape == (3, 32) and np.allclose(np.linalg.norm(e, axis=1), 1, atol=1e-4)
