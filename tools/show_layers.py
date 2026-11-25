"""Print the layers of the models used in this project (viva: "how many layers, which ones did you train?").

    python tools/show_layers.py             # DINOv2 embedder in all four modes + the YOLO models found in this folder
    python tools/show_layers.py --no-yolo

Only the architecture is shown, so random weights are used (nothing is downloaded)."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.embedder import Embedder  # noqa: E402


def embedder():
    base = Embedder(mode="frozen", pretrained=False)
    bb = base.backbone
    blk = bb.blocks[0]
    print("=" * 78)
    print("DINOv2 ViT-S/14 embedder  (src/embedder.py, class Embedder)")
    print("=" * 78)
    print(f"  patch embedding        : {bb.patch_embed.patch_size[0]}x{bb.patch_embed.patch_size[1]} patches -> "
          f"{bb.patch_embed.grid_size[0]}x{bb.patch_embed.grid_size[1]} = {bb.patch_embed.num_patches} tokens (+1 CLS token)")
    print(f"  transformer blocks     : {len(bb.blocks)}")
    print(f"  per block              : LayerNorm, attention ({blk.attn.num_heads} heads, qkv {blk.attn.qkv.in_features}->"
          f"{blk.attn.qkv.out_features}), LayerNorm, MLP ({blk.mlp.fc1.in_features}->{blk.mlp.fc1.out_features}->{blk.mlp.fc2.out_features})")
    print(f"  feature size           : {bb.num_features}")
    print(f"  metric head            : {base.head}")
    print(f"  backbone parameters    : {sum(p.numel() for p in bb.parameters()):,}")
    print(f"  head parameters        : {sum(p.numel() for p in base.head.parameters()):,}")
    print(f"  all parameters         : {sum(p.numel() for p in base.parameters()):,}")
    print()
    print(f"  {'mode':8s} {'trainable params':>17s}  trainable layers")
    for mode in ("frozen", "partial", "lora", "full"):
        m = Embedder(mode=mode, pretrained=False)
        blocks = [i for i, b in enumerate(m.backbone.blocks) if any(p.requires_grad for p in b.parameters())]
        lora = sum(1 for x in m.modules() if x.__class__.__name__ == "LoRALinear")
        if mode == "frozen":
            what = "metric head only (the frozen baseline B does not even use the head)"
        elif mode == "partial":
            what = f"blocks {blocks[0]}-{blocks[-1]} (the last {len(blocks)} of {len(m.backbone.blocks)}) + final norm + head"
        elif mode == "lora":
            what = f"LoRA adapters (rank 8) on the attention qkv of {lora} blocks + head; original weights frozen"
        else:
            what = "everything"
        print(f"  {mode:8s} {m.n_trainable():>17,}  {what}")
    print()


def yolo():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed: YOLO layers skipped")
        return
    print("=" * 78)
    print("YOLO detectors (architecture comes from the ultralytics library, not from our code)")
    print("=" * 78)
    for w in ("yolo11n.pt", "yolo26n.pt", "yolo26s.pt"):
        if os.path.exists(w):
            print(w)
            YOLO(w).info(detailed=False, verbose=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-yolo", action="store_true")
    a = ap.parse_args()
    embedder()
    if not a.no_yolo:
        yolo()
