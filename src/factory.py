"""Build the full system from config + files (used by CLI, live app, evals)."""
import os
from typing import Optional

from .speaker import VoiceSpeaker
from .config import ROOT, get_device, load_config
from .memory import PersonalMemory
from .pipeline import Sahayak

DEFAULT_MEMORY_DIR = os.path.join(ROOT, "data", "memory")
DEFAULT_CKPT = os.path.join(ROOT, "checkpoints", "embedder.pt")


def build_embedder(cfg, ckpt: Optional[str] = None, device=None, frozen_only=False):
    from .embedder import Embedder
    dev = device or get_device(cfg["train"].get("device", "auto"))
    ckpt = ckpt or DEFAULT_CKPT
    if not frozen_only and os.path.exists(ckpt):
        print(f"[init] fine-tuned embedder: {ckpt}")
        return Embedder.load(ckpt, device=dev)
    b = cfg["backbone"]
    print("[init] no fine-tuned checkpoint found -> frozen pretrained backbone (baseline B)")
    return Embedder(b["name"], b["embed_dim"], b["img_size"], mode="frozen",
                    pretrained=True, use_head=False).to(dev).eval()


def build_system(cfg_path=None, ckpt=None, memory_dir=None, speak=True,
                 use_detector=True, device=None, frozen_only=False,
                 use_depth: Optional[bool] = None) -> Sahayak:
    cfg = load_config(cfg_path)
    dev = device or get_device(cfg["train"].get("device", "auto"))
    emb = build_embedder(cfg, ckpt, dev, frozen_only)
    memory_dir = memory_dir or DEFAULT_MEMORY_DIR
    if os.path.exists(os.path.join(memory_dir, "memory.json")):
        mem = PersonalMemory.load(memory_dir)
        print(f"[init] loaded memory with {len(mem)} objects")
    else:
        mem = PersonalMemory.from_config(cfg["memory"])
    det = None
    if use_detector:
        try:
            from .detector import Detector
            d = cfg["detector"]
            det = Detector(d["weights"], d["conf"], d["max_det"], device=None)
        except Exception as e:              # keeps the project usable without ultralytics
            print(f"[warn] detector unavailable ({e}); using centre-crop mode")
    depth_model = None
    want_depth = cfg["depth"].get("use_depth_model", False) if use_depth is None else use_depth
    if want_depth:
        try:
            from .depth import RelativeDepth
            depth_model = RelativeDepth(device=dev)
        except Exception as e:
            print(f"[warn] depth model unavailable ({e}); using size heuristic only")
    spk = VoiceSpeaker(enabled=speak)
    system = Sahayak(emb, mem, det, depth_model, spk, cfg)
    system.memory_dir = memory_dir
    return system


def save_memory(system: Sahayak):
    system.mem.save(system.memory_dir)
