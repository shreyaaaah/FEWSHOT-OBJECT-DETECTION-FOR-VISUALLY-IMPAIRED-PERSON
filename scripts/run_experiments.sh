#!/usr/bin/env bash
# Full ablation: every row of the results table.  Usage:
#   bash scripts/run_experiments.sh data/orbit_instances [EPISODES]
# One GPU (Colab/Kaggle T4) needs roughly 20-40 min per training run at 3000 episodes.
set -euo pipefail
ROOT=${1:?path to instance folder}
EPS=${2:-3000}
EV="--root $ROOT --split test --episodes 300 --k-shots 1 3 5 10"

echo "== Stage 0: hand-crafted baseline (no deep learning)"
python -m src.eval.episodic $EV --frozen-backbone hsv_hist --tag A0_hsv_hist
echo "== Stage A: frozen ImageNet ResNet-50 (nearest centroid)"
python -m src.eval.episodic $EV --frozen-backbone resnet50.a1_in1k --tag A_resnet50_frozen
echo "== Stage B: frozen DINOv2 ViT-S/14"
python -m src.eval.episodic $EV --tag B_dinov2_frozen

train () {  # tag, mode, loss
  python -m src.train.train --root "$ROOT" --mode "$2" --loss "$3" --episodes "$EPS" --tag "$1" --amp
  python -m src.eval.episodic $EV --ckpt "checkpoints/embedder_$1.pt" --tag "$1"
}
echo "== Stage C (MAIN): last 2 blocks + head, open-set prototypical loss"
train C_partial_open partial open
echo "== Stage D: LoRA"
train D_lora_open lora open
echo "== Loss ablation (same partial fine-tuning)"
train C_partial_proto partial proto
train C_partial_supcon partial supcon
train C_partial_triplet partial triplet
echo "== Frozen backbone + trained head only"
train B_head_only frozen open
echo "== Full fine-tuning (expected to overfit)"
train X_full_open full open

echo "== Robustness of the main model"
python -m src.eval.robustness --root "$ROOT" --split test --ckpt checkpoints/embedder_C_partial_open.pt --tag C_partial_open
python -m src.eval.robustness --root "$ROOT" --split test --tag B_dinov2_frozen
echo "All reports are in reports/*.md"
