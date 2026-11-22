"""Fine-tune a selected pretrained YOLO checkpoint on Sahayak-28."""
import argparse
from pathlib import Path
import time

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = PROJECT_ROOT / "datasets" / "sahayak28" / "data.yaml"
RUNS_DIR = PROJECT_ROOT / "runs" / "finetune"
ALLOWED_MODELS = ("yolo11n.pt", "yolo26n.pt", "yolo26s.pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=ALLOWED_MODELS, default="yolo26n.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0 < args.fraction <= 1:
        parser.error("--fraction must be greater than 0 and at most 1")
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}; run eval/make_dataset.py first")

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print(f"CUDA device: {device_name}")
    else:
        print("WARNING: CUDA is unavailable; CPU training will be very slow.")

    stem = Path(args.model).stem
    run_dir = RUNS_DIR / stem
    last_checkpoint = run_dir / "weights" / "last.pt"
    resume_checkpoint = args.resume and last_checkpoint.exists()
    if resume_checkpoint:
        print(f"Resuming from: {last_checkpoint}")
        model = YOLO(str(last_checkpoint))
    else:
        if args.resume:
            print(f"No resume checkpoint at {last_checkpoint}; starting from {args.model}.")
        model = YOLO(args.model)

    epoch_started = None

    def on_epoch_start(trainer):
        nonlocal epoch_started
        if trainer.epoch == 0:
            epoch_started = time.perf_counter()

    def on_epoch_end(trainer):
        if trainer.epoch == 0 and epoch_started is not None:
            elapsed = time.perf_counter() - epoch_started
            print(f"First epoch time: {elapsed:.2f} seconds")

    model.add_callback("on_train_epoch_start", on_epoch_start)
    model.add_callback("on_train_epoch_end", on_epoch_end)
    model.train(
        data=str(DATA_YAML),
        imgsz=640,
        batch=16,
        epochs=args.epochs,
        fraction=args.fraction,
        patience=15,
        lr0=0.005,
        cos_lr=True,
        close_mosaic=10,
        project=str(RUNS_DIR),
        name=stem,
        resume=bool(resume_checkpoint),
    )


if __name__ == "__main__":
    main()
