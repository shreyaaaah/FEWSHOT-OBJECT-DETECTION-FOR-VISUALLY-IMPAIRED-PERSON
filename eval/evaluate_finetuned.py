"""Evaluate available fine-tuned checkpoints using validation-selected thresholds."""
import argparse
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from eval.compare_models import (
    COCO_IDS,
    DATA_YAML,
    RESULTS,
    best_f1_threshold,
    collect_frames,
    image_paths,
    test_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs" / "finetune"
DEFAULT_MODELS = ("yolo11n", "yolo26n", "yolo26s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    args = parser.parse_args()
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"COCO-80 dataset YAML not found: {DATA_YAML}; run eval/make_dataset.py first")

    RESULTS.mkdir(parents=True, exist_ok=True)
    metric_rows, class_rows = [], []
    for stem in args.models:
        checkpoint = RUNS_DIR / stem / "weights" / "best.pt"
        if not checkpoint.exists():
            print(f"Skipping {stem}: checkpoint not found at {checkpoint}")
            continue
        model = YOLO(str(checkpoint))
        threshold = best_f1_threshold(collect_frames(model, "val"))
        # The threshold is selected only from val: confidence candidates 0.01-0.99
        # maximize micro F1 after greedy same-class matching at IoU >= 0.50; ties
        # choose the higher threshold. Test P/R/F1 use the same match rule once.
        model.val(
            data=str(DATA_YAML), split="test", classes=COCO_IDS, conf=threshold,
            device="cpu", imgsz=640, plots=False, verbose=False,
            project=str(RESULTS / "ultralytics_finetuned"), name=stem,
        )
        scores, per_class = test_metrics(collect_frames(model, "test"), threshold)
        metric_rows.append({"model": stem, "threshold": threshold, **scores})
        for row in per_class:
            class_rows.append({"model": stem, **row})
        print(f"{stem}: threshold={threshold:.2f}, P={scores['precision']:.4f}, "
              f"R={scores['recall']:.4f}, F1={scores['f1']:.4f}, "
              f"mAP50={scores['map50']:.4f}, mAP50-95={scores['map50_95']:.4f}")

    pd.DataFrame(metric_rows, columns=[
        "model", "threshold", "precision", "recall", "f1", "map50", "map50_95",
    ]).to_csv(RESULTS / "finetuned_metrics.csv", index=False)
    pd.DataFrame(class_rows, columns=[
        "model", "class", "ap50", "precision", "recall", "f1", "targets",
    ]).to_csv(RESULTS / "per_class_metrics.csv", index=False)
    print(pd.DataFrame(metric_rows).to_string(index=False) if metric_rows else "No fine-tuned checkpoints found.")


if __name__ == "__main__":
    main()
