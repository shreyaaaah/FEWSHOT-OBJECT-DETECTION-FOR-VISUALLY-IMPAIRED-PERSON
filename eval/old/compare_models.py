"""Compare the three pretrained YOLO models on the prepared COCO-80 label export."""
from pathlib import Path
import csv
import time

import numpy as np
import pandas as pd
from PIL import Image
import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET = PROJECT_ROOT / "datasets" / "sahayak28_coco80"
DATA_YAML = DATASET / "data.yaml"
RESULTS = PROJECT_ROOT / "results"
MODELS = ("yolo11n.pt", "yolo26n.pt", "yolo26s.pt")
CLASSES = [
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "stop sign",
    "fire hydrant", "bench", "dog", "cat", "backpack", "umbrella", "suitcase",
    "bottle", "cup", "bowl", "banana", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "keyboard", "cell phone",
]
COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]
COCO_IDS = [COCO80.index(name) for name in CLASSES]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_paths(split):
    directory = DATASET / "images" / split
    paths = sorted(path for path in directory.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise FileNotFoundError(f"No {split} images found under {directory}")
    return paths


def label_path(image_path, split):
    relative = image_path.relative_to(DATASET / "images" / split)
    return (DATASET / "labels" / split / relative).with_suffix(".txt")


def xywhn_to_xyxy(cx, cy, width, height, image_width, image_height):
    cx, width = cx * image_width, width * image_width
    cy, height = cy * image_height, height * image_height
    return (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)


def load_targets(path, width, height):
    targets = []
    if not path.exists():
        return targets
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 5:
            continue
        class_id = int(float(fields[0]))
        if class_id in COCO_IDS:
            box = xywhn_to_xyxy(*map(float, fields[1:]), width, height)
            targets.append((class_id, box))
    return targets


def collect_frames(model, split):
    frames = []
    for path in image_paths(split):
        with Image.open(path) as image:
            width, height = image.size
        targets = load_targets(label_path(path, split), width, height)
        result = model.predict(
            source=str(path), conf=0.001, classes=COCO_IDS, device="cpu",
            imgsz=640, verbose=False,
        )[0]
        predictions = []
        if result.boxes is not None:
            coords = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            predictions = [
                (int(class_id), float(score), tuple(map(float, box)))
                for box, score, class_id in zip(coords, scores, class_ids)
                if int(class_id) in COCO_IDS
            ]
        frames.append({"path": path, "targets": targets, "predictions": predictions})
    return frames


def iou(box_a, box_b):
    x1, y1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    x2, y2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def class_counts(frames, class_id, confidence, iou_threshold=0.5):
    total_targets = sum(sum(target_id == class_id for target_id, _ in f["targets"]) for f in frames)
    true_positive = false_positive = 0
    for frame in frames:
        targets = [box for target_id, box in frame["targets"] if target_id == class_id]
        matched = set()
        predictions = sorted(
            (pred for pred in frame["predictions"] if pred[0] == class_id and pred[1] >= confidence),
            key=lambda pred: pred[1], reverse=True,
        )
        for _, _, pred_box in predictions:
            candidates = [(iou(pred_box, box), index) for index, box in enumerate(targets) if index not in matched]
            best_iou, best_index = max(candidates, default=(0.0, None))
            if best_index is not None and best_iou >= iou_threshold:
                matched.add(best_index)
                true_positive += 1
            else:
                false_positive += 1
    return true_positive, false_positive, total_targets


def scores_at(frames, confidence, class_ids=COCO_IDS):
    true_positive = false_positive = total_targets = 0
    for class_id in class_ids:
        tp, fp, targets = class_counts(frames, class_id, confidence)
        true_positive += tp
        false_positive += fp
        total_targets += targets
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / total_targets if total_targets else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def best_f1_threshold(validation_frames):
    # For each threshold, detections are confidence-filtered, matched greedily to
    # same-class ground truth at IoU >= 0.50, and aggregated micro-wise. Choose the
    # highest F1; ties select the larger threshold to prefer fewer false positives.
    candidates = [round(value / 100, 2) for value in range(1, 100)]
    scored = [(scores_at(validation_frames, threshold)[2], threshold) for threshold in candidates]
    return max(scored, key=lambda pair: (pair[0], pair[1]))[1]


def average_precision(frames, class_id, confidence, iou_threshold):
    ground_truth = {}
    detections = []
    for frame_index, frame in enumerate(frames):
        ground_truth[frame_index] = [
            box for target_id, box in frame["targets"] if target_id == class_id
        ]
        detections.extend(
            (score, frame_index, box)
            for pred_id, score, box in frame["predictions"]
            if pred_id == class_id and score >= confidence
        )
    total = sum(len(boxes) for boxes in ground_truth.values())
    if total == 0:
        return None
    matched = {index: set() for index in ground_truth}
    true_flags, false_flags = [], []
    for _, frame_index, box in sorted(detections, key=lambda item: item[0], reverse=True):
        candidates = [
            (iou(box, target), index)
            for index, target in enumerate(ground_truth[frame_index])
            if index not in matched[frame_index]
        ]
        best_iou, best_index = max(candidates, default=(0.0, None))
        if best_index is not None and best_iou >= iou_threshold:
            matched[frame_index].add(best_index)
            true_flags.append(1)
            false_flags.append(0)
        else:
            true_flags.append(0)
            false_flags.append(1)
    if not true_flags:
        return 0.0
    tp = np.cumsum(true_flags)
    fp = np.cumsum(false_flags)
    recall = tp / total
    precision = tp / np.maximum(tp + fp, 1)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.mean([
        precision[recall >= target].max() if np.any(recall >= target) else 0.0
        for target in np.linspace(0.0, 1.0, 101)
    ]))


def test_metrics(frames, confidence):
    per_class = []
    aps_50_95 = []
    for name, class_id in zip(CLASSES, COCO_IDS):
        tp, fp, targets = class_counts(frames, class_id, confidence)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / targets if targets else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        ap50 = average_precision(frames, class_id, 0.001, 0.50)
        aps = [average_precision(frames, class_id, 0.001, threshold)
               for threshold in np.arange(0.50, 0.96, 0.05)]
        per_class.append({"class": name, "ap50": ap50, "precision": precision,
                          "recall": recall, "f1": f1, "targets": targets})
        aps_50_95.extend(value for value in aps if value is not None)
    precision, recall, f1 = scores_at(frames, confidence)
    valid_ap50 = [row["ap50"] for row in per_class if row["ap50"] is not None]
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": float(np.mean(valid_ap50)) if valid_ap50 else 0.0,
        "map50_95": float(np.mean(aps_50_95)) if aps_50_95 else 0.0,
    }, per_class


def cpu_latency(model, paths):
    if len(paths) < 100:
        raise RuntimeError(f"Latency measurement requires 100 test images; found {len(paths)}")
    warmup_path = str(paths[0])
    for _ in range(10):
        model.predict(source=warmup_path, conf=0.001, classes=COCO_IDS,
                      device="cpu", imgsz=640, verbose=False)
    started = time.perf_counter()
    for path in paths[:100]:
        model.predict(source=str(path), conf=0.001, classes=COCO_IDS,
                      device="cpu", imgsz=640, verbose=False)
    return (time.perf_counter() - started) * 1000 / 100


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}; run eval/make_dataset.py first")
    RESULTS.mkdir(parents=True, exist_ok=True)
    metrics_rows, class_rows = [], []
    for weight in MODELS:
        model = YOLO(weight)
        validation_frames = collect_frames(model, "val")
        threshold = best_f1_threshold(validation_frames)
        # Run Ultralytics validation on the held-out test split at the selected
        # threshold and with only the matching COCO-80 class IDs. AP is computed
        # from all retained confidence-ranked predictions below for a consistent
        # per-class report; P/R/F1 are micro-aggregated at this threshold, IoU .50.
        test_result = model.val(
            data=str(DATA_YAML), split="test", classes=COCO_IDS, conf=threshold,
            device="cpu", imgsz=640, plots=False, verbose=False,
            project=str(RESULTS / "ultralytics"), name=Path(weight).stem,
        )
        test_frames = collect_frames(model, "test")
        scores, per_class = test_metrics(test_frames, threshold)
        scores["map50"] = float(test_result.box.map50)
        scores["map50_95"] = float(test_result.box.map)
        paths = image_paths("test")
        latency = cpu_latency(model, paths)
        parameter_count = sum(parameter.numel() for parameter in model.model.parameters())
        metrics_rows.append({
            "model": Path(weight).stem,
            "threshold": threshold,
            **scores,
            "parameters": parameter_count,
            "cpu_ms_per_image": latency,
        })
        for row in per_class:
            class_rows.append({"model": Path(weight).stem, **row})
        print(f"{Path(weight).stem}: threshold={threshold:.2f}, "
              f"P={scores['precision']:.4f}, R={scores['recall']:.4f}, "
              f"F1={scores['f1']:.4f}, mAP50={scores['map50']:.4f}, "
              f"mAP50-95={scores['map50_95']:.4f}, params={parameter_count}, "
              f"CPU={latency:.2f} ms/image")

    pd.DataFrame(metrics_rows).to_csv(RESULTS / "baseline_metrics.csv", index=False)
    pd.DataFrame(class_rows).to_csv(RESULTS / "baseline_per_class.csv", index=False)
    print(pd.DataFrame(metrics_rows).to_string(index=False))


if __name__ == "__main__":
    main()
