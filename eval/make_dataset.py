"""Download and export COCO-2017 train/validation data for Sahayak's 28 classes."""
from pathlib import Path
import random

import fiftyone as fo
import fiftyone.zoo as foz
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_28 = PROJECT_ROOT / "datasets" / "sahayak28"
DATASET_COCO80 = PROJECT_ROOT / "datasets" / "sahayak28_coco80"

CLASSES = [
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "stop sign",
    "fire hydrant", "bench", "dog", "cat", "backpack", "umbrella", "suitcase",
    "bottle", "cup", "bowl", "banana", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "keyboard", "cell phone",
]

# COCO's contiguous YOLO class order (category IDs 0 through 79).
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


def detections(sample):
    labels = sample["ground_truth"]
    return labels.detections if labels is not None else []


def count_boxes(collection):
    counts = {name: 0 for name in CLASSES}
    for sample in collection.iter_samples():
        for detection in detections(sample):
            if detection.label in counts:
                counts[detection.label] += 1
    return counts


def export_yolo(collection, destination, split, class_names):
    collection.export(
        export_dir=str(destination),
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="ground_truth",
        split=split,
        classes=list(class_names),
    )


def write_yaml(destination, root, class_names, include_train):
    content = {
        "path": str(root.resolve()),
        "train": "images/train" if include_train else None,
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(class_names)},
    }
    if not include_train:
        content.pop("train")
    with (destination / "data.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(content, stream, sort_keys=False, allow_unicode=True)


def report_counts(collections):
    train_counts = None
    for split, collection in collections.items():
        per_class = count_boxes(collection)
        box_total = sum(per_class.values())
        print(f"{split}: {len(collection)} images, {box_total} boxes")
        for name in CLASSES:
            print(f"  {name}: {per_class[name]} boxes")
        if split == "train":
            train_counts = per_class
    if train_counts is not None:
        low = [name for name in CLASSES if train_counts[name] < 50]
        if low:
            print("WARNING: fewer than 50 training boxes: " + ", ".join(low))
        else:
            print("All classes have at least 50 training boxes.")


def exported_image_names(dataset_root, split):
    directory = dataset_root / "images" / split
    return {path.name for path in directory.rglob("*") if path.is_file()}


def report_split_sizes(dataset_root):
    names = {split: exported_image_names(dataset_root, split) for split in ("train", "val", "test")}
    counts = {split: len(split_names) for split, split_names in names.items()}
    total = sum(counts.values())
    print("Exported image counts and percentages:")
    for split in ("train", "val", "test"):
        percentage = 100.0 * counts[split] / total if total else 0.0
        print(f"  {split}: {counts[split]} images ({percentage:.2f}%)")

    val_test_overlap = names["val"] & names["test"]
    train_val_overlap = names["train"] & names["val"]
    train_test_overlap = names["train"] & names["test"]
    if val_test_overlap:
        raise RuntimeError(f"Val/test image filename overlap: {sorted(val_test_overlap)[:10]}")
    if train_val_overlap or train_test_overlap:
        overlap = sorted(train_val_overlap | train_test_overlap)
        raise RuntimeError(f"Train and validation image filenames overlap: {overlap[:10]}")
    print("Filename overlap check: val/test do not overlap; train is disjoint from val/test.")


def main():
    train = foz.load_zoo_dataset(
        "coco-2017",
        split="train",
        classes=CLASSES,
        only_matching=True,
        max_samples=8000,
        dataset_name="sahayak28-coco-train",
    )
    validation = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        classes=CLASSES,
        only_matching=True,
        max_samples=2000,
        shuffle=True,
        seed=42,
        dataset_name="sahayak28-coco-validation",
    )

    validation_ids = [sample.id for sample in validation]
    random.Random(42).shuffle(validation_ids)
    validation_count = len(validation_ids)
    if validation_count >= 2000:
        val_count = test_count = 1000
    else:
        val_count = (validation_count + 1) // 2
        test_count = validation_count - val_count
        print(
            f"Fewer than 2000 matching COCO validation images found ({validation_count}); "
            f"using all images split as val={val_count}, test={test_count}."
        )
    val = validation.select(validation_ids[:val_count])
    test = validation.select(validation_ids[val_count:val_count + test_count])

    DATASET_28.mkdir(parents=True, exist_ok=True)
    DATASET_COCO80.mkdir(parents=True, exist_ok=True)
    export_yolo(train, DATASET_28, "train", CLASSES)
    export_yolo(val, DATASET_28, "val", CLASSES)
    export_yolo(test, DATASET_28, "test", CLASSES)
    write_yaml(DATASET_28, DATASET_28, CLASSES, include_train=True)

    # These exports contain the same val/test images and filtered labels, but use
    # the original contiguous COCO-80 class indices for pretrained 80-class models.
    export_yolo(val, DATASET_COCO80, "val", COCO80)
    export_yolo(test, DATASET_COCO80, "test", COCO80)
    write_yaml(DATASET_COCO80, DATASET_COCO80, COCO80, include_train=False)

    report_counts({"train": train, "val": val, "test": test})
    report_split_sizes(DATASET_28)
    print(f"28-class dataset: {DATASET_28}")
    print(f"COCO-80 val/test dataset: {DATASET_COCO80}")


if __name__ == "__main__":
    main()
