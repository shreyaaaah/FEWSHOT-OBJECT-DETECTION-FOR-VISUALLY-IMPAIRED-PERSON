# Precision / recall / F1 at different confidence thresholds (test split, macro over 28 classes)

Computed from the saved `curves_*.npz` of the Colab run (`python tools/operating_points.py lab_out`). Same test images as REPORT.md.
Precision and recall trade off against each other through the confidence threshold: always quote the threshold and the recall with any precision.
These thresholds are read off the test-split curves to describe the trade-off; they are not a tuned result (the official numbers in REPORT.md use a threshold chosen on the validation split).

## Pretrained models (zero-shot COCO weights)

| model | conf | precision | recall | F1 |
|---|---|---|---|---|
| yolo26n | 0.15 (demo) | 0.592 | 0.591 | 0.585 |
| yolo26n | 0.25 | 0.718 | 0.526 | 0.593 |
| **yolo26n** | **0.284** | **0.750** | **0.509** | **0.590** |
| yolo26n | 0.30 | 0.763 | 0.501 | 0.587 |
| yolo26n | 0.40 | 0.827 | 0.449 | 0.557 |
| yolo26n | 0.50 | 0.877 | 0.402 | 0.518 |
| yolo11n | 0.25 | 0.650 | 0.530 | 0.576 |
| yolo11n | 0.362 (P=0.75) | 0.750 | 0.469 | 0.558 |
| yolo26s | 0.25 | 0.702 | 0.643 | 0.666 |
| yolo26s | 0.316 (P=0.75) | 0.750 | 0.607 | 0.664 |
| yolo26s | 0.50 | 0.848 | 0.508 | 0.619 |

## Fine-tuned models, first threshold with precision >= 0.75

| model | conf | precision | recall | F1 |
|---|---|---|---|---|
| yolo11n | 0.477 | 0.750 | 0.373 | 0.473 |
| yolo26n | 0.376 | 0.750 | 0.467 | 0.556 |
| yolo26n (backbone frozen) | 0.363 | 0.751 | 0.479 | 0.567 |
| yolo26s | 0.401 | 0.751 | 0.524 | 0.603 |

At equal precision 0.75, pretrained yolo26n keeps higher recall (0.509) than yolo11n (0.469) and than fine-tuned yolo26n (0.467):
YOLO26n is the better of the two nano models, and fine-tuning on the COCO subset did not help.
