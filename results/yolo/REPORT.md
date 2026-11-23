# Sahayak - detector evaluation (YOLO11n vs YOLO26n vs YOLO26s)

All numbers use the same held-out test split, the same matching rule (IoU>=0.5 for P/R/F1) and the same code for every model; mAP/P/R come from the official Ultralytics validator. The confidence threshold is chosen on the validation split (best macro-F1) and then frozen; all numbers are on the test split. P and R are macro-averages over classes at that threshold, F1 is the macro-average of the per-class F1 (as in Ultralytics), so it is not 2PR/(P+R) of the averaged P and R.

| model | stage | P | R | F1 | mAP@0.5 | mAP@0.5:0.95 | best conf |
|---|---|---|---|---|---|---|---|
| yolo11n | zero-shot COCO | 0.635 | 0.539 | 0.575 | 0.588 | 0.433 | 0.23 |
| yolo26n | zero-shot COCO | 0.680 | 0.544 | 0.594 | 0.608 | 0.447 | 0.22 |
| yolo26s | zero-shot COCO | 0.712 | 0.638 | 0.668 | 0.695 | 0.525 | 0.26 |
| yolo11n | fine-tuned | 0.579 | 0.493 | 0.520 | 0.520 | 0.363 | 0.26 |
| yolo26n | fine-tuned | 0.661 | 0.519 | 0.568 | 0.577 | 0.418 | 0.27 |
| yolo26n | fine-tuned (backbone frozen) | 0.683 | 0.520 | 0.576 | 0.588 | 0.422 | 0.28 |
| yolo26s | fine-tuned | 0.696 | 0.563 | 0.614 | 0.627 | 0.458 | 0.32 |

Test set: 1000 images, 6031 labelled objects, 28 classes: person, bicycle, car, motorcycle, bus, truck, stop sign, fire hydrant, bench, dog, cat, backpack, umbrella, suitcase, bottle, cup, bowl, banana, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, keyboard, cell phone.

## Speed and size (this computer, CPU, batch 1)

| model | params (M) | GFLOPs | ms/frame (total) | ms (network only) |
|---|---|---|---|---|
| yolo11n | 2.62 | 6.5 | 94.2 | 88.9 |
| yolo26n | 2.57 | 5.5 | 120.1 | 113.6 |
| yolo26s | 10.01 | 20.9 | 235.2 | 231.1 |

## Training behaviour

Loss values are only comparable within one model family: YOLO11 and YOLO26 use different loss terms (YOLO26 has no DFL term), so read each curve against itself over the epochs.

| run | epochs run | train loss first -> last | val loss first -> last | best val mAP@0.5 |
|---|---|---|---|---|
| ft_yolo11n | 20 | 3.653 -> 3.163 | 3.595 -> 3.587 | 0.524 |
| ft_yolo26n | 20 | 3.041 -> 2.540 | 2.777 -> 2.837 | 0.576 |
| ft_yolo26n_head | 20 | 2.986 -> 2.510 | 2.737 -> 2.758 | 0.586 |
| ft_yolo26s | 20 | 2.645 -> 2.045 | 2.493 -> 2.551 | 0.616 |

## What the numbers say

* **yolo11n** after fine-tuning: mAP@0.5 0.520, F1 0.520 (zero-shot baseline: mAP@0.5 0.588, F1 0.575) -> target mAP@0.5 >= 0.85: **NOT REACHED**, F1 >= 0.85: **NOT REACHED**.
* **yolo26n** after fine-tuning: mAP@0.5 0.577, F1 0.568 (zero-shot baseline: mAP@0.5 0.608, F1 0.594) -> target mAP@0.5 >= 0.85: **NOT REACHED**, F1 >= 0.85: **NOT REACHED**.
* **yolo26s** after fine-tuning: mAP@0.5 0.627, F1 0.614 (zero-shot baseline: mAP@0.5 0.695, F1 0.668) -> target mAP@0.5 >= 0.85: **NOT REACHED**, F1 >= 0.85: **NOT REACHED**.
* YOLO26n vs YOLO11n accuracy difference: +0.057 mAP@0.5. Differences below about 0.01-0.02 are within noise for a test set of this size.
* CPU speed: YOLO26n 120.1 ms vs YOLO11n 94.2 ms per frame (YOLO26n is not faster on this machine). YOLO26n is end-to-end (no NMS step), which removes post-processing time.
* Size trade-off: YOLO26s is 2.0x slower than YOLO26n on CPU (235.2 vs 120.1 ms).

Figures: fig1_loss_curves, fig2_metrics_per_epoch, fig3_before_after, fig4_pr_f1_curves, fig5_confusion, fig6_per_class_ap50, fig7_speed_vs_accuracy (all .png in this folder).