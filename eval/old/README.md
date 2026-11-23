# Superseded scripts

`compare_models.py`, `make_report.py` and `train.py` were the first version of the detector comparison.
They are superseded by `lab.py` in the project root (same protocol, plus val-chosen threshold, the official
Ultralytics validator, PR/F1 curves and an auto-written report). Kept only for history; do not use them for results.
`eval/make_dataset.py` (builds the 28-class COCO dataset) is still used.
