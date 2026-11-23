# Sahayak evaluation workflow

The detector study is run by `lab.py` in the project root (see the main README, "Detector study").
The only script here still in use is `make_dataset.py`:

```powershell
python -m eval.make_dataset
```

It downloads and exports up to 8,000 matching COCO-2017 train images and 2,000 matching validation images
(randomly split with seed 42 into 1,000 `val` and 1,000 `test`) for the 28 classes used.
Older comparison/training scripts are in `eval/old/` (superseded by `lab.py`).
`evaluate_finetuned.py` is disabled: it scored 28-class models against COCO-80 ids.
