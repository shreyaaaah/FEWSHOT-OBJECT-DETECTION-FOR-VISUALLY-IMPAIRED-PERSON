# YOLO detector study results (28 COCO classes, test split of 1000 images, 6031 boxes)

`REPORT.md` and `fig1`-`fig7` are the unedited outputs of `lab.py` run on Google Colab (T4 GPU for training, Colab CPU for the speed table).
Headline: the 85% target is NOT reached (best mAP@0.5 0.627 fine-tuned, 0.695 zero-shot yolo26s); fine-tuning on the COCO subset did not beat the pretrained weights.

Speed note: the speed table in `REPORT.md` and `fig7` was measured on the Colab CPU (yolo11n 94.2, yolo26n 120.1, yolo26s 235.2 ms/frame).
The same measurement on the project laptop is in `speed_laptop.md` (118.6 / 108.7 / 283.2 ms) and the ranking of the two nano models is reversed,
so speed is machine-dependent; quote the laptop numbers for the demo machine.
