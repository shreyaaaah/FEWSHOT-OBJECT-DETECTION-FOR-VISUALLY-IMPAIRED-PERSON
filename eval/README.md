# Sahayak evaluation workflow

Run each command from the project root. The scripts use project-relative paths.
Install the required Python packages and place the pretrained `.pt` weights in
the working directory before running model scripts. Dataset/model commands may
download data or weights when you run them yourself.

1. Download and export up to 8,000 matching COCO-2017 train images and 2,000
   matching COCO-2017 validation images, randomly split with seed 42 into 1,000
   `val` and 1,000 `test` images. If fewer than 2,000 matching validation images
   are available, all are split as evenly as possible between `val` and `test`.
   The same validation/test image lists are exported to both label layouts:

   ```powershell
   python -m eval.make_dataset
   ```

2. Compare pretrained baseline models:

   ```powershell
   python -m eval.compare_models
   ```

3. Fine-tune each model (omit a command if that model is not needed):

   ```powershell
   python -m eval.train --model yolo11n.pt --epochs 60 --fraction 1.0
   python -m eval.train --model yolo26n.pt --epochs 60 --fraction 1.0
   python -m eval.train --model yolo26s.pt --epochs 60 --fraction 1.0
   ```

   Resume an interrupted run with the same model name:

   ```powershell
   python -m eval.train --model yolo26n.pt --resume
   ```

4. Evaluate available fine-tuned checkpoints on validation-selected thresholds:

   ```powershell
   python -m eval.evaluate_finetuned --models yolo11n yolo26n yolo26s
   ```

5. Build plots and `results/RESULTS.md` from the generated CSVs and run artifacts:

   ```powershell
   python -m eval.make_report
   ```
