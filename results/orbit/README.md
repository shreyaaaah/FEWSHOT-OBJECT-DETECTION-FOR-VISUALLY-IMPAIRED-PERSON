# ORBIT personal-object recogniser: results obtained so far (test split)

Real numbers copied unedited from the Kaggle logs in `logs/`. Protocol: user-disjoint split (214 train / 59 val / 55 test objects), test = 55 objects,
47,569 frames; 5-way episodes + 3 unknown objects, 200 episodes; support = K frames from clean videos, queries = frames from cluttered videos
(`--cross-video`). Chance level for closed-set accuracy is 20%. Mean +- 95% CI.

| model | metric | K=1 | K=3 | K=5 | K=10 |
|---|---|---|---|---|---|
| A0 colour histogram (no deep learning) | closed-set acc % | 32.0 +- 1.7 | 36.0 +- 1.9 | 37.5 +- 2.2 | 42.1 +- 2.5 |
| | open-set AUROC (max-cosine) | 0.473 | 0.508 | 0.504 | 0.519 |
| A ResNet-50 (ImageNet, frozen) | closed-set acc % | 50.1 +- 2.3 | 62.7 +- 2.1 | 69.8 +- 1.9 | 69.7 +- 2.0 |
| | open-set AUROC (max-cosine) | 0.683 | 0.717 | 0.734 | 0.762 |
| B DINOv2 ViT-S/14 (frozen) | closed-set acc % | 61.2 +- 2.1 | 75.1 +- 1.7 | 80.7 +- 1.5 | 81.3 +- 1.7 |
| | open-set AUROC (max-cosine) | 0.643 | 0.711 | 0.733 | 0.723 |
| C DINOv2 fine-tuned (last 2 blocks, open-set loss, 1500 episodes) | closed-set acc % | 61.7 +- 1.9 | 73.5 +- 1.6 | 76.8 +- 1.6 | 77.0 +- 1.6 |
| | open-set AUROC (max-cosine) | 0.689 | 0.761 | 0.763 | 0.775 |

## Honest reading
* Learned features beat the colour histogram clearly; frozen DINOv2 is the best closed-set model (81% at K=10); the colour histogram cannot reject unknown objects (AUROC about 0.5).
* Fine-tuning (C) improved unknown-object rejection (AUROC +0.03 to +0.05 over frozen DINOv2) but did NOT improve closed-set accuracy (equal at K=1, 2-4 points lower at K>=3).
* The calibrated memory ("ours") accepted NOTHING (correct-ID rate 0.0%) for every model in this clean-to-clutter protocol: its per-object thresholds, calibrated on the enrolment video,
  are too strict for cluttered query frames. This is a real negative finding; the fixed-threshold rows in the logs are better. A fix, tuned on the validation split only, is in progress.
* Training episodes of C mixed clean and clutter frames while the test enrols from clean and queries clutter (domain shift); a cross-video training option (`--cross-video`) was added afterwards and has not been evaluated yet.
* The plain-prototypical-loss training run (`logs/stageC_proto_train_log_eval_not_completed.txt`) finished training (best val score 0.880) but its test evaluation was interrupted; the LoRA run's test evaluation is not available either.
* Checkpoints and JSON reports from these runs were lost when the Kaggle session was reset; only the logs here survive.
