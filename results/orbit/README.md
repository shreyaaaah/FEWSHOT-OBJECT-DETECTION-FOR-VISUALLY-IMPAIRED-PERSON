# ORBIT personal-object recogniser: results (test split)

Real numbers copied unedited from the Kaggle logs in `logs/`. Protocol: user-disjoint split (214 train / 59 val / 55 test objects), test = 55 objects,
47,569 frames; 5-way episodes + 3 unknown objects, 200 episodes; support = K frames from clean videos, queries = frames from cluttered videos
(`--cross-video`). Chance level for closed-set accuracy is 20%. Mean +- 95% CI. `orbit_results.png` plots the same numbers.

| model | metric | K=1 | K=3 | K=5 | K=10 |
|---|---|---|---|---|---|
| A0 colour histogram (no deep learning) | closed-set acc % | 32.0 +- 1.7 | 36.0 +- 1.9 | 37.5 +- 2.2 | 42.1 +- 2.5 |
| | open-set AUROC (max-cosine) | 0.473 | 0.508 | 0.504 | 0.519 |
| A ResNet-50 (ImageNet, frozen) | closed-set acc % | 50.1 +- 2.3 | 62.7 +- 2.1 | 69.8 +- 1.9 | 69.7 +- 2.0 |
| | open-set AUROC (max-cosine) | 0.683 | 0.717 | 0.734 | 0.762 |
| B DINOv2 ViT-S/14 (frozen) | closed-set acc % | 61.2 +- 2.1 | 75.1 +- 1.7 | 80.7 +- 1.5 | 81.3 +- 1.7 |
| | open-set AUROC (max-cosine) | 0.643 | 0.711 | 0.733 | 0.723 |
| C fine-tuned, last 2 blocks, open-set loss | closed-set acc % | 61.1 +- 2.0 | 74.0 +- 1.6 | 76.5 +- 1.6 | 77.0 +- 1.7 |
| | open-set AUROC (max-cosine) | 0.673 | 0.763 | 0.760 | 0.765 |
| C-xv same, trained with cross-video episodes | closed-set acc % | 62.4 +- 2.2 | 70.6 +- 1.9 | 75.5 +- 1.8 | 76.5 +- 1.8 |
| | open-set AUROC (max-cosine) | 0.706 | 0.762 | 0.790 | 0.780 |
| D LoRA (rank-limited, 0.48 M trainable params) | closed-set acc % | 59.1 +- 1.9 | 68.6 +- 1.7 | 73.3 +- 1.5 | 73.7 +- 1.8 |
| | open-set AUROC (max-cosine) | 0.668 | 0.745 | 0.752 | 0.743 |

## Honest reading
* Learned features beat the colour histogram clearly. **Frozen DINOv2 (B) is the best closed-set model** (81% at K=10): none of our fine-tuned variants (C, C-xv, D) improved closed-set accuracy over it (differences 0 to -5 points; at K>=5 C, C-xv and D are lower).
* Fine-tuning helps **unknown-object rejection**: open-set AUROC (max-cosine) at K=5/10 is 0.76-0.79 for C/C-xv vs 0.73/0.72 for frozen DINOv2; C-xv (trained with cross-video episodes) is the best at K=5 and 10 (0.790 / 0.780).
  Cross-video training did not change closed-set accuracy (62.4 vs 61.1 at K=1, 75.5 vs 76.5 at K=5: inside the confidence intervals).
* LoRA (D) was lower than partial fine-tuning (C) on closed-set accuracy (73.3 vs 76.5 at K=5); its AUROC is about the same as C (differences inside the confidence intervals).
* Fine-tuning cost vs benefit in one sentence: it buys better rejection of objects the user has NOT enrolled, not better identification of enrolled ones.

## Evaluation bug found and fixed (important)
In the first evaluation logs (`logs/pipeline_log_eval_*`, `logs/stageC_*`, `logs/baselines_*`), the row **"ours" (calibrated memory) shows a correct-ID rate of exactly 0.0 +- 0.0. Those "ours" numbers are INVALID.**
`src/eval/episodic.py` compared the memory's lower-cased object name with the raw ORBIT name (`P189__keys`), so a correct identification could never match.
Synthetic tests passed because their names were lower case. Fixed (`norm_name`), with a regression test (`tests/test_episodic_names.py`).
The checkpoints were re-evaluated after the fix (`logs/reeval_fixed_eval_tables_xv_C_B_D.txt`, Colab, same 200 episodes). Closed-set accuracy, AUROC and the fixed-threshold rows were not affected by the bug
(re-evaluated closed-set accuracy differs by about 1 point from the first run: C-xv 70.8/75.0/75.6, C 73.0/75.1/75.6, B 74.7/80.2/80.3, D 68.2/72.6/72.8 at K=3/5/10; the K=1 lines were not in the pasted log).

## Corrected test results of the calibrated memory ("ours") vs fixed cosine thresholds (correct-ID rate % / false-accept rate % on unknown objects)
| model | K | ours (calibrated memory) | fixed threshold 0.5 | fixed threshold 0.6 |
|---|---|---|---|---|
| B frozen DINOv2 | 5 | 34.2 / 10.3 | 32.3 / 9.0 | 15.8 / 1.8 |
| | 10 | 38.4 / 14.5 | 36.5 / 12.9 | 18.4 / 4.1 |
| C fine-tuned | 5 | 41.8 / 16.3 | 52.1 / 22.5 | 36.4 / 9.2 |
| | 10 | 45.7 / 21.0 | 53.4 / 24.4 | 37.2 / 12.0 |
| C-xv (cross-video training) | 5 | 48.1 / 18.4 | 62.1 / 32.4 | 46.4 / 13.7 |
| | 10 | 49.2 / 22.8 | 63.2 / 36.3 | 49.2 / 17.8 |
| D LoRA | 5 | 36.9 / 15.4 | 46.4 / 20.1 | 28.5 / 7.8 |
| | 10 | 41.1 / 20.8 | 48.1 / 23.1 | 27.9 / 10.6 |

### Honest reading
* After the fix the memory works (correct-ID 14-17% at K=1 up to 49% at K=10), and it chooses an operating point **between the fixed thresholds 0.5 and 0.6 without any hand-picked global threshold**.
* **It does NOT beat a well-chosen fixed threshold.** At the same false-accept rate its correct-ID rate lies on, or slightly below, the line through the fixed-threshold points (e.g. C-xv K=10: ours 49.2% at FAR 22.8%, fixed 0.6 gives 49.2% at FAR 17.8%; for frozen DINOv2 ours is about equal to threshold 0.5).
  The calibrated probability also ranks known vs unknown objects slightly worse (AUROC 0.69-0.75) than the raw max-cosine score (0.72-0.78). The hypothesis "calibrated memory is better than a fixed threshold" is therefore **not supported** on the test split; what we can claim is that it picks a reasonable operating point automatically.
* Absolute performance is modest for this hard setting (unseen users, clean enrolment, cluttered queries): about 40-50% correct identification at 15-23% false acceptance for the fine-tuned models, 38% at 14% for frozen DINOv2.
* The earlier validation-split diagnosis (`logs/pipeline_log_diag_*`) suggested the memory beats the best single global threshold; it measured acceptance only (not identity or margin) and used FAR-matched global thresholds chosen on that same split, so it was too optimistic. Do not quote it as evidence.
* Run-to-run: re-evaluating the same checkpoints on a different machine changed closed-set accuracy by up to about 1 point, so differences smaller than that between models are not meaningful.

## Other notes
* The plain-prototypical-loss run's test evaluation was not completed and the earlier checkpoints were lost with the first Kaggle session; the numbers above come from the final committed run (`pipeline_*` logs); the earlier run's C results (61.7/73.5/76.8/77.0) agree within the confidence intervals.
