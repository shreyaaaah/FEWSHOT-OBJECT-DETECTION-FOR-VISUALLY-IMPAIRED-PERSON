# Sahayak: Few-Shot Personal Object Recogniser for the Visually Impaired

A capstone-level computer-vision system. A blind or low-vision user **teaches** the system
5-10 personal objects with a short video; afterwards they can ask *"What is this?"*,
*"Where is my bottle?"* or *"What is around me?"* and get a **spoken answer with direction and
distance**, without any retraining and with explicit protection against confident wrong answers.

```
 enroll (5-10 s video)                       live use
 ┌──────────────────────────┐        ┌────────────────────────────────────────┐
 │ frames → blur filter     │        │ camera → YOLO (COCO) boxes             │
 │ → crop → augment         │        │   ├─ crop → embedder → memory.query    │
 │ → embedder (fine-tuned)  │        │   ├─ detector miss? grid search scored │
 │ → PersonalMemory.enroll  │        │   │   against the stored prototypes    │
 │   · multi-prototype bank │        │   └─ size-anchored depth → distance    │
 │   · blocked-CV threshold │        │ → spatial phrase + confidence wording  │
 │   · confusable warning   │        │ → text-to-speech (+ voice commands)    │
 └──────────────────────────┘        └────────────────────────────────────────┘
```

## 1. What is new in this project (and what is not)

Individual building blocks (prototypical networks, DINOv2, LoRA, YOLO, Depth-Anything) are
existing work. **The contribution is the combination and these five design decisions.**
Check each against the literature before you claim it in your report, and phrase them as
"we propose / we study" rather than "first".

| # | Contribution | Where | What to measure |
|---|---|---|---|
| 1 | **Open-set episodic training**: prototypical loss with a *learned reject logit*; every episode contains unknown objects that must be rejected; small SupCon auxiliary term | `src/train/losses.py` | AUROC / FAR vs closed-set prototypical, SupCon, triplet |
| 2 | **Self-calibrating personal memory**: multi-prototype bank, per-object threshold from *blocked cross-validation over the enrollment video*, impostor scores from the user's *other objects + background patches*, re-calibration on every enrollment, margin test, stricter threshold for *safety-critical* objects (medicine) | `src/memory.py` | FAR at matched correct-ID rate vs a global fixed threshold |
| 3 | **Query-conditioned, detector-free search**: "where are my keys?" works for objects that are not COCO classes, by scoring multi-scale windows directly against the stored prototypes when the detector gives nothing | `src/pipeline.py` | Recall on non-COCO objects (keys, wallet, spectacles) |
| 4 | **Known-size-anchored monocular depth**: objects with known real size (COCO table or size the user gives) turn relative depth into metric scale; every object, including unknown-size ones, gets a distance, spoken in coarse bins | `src/depth.py` | MAE and spoken-bin accuracy vs tape measure |
| 5 | **Uncertainty-aware speech + guidance**: wording changes with calibrated confidence ("This is / might be / could be A or B"), framing hints, "turn left" guidance, enrollment-time warning "this looks like your X" | `src/phrases.py`, `src/spatial.py`, `src/memory.py` | Qualitative + small user study |

Plus: degradations that imitate blind-user footage (motion blur, low light, occlusion, tiny objects)
are used both in training and in `src/eval/robustness.py`.

### Honest status of the claims
* **#2 was tested on real ORBIT data and the hypothesis is NOT supported.** After fixing an evaluation bug, the
  calibrated memory lands on, or slightly below, the correct-ID vs false-accept curve of fixed cosine thresholds
  (e.g. C-xv, K=10: ours 49.2% correct / 22.8% false-accept; fixed 0.6 gives 49.2% / 17.8%). What it does offer is an automatic operating
  point with no hand-tuned global threshold. Details: `results/orbit/README.md`.
* **Cross-validation inside a video is optimistic** because neighbouring frames are near-duplicates.
  `cv_folds` / `cv_gap` in `config.yaml` control how pessimistic calibration is. Tune it on validation data.

## 2. Transfer-learning plan (your ablation table)

| Stage | Backbone state | Command tag |
|---|---|---|
| 0 | hand-crafted HSV histogram (no deep learning) | `A0_hsv_hist` |
| A | frozen ImageNet ResNet-50, nearest centroid | `A_resnet50_frozen` |
| B | frozen DINOv2 ViT-S/14, nearest centroid | `B_dinov2_frozen` |
| B' | frozen backbone + trained projection head | `B_head_only` |
| **C (main)** | **last 2 blocks + head trained, open-set loss** | `C_partial_open` |
| D | LoRA (rank 8) on attention qkv + head | `D_lora_open` |
| X | full fine-tuning (expected to overfit) | `X_full_open` |
| losses | open-proto vs proto vs SupCon vs triplet (same Stage C) | `C_partial_*` |

Training detail: backbone LR 1e-5, head LR 1e-3, AdamW + cosine schedule with warm-up, gradient
clipping, L2-normalised 256-d embeddings, heavy augmentation, BatchNorm statistics frozen where used.
Datasets are **split by user** (`splits.json`) so one person never appears in both train and test.

## 3. Data

| Dataset | Role | How |
|---|---|---|
| **ORBIT** (Microsoft, videos by blind/low-vision users) | main train/val/test | download the *benchmark frames* release, then `python tools/prepare_orbit.py --src <orbit> --out data/orbit_instances --every 5 --max-side 320 --dry-run` (drop `--dry-run` when the listing looks right) |
| **Your own recordings** (15-20 objects, 3-4 rooms) | real-world validation | name videos `keys__kitchen.mp4`; `python tools/videos_to_instances.py --videos my_videos --out data/own --user me` |
| **COCO** | *not trained on by you*; it enters through the pretrained YOLO detector and the size table | automatic |
| Distance photos (≈ 60) | distance evaluation | CSV format in `src/eval/distance_eval.py` |

`tools/prepare_orbit.py` was written from ORBIT's documented layout and tested only on a mock
tree. Run it with `--dry-run` and check that users and objects look right before training.

## 4. Quick start

```bash
pip install -r requirements.txt
pytest -q                                    # 66 tests, no GPU / downloads needed

# smoke test the whole training + eval stack in ~1 minute (random weights, numbers meaningless)
python tools/make_synthetic.py --out data/synthetic --instances 24 --frames 20
python -m src.train.train --root data/synthetic --mode partial --loss open --no-pretrained \
       --img-size 56 --episodes 40 --n-way 4 --k-shot 3 --q-query 5 --val-every 20 --tag smoke

# real run (GPU recommended: Colab / Kaggle T4 is enough)
python -m src.train.train --root data/orbit_instances --mode partial --loss open --amp --tag main
python -m src.eval.episodic --root data/orbit_instances --ckpt checkpoints/embedder_main.pt --cross-video
bash scripts/run_experiments.sh data/orbit_instances      # the whole ablation table

# the assistant
cp checkpoints/embedder_main.pt checkpoints/embedder.pt
python -m app.cli enroll --name keys --video clips/keys.mp4 --size-cm 10
python -m app.cli ask --image room.jpg --text "where is my keys"
python -m app.live                                        # webcam + voice
```

Spoken commands: *where is my bottle · what is this · what is around me · save this as keys ·
forget keys · what do you know · help*. Microphone input uses `SpeechRecognition` and falls back to
typing, so everything also works without a microphone.

## 5. Code map: where is what (use this in the viva)

Line numbers are for the commit that added this section; if a file was edited later, search the function name.
To *show the layers live*, run `python tools/show_layers.py` (prints every layer count below from the real code).

### 5.1 "Show me where you did ..."

| Question | File | What to open |
|---|---|---|
| Where is the model created? | `src/embedder.py` | `Embedder` L47 (its `__init__`): DINOv2 ViT-S/14 backbone L60 (`timm.create_model`), metric head L63 (`Linear 384-512, GELU, Linear 512-256`) |
| Which layers did you train? | `src/embedder.py`, `config.yaml` | `_set_trainable` L68. **Partial**: `_unfreeze_last` L85 = last 2 of 12 blocks + final norm + head (`backbone.unfreeze_last_blocks: 2` in `config.yaml`). **LoRA**: `apply_lora` L34 + `LoRALinear` L21 = rank 8 adapters on the qkv of all 12 blocks. **Frozen**: nothing in the backbone |
| Output vector | `src/embedder.py` | `forward` L127: backbone -> head -> L2 normalise (256 numbers, unit length) |
| Learning rates, optimiser | `src/embedder.py`, `src/train/train.py`, `config.yaml` | `param_groups` L113 (backbone 1e-5, head 1e-3 from `config.yaml` `train:`); AdamW `train.py` L130; warm-up 50 steps + cosine decay L131; fp16 `GradScaler` L133; gradient clipping L162 |
| The loss | `src/train/losses.py` | `OpenProtoLoss` L26: prototypes L21, learned reject score `self.thr` L31, logit scale = 1/temperature = 10; `supcon_loss` L60 (weight 0.2, `--lam-supcon`); `batch_hard_triplet` L75 (ablation only). Used in `train.py` L126 and L153 |
| Episodes (N-way K-shot) | `src/train/datasets.py` | `EpisodeSampler` L160, `sample` L181; cross-video support/query from clean/clutter videos: `_kind` L155 and `cross_video` flag |
| User-disjoint splits | `src/train/datasets.py` | `group_of` L40 (user id), `split_instances` L45, `ensure_splits` L74 |
| Augmentation for training | `src/train/datasets.py` | `MotionBlur` L95, `PadSquare` L86, `train_transform` L115 |
| Training loop | `src/train/train.py` | `main` L62, `embed_episode` L32, `evaluate` L43 (validation every 200 episodes) |
| Teaching an object | `src/pipeline.py`, `src/frames.py`, `src/augment.py` | `enroll_from_frames` L113; `select_frames` L42 (25 sharpest); `augment_crop` L47; `random_background_crops` L75 |
| Prototypes (several per object) | `src/memory.py` | `spherical_kmeans` L45, `build_bank` L73 (max 8), `bank_score` L84 |
| Personal threshold (blocked CV) | `src/memory.py` | `_cv_genuine` L234 (10 blocked folds), `_calibrate_all` L256 (5th / 99th percentile, clipped 0.45-0.90, critical +0.05) |
| Decision: high / medium / ambiguous / unknown | `src/memory.py` | `PersonalMemory.query` L322 (margin 0.04); `enroll` L199; `forget` L225; `save` L350 / `load` L371; thresholds set in `config.yaml` `memory:` |
| Identify the held object, scene, "where is my X" | `src/pipeline.py` | `_locate_held_object` L64, `identify_held` L140, `scene` L152, `find_personal` L176 (detector-free sliding windows, `grid_boxes` in `src/imgutil.py` L67), `answer_find` L220, `respond` L267 |
| Spoken wording | `src/phrases.py`, `src/spatial.py` | `identity_phrase` L8 ("This is / might be / not sure"); `side_phrase` L51, `distance_phrase` L26 |
| Distance and direction | `src/depth.py`, `src/spatial.py`, `live.py` | pinhole formula `size_distance` L38; `focal_px` L36, `horizontal_angle` L40; optional depth model `RelativeDepth` L94; general demo: `analyze` in `live.py` L74 |
| Reduce flicker | `live.py`, `src/smoothing.py` | `Announcer` L154 (4 of last 6 frames); personal app: `VoteSmoother` L6 |
| Confidence thresholds of the demo | `live.py` | `keep` L210 (0.30 shown, 0.40 spoken, 0.55 for person) |
| "Where is my laptop" (voice or typed) | `live.py` | `find_target` L215, `answer` L230, `Segmenter` L247 (cuts speech from the microphone), `start_question_listeners` L286 (Google recogniser via SpeechRecognition) |
| Two phones: "your mobile" vs "a cell phone" | `src/personal_scene.py`, `live.py` | `PersonalTagger` L14 (`tag` L32) re-names YOLO boxes using the personal memory; wiring in `live.py` `main` L379 |
| Speech output | `src/tts.py`, `src/speaker.py` | `Voice` L9 (SAPI, pyttsx3, PowerShell fallbacks); `VoiceSpeaker` L9 used by the personal app |
| Voice commands of the personal app | `src/voice.py`, `app/live.py` | `parse_command` L64, `resolve_target` L104, `listen_once` L129; `LiveApp` L27 (keys E teach, V command, I identify, S scene) |
| Detector wrapper (YOLO) | `src/detector.py`, `config.yaml` | `Detector` L14; weights `yolo26n.pt` set in `config.yaml` `detector:` |

### 5.2 How many layers, and where

| Model | Layers | Parameters | Where it comes from | What we trained |
|---|---|---|---|---|
| YOLO11n | 181 | 2.62 M | `ultralytics` library (`yolo11n.pt`) | Zero-shot, and fine-tuned with `lab.py` `train_one` L314 |
| YOLO26n | 260 (120 after fusing for inference) | 2.57 M | `ultralytics` (`yolo26n.pt`) | same; **used in the demo** |
| YOLO26s | 260 | 10.01 M | `ultralytics` (`yolo26s.pt`) | same |
| DINOv2 ViT-S/14 backbone | patch embedding + 12 transformer blocks + final LayerNorm; each block = LayerNorm, attention (6 heads, qkv 384->1152), LayerNorm, MLP (384->1536->384) | 21,628,800 | `timm` model `vit_small_patch14_dinov2.lvd142m` (pretrained by Meta) | Partial: blocks 10-11; LoRA: qkv of all 12; Frozen: none |
| Metric head | 2 linear layers (384->512, GELU, 512->256) + L2 normalise | 328,448 | `src/embedder.py` L63 | always trained (except baseline B) |
| Trainable parameters | frozen head only 328,448; **partial 3,879,680**; **LoRA 475,904**; full 21,957,248 | | `Embedder.n_trainable` L123 | |

The YOLO layers are defined inside the ultralytics library, not in our code. Our code in `lab.py` loads those weights, evaluates and fine-tunes them.

### 5.3 Plots and result files: which code made them

| Output | Made by | What it shows |
|---|---|---|
| `results/yolo/fig1_loss_curves.png` | `lab.py` `cmd_report` L394 (savefig L424) | training and validation loss per epoch for the fine-tuned detectors |
| `fig2_metrics_per_epoch.png` | `lab.py` L441 | validation precision, recall and mAP per epoch |
| `fig3_before_after.png` | `lab.py` L463 | zero-shot COCO weights vs fine-tuned, test set, per metric |
| `fig4_pr_f1_curves.png` | `lab.py` L483 | precision-recall and F1-vs-confidence curves (used to choose the threshold on validation) |
| `fig5_confusion.png` | `lab.py` L508 | confusion matrix per model (which classes are mistaken for which) |
| `fig6_per_class_ap50.png` | `lab.py` L525 | AP@0.5 per class |
| `fig7_speed_vs_accuracy.png` | `lab.py` L542 (speed from `cmd_speed` L283) | accuracy against CPU milliseconds per frame |
| `results/yolo/REPORT.md` | `lab.py` `cmd_report` | all detector tables (P / R / F1 / mAP / speed), written automatically |
| `results/yolo/operating_points.md` | `tools/operating_points.py` | precision / recall / F1 at chosen confidence thresholds |
| `results/yolo/speed_laptop.md` | `lab.py speed` | speed on the project laptop CPU |
| `results/orbit/orbit_results.png` | plot cell of `colab/sahayak_orbit_pipeline_colab.ipynb` (reads `reports/episodic_*_test.json`) | left: closed-set accuracy vs K; right: open-set AUROC vs K, for all models |
| `results/orbit/README.md` tables | copied from `results/orbit/logs/*.txt`, which are the outputs of `src/eval/episodic.py` `run` L50 (`main` L126) | accuracy, AUROC, correct-ID and false-accept rates per K |
| ORBIT baselines A0, A, B | same script: `--frozen-backbone hsv_hist` (A0, `HistEmbedder` in `src/eval/common.py` L63), `--frozen-backbone resnet50.a1_in1k` (A), `--frozen-backbone vit_small_patch14_dinov2.lvd142m` (B, frozen DINOv2); fine-tuned models: `--ckpt <file>.pt` | |
| Robustness (blur, low light, occlusion) | `src/eval/robustness.py` | code ready, not yet run on real data |
| Distance against a tape measure | `src/eval/distance_eval.py` | code ready, not yet run on real data |
| Memory diagnosis | `tools/diagnose_memory.py` | read-only check of thresholds and score ranges |

### 5.4 Pipeline of the experiments

1. Data: `eval/make_dataset.py` (COCO 28 classes for YOLO) and `tools/prepare_orbit.py` (ORBIT frames into `<user>__<object>/` folders; clean_ / clutter_ prefixes).
2. Detector study: `lab.py all` (train, evaluate, speed, report) via `colab/sahayak_yolo_study.ipynb`.
3. Train the embedder: `src/train/train.py` (`--mode frozen|partial|lora|full`, `--loss open`, `--cross-video`, `--episodes 1500`; unspecified values come from `config.yaml` `train:`).
4. Evaluate: `src/eval/episodic.py` (`--k-shots 1 3 5 10`, 200 episodes, 5-way + 3 unknown objects by default, `--cross-video`).
5. Notebooks that run steps 1-4 end to end: `colab/sahayak_orbit_pipeline_colab.ipynb` (and `colab/sahayak_orbit_reeval_colab.ipynb` for evaluation only).
6. Demos: `live.py` (general objects + your taught objects + questions) and `app/live.py` (teach and identify).

### 5.5 File index

```
config.yaml                              all hyper-parameters (backbone, memory, enrolment, training, detector)
live.py                                  general live demo: YOLO + direction + distance + voting + questions + taught names
lab.py                                   YOLO detector study (train, evaluate, speed, figures, REPORT.md)
src/embedder.py                          DINOv2 backbone + metric head; frozen / partial / LoRA / full
src/train/{datasets,losses,train}.py     episodic training, losses, splits
src/memory.py                            personal memory: prototypes, thresholds, decisions
src/pipeline.py                          teach, identify, find, scene, answers
src/personal_scene.py                    names YOLO boxes with taught objects
src/detector.py  src/scene.py            YOLO wrapper; scene description (analyze, summary) used by app/detect.py
src/spatial.py  src/depth.py             direction, distance phrases, pinhole / depth
src/phrases.py  src/voice.py             spoken wording; command parsing and microphone
src/tts.py  src/speaker.py  src/audio.py text-to-speech (tts + speaker are the working ones; audio.py is the older pyttsx3 version, now only used for the `beep` sound)
src/factory.py  src/config.py  src/schema.py   wiring of the pipeline, config loader, small data classes (Det, Match)
src/frames.py  src/augment.py  src/imgutil.py   frame selection, augmentation, image helpers
src/smoothing.py                         temporal vote smoothing for the personal app
src/eval/{episodic,robustness,distance_eval,common}.py   all evaluations
app/{cli,live,detect}.py                 user interfaces
eval/make_dataset.py                     exports the 28-class COCO set (8k train / 1k val / 1k test) for the YOLO study
tools/                                   prepare_orbit, diagnose_memory, operating_points, show_layers, videos_to_instances, make_synthetic
tests/                                   83 unit and pipeline tests
results/yolo, results/orbit              real results, figures and logs
colab/ kaggle/                           notebooks that produced the results (YOLO study, ORBIT pipeline, re-evaluation)
```

## 6. Evaluation checklist for the report

1. **Few-shot**: K = 1, 3, 5, 10, 95% CI over episodes (`episodic.py`).
2. **Open-set safety**: AUROC and false-accept rate on never-enrolled objects (the key safety metric).
3. **Cross-video**: enrol on clean videos, test on cluttered ones (`--cross-video`).
4. **Robustness** curves: blur, motion blur, low light, occlusion, small objects.
5. **Distance**: MAE and spoken-bin accuracy; size-only vs fused with depth (`--depth`).
6. **Detector-free search recall** on non-COCO objects, with and without the grid fallback.
7. **Latency**: per-frame time on your target machine (identify, scene, enroll).
8. **Ablations**: the table in section 2, plus memory calibration on/off, temporal smoothing on/off.
9. **User study** (optional): 3-5 volunteers, blindfolded or simulated low vision, SUS score.
10. **Ethics**: privacy (everything runs on-device), failure handling, over-trust, never use as the
    sole safeguard for medicine.

## 7. 12-week schedule

| Week | Milestone |
|---|---|
| 1 | Literature (ProtoNets, SimpleShot, ORBIT, DINOv2, LoRA, open-set few-shot), set up, run `pytest` |
| 2 | Download ORBIT, `prepare_orbit.py`, check splits, smoke-run training |
| 3 | Baselines 0, A, B with `episodic.py` |
| 4-5 | Stage C training; loss ablation |
| 6 | LoRA, head-only, full fine-tune; tune `cv_folds`, thresholds |
| 7 | Enrollment pipeline on real videos; detector choice (YOLO vs YOLO-World) |
| 8 | Depth: tape-measure data collection, `distance_eval.py` |
| 9 | Live app, audio polish, latency optimisation |
| 10 | Own dataset recording and real-world tests |
| 11 | Robustness, error analysis, user study |
| 12 | Report, slides, demo video |

## 8. Known limitations (state these in your viva)

* A single camera cannot give exact distance; the system reports coarse bins and says "roughly" when unsure.
* COCO has no keys / wallet / medicine: these rely on contribution #3 or on YOLO-World.
* Frames from one video are highly correlated; cross-video evaluation is the honest protocol.
* The system must not be the only safeguard for safety-critical items.
* Windows/macOS/Linux TTS voices differ; `pyttsx3` quality is basic (swap in Coqui TTS if needed).

## 9. Likely viva questions

* Why metric learning rather than a softmax classifier? (new objects need no retraining)
* Why do the last blocks help, and when does fine-tuning hurt? (see Stage X vs C vs D)
* How do you reject unknown objects, and how is the threshold chosen?
* What happens with two similar mugs? (margin test, enrollment warning)
* How is distance computed and how wrong can it be?
* What did *not* work?

## Detector study: YOLO11n vs YOLO26n vs YOLO26s
`lab.py` compares the three detectors on 28 COCO classes relevant to blind users: zero-shot baseline vs
fine-tuned, with precision, recall, F1, mAP@0.5, mAP@0.5:0.95, CPU speed, loss curves, PR/F1 curves,
confusion matrix and an auto-written `lab_out/REPORT.md`. The confidence threshold is chosen on the
validation split and all numbers are reported on the test split. Run it on Colab with
`colab/sahayak_yolo_study.ipynb` (or `python lab.py all --data datasets/sahayak28/data.yaml --epochs 20`).
Build the dataset first with `python eval/make_dataset.py`. `eval/evaluate_finetuned.py` is deprecated; older scripts are in `eval/old/`.
Real results go in `results/yolo/` (speed on the project laptop: `results/yolo/speed_laptop.md`).

### Live demo (general objects)
`python live.py --weights yolo26n.pt` opens the camera and speaks by itself: it announces the object, direction
and rough distance when the scene changes (and every ~20 s while objects stay in view). An object is announced only
after it was seen in 3 of the last 5 frames. Default `--conf 0.15`; uncertain items are spoken as "maybe". Press Q to quit.
Just say "where is my laptop" (microphone; one-time `pip install SpeechRecognition sounddevice`; the recogniser is Google's online service, so it needs internet) or type it in the console: it answers with direction and distance and keeps looking for 25 s if the object is not in view. It does not listen while it is speaking. `--no-mic` turns the microphone off. The microphone path is unit-tested only for the phrase cutting; it has not been tried on real hardware yet.
Taught personal objects (from `python -m app.live`) are also recognised in this view: each detected box is checked against your memory, so one phone can be announced as "your mobile" and another as "a cell phone" (`--no-personal` turns it off). Objects the detector does not box are only found when held in the middle of the picture.
False alarms are reduced by `--conf 0.30` (shown) / `--speak-conf 0.40` (spoken) and a stricter 0.55 minimum for 'person'.
This demo is YOLO only; personal objects are taught with `python -m app.cli enroll ...` (see below).

## Personal-object recogniser on ORBIT (the novelty)
Run on real ORBIT data (user-disjoint split, enrol from clean videos, query cluttered videos, 5-way + 3 unknown objects, 200 episodes).
Real results, the honest reading and the logs are in `results/orbit/` (start with `results/orbit/README.md`).
Summary (test): frozen DINOv2 gives the best closed-set accuracy (81% at K=10; colour histogram 42%, ResNet-50 70%); fine-tuning
(last 2 blocks, cross-video episodes) improves rejection of unknown objects (AUROC 0.77-0.78 vs 0.72-0.73) but not identification;
the calibrated memory does not beat a well-chosen fixed threshold. Reproduce with `colab/sahayak_orbit_pipeline_colab.ipynb`
(resume-safe, saves to Google Drive) or `colab/sahayak_orbit_reeval_colab.ipynb` (evaluation only).
