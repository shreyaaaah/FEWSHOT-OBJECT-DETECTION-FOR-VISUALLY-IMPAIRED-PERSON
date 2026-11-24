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

## 5. Code map

```
config.yaml                    all hyper-parameters
src/memory.py                  novelty #2 (+ AUROC, spherical k-means)
src/train/{datasets,losses,train}.py   episodic transfer learning, novelty #1
src/embedder.py                backbone + head; frozen / partial / LoRA / full
src/pipeline.py                enrollment, identify, find (novelty #3), scene, answers
src/depth.py  src/spatial.py   novelty #4 and direction / clock / distance phrases
src/phrases.py src/voice.py    novelty #5, command parsing, microphone
src/detector.py                COCO YOLO or open-vocabulary YOLO-World
src/eval/{episodic,robustness,distance_eval}.py   all evaluations
app/{cli,live}.py              user interfaces
tools/                         dataset preparation
tests/                         unit + pipeline tests
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
