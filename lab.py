"""Sahayak detector lab: compare YOLO11n vs YOLO26n, fine-tune, evaluate, plot.

  python lab.py speed                                   # params, GFLOPs, CPU ms per frame
  python lab.py eval  --data data/objects/data.yaml     # zero-shot baseline (COCO weights)
  python lab.py all   --data data/objects/data.yaml --epochs 60      # EVERYTHING + report
  python lab.py train --data D --model yolo26n.pt --tag ft_yolo26n   # one training run
  python lab.py report                                  # figures + REPORT.md from saved results

Same data, same settings, same metric code for every model, so the comparison is fair.
Outputs: lab_out/ (numbers, figures, REPORT.md)   lab_runs/ (training runs, weights)
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import yaml

OUT = Path("lab_out")
RUNS = Path("lab_runs").resolve()
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# dataset class name -> COCO class name (only used for the zero-shot baseline)
ALIAS = {"phone": "cell phone", "mobile": "cell phone", "mobile phone": "cell phone",
         "remote control": "remote", "plant": "potted plant", "table": "dining table",
         "sofa": "couch", "television": "tv", "bag": "handbag"}
_trapz = getattr(np, "trapezoid", None) or np.trapz


def default_device():
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def family(tag):
    m = re.search(r"yolo\d+[nsmlx]", tag)
    return m.group(0) if m else tag


# ----------------------------------------------------------------------------- data
def load_split(data_yaml, split):
    d = yaml.safe_load(open(data_yaml, encoding="utf-8"))
    base = Path(data_yaml).resolve().parent
    key = split if d.get(split) else "val"
    sp = d[key][0] if isinstance(d[key], (list, tuple)) else d[key]
    root = Path(d.get("path", "."))
    if not root.is_absolute():
        cands = [(base / root).resolve(), (Path.cwd() / root).resolve()]
        root = next((c for c in cands if (c / sp).exists()), cands[0])
    folder = root / sp
    imgs = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXT)
    names = d["names"]
    names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    from PIL import Image
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    gts = []
    for p in imgs:
        lp = Path(sb.join(str(p).rsplit(sa, 1))).with_suffix(".txt")
        rows = []
        if lp.exists():
            W, H = Image.open(p).size
            for ln in lp.read_text().splitlines():
                v = ln.split()
                if len(v) >= 5:
                    c, cx, cy, w, h = int(v[0]), *map(float, v[1:5])
                    rows.append([c, (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
        gts.append(np.array(rows, float).reshape(-1, 5))
    return imgs, gts, names, key


def build_map(model_names, ds_names):
    inv = {str(v).lower(): int(k) for k, v in model_names.items()}
    m = {}
    for j, n in enumerate(ds_names):
        key = n.lower().replace("_", " ")
        key = ALIAS.get(key, key)
        if key in inv:
            m[inv[key]] = j
    return m


# ------------------------------------------------------------------------- metrics
def _ap(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([1.0], prec, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(_trapz(np.interp(x, mrec, mpre), x)), mpre, mrec


def curves(tp, conf, pk, tk, nc):
    px = np.linspace(0, 1, 1001)
    ap = np.zeros((nc, 10))
    P, R, PR = (np.zeros((nc, 1001)) for _ in range(3))
    n_gt = np.bincount(tk, minlength=nc)
    o = np.argsort(-conf)
    tp, conf, pk = tp[o], conf[o], pk[o]
    for c in range(nc):
        m = pk == c
        if n_gt[c] == 0 or m.sum() == 0:
            continue
        tpc = np.cumsum(tp[m], 0).astype(float)
        fpc = np.cumsum(~tp[m], 0).astype(float)
        rec, prec = tpc / (n_gt[c] + 1e-16), tpc / (tpc + fpc + 1e-16)
        R[c] = np.interp(-px, -conf[m], rec[:, 0], left=0)
        P[c] = np.interp(-px, -conf[m], prec[:, 0], left=1)
        for j in range(10):
            ap[c, j], mpre, mrec = _ap(rec[:, j], prec[:, j])
            if j == 0:
                PR[c] = np.interp(px, mrec, mpre)
    F1 = 2 * P * R / (P + R + 1e-16)
    return dict(px=px, ap=ap, P=P, R=R, F1=F1, PR=PR, n_gt=n_gt)


def summarize(cv, idx, k=None):
    idx = [i for i in idx if cv["n_gt"][i] > 0]
    if not idx:
        return None
    f1 = cv["F1"][idx].mean(0)
    k = int(f1.argmax()) if k is None else k
    return dict(mAP50=float(cv["ap"][idx, 0].mean()), mAP5095=float(cv["ap"][idx].mean()),
                P=float(cv["P"][idx, k].mean()), R=float(cv["R"][idx, k].mean()),
                F1=float(cv["F1"][idx, k].mean()), conf=float(cv["px"][k]))


def confusion_update(cm, pb, pc, pk, gb, gk, thr, nc):
    import torch
    from ultralytics.utils.metrics import box_iou
    sel = pc >= thr
    pb, pk = pb[sel], pk[sel]
    used_g, used_p = set(), set()
    if len(pb) and len(gb):
        iou = box_iou(torch.as_tensor(gb, dtype=torch.float32), torch.as_tensor(pb, dtype=torch.float32)).numpy()
        gi, pi = np.nonzero(iou >= 0.5)
        for _, g, p in sorted(zip(iou[gi, pi], gi, pi), reverse=True):
            if g in used_g or p in used_p:
                continue
            used_g.add(g)
            used_p.add(p)
            cm[pk[p], int(gk[g])] += 1
    for g in range(len(gk)):
        if g not in used_g:
            cm[nc, int(gk[g])] += 1
    for p in range(len(pk)):
        if p not in used_p:
            cm[pk[p], nc] += 1


def make_coco80(data_yaml, coco_names):
    """For zero-shot COCO models: a copy of val/test whose label ids are the 80 COCO ids (only our classes kept)."""
    d = yaml.safe_load(open(data_yaml, encoding="utf-8"))
    names = d["names"]
    names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    cmap = {j: k for k, j in build_map(coco_names, names).items()}          # ds idx -> coco id
    base = Path(data_yaml).resolve().parent
    out = base.parent / (base.name + "_coco80_lab")
    for sp in ("val", "test"):
        if not d.get(sp):
            continue
        imgs, _, _, _ = load_split(data_yaml, sp)
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)
        sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
        for p in imgs:
            dst = out / "images" / sp / p.name
            if not dst.exists():
                try:
                    os.link(p, dst)
                except OSError:
                    import shutil
                    shutil.copy2(p, dst)
            lp = Path(sb.join(str(p).rsplit(sa, 1))).with_suffix(".txt")
            rows = []
            if lp.exists():
                for ln in lp.read_text().splitlines():
                    v = ln.split()
                    if len(v) >= 5 and int(v[0]) in cmap:
                        rows.append(" ".join([str(cmap[int(v[0])])] + v[1:5]))
            (out / "labels" / sp / (p.stem + ".txt")).write_text("\n".join(rows))
    ynames = "".join(f"  {k}: {v}\n" for k, v in sorted(coco_names.items()))
    (out / "data.yaml").write_text(f"path: {out}\ntrain: images/val\nval: images/val\ntest: images/test\nnames:\n{ynames}")
    return out / "data.yaml"


def val_stats(model, ds_yaml, split, imgsz, device):
    """Run the official Ultralytics validator (multi-label NMS, conf 0.001) and capture its per-prediction
    match table (tp at 10 IoU thresholds, confidence, predicted class, target class)."""
    snap = {}

    def grab(v):
        snap["s"] = {k: list(x) for k, x in v.metrics.stats.items()}

    model.add_callback("on_val_batch_end", grab)
    model.val(data=str(ds_yaml), split=split, imgsz=imgsz, batch=16, conf=0.001, iou=0.7, device=device,
              plots=False, verbose=False, workers=0)
    if "s" not in snap:
        raise RuntimeError("could not capture validator statistics (Ultralytics version too different)")

    def cat(k, shape):
        parts = [np.asarray(x) for x in snap["s"][k] if len(x)]
        return np.concatenate(parts) if parts else np.zeros(shape)

    return (cat("tp", (0, 10)).astype(bool), cat("conf", (0,)), cat("pred_cls", (0,)).astype(int),
            cat("target_cls", (0,)).astype(int))


def confusion_run(model, imgs, gts, cmap, nc, imgsz, device, thr):
    """Confusion matrix of what the app would output (one class per box, confidence >= thr)."""
    cm = np.zeros((nc + 1, nc + 1), int)
    for p, g in zip(imgs, gts):
        r = model.predict(str(p), imgsz=imgsz, conf=thr, device=device, verbose=False)[0]
        b = r.boxes
        pb, pc, pk = b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int)
        keep = np.array([k in cmap for k in pk], bool)
        pb, pc = pb[keep], pc[keep]
        pk = np.array([cmap[k] for k in pk[keep]], int)
        confusion_update(cm, pb, pc, pk, g[:, 1:], g[:, 0].astype(int), thr, nc)
    return cm


def evaluate(weights, data, split, tag, imgsz, device, conf_fixed=0.25, data80=None):
    """Numbers come from the official Ultralytics validator. The confidence threshold is chosen on the VAL
    split (best macro-F1) and every number is reported on `split` (test) at that fixed threshold.
    mAP does not depend on any threshold."""
    from ultralytics import YOLO
    model = YOLO(str(weights))
    v_imgs, v_gts, names, vkey = load_split(data, "val")
    t_imgs, t_gts, _, tkey = load_split(data, split)
    nc = len(names)
    cmap = build_map(model.names, names)
    own = len(model.names) == nc and all(str(model.names[i]) == names[i] for i in range(nc))
    ds = data if own else (data80 or make_coco80(data, model.names))
    covered = [j in cmap.values() for j in range(nc)]
    cov_idx = [j for j in range(nc) if covered[j]]
    lut = np.full(max(int(k) for k in model.names) + 1, -1)
    for k, j in cmap.items():
        lut[k] = j

    def run(sp):
        tp, conf, pk, tk = val_stats(model, ds, sp, imgsz, device)
        pk, tk = lut[pk], lut[tk]
        keep = pk >= 0
        return curves(tp[keep], conf[keep], pk[keep], tk[tk >= 0], nc)

    t0 = time.time()
    cv = run(split if tkey == split else "val")
    same = vkey == tkey
    cv_val = cv if same else run("val")
    if same:
        print("  WARNING: no separate test split -> threshold chosen on the same data it is reported on")
    k_all = int(round(summarize(cv_val, range(nc))["conf"] * 1000))
    sv = summarize(cv_val, cov_idx) if cov_idx else None
    k_cov = int(round(sv["conf"] * 1000)) if sv else None
    allc = summarize(cv, range(nc), k_all)
    cm = confusion_run(model, t_imgs, t_gts, cmap, nc, imgsz, device, conf_fixed)
    per = {}
    for j, n in enumerate(names):
        per[n] = dict(n=int(cv["n_gt"][j]), covered=covered[j], AP50=float(cv["ap"][j, 0]),
                      AP5095=float(cv["ap"][j].mean()), P=float(cv["P"][j, k_all]), R=float(cv["R"][j, k_all]),
                      F1=float(cv["F1"][j, k_all]))
    out = dict(tag=tag, weights=str(weights), split=tkey, threshold_from=vkey, imgsz=imgsz,
               n_images=len(t_imgs), n_labels=int(cv["n_gt"].sum()), classes=names, covered=covered,
               all=allc, covered_only=summarize(cv, cov_idx, k_cov) if sv else None,
               fixed=summarize(cv, range(nc), int(conf_fixed * 1000)), per_class=per,
               seconds=round(time.time() - t0, 1))
    OUT.mkdir(exist_ok=True)
    (OUT / f"metrics_{tag}.json").write_text(json.dumps(out, indent=1))
    np.savez(OUT / f"curves_{tag}.npz", px=cv["px"], P=cv["P"], R=cv["R"], F1=cv["F1"], PR=cv["PR"],
             n_gt=cv["n_gt"], cm=cm)
    a = allc
    print(f"[{tag}] test={len(t_imgs)} imgs  P={a['P']:.3f} R={a['R']:.3f} F1={a['F1']:.3f} "
          f"mAP50={a['mAP50']:.3f} mAP50-95={a['mAP5095']:.3f}  (conf {a['conf']:.2f} chosen on {vkey})")
    return out


# --------------------------------------------------------------------------- speed
def cmd_speed(a):
    from ultralytics import YOLO
    from ultralytics.utils import ASSETS
    img = str(Path(a.image) if a.image else ASSETS / "bus.jpg")
    rows = {}
    for w in a.models:
        m = YOLO(w)
        params = sum(p.numel() for p in m.model.parameters())
        try:
            from ultralytics.utils.torch_utils import get_flops
            m.fuse()
            gflops = float(get_flops(m.model, a.imgsz))
        except Exception:
            gflops = float("nan")
        for _ in range(5):
            m.predict(img, imgsz=a.imgsz, device="cpu", verbose=False)
        tot, inf = [], []
        for _ in range(a.runs):
            sp = m.predict(img, imgsz=a.imgsz, device="cpu", verbose=False)[0].speed
            tot.append(sum(sp.values()))
            inf.append(sp["inference"])
        rows[Path(w).stem] = dict(params_M=params / 1e6, GFLOPs=float(gflops), imgsz=a.imgsz,
                                  cpu_ms_total=float(np.median(tot)), cpu_ms_inference=float(np.median(inf)))
        r = rows[Path(w).stem]
        print(f"[speed] {Path(w).stem}: {r['params_M']:.2f} M params, {r['GFLOPs']:.1f} GFLOPs, "
              f"{r['cpu_ms_total']:.1f} ms/frame on CPU (model {r['cpu_ms_inference']:.1f} + pre/post)")
    OUT.mkdir(exist_ok=True)
    (OUT / "speed.json").write_text(json.dumps(rows, indent=1))


# -------------------------------------------------------------------------- train
def train_one(a, model_path, tag, freeze=None):
    from ultralytics import YOLO
    m = YOLO(model_path)
    t0 = time.time()
    kw = dict(data=str(Path(a.data).resolve()), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
              device=a.device, project=str(RUNS), name=tag, exist_ok=True, patience=a.patience,
              workers=a.workers, seed=0, plots=True, verbose=False)
    if freeze:
        kw["freeze"] = freeze
    m.train(**kw)
    best = Path(m.trainer.save_dir) / "weights" / "best.pt"
    OUT.mkdir(exist_ok=True)
    (OUT / f"train_{tag}.json").write_text(json.dumps(dict(
        tag=tag, start=model_path, freeze=freeze, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
        seconds=round(time.time() - t0), run_dir=str(m.trainer.save_dir), best=str(best)), indent=1))
    return best


def cmd_train(a):
    tag = a.tag or f"ft_{Path(a.model).stem}"
    best = train_one(a, a.model, tag, a.freeze)
    evaluate(best, a.data, a.split, tag, a.imgsz, a.device)


def cmd_eval(a):
    for w in a.weights:
        tag = a.tag or f"baseline_{Path(w).stem}"
        evaluate(w, a.data, a.split, tag, a.imgsz, a.device, data80=a.data80)


def cmd_all(a):
    cmd_speed(a)
    for w in a.models:
        evaluate(w, a.data, a.split, f"baseline_{Path(w).stem}", a.imgsz, a.device, data80=a.data80)
    for w in a.models:
        tag = f"ft_{Path(w).stem}"
        evaluate(train_one(a, w, tag), a.data, a.split, tag, a.imgsz, a.device)
    if a.ablate:
        w = "yolo26n.pt"
        tag = "ft_yolo26n_head"
        evaluate(train_one(a, w, tag, freeze=10), a.data, a.split, tag, a.imgsz, a.device)
    cmd_report(a)


# -------------------------------------------------------------------------- report
def _style(tag):
    base = {"yolo11n": "#4c72b0", "yolo26n": "#dd8452", "yolo26s": "#55a868"}.get(family(tag), "#8172b3")
    if tag.startswith("baseline"):
        return base, "--", 0.55
    if tag.endswith("_head"):
        return base, ":", 1.0
    return base, "-", 1.0


def _load_all():
    M = {}
    for f in sorted(OUT.glob("metrics_*.json")):
        d = json.loads(f.read_text())
        M[d["tag"]] = d
    C = {t: np.load(OUT / f"curves_{t}.npz") for t in M if (OUT / f"curves_{t}.npz").exists()}
    return M, C


def _runs(M):
    import pandas as pd
    R = {}
    for t in M:
        f = RUNS / t / "results.csv"
        if t.startswith("ft_") and f.exists():
            df = pd.read_csv(f)
            df.columns = [c.strip() for c in df.columns]
            R[t] = df
    return R


def _loss(df, pre):
    cols = [c for c in df.columns if c.startswith(pre + "/") and c.endswith("loss")]
    return df[cols].sum(axis=1) if cols else None


def cmd_report(a=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    M, C = _load_all()
    if not M:
        sys.exit("nothing to report yet: run `python lab.py all --data ...` first")
    R = _runs(M)
    plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .3, "axes.spines.top": False,
                         "axes.spines.right": False})
    # 1. loss curves
    if R:
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        for t, df in R.items():
            c, ls, al = _style(t)
            for a_, pre, ttl in zip(ax[:2], ("train", "val"), ("Training loss (sum of loss terms)", "Validation loss")):
                y = _loss(df, pre)
                if y is not None:
                    a_.plot(df["epoch"], y, color=c, ls=ls, lw=2, label=t)
                    a_.set_title(ttl)
                    a_.set_xlabel("epoch")
            for k, lab in (("box", "box"), ("cls", "cls")):
                col = f"val/{k}_loss"
                if col in df:
                    ax[2].plot(df["epoch"], df[col], color=c, ls=ls if k == "box" else "-.", lw=1.6, label=f"{t} {lab}")
        ax[2].set_title("Validation loss parts")
        ax[2].set_xlabel("epoch")
        for a_ in ax:
            a_.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(OUT / "fig1_loss_curves.png")
        plt.close(fig)
        # 2. metric curves per epoch
        fig, ax = plt.subplots(2, 2, figsize=(11, 7))
        for t, df in R.items():
            c, ls, al = _style(t)
            for a_, col, ttl in zip(ax.ravel(), ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)",
                                                 "metrics/mAP50-95(B)"),
                                    ("Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95")):
                if col in df:
                    a_.plot(df["epoch"], df[col], color=c, ls=ls, lw=2, label=t)
                    a_.set_title(ttl + " (validation set, per epoch)")
                    a_.set_xlabel("epoch")
                    a_.set_ylim(0, 1.02)
        for a_ in ax.ravel():
            a_.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / "fig2_metrics_per_epoch.png")
        plt.close(fig)
    # 3. before / after bars
    keys = [("P", "Precision"), ("R", "Recall"), ("F1", "F1"), ("mAP50", "mAP@0.5"), ("mAP5095", "mAP@0.5:0.95")]
    tags = [t for t in M if "_head" not in t]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    w = 0.8 / max(len(tags), 1)
    for i, t in enumerate(tags):
        c, ls, al = _style(t)
        vals = [M[t]["all"][k] for k, _ in keys]
        b = ax.bar(np.arange(len(keys)) + i * w, vals, w, color=c, alpha=al, label=t,
                   hatch="//" if t.startswith("baseline") else None, edgecolor="white")
        for x, v in zip(b, vals):
            ax.text(x.get_x() + x.get_width() / 2, v + .01, f"{v:.2f}", ha="center", fontsize=7)
    ax.axhline(.85, color="crimson", lw=1, ls="--")
    ax.text(len(keys) - .55, .86, "target 0.85", color="crimson", fontsize=8)
    ax.set_xticks(np.arange(len(keys)) + w * (len(tags) - 1) / 2)
    ax.set_xticklabels([n for _, n in keys])
    ax.set_ylim(0, 1.08)
    ax.set_title("Before fine-tuning (hatched, COCO weights) vs after fine-tuning - test set, all classes")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_before_after.png")
    plt.close(fig)
    # 4. PR + F1 curves
    if C:
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        for t, cv in C.items():
            c, ls, al = _style(t)
            idx = [i for i in range(len(cv["n_gt"])) if cv["n_gt"][i] > 0]
            ax[0].plot(cv["px"], cv["PR"][idx].mean(0), color=c, ls=ls, lw=2, label=f"{t} (mAP50 {M[t]['all']['mAP50']:.2f})")
            ax[1].plot(cv["px"], cv["F1"][idx].mean(0), color=c, ls=ls, lw=2, label=f"{t} (best {M[t]['all']['F1']:.2f} @ {M[t]['all']['conf']:.2f})")
            ax[2].plot(cv["px"], cv["P"][idx].mean(0), color=c, ls=ls, lw=1.6, label=f"{t} P")
            ax[2].plot(cv["px"], cv["R"][idx].mean(0), color=c, ls="-.", lw=1.2, alpha=.7)
        ax[0].set(title="Precision-Recall curve (IoU 0.5, mean of classes)", xlabel="recall", ylabel="precision")
        ax[1].set(title="F1 vs confidence threshold", xlabel="confidence")
        ax[2].set(title="Precision (solid) / Recall (dash-dot) vs confidence", xlabel="confidence")
        for a_ in ax:
            a_.legend(fontsize=7)
            a_.set_xlim(0, 1)
            a_.set_ylim(0, 1.02)
        fig.tight_layout()
        fig.savefig(OUT / "fig4_pr_f1_curves.png")
        plt.close(fig)
        # 5. confusion matrices (fine-tuned)
        fts = [t for t in C if t.startswith("ft_") and "_head" not in t]
        if fts:
            fig, ax = plt.subplots(1, len(fts), figsize=(6.4 * len(fts), 5.6), squeeze=False)
            for a_, t in zip(ax[0], fts):
                cm = C[t]["cm"].astype(float)
                cmn = cm / np.maximum(cm.sum(0, keepdims=True), 1)
                names = M[t]["classes"] + ["background"]
                a_.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
                a_.set_xticks(range(len(names)))
                a_.set_yticks(range(len(names)))
                a_.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
                a_.set_yticklabels(names, fontsize=7)
                a_.set_xlabel("true class")
                a_.set_ylabel("predicted class")
                a_.set_title(f"Confusion matrix (normalised by true class) - {t}")
                a_.grid(False)
                for i in range(len(names)):
                    for j in range(len(names)):
                        if cm[i, j]:
                            a_.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=6,
                                    color="white" if cmn[i, j] > .5 else "black")
            fig.tight_layout()
            fig.savefig(OUT / "fig5_confusion.png")
            plt.close(fig)
    # 6. per-class AP50
    names = next(iter(M.values()))["classes"]
    tg = [t for t in M if "_head" not in t]
    fig, ax = plt.subplots(figsize=(max(8, .9 * len(names) + 3), 4.6))
    w = 0.8 / max(len(tg), 1)
    for i, t in enumerate(tg):
        c, ls, al = _style(t)
        ax.bar(np.arange(len(names)) + i * w, [M[t]["per_class"][n]["AP50"] for n in names], w, color=c,
               alpha=al, hatch="//" if t.startswith("baseline") else None, edgecolor="white", label=t)
    ax.set_xticks(np.arange(len(names)) + w * (len(tg) - 1) / 2)
    ax.set_xticklabels(names, rotation=40, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-class AP@0.5 (classes COCO does not know score 0 before fine-tuning)")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_per_class_ap50.png")
    plt.close(fig)
    # 7. speed vs accuracy
    S = json.loads((OUT / "speed.json").read_text()) if (OUT / "speed.json").exists() else {}
    if S:
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        for t in M:
            fam = family(t)
            if t.startswith("ft_") and "_head" not in t and fam in S:
                c, _, _ = _style(t)
                ax.scatter(S[fam]["cpu_ms_total"], M[t]["all"]["mAP50"], s=60 + 90 * S[fam]["params_M"], color=c)
                ax.annotate(f"{fam}\n{S[fam]['params_M']:.2f} M params", (S[fam]["cpu_ms_total"], M[t]["all"]["mAP50"]),
                            textcoords="offset points", xytext=(8, -4), fontsize=8)
        ax.set(xlabel="CPU time per frame (ms, lower = faster)", ylabel="mAP@0.5 on test set",
               title="Speed vs accuracy (fine-tuned models)")
        ax.margins(.3)
        fig.tight_layout()
        fig.savefig(OUT / "fig7_speed_vs_accuracy.png")
        plt.close(fig)
    _write_report(M, S, R)
    print("report written to", OUT.resolve())


def _write_report(M, S, R):
    L = ["# Sahayak - detector evaluation (YOLO11n vs YOLO26n vs YOLO26s)", ""]
    L += ["All numbers use the same held-out test split, the same matching rule (IoU>=0.5 for P/R/F1) and the "
          "same code for every model; mAP/P/R come from the official Ultralytics validator. The confidence threshold is chosen on the validation split (best macro-F1) and then frozen; all numbers are on the test split. P and R are macro-averages over classes at that threshold, F1 is the macro-average of the per-class F1 (as in Ultralytics), so it is not 2PR/(P+R) of the averaged P and R.", ""]
    L += ["| model | stage | P | R | F1 | mAP@0.5 | mAP@0.5:0.95 | best conf |", "|---|---|---|---|---|---|---|---|"]
    for t, d in M.items():
        a = d["all"]
        L.append(f"| {family(t)} | {'zero-shot COCO' if t.startswith('baseline') else 'fine-tuned' + (' (backbone frozen)' if t.endswith('_head') else '')} "
                 f"| {a['P']:.3f} | {a['R']:.3f} | {a['F1']:.3f} | {a['mAP50']:.3f} | {a['mAP5095']:.3f} | {a['conf']:.2f} |")
    first = next(iter(M.values()))
    L += ["", f"Test set: {first['n_images']} images, {first['n_labels']} labelled objects, "
          f"{len(first['classes'])} classes: {', '.join(first['classes'])}.", ""]
    cov = [c for c, ok in zip(first["classes"], first["covered"]) if ok]
    partial = len(cov) < len(first["classes"])
    if partial:
        L += [f"Classes that COCO already knows (zero-shot baseline can detect them): {', '.join(cov) or 'none'}. "
              "All other classes score 0 before fine-tuning by construction.", ""]
    if partial and any(d.get("covered_only") for d in M.values()):
        L += ["Same comparison restricted to COCO-known classes (a fairer view of what fine-tuning adds there):", "",
              "| run | F1 | mAP@0.5 |", "|---|---|---|"]
        for t, d in M.items():
            if d.get("covered_only"):
                L.append(f"| {t} | {d['covered_only']['F1']:.3f} | {d['covered_only']['mAP50']:.3f} |")
        L.append("")
    if S:
        L += ["## Speed and size (this computer, CPU, batch 1)", "", "| model | params (M) | GFLOPs | ms/frame (total) | ms (network only) |", "|---|---|---|---|---|"]
        for k, v in S.items():
            L.append(f"| {k} | {v['params_M']:.2f} | {v['GFLOPs']:.1f} | {v['cpu_ms_total']:.1f} | {v['cpu_ms_inference']:.1f} |")
        L.append("")
    if R:
        L += ["## Training behaviour", "", "Loss values are only comparable within one model family: YOLO11 and YOLO26 use different loss terms "
              "(YOLO26 has no DFL term), so read each curve against itself over the epochs.", "", "| run | epochs run | train loss first -> last | val loss first -> last | best val mAP@0.5 |", "|---|---|---|---|---|"]
        for t, df in R.items():
            tl, vl = _loss(df, "train"), _loss(df, "val")
            best = df["metrics/mAP50(B)"].max() if "metrics/mAP50(B)" in df else float("nan")
            L.append(f"| {t} | {len(df)} | {tl.iloc[0]:.3f} -> {tl.iloc[-1]:.3f} | {vl.iloc[0]:.3f} -> {vl.iloc[-1]:.3f} | {best:.3f} |")
        L.append("")
    # data-driven conclusion (no pre-written winner)
    ft = {family(t): d["all"] for t, d in M.items() if t.startswith("ft_") and "_head" not in t}
    bl = {family(t): d["all"] for t, d in M.items() if t.startswith("baseline")}
    L += ["## What the numbers say", ""]
    for fam_, a in ft.items():
        s0 = f" (zero-shot baseline: mAP@0.5 {bl[fam_]['mAP50']:.3f}, F1 {bl[fam_]['F1']:.3f})" if fam_ in bl else ""
        L.append(f"* **{fam_}** after fine-tuning: mAP@0.5 {a['mAP50']:.3f}, F1 {a['F1']:.3f}{s0}"
                 f" -> target mAP@0.5 >= 0.85: **{'PASS' if a['mAP50'] >= .85 else 'NOT REACHED'}**, "
                 f"F1 >= 0.85: **{'PASS' if a['F1'] >= .85 else 'NOT REACHED'}**.")
    if "yolo26n" in ft and "yolo11n" in ft:
        d = ft["yolo26n"]["mAP50"] - ft["yolo11n"]["mAP50"]
        L.append(f"* YOLO26n vs YOLO11n accuracy difference: {d:+.3f} mAP@0.5. Differences below about 0.01-0.02 "
                 "are within noise for a test set of this size.")
    if "yolo26n" in S and "yolo11n" in S:
        a_, b_ = S["yolo26n"]["cpu_ms_total"], S["yolo11n"]["cpu_ms_total"]
        L.append(f"* CPU speed: YOLO26n {a_:.1f} ms vs YOLO11n {b_:.1f} ms per frame ("
                 + (f"YOLO26n is {(b_ - a_) / b_ * 100:.0f}% faster" if a_ < b_ else "YOLO26n is not faster on this machine")
                 + "). YOLO26n is end-to-end (no NMS step), which removes post-processing time.")
    if "yolo26s" in S and "yolo26n" in S:
        L.append(f"* Size trade-off: YOLO26s is {S['yolo26s']['cpu_ms_total'] / S['yolo26n']['cpu_ms_total']:.1f}x slower than "
                 f"YOLO26n on CPU ({S['yolo26s']['cpu_ms_total']:.1f} vs {S['yolo26n']['cpu_ms_total']:.1f} ms).")
    L += ["", "Figures: fig1_loss_curves, fig2_metrics_per_epoch, fig3_before_after, fig4_pr_f1_curves, "
          "fig5_confusion, fig6_per_class_ap50, fig7_speed_vs_accuracy (all .png in this folder)."]
    (OUT / "REPORT.md").write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, models=True):
        p.add_argument("--data", default="data/objects/data.yaml")
        p.add_argument("--split", default="test", help="test (default) or val")
        p.add_argument("--imgsz", type=int, default=640)
        p.add_argument("--device", default=default_device())
        p.add_argument("--data80", default=None, help="optional: COCO-80-id copy of val/test for zero-shot models (auto-built if absent)")
        if models:
            p.add_argument("--models", nargs="+", default=["yolo11n.pt", "yolo26n.pt", "yolo26s.pt"])
            p.add_argument("--image", default=None, help="image used for the speed test")
            p.add_argument("--runs", type=int, default=40)

    def trainargs(p):
        p.add_argument("--epochs", type=int, default=60)
        p.add_argument("--batch", type=int, default=16)
        p.add_argument("--patience", type=int, default=20)
        p.add_argument("--workers", type=int, default=0 if sys.platform.startswith("win") else 4)

    p = sub.add_parser("speed"); common(p)
    p = sub.add_parser("eval"); common(p, False); p.add_argument("--weights", nargs="+", default=["yolo11n.pt", "yolo26n.pt", "yolo26s.pt"]); p.add_argument("--tag")
    p = sub.add_parser("train"); common(p, False); trainargs(p); p.add_argument("--model", default="yolo26n.pt"); p.add_argument("--tag"); p.add_argument("--freeze", type=int)
    p = sub.add_parser("all"); common(p); trainargs(p); p.add_argument("--ablate", action="store_true", help="also train YOLO26n with frozen backbone")
    p = sub.add_parser("report")
    a = ap.parse_args()
    {"speed": cmd_speed, "eval": cmd_eval, "train": cmd_train, "all": cmd_all, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
