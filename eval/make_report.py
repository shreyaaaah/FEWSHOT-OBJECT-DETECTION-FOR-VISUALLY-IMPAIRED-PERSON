"""Build plots and a results report from saved evaluation and training CSV files."""
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"
RUNS_DIR = PROJECT_ROOT / "runs" / "finetune"
STEMS = ("yolo11n", "yolo26n", "yolo26s")
METRICS = ("precision", "recall", "f1", "map50")
METRIC_LABELS = ("P", "R", "F1", "mAP50")
RUN_PLOTS = ("confusion_matrix_normalized.png", "PR_curve.png", "F1_curve.png")


def read_metrics(path):
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip() for column in frame.columns]
    if "model" in frame:
        frame["model"] = frame["model"].astype(str).map(lambda value: Path(value).stem)
    return frame


def run_csv_plots():
    loss_fig, loss_ax = plt.subplots(figsize=(11, 6))
    map_fig, map_ax = plt.subplots(figsize=(9, 5))
    loss_lines = map_lines = 0
    for stem in STEMS:
        path = RUNS_DIR / stem / "results.csv"
        if not path.exists():
            print(f"No training CSV for {stem}: {path}")
            continue
        frame = pd.read_csv(path)
        frame.columns = [str(column).strip() for column in frame.columns]
        epoch_column = next((column for column in frame.columns if column.lower() == "epoch"), None)
        epochs = frame[epoch_column] if epoch_column else np.arange(1, len(frame) + 1)
        for column in frame.columns:
            lowered = column.lower()
            if "loss" in lowered and pd.api.types.is_numeric_dtype(frame[column]):
                loss_ax.plot(epochs, frame[column], label=f"{stem} {column}")
                loss_lines += 1
            if "map50" in lowered and pd.api.types.is_numeric_dtype(frame[column]):
                map_ax.plot(epochs, frame[column], label=f"{stem} {column}")
                map_lines += 1
    loss_ax.set(title="Training and validation losses", xlabel="Epoch", ylabel="Loss")
    map_ax.set(title="mAP50 by epoch", xlabel="Epoch", ylabel="mAP50")
    if loss_lines:
        loss_ax.legend(fontsize="small")
    if map_lines:
        map_ax.legend(fontsize="small")
    loss_fig.tight_layout()
    map_fig.tight_layout()
    loss_fig.savefig(RESULTS / "loss_curves.png", dpi=160)
    map_fig.savefig(RESULTS / "map50_by_epoch.png", dpi=160)
    plt.close(loss_fig)
    plt.close(map_fig)


def comparison_plots(baseline, finetuned):
    baseline = baseline.copy()
    finetuned = finetuned.copy()
    baseline["model"] = baseline["model"].map(lambda value: Path(str(value)).stem)
    finetuned["model"] = finetuned["model"].map(lambda value: Path(str(value)).stem)
    merged = baseline.merge(finetuned, on="model", suffixes=("_baseline", "_finetuned"))
    if not merged.empty:
        positions = np.arange(len(METRICS) * len(merged))
        width = 0.38
        before, after = [], []
        labels = []
        for _, row in merged.iterrows():
            for metric, label in zip(METRICS, METRIC_LABELS):
                before.append(float(row[f"{metric}_baseline"]))
                after.append(float(row[f"{metric}_finetuned"]))
                labels.append(f"{row['model']}\n{label}")
        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.8), 6))
        ax.bar(positions - width / 2, before, width, label="Baseline")
        ax.bar(positions + width / 2, after, width, label="Fine-tuned")
        ax.set_xticks(positions, labels, rotation=35, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.set_title("Baseline vs fine-tuned metrics")
        ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS / "baseline_vs_finetuned.png", dpi=160)
        plt.close(fig)
    else:
        print("No model appears in both baseline and fine-tuned metrics; paired bars skipped.")

    if "cpu_ms_per_image" in baseline.columns and "map50" in baseline.columns:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(baseline["cpu_ms_per_image"], baseline["map50"])
        for _, row in baseline.iterrows():
            ax.annotate(str(row["model"]), (row["cpu_ms_per_image"], row["map50"]))
        ax.set(xlabel="CPU ms per image", ylabel="mAP50", title="Accuracy vs CPU inference time")
        fig.tight_layout()
        fig.savefig(RESULTS / "accuracy_vs_cpu.png", dpi=160)
        plt.close(fig)


def copy_run_plots():
    copied = []
    for stem in STEMS:
        for filename in RUN_PLOTS:
            source = RUNS_DIR / stem / filename
            if source.exists():
                target = RESULTS / f"{stem}_{filename}"
                shutil.copy2(source, target)
                copied.append(target.name)
            else:
                print(f"Plot not found, skipped: {source}")
    return copied


def metric_table(frame, columns):
    headers = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [headers, divider]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row.get(column, "")
            if pd.isna(value):
                values.append("")
            elif isinstance(value, (int, float, np.number)):
                values.append(f"{float(value):.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_markdown(baseline, finetuned, baseline_classes, finetuned_classes, copied):
    lines = ["# Sahayak model evaluation", "", "## Baseline metrics", ""]
    if baseline.empty:
        lines.append("No baseline metrics CSV rows were available.")
    else:
        columns = [column for column in (
            "model", "threshold", "precision", "recall", "f1", "map50", "map50_95",
            "parameters", "cpu_ms_per_image",
        ) if column in baseline.columns]
        lines.append(metric_table(baseline, columns))
    lines.extend(["", "## Fine-tuned metrics", ""])
    if finetuned.empty:
        lines.append("No fine-tuned metrics CSV rows were available.")
    else:
        columns = [column for column in (
            "model", "threshold", "precision", "recall", "f1", "map50", "map50_95",
        ) if column in finetuned.columns]
        lines.append(metric_table(finetuned, columns))

    candidates = []
    if not finetuned.empty:
        candidates = [("fine-tuned", row) for _, row in finetuned.iterrows()]
    elif not baseline.empty:
        candidates = [("baseline", row) for _, row in baseline.iterrows()]
    if candidates:
        best_kind, best = max(candidates, key=lambda pair: float(pair[1].get("f1", 0.0)))
        f1 = float(best.get("f1", 0.0))
        map50 = float(best.get("map50", 0.0))
        lines.extend(["", "## 85% target", ""])
        lines.append(
            f"Best available {best_kind} model by F1: **{best['model']}** "
            f"(F1 {f1:.4f}, mAP50 {map50:.4f})."
        )
        if f1 >= 0.85 and map50 >= 0.85:
            lines.append("Both F1 and mAP50 reached the 0.85 target.")
        else:
            lines.append(f"The 0.85 target was not reached for both metrics: F1 is {f1:.4f} and mAP50 is {map50:.4f}.")
        model_name = Path(str(best["model"])).stem
        class_frame = finetuned_classes if best_kind == "fine-tuned" else baseline_classes
        if not class_frame.empty and "model" in class_frame:
            class_frame = class_frame[class_frame["model"].map(lambda value: Path(str(value)).stem) == model_name]
            if "f1" in class_frame:
                class_frame = class_frame.sort_values("f1", ascending=True)
                lines.extend(["", f"## Per-class results for {model_name} (weakest to strongest)", ""])
                columns = [column for column in ("class", "ap50", "precision", "recall", "f1")
                           if column in class_frame.columns]
                lines.append(metric_table(class_frame, columns))

    plots = [
        ("loss_curves.png", "Train and validation loss curves"),
        ("map50_by_epoch.png", "mAP50 by epoch"),
        ("baseline_vs_finetuned.png", "Baseline vs fine-tuned metrics"),
        ("accuracy_vs_cpu.png", "mAP50 vs CPU inference time"),
    ]
    lines.extend(["", "## Plots", ""])
    for filename, title in plots:
        if (RESULTS / filename).exists():
            lines.extend([f"### {title}", "", f"![{title}]({filename})", ""])
    for filename in copied:
        lines.extend([f"### {filename}", "", f"![{filename}]({filename})", ""])
    (RESULTS / "RESULTS.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    baseline_path = RESULTS / "baseline_metrics.csv"
    finetuned_path = RESULTS / "finetuned_metrics.csv"
    baseline_classes_path = RESULTS / "baseline_per_class.csv"
    finetuned_classes_path = RESULTS / "per_class_metrics.csv"
    baseline = read_metrics(baseline_path) if baseline_path.exists() else pd.DataFrame()
    finetuned = read_metrics(finetuned_path) if finetuned_path.exists() else pd.DataFrame()
    baseline_classes = read_metrics(baseline_classes_path) if baseline_classes_path.exists() else pd.DataFrame()
    finetuned_classes = read_metrics(finetuned_classes_path) if finetuned_classes_path.exists() else pd.DataFrame()
    run_csv_plots()
    if not baseline.empty and not finetuned.empty:
        comparison_plots(baseline, finetuned)
    elif "cpu_ms_per_image" in baseline.columns:
        comparison_plots(baseline, finetuned)
    copied = copy_run_plots()
    write_markdown(baseline, finetuned, baseline_classes, finetuned_classes, copied)
    print(f"Wrote report and available plots under {RESULTS}")


if __name__ == "__main__":
    main()
