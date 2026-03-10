#!/usr/bin/env python3
"""
Evaluate per-fold validation results from nnU-Net training.

Reads validation/summary.json for each fold and prints Dice + IoU per label,
cross-fold summary (mean ± std), and per-case boxplots.

Usage:
    python scripts/analysis/evaluation/evaluate_folds.py --dataset Dataset001_GroundTruth
    python scripts/analysis/evaluation/evaluate_folds.py --dataset Dataset002_Karies
    python scripts/analysis/evaluation/evaluate_folds.py --dataset Dataset001_GroundTruth --folds 0 1 2
"""

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ──────────────────────────────────────────────────────────────────────────────
# Helpers (shared with evaluate_ensemble.py)
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset_labels(dataset: str) -> dict[str, str]:
    """Return {label_id_str: label_name} from dataset.json (skip background)."""
    path = PROJECT_ROOT / "data" / "nnUNet_raw" / dataset / "dataset.json"
    if not path.exists():
        return {}
    with open(path) as f:
        d = json.load(f)
    return {str(v): k for k, v in d.get("labels", {}).items() if k != "background"}


def load_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def extract_metrics(summary: dict, label_map: dict[str, str]) -> dict:
    """Extract per-label Dice + IoU + foreground mean from summary.json."""
    result = {}
    for lid, name in label_map.items():
        m = summary.get("mean", {}).get(lid, {})
        result[name] = {"Dice": m.get("Dice"), "IoU": m.get("IoU")}
    fm = summary.get("foreground_mean", {})
    result["__fg__"] = {"Dice": fm.get("Dice"), "IoU": fm.get("IoU")}
    return result


def per_case_dice(summary: dict, label_map: dict[str, str]) -> dict[str, dict[str, float]]:
    out = {}
    for entry in summary.get("metric_per_case", []):
        case_id = Path(entry.get("reference_file", "")).stem.replace(".nii", "")
        out[case_id] = {label_map.get(lid, lid): m.get("Dice")
                        for lid, m in entry.get("metrics", {}).items()}
    return out


def fmt(v) -> str:
    return "  N/A " if v is None else f"{v:.4f}"


def print_table(title: str, rows: list[tuple], headers: list[str]):
    col_w = [max(len(str(r[i])) for r in rows + [tuple(headers)]) for i in range(len(headers))]
    sep = "  ".join("-" * w for w in col_w)
    hdr = "  ".join(str(h).ljust(col_w[i]) for i, h in enumerate(headers))
    print(f"\n{'='*60}\n  {title}\n{'='*60}")
    print(hdr)
    print(sep)
    for row in rows:
        print("  ".join(str(v).ljust(col_w[i]) for i, v in enumerate(row)))


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_fold_comparison(fold_metrics: dict, label_names: list[str], out_path: Path, dataset: str):
    """Grouped bar chart: Dice per label for each fold."""
    folds = sorted(fold_metrics.keys())
    x = np.arange(len(label_names))
    width = 0.8 / max(len(folds), 1)
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = plt.cm.tab10.colors
    for i, fold in enumerate(folds):
        vals = [fold_metrics[fold].get(lbl, {}).get("Dice") or 0 for lbl in label_names]
        ax.bar(x + i * width - (len(folds) - 1) * width / 2, vals,
               width=width * 0.9, label=f"Fold {fold}", color=colors[i % 10])
    ax.set_xticks(x)
    ax.set_xticklabels(label_names)
    ax.set_ylabel("Dice")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{dataset} – Validation Dice per Fold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved: {out_path}")


def plot_fold_mean_std(fold_metrics: dict, label_names: list[str], out_path: Path, dataset: str):
    """Bar chart of mean Dice across folds with std error bars."""
    means, stds = [], []
    for lbl in label_names:
        vals = [fold_metrics[f].get(lbl, {}).get("Dice") for f in sorted(fold_metrics)
                if fold_metrics[f].get(lbl, {}).get("Dice") is not None]
        means.append(float(np.mean(vals)) if vals else 0)
        stds.append(float(np.std(vals)) if vals else 0)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(label_names))
    ax.bar(x, means, yerr=stds, capsize=6, color="steelblue", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(label_names)
    ax.set_ylabel("Dice (mean ± std)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{dataset} – Mean Validation Dice ± Std (all folds)")
    ax.grid(axis="y", alpha=0.3)
    for xi, (m, s) in enumerate(zip(means, stds)):
        ax.text(xi, m + s + 0.01, f"{m:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved: {out_path}")


def plot_per_case_boxplot(per_case: dict, label_names: list[str], fold: int,
                          out_path: Path, dataset: str):
    """Boxplot of per-case Dice distribution within a fold."""
    data = [[v.get(lbl) for v in per_case.values() if v.get(lbl) is not None]
            for lbl in label_names]
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, tick_labels=label_names, patch_artist=True, notch=False)
    colors = ["#4878cf", "#6acc65", "#d65f5f"]
    for patch, color in zip(bp["boxes"], colors[:len(label_names)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Dice")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{dataset} – Fold {fold}: Per-Case Dice Distribution ({len(per_case)} cases)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved: {out_path}")


def plot_all_folds_boxplot(fold_per_case: dict, label_names: list[str],
                           out_path: Path, dataset: str):
    """Combined boxplot: all folds together per label."""
    n_folds = len(fold_per_case)
    n_labels = len(label_names)
    fig, axes = plt.subplots(1, n_labels, figsize=(5 * n_labels, 5), sharey=True)
    if n_labels == 1:
        axes = [axes]
    for ax, lbl in zip(axes, label_names):
        data = [[v.get(lbl) for v in fold_per_case[f].values() if v.get(lbl) is not None]
                for f in sorted(fold_per_case)]
        bp = ax.boxplot(data, tick_labels=[f"F{f}" for f in sorted(fold_per_case)],
                        patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("steelblue")
            patch.set_alpha(0.7)
        ax.set_title(lbl)
        ax.set_ylabel("Dice" if ax == axes[0] else "")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"{dataset} – Per-Case Dice per Fold", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate per-fold nnU-Net validation results.")
    parser.add_argument("--dataset", default="Dataset001_GroundTruth")
    parser.add_argument("--config", default="3d_fullres")
    parser.add_argument("--trainer", default="nnUNetTrainer")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--no_save", action="store_true")
    args = parser.parse_args()

    results_base = (PROJECT_ROOT / "data" / "nnUNet_results" / args.dataset /
                    f"{args.trainer}__nnUNetPlans__{args.config}")
    output_dir = Path(args.output_dir) if args.output_dir else \
                 PROJECT_ROOT / "analysis_results" / "evaluation" / args.dataset / "folds"
    if not args.no_save:
        output_dir.mkdir(parents=True, exist_ok=True)

    label_map   = load_dataset_labels(args.dataset)
    label_names = list(label_map.values())

    print(f"\nDataset : {args.dataset}")
    print(f"Config  : {args.config}")
    print(f"Labels  : {', '.join(label_names)}")

    # ── Load fold summaries ───────────────────────────────────────────────────
    fold_metrics:   dict[int, dict] = {}
    fold_per_case:  dict[int, dict] = {}

    for fold in args.folds:
        path = results_base / f"fold_{fold}" / "validation" / "summary.json"
        s = load_summary(path)
        if s is None:
            print(f"[WARN] No summary.json for fold {fold} ({path})")
            continue
        fold_metrics[fold]  = extract_metrics(s, label_map)
        fold_per_case[fold] = per_case_dice(s, label_map)

    if not fold_metrics:
        print("[ERROR] No summaries found. Run training + validation first.")
        sys.exit(1)

    # ── Per-fold tables ───────────────────────────────────────────────────────
    for fold, metrics in fold_metrics.items():
        rows = [(lbl, fmt(metrics[lbl]["Dice"]), fmt(metrics[lbl]["IoU"])) for lbl in label_names]
        rows.append(("MEAN (fg)", fmt(metrics["__fg__"]["Dice"]), fmt(metrics["__fg__"]["IoU"])))
        print_table(f"Fold {fold} – Validation ({len(fold_per_case.get(fold, {}))} cases)",
                    rows, ["Label", "Dice", "IoU"])

    # ── Cross-fold summary ────────────────────────────────────────────────────
    all_labels = label_names + ["MEAN (fg)"]
    headers = ["Label"] + [f"Fold {f}" for f in sorted(fold_metrics)] + ["Mean", "Std"]
    rows = []
    for lbl in all_labels:
        key = "__fg__" if lbl == "MEAN (fg)" else lbl
        vals = [fold_metrics[f][key]["Dice"] for f in sorted(fold_metrics)
                if fold_metrics[f].get(key, {}).get("Dice") is not None]
        mean = float(np.mean(vals)) if vals else None
        std  = float(np.std(vals)) if vals else None
        fold_vals = [fold_metrics[f].get(key, {}).get("Dice") for f in sorted(fold_metrics)]
        rows.append((lbl, *[fmt(v) for v in fold_vals], fmt(mean), fmt(std)))
    print_table("Cross-Fold Summary – Dice (mean ± std)", rows, headers)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    if not args.no_save:
        csv_path = output_dir / f"folds_{args.dataset}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["fold", "label", "Dice", "IoU"])
            for fold, metrics in fold_metrics.items():
                for lbl in label_names:
                    writer.writerow([fold, lbl, metrics[lbl]["Dice"], metrics[lbl]["IoU"]])
                writer.writerow([fold, "foreground_mean",
                                 metrics["__fg__"]["Dice"], metrics["__fg__"]["IoU"]])
        print(f"\n[INFO] CSV saved: {csv_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not args.no_save and HAS_MPL:
        if len(fold_metrics) > 1:
            plot_fold_comparison(fold_metrics, label_names,
                                 output_dir / f"fold_comparison_{args.dataset}.png", args.dataset)
            plot_fold_mean_std(fold_metrics, label_names,
                               output_dir / f"fold_mean_std_{args.dataset}.png", args.dataset)
            plot_all_folds_boxplot(fold_per_case, label_names,
                                   output_dir / f"fold_boxplot_all_{args.dataset}.png", args.dataset)
        for fold, per_case in fold_per_case.items():
            if per_case:
                plot_per_case_boxplot(per_case, label_names, fold,
                                      output_dir / f"fold{fold}_per_case_{args.dataset}.png",
                                      args.dataset)
        print(f"\n[INFO] All outputs in: {output_dir}")


if __name__ == "__main__":
    main()
