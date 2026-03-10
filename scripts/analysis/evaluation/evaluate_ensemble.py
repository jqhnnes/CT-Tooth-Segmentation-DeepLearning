#!/usr/bin/env python3
"""
Evaluate ensemble predictions against ground truth (test set / labelsTs).

Runs nnUNetv2_evaluate_folder on ensemble/single-fold predictions and produces:
  - Per-label Dice + IoU table
  - Comparison: ensemble vs. per-fold validation mean
  - Per-case Dice bar chart (ranked worst → best)
  - Per-label Dice distribution (boxplot)
  - Summary CSV

Usage:
    # Ensemble predictions vs. test set
    python scripts/analysis/evaluation/evaluate_ensemble.py \
        --dataset Dataset001_GroundTruth \
        --pred_dir ensemble_predictions/Dataset001_GroundTruth_3d_fullres \
        --labels_dir data/nnUNet_raw/Dataset001_GroundTruth/labelsTs

    # Single-fold predictions vs. test set
    python scripts/analysis/evaluation/evaluate_ensemble.py \
        --dataset Dataset001_GroundTruth \
        --pred_dir predictions/Dataset001_GroundTruth/fold_0 \
        --labels_dir data/nnUNet_raw/Dataset001_GroundTruth/labelsTs \
        --name fold_0_test
"""

import argparse
import csv
import json
import subprocess
import sys
import tempfile
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
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset_labels(dataset: str) -> dict[str, str]:
    path = PROJECT_ROOT / "data" / "nnUNet_raw" / dataset / "dataset.json"
    if not path.exists():
        return {}
    with open(path) as f:
        d = json.load(f)
    return {str(v): k for k, v in d.get("labels", {}).items() if k != "background"}


def load_fold_means(dataset: str, config: str, trainer: str,
                    folds: list[int], label_map: dict[str, str]) -> dict[str, float] | None:
    """Load mean validation Dice per label from fold summaries (for comparison)."""
    base = PROJECT_ROOT / "data" / "nnUNet_results" / dataset / f"{trainer}__nnUNetPlans__{config}"
    label_sums: dict[str, list[float]] = {name: [] for name in label_map.values()}
    label_sums["__fg__"] = []
    found = 0
    for fold in folds:
        path = base / f"fold_{fold}" / "validation" / "summary.json"
        if not path.exists():
            continue
        with open(path) as f:
            s = json.load(f)
        for lid, name in label_map.items():
            v = s.get("mean", {}).get(lid, {}).get("Dice")
            if v is not None:
                label_sums[name].append(v)
        fg = s.get("foreground_mean", {}).get("Dice")
        if fg is not None:
            label_sums["__fg__"].append(fg)
        found += 1
    if not found:
        return None
    return {k: float(np.mean(v)) for k, v in label_sums.items() if v}


def run_nnunet_evaluate(pred_dir: Path, labels_dir: Path, dataset_json: Path,
                        plans_file: Path) -> dict | None:
    """Run nnUNetv2_evaluate_folder and return the summary dict."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "summary.json"
        cmd = [
            "nnUNetv2_evaluate_folder",
            str(labels_dir),
            str(pred_dir),
            "-djfile", str(dataset_json),
            "-pfile",  str(plans_file),
            "-o",      str(out_file),
            "--chill",
        ]
        print(f"[INFO] Running: {' '.join(cmd)}\n")
        try:
            subprocess.run(cmd, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"[ERROR] Evaluation failed: {e}")
            return None
        if out_file.exists():
            with open(out_file) as f:
                return json.load(f)
    return None


def extract_metrics(summary: dict, label_map: dict[str, str]) -> dict:
    result = {}
    for lid, name in label_map.items():
        m = summary.get("mean", {}).get(lid, {})
        result[name] = {"Dice": m.get("Dice"), "IoU": m.get("IoU"),
                        "FP": m.get("FP"), "FN": m.get("FN")}
    fm = summary.get("foreground_mean", {})
    result["__fg__"] = {"Dice": fm.get("Dice"), "IoU": fm.get("IoU")}
    return result


def per_case_metrics(summary: dict, label_map: dict[str, str]) -> dict:
    """Return {case_id: {label_name: {Dice, IoU}}} per case."""
    out = {}
    for entry in summary.get("metric_per_case", []):
        case_id = Path(entry.get("reference_file", "")).stem.replace(".nii", "")
        out[case_id] = {label_map.get(lid, lid): m
                        for lid, m in entry.get("metrics", {}).items()}
    return out


def fmt(v) -> str:
    return "  N/A " if v is None else f"{v:.4f}"


def print_table(title: str, rows, headers):
    col_w = [max(len(str(r[i])) for r in list(rows) + [tuple(headers)]) for i in range(len(headers))]
    sep = "  ".join("-" * w for w in col_w)
    hdr = "  ".join(str(h).ljust(col_w[i]) for i, h in enumerate(headers))
    print(f"\n{'='*65}\n  {title}\n{'='*65}")
    print(hdr)
    print(sep)
    for row in rows:
        print("  ".join(str(v).ljust(col_w[i]) for i, v in enumerate(row)))


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_label_comparison(metrics: dict, fold_means: dict | None,
                          label_names: list[str], out_path: Path, name: str):
    """Bar chart: ensemble Dice per label, optionally with val mean overlay."""
    x = np.arange(len(label_names))
    fig, ax = plt.subplots(figsize=(9, 5))
    ens_vals = [metrics.get(lbl, {}).get("Dice") or 0 for lbl in label_names]
    ax.bar(x, ens_vals, width=0.5, label="Ensemble / Test", color="steelblue", alpha=0.85)
    if fold_means:
        val_vals = [fold_means.get(lbl) or 0 for lbl in label_names]
        ax.scatter(x, val_vals, color="tomato", zorder=5, s=80, label="Val mean (folds)", marker="D")
    ax.set_xticks(x)
    ax.set_xticklabels(label_names)
    ax.set_ylabel("Dice")
    ax.set_ylim(0, 1.08)
    ax.set_title(f"{name} – Dice per Label")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for xi, v in enumerate(ens_vals):
        ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved: {out_path}")


def plot_per_case_ranked(per_case: dict, label_names: list[str],
                         out_path: Path, name: str):
    """Per-case mean foreground Dice, sorted worst → best."""
    case_means = {}
    for case_id, lbls in per_case.items():
        vals = [lbls[lbl].get("Dice") for lbl in label_names if lbls.get(lbl, {}).get("Dice") is not None]
        if vals:
            case_means[case_id] = float(np.mean(vals))
    sorted_cases = sorted(case_means.items(), key=lambda x: x[1])
    ids   = [c for c, _ in sorted_cases]
    vals  = [v for _, v in sorted_cases]
    colors = ["tomato" if v < 0.7 else "orange" if v < 0.85 else "steelblue" for v in vals]
    fig, ax = plt.subplots(figsize=(max(10, len(ids) * 0.45), 5))
    ax.bar(range(len(ids)), vals, color=colors)
    ax.axhline(np.mean(vals), color="black", linestyle="--", linewidth=1.2,
               label=f"Mean: {np.mean(vals):.3f}")
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Mean Foreground Dice")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{name} – Per-Case Dice (sorted, {len(ids)} cases)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved: {out_path}")


def plot_per_case_heatmap(per_case: dict, label_names: list[str],
                          out_path: Path, name: str):
    """Heatmap: cases × labels Dice scores."""
    sorted_cases = sorted(per_case.keys())
    matrix = np.array([[per_case[c].get(lbl, {}).get("Dice") or 0 for lbl in label_names]
                       for c in sorted_cases])
    fig, ax = plt.subplots(figsize=(max(6, len(label_names) * 2), max(8, len(sorted_cases) * 0.35)))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(label_names)))
    ax.set_xticklabels(label_names)
    ax.set_yticks(range(len(sorted_cases)))
    ax.set_yticklabels(sorted_cases, fontsize=7)
    for i, case in enumerate(sorted_cases):
        for j, lbl in enumerate(label_names):
            v = per_case[case].get(lbl, {}).get("Dice")
            if v is not None:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color="black" if 0.3 < v < 0.85 else "white")
    plt.colorbar(im, ax=ax, label="Dice")
    ax.set_title(f"{name} – Dice Heatmap (cases × labels)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved: {out_path}")


def plot_per_label_boxplot(per_case: dict, label_names: list[str],
                           out_path: Path, name: str):
    """Boxplot of per-case Dice per label."""
    data = [[per_case[c].get(lbl, {}).get("Dice") for c in per_case
             if per_case[c].get(lbl, {}).get("Dice") is not None]
            for lbl in label_names]
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, tick_labels=label_names, patch_artist=True, notch=False)
    colors = ["#4878cf", "#6acc65", "#d65f5f", "#9467bd", "#e377c2"]
    for patch, color in zip(bp["boxes"], colors[:len(label_names)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_ylabel("Dice")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{name} – Per-Case Dice Distribution ({len(per_case)} cases)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate ensemble/test-set predictions.")
    parser.add_argument("--dataset", default="Dataset001_GroundTruth")
    parser.add_argument("--pred_dir", required=True,
                        help="Prediction folder (relative to project root)")
    parser.add_argument("--labels_dir", required=True,
                        help="Ground truth labels folder (relative to project root)")
    parser.add_argument("--config", default="3d_fullres")
    parser.add_argument("--trainer", default="nnUNetTrainer")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                        help="Fold indices used for val comparison (default: 0 1 2 3 4)")
    parser.add_argument("--name", default=None,
                        help="Label for outputs (default: derived from pred_dir)")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--no_save", action="store_true")
    args = parser.parse_args()

    pred_dir   = PROJECT_ROOT / args.pred_dir
    labels_dir = PROJECT_ROOT / args.labels_dir
    name       = args.name or pred_dir.name

    output_dir = Path(args.output_dir) if args.output_dir else \
                 PROJECT_ROOT / "analysis_results" / "evaluation" / args.dataset / "ensemble"
    if not args.no_save:
        output_dir.mkdir(parents=True, exist_ok=True)

    label_map   = load_dataset_labels(args.dataset)
    label_names = list(label_map.values())

    print(f"\nDataset    : {args.dataset}")
    print(f"Predictions: {pred_dir}")
    print(f"Labels     : {labels_dir}")
    print(f"Labels     : {', '.join(label_names)}")

    if not pred_dir.exists():
        print(f"[ERROR] Prediction directory not found: {pred_dir}")
        sys.exit(1)

    # ── Find plans.json ───────────────────────────────────────────────────────
    plans_file = pred_dir / "plans.json"
    if not plans_file.exists():
        plans_file = (PROJECT_ROOT / "data" / "nnUNet_preprocessed" /
                      args.dataset / "nnUNetPlans.json")
    if not plans_file.exists():
        print(f"[ERROR] plans.json not found. Copy it to {pred_dir}/plans.json")
        sys.exit(1)

    dataset_json = PROJECT_ROOT / "data" / "nnUNet_raw" / args.dataset / "dataset.json"

    # ── Run evaluation ────────────────────────────────────────────────────────
    summary = run_nnunet_evaluate(pred_dir, labels_dir, dataset_json, plans_file)
    if summary is None:
        sys.exit(1)

    metrics   = extract_metrics(summary, label_map)
    per_case  = per_case_metrics(summary, label_map)
    fold_means = load_fold_means(args.dataset, args.config, args.trainer,
                                 args.folds, label_map)

    # ── Main metrics table ────────────────────────────────────────────────────
    rows = [(lbl, fmt(metrics[lbl]["Dice"]), fmt(metrics[lbl]["IoU"]),
             fmt(metrics[lbl].get("FP")), fmt(metrics[lbl].get("FN")))
            for lbl in label_names]
    rows.append(("MEAN (fg)", fmt(metrics["__fg__"]["Dice"]),
                 fmt(metrics["__fg__"]["IoU"]), "", ""))
    print_table(f"{name} – Results ({len(per_case)} cases)",
                rows, ["Label", "Dice", "IoU", "FP (vox)", "FN (vox)"])

    # ── Comparison vs. validation ─────────────────────────────────────────────
    if fold_means:
        headers = ["Label", "Test (Ensemble)", "Val Mean (folds)", "Δ Dice"]
        rows = []
        for lbl in label_names:
            ens  = metrics[lbl]["Dice"]
            val  = fold_means.get(lbl)
            diff = f"{ens - val:+.4f}" if ens is not None and val is not None else "  N/A "
            rows.append((lbl, fmt(ens), fmt(val), diff))
        fg_ens = metrics["__fg__"]["Dice"]
        fg_val = fold_means.get("__fg__")
        diff   = f"{fg_ens - fg_val:+.4f}" if fg_ens and fg_val else "  N/A "
        rows.append(("MEAN (fg)", fmt(fg_ens), fmt(fg_val), diff))
        print_table("Comparison: Test vs. Validation", rows, headers)

    # ── Per-case worst/best ───────────────────────────────────────────────────
    case_means = {}
    for cid, lbls in per_case.items():
        vals = [lbls[lbl].get("Dice") for lbl in label_names if lbls.get(lbl, {}).get("Dice") is not None]
        if vals:
            case_means[cid] = float(np.mean(vals)) if HAS_MPL else sum(vals) / len(vals)
    if case_means:
        sorted_c = sorted(case_means.items(), key=lambda x: x[1])
        worst5 = sorted_c[:5]
        best5  = sorted_c[-5:][::-1]
        print("\n── Worst 5 cases (mean foreground Dice) ──")
        for cid, v in worst5:
            per_lbl = "  ".join(f"{lbl}: {fmt(per_case[cid].get(lbl, {}).get('Dice'))}"
                                for lbl in label_names)
            print(f"  {cid:<30}  mean={v:.4f}  [{per_lbl}]")
        print("\n── Best 5 cases ──")
        for cid, v in best5:
            per_lbl = "  ".join(f"{lbl}: {fmt(per_case[cid].get(lbl, {}).get('Dice'))}"
                                for lbl in label_names)
            print(f"  {cid:<30}  mean={v:.4f}  [{per_lbl}]")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    if not args.no_save:
        csv_path = output_dir / f"ensemble_{name}_{args.dataset}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["case_id", "label", "Dice", "IoU"])
            for cid, lbls in sorted(per_case.items()):
                for lbl in label_names:
                    m = lbls.get(lbl, {})
                    writer.writerow([cid, lbl, m.get("Dice", ""), m.get("IoU", "")])
        print(f"\n[INFO] CSV saved: {csv_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not args.no_save and HAS_MPL:
        plot_label_comparison(metrics, fold_means, label_names,
                              output_dir / f"ensemble_label_dice_{name}_{args.dataset}.png", name)
        plot_per_label_boxplot(per_case, label_names,
                               output_dir / f"ensemble_boxplot_{name}_{args.dataset}.png", name)
        plot_per_case_ranked(per_case, label_names,
                             output_dir / f"ensemble_ranked_{name}_{args.dataset}.png", name)
        plot_per_case_heatmap(per_case, label_names,
                              output_dir / f"ensemble_heatmap_{name}_{args.dataset}.png", name)
        print(f"\n[INFO] All outputs in: {output_dir}")


if __name__ == "__main__":
    main()
