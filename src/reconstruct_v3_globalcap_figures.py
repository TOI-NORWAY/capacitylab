#!/usr/bin/env python3
"""Reconstruct v3 global-capacity flow+speed summary figures from saved logs.

This is mainly for runs produced before train_flow_speed_v3_globalcap.py saved
history CSVs and the standard learning/model-comparison figures directly.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


EPOCH_RE = re.compile(
    r"\b(?P<kind>FF|GNN) Epoch\s+(?P<epoch>\d+): "
    r"train=(?P<train>[0-9.]+), val=(?P<val>[0-9.]+), "
    r"time=(?P<time>[0-9.]+)s, samples/s=(?P<samples>[0-9,]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct v3 global-capacity flow+speed figures.")
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/flowspeed_v3_globalcap"))
    args = parser.parse_args()
    args.run_dir = project_path(args.run_dir)
    return args


def parse_history(log_file: Path) -> pd.DataFrame:
    rows = []
    for line in log_file.read_text(encoding="utf-8").splitlines():
        match = EPOCH_RE.search(line)
        if not match:
            continue
        model = "FF-GlobalCapacity" if match.group("kind") == "FF" else "GNN-GlobalCapacity"
        rows.append(
            {
                "model": model,
                "epoch": int(match.group("epoch")),
                "train_loss": float(match.group("train")),
                "val_loss": float(match.group("val")),
                "epoch_time_sec": float(match.group("time")),
                "samples_per_sec": float(match.group("samples").replace(",", "")),
            }
        )
    return pd.DataFrame(rows)


def plot_learning_curves(run_dir: Path, history_df: pd.DataFrame) -> None:
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    colors = {"FF-GlobalCapacity": "#4ECDC4", "GNN-GlobalCapacity": "#45B7D1"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.ravel()

    for model, group in history_df.groupby("model"):
        group = group.sort_values("epoch")
        label = model.replace("-GlobalCapacity", "")
        axes[0].plot(group["epoch"], group["train_loss"], color=colors[model], linewidth=2, marker="o", label=label)
        axes[1].plot(group["epoch"], group["val_loss"], color=colors[model], linewidth=2, marker="o", label=label)
        axes[2].plot(
            group["epoch"],
            group["val_loss"] - group["train_loss"],
            color=colors[model],
            linewidth=2,
            marker="o",
            label=label,
        )
        axes[3].plot(group["epoch"], group["samples_per_sec"], color=colors[model], linewidth=2, marker="o", label=label)

    titles = [
        "Training Loss - Combined Flow + Speed",
        "Validation Loss - Combined Flow + Speed",
        "Generalization Gap",
        "Training Throughput",
    ]
    ylabels = ["MSE", "MSE", "Val Loss - Train Loss", "Samples/s"]
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[2].axhline(y=0, color="black", linestyle=":", linewidth=1)

    fig.suptitle("Sparse reconstruction from train.log (logged every print interval)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(figures_dir / "learning_curves_flow_speed.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(run_dir: Path, metrics_df: pd.DataFrame) -> None:
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    model_order = [m for m in ["FF-GlobalCapacity", "GNN-GlobalCapacity"] if m in set(metrics_df["model"])]
    colors = ["#4ECDC4", "#45B7D1"][: len(model_order)]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    metric_specs = [("r2", "R2 Score", "higher is better"), ("rmse", "RMSE", "lower is better"), ("mae", "MAE", "lower is better")]

    for row_idx, target in enumerate(["flow", "speed"]):
        target_df = metrics_df[metrics_df["target"] == target].set_index("model")
        for col_idx, (metric, label, hint) in enumerate(metric_specs):
            ax = axes[row_idx, col_idx]
            values = [float(target_df.loc[model, metric]) for model in model_order]
            bars = ax.bar(range(len(model_order)), values, color=colors, alpha=0.85, edgecolor="black", linewidth=1.5)
            ax.set_title(f"{target.upper()}: {label} ({hint})", fontsize=13, fontweight="bold")
            ax.set_xticks(range(len(model_order)))
            ax.set_xticklabels(["FF", "GNN"][: len(model_order)])
            ax.grid(True, alpha=0.3, axis="y")
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    fig.savefig(figures_dir / "model_comparison_flow_speed.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    log_file = run_dir / "logs" / "train.log"
    metrics_file = run_dir / "data" / "metrics_flowspeed_v3_globalcap.csv"
    history_file = run_dir / "data" / "history_flowspeed_v3_globalcap_reconstructed.csv"

    history_df = parse_history(log_file)
    if history_df.empty:
        raise RuntimeError(f"No epoch history lines found in {log_file}")
    history_file.parent.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(history_file, index=False)
    plot_learning_curves(run_dir, history_df)

    metrics_df = pd.read_csv(metrics_file)
    plot_model_comparison(run_dir, metrics_df)

    print(f"Saved reconstructed history to {history_file}")
    print(f"Saved figures to {run_dir / 'figures'}")


if __name__ == "__main__":
    main()
