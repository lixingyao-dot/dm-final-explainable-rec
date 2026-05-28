"""Training visualization utilities."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _ensure_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _set_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "lines.linewidth": 1.5,
        "figure.figsize": (10, 5),
    })


def plot_training_history(train_losses, val_losses, lr_history, save_path,
                           val_hitrates=None):
    """3-panel figure: loss curve (top) + HitRate (mid) + learning rate (bottom)."""
    _ensure_dir(save_path)
    _set_style()

    n_panels = 3 if val_hitrates else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 4 + 2 * n_panels), sharex=True)

    epochs = range(1, len(train_losses) + 1)

    # Panel 1: Loss
    ax_loss = axes[0]
    ax_loss.plot(epochs, train_losses, "b-o", label="Train Loss", markersize=3)
    if val_losses:
        ax_loss.plot(epochs, val_losses, "r-s", label="Val Loss", markersize=3)
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training & Validation Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    # Panel 2: HitRate or LR
    if val_hitrates and len(val_hitrates) == len(train_losses):
        ax_hr = axes[1]
        ax_hr.plot(epochs, val_hitrates, "g-^", label="HitRate@10", markersize=3)
        ax_hr.set_ylabel("HitRate@10")
        ax_hr.set_title("Validation HitRate@10 (early stopping metric)")
        ax_hr.legend()
        ax_hr.grid(True, alpha=0.3)
        ax_lr = axes[2] if n_panels == 3 else axes[1]
    else:
        ax_lr = axes[1]

    # Last panel: LR
    ax_lr.plot(epochs, lr_history, "m-d", markersize=3)
    ax_lr.set_xlabel("Epoch")
    ax_lr.set_ylabel("Learning Rate")
    ax_lr.set_title("Learning Rate Schedule")
    ax_lr.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Training chart saved to {save_path}")


def plot_metrics_comparison(metrics_dict, title, save_path):
    """Grouped bar chart comparing multiple models across metrics."""
    _ensure_dir(save_path)
    _set_style()

    models = list(metrics_dict.keys())
    metrics = [k for k in list(metrics_dict.values())[0].keys() if k not in ("time",)]

    x = np.arange(len(metrics))
    n_models = len(models)
    bar_width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(max(8, len(metrics) * 2), 5))

    for i, model_name in enumerate(models):
        values = [metrics_dict[model_name][m] for m in metrics]
        offset = (i - n_models / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, values, bar_width, label=model_name)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.4f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Metrics chart saved to {save_path}")


def plot_alpha_ablation(alpha_results, metric_names, save_path):
    """Line plot: metrics vs alpha values for NCF+Review ablation."""
    _ensure_dir(save_path)
    _set_style()

    alphas = sorted(alpha_results.keys())
    fig, ax = plt.subplots(figsize=(8, 5))

    markers = ["o", "s", "^", "D", "v"]
    for i, metric in enumerate(metric_names):
        values = [alpha_results[a][metric] for a in alphas]
        ax.plot(alphas, values, f"-{markers[i % len(markers)]}",
                label=metric, markersize=6)

    ax.set_xlabel("Alpha (fusion weight)")
    ax.set_ylabel("Score")
    ax.set_title("NCF+Review: Alpha Ablation Study")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Alpha ablation chart saved to {save_path}")


def plot_experiment_summary(all_results, title, save_path):
    """Full comparison across all experiments (combined bar chart)."""
    _ensure_dir(save_path)
    _set_style()

    models = list(all_results.keys())
    metrics = [k for k in list(all_results.values())[0].keys() if k not in ("time",)]

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 4),
                             squeeze=False)

    for j, metric in enumerate(metrics):
        ax = axes[0, j]
        names = []
        values = []
        for model_name in models:
            names.append(model_name)
            values.append(all_results[model_name][metric])

        colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
        bars = ax.bar(range(len(names)), values, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_title(metric)
        ax.grid(True, alpha=0.3, axis="y")
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.4f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Experiment summary chart saved to {save_path}")