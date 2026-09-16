"""
Utility functions: metrics, plotting, reproducibility, device management.
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots

import config


# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
def set_seed(seed: int = config.RANDOM_SEED):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return the best available device (CUDA > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ──────────────────────────────────────────────
# Evaluation Metrics
# ──────────────────────────────────────────────
def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error.
    Only computed on non-zero actuals to avoid division-by-zero.
    Returns percentage (0-100 scale).
    """
    nonzero_mask = np.abs(y_true) > eps
    if nonzero_mask.sum() == 0:
        return 0.0
    return float(
        np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100
    )


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute all evaluation metrics and return as a dictionary."""
    return {
        "RMSE": compute_rmse(y_true, y_pred),
        "MAE": compute_mae(y_true, y_pred),
        "MAPE": compute_mape(y_true, y_pred),
    }


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────
def plot_imputation_sample(
    original: np.ndarray,
    masked: np.ndarray,
    imputed: np.ndarray,
    mask: np.ndarray,
    model_name: str,
    mechanism: str,
    rate: float,
    save_path: str | None = None,
):
    """
    Plot a single window showing original, masked, and imputed values.

    Args:
        original: true values (window_size,)
        masked: values with missingness applied (window_size,)
        imputed: model predictions (window_size,)
        mask: binary mask, 1=observed, 0=missing (window_size,)
        model_name: e.g. "LSTM", "GRU", "ResNet"
        mechanism: e.g. "MCAR", "MAR", "MNAR"
        rate: missingness rate
        save_path: if provided, save the figure here
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    timesteps = np.arange(len(original))

    # Original full signal
    ax.plot(timesteps, original, color="gray", alpha=0.5, linewidth=1, label="Original (true)")

    # Observed points
    obs_idx = mask == 1
    ax.scatter(timesteps[obs_idx], original[obs_idx], color="blue", s=12, zorder=3, label="Observed")

    # Missing positions: show true vs imputed
    miss_idx = mask == 0
    if miss_idx.any():
        ax.scatter(
            timesteps[miss_idx], original[miss_idx],
            color="green", marker="x", s=30, zorder=4, label="True (missing)"
        )
        ax.scatter(
            timesteps[miss_idx], imputed[miss_idx],
            color="red", marker="o", s=20, zorder=5, alpha=0.7, label="Imputed"
        )

    ax.set_xlabel("Timestep (15-min intervals)")
    ax.set_ylabel("kWh")
    ax.set_title(f"{model_name} Imputation — {mechanism} @ {rate*100:.0f}% missing")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved to {save_path}")
    plt.close(fig)


def plot_training_loss(
    train_losses: list,
    val_losses: list,
    model_name: str,
    save_path: str | None = None,
):
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train Loss", color="blue")
    ax.plot(epochs, val_losses, label="Val Loss", color="orange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"{model_name} — Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_overlay_comparison(
    original: np.ndarray,
    preds_dict: dict,
    mask: np.ndarray,
    mechanism: str,
    rate: float,
    save_path: str | None = None,
):
    """
    Plot all models (LSTM, GRU, ResNet) simultaneously against ground truth.
    Highlights missing regions with gray vertical shading.

    Args:
        original: true values (window_size,)
        preds_dict: dict of {model_name: imputed_array}
        mask: binary mask (1=observed, 0=missing)
        mechanism: missingness mechanism (MCAR, MAR, MNAR)
        rate: missingness rate
        save_path: path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    timesteps = np.arange(len(original))

    # Ground truth
    ax.plot(timesteps, original, color="black", linewidth=2.0, label="Ground Truth", zorder=3)

    # Missing regions highlight
    miss_idx = np.where(mask == 0)[0]
    for idx in miss_idx:
        ax.axvspan(idx - 0.5, idx + 0.5, color="lightgray", alpha=0.35, zorder=1)

    colors = {"LSTM": "#1f77b4", "GRU": "#2ca02c", "ResNet": "#d62728"}
    linestyles = {"LSTM": "--", "GRU": "-.", "ResNet": "-"}

    for name, pred in preds_dict.items():
        c = colors.get(name, "purple")
        ls = linestyles.get(name, "-")
        lw = 2.2 if name == "ResNet" else 1.5
        ax.plot(timesteps, pred, color=c, linestyle=ls, linewidth=lw, label=f"{name} Imputed", zorder=4)

    # Mark observed points lightly
    obs_idx = np.where(mask == 1)[0]
    ax.scatter(timesteps[obs_idx], original[obs_idx], color="gray", s=10, alpha=0.5, zorder=2, label="Observed Points")

    ax.set_xlabel("Time Interval (15-min steps across 24h Day)", fontsize=11)
    ax.set_ylabel("Normalized Energy (kWh)", fontsize=11)
    ax.set_title(
        f"Comparative Imputation: ResNet vs. LSTM & GRU Baselines\n({mechanism} Missingness @ {int(rate*100)}% — Gray spans = Missing Intervals)",
        fontsize=13, fontweight="bold"
    )
    ax.legend(loc="upper right", framealpha=0.9, fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  Overlay comparison plot saved to {save_path}")
    plt.close(fig)

