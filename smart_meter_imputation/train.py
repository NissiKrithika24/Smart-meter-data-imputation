"""
Unified training loop for all imputation models (LSTM, GRU, ResNet).

Supports:
- Training with masked loss (MSE only on missing positions)
- Early stopping with patience
- Training/validation loss tracking
- Model checkpointing
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import config
from utils import set_seed, get_device, plot_training_loss
from data_loader import create_dataloader
from missingness import get_mask_fn
from models.lstm_imputer import LSTMImputer
from models.gru_imputer import GRUImputer
from models.resnet_imputer import ResNetImputer


# ──────────────────────────────────────────────
# Model factory
# ──────────────────────────────────────────────
def create_model(model_name: str) -> nn.Module:
    """Create a model instance by name."""
    models = {
        "LSTM": LSTMImputer,
        "GRU": GRUImputer,
        "ResNet": ResNetImputer,
    }
    if model_name not in models:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {list(models.keys())}")
    return models[model_name]()


# ──────────────────────────────────────────────
# Masked MSE Loss
# ──────────────────────────────────────────────
def masked_mse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    """
    Compute MSE loss only on missing positions (mask == 0).
    Also includes a small reconstruction loss on observed positions to
    stabilize training.

    Args:
        predictions: (batch, window_size)
        targets: (batch, window_size)
        masks: (batch, window_size) — 1=observed, 0=missing
    """
    # Loss on missing positions (primary)
    missing_mask = 1.0 - masks  # 1 where missing
    missing_count = missing_mask.sum()

    if missing_count > 0:
        missing_loss = ((predictions - targets) ** 2 * missing_mask).sum() / missing_count
    else:
        missing_loss = torch.tensor(0.0, device=predictions.device)

    # Small auxiliary loss on observed positions (stabilizes reconstruction)
    observed_count = masks.sum()
    if observed_count > 0:
        observed_loss = ((predictions - targets) ** 2 * masks).sum() / observed_count
    else:
        observed_loss = torch.tensor(0.0, device=predictions.device)

    # Weighted combination: primarily optimize for missing positions
    return 0.8 * missing_loss + 0.2 * observed_loss


# ──────────────────────────────────────────────
# Training function
# ──────────────────────────────────────────────
def train_model(
    model: nn.Module,
    train_windows: list,
    val_windows: list,
    mechanism: str,
    missing_rate: float,
    num_epochs: int = config.NUM_EPOCHS,
    lr: float = config.LEARNING_RATE,
    patience: int = config.EARLY_STOP_PATIENCE,
    save_dir: str = config.MODEL_DIR,
) -> dict:
    """
    Train a model on the given data with specified missingness.

    Returns a dict with:
        - 'train_losses': list of per-epoch training losses
        - 'val_losses': list of per-epoch validation losses
        - 'best_epoch': epoch with best validation loss
        - 'model_path': path to saved best model
        - 'training_time': total training time in seconds
    """
    device = get_device()
    model = model.to(device)

    mask_fn = get_mask_fn(mechanism)

    # Create data loaders
    train_loader = create_dataloader(train_windows, mask_fn, missing_rate, shuffle=True)
    val_loader = create_dataloader(val_windows, mask_fn, missing_rate, shuffle=False)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    # Model save path
    model_name = model.model_name if hasattr(model, 'model_name') else 'model'
    save_path = os.path.join(save_dir, f"{model_name}_{mechanism}_{int(missing_rate*100)}.pt")
    os.makedirs(save_dir, exist_ok=True)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        # ── Training ──
        model.train()
        epoch_train_loss = 0.0
        n_train_batches = 0

        for features, targets, masks in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            predictions = model(features)
            loss = masked_mse_loss(predictions, targets, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_train_loss += loss.item()
            n_train_batches += 1

        avg_train_loss = epoch_train_loss / max(n_train_batches, 1)
        train_losses.append(avg_train_loss)

        # ── Validation ──
        model.eval()
        epoch_val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for features, targets, masks in val_loader:
                features = features.to(device)
                targets = targets.to(device)
                masks = masks.to(device)

                predictions = model(features)
                loss = masked_mse_loss(predictions, targets, masks)

                epoch_val_loss += loss.item()
                n_val_batches += 1

        avg_val_loss = epoch_val_loss / max(n_val_batches, 1)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        # ── Early Stopping ──
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{num_epochs} | "
                f"Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | "
                f"Best: {best_val_loss:.6f} (ep {best_epoch})"
            )

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch} (patience={patience})")
            break

    training_time = time.time() - start_time

    # Load best model
    model.load_state_dict(torch.load(save_path, weights_only=True))

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
        "model_path": save_path,
        "training_time": training_time,
    }


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from data_loader import prepare_data

    parser = argparse.ArgumentParser(description="Train an imputation model")
    parser.add_argument("--model", type=str, default="LSTM", choices=["LSTM", "GRU", "ResNet"])
    parser.add_argument("--mechanism", type=str, default="MCAR", choices=["MCAR", "MAR", "MNAR"])
    parser.add_argument("--rate", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    args = parser.parse_args()

    set_seed()
    print(f"Training {args.model} with {args.mechanism} @ {args.rate*100:.0f}% missing")

    _, _, train_w, val_w, _ = prepare_data()
    model = create_model(args.model)

    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    result = train_model(model, train_w, val_w, args.mechanism, args.rate, args.epochs)

    print(f"\nDone! Best epoch: {result['best_epoch']}, Time: {result['training_time']:.1f}s")
    print(f"Model saved to: {result['model_path']}")

    # Plot training curves
    plot_training_loss(
        result["train_losses"],
        result["val_losses"],
        f"{args.model}_{args.mechanism}_{int(args.rate*100)}",
        os.path.join(config.PLOT_DIR, f"loss_{args.model}_{args.mechanism}_{int(args.rate*100)}.png"),
    )
