"""
Evaluation module: compute RMSE, MAE, MAPE on test set and generate comparison plots.

Evaluates a trained model on the test windows with the specified missingness,
computing metrics only on the artificially masked positions.
"""

import os
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

import config
from utils import get_device, compute_all_metrics, plot_imputation_sample, plot_overlay_comparison
from data_loader import create_dataloader
from missingness import get_mask_fn
from train import create_model


def evaluate_model(
    model: torch.nn.Module,
    test_windows: list,
    mechanism: str,
    missing_rate: float,
    scalers: dict = None,
) -> dict:
    """
    Evaluate a trained model on test data.

    Returns:
        dict with RMSE, MAE, MAPE computed on the masked positions.
    """
    device = get_device()
    model = model.to(device)
    model.eval()

    mask_fn = get_mask_fn(mechanism)
    test_loader = create_dataloader(
        test_windows, mask_fn, missing_rate,
        batch_size=config.BATCH_SIZE, shuffle=False,
    )

    all_true = []
    all_pred = []

    with torch.no_grad():
        for features, targets, masks in test_loader:
            features = features.to(device)
            predictions = model(features).cpu().numpy()  # (batch, window_size)
            targets = targets.numpy()                     # (batch, window_size)
            masks = masks.numpy()                         # (batch, window_size)

            # Collect only the missing positions
            missing_mask = (masks == 0)
            for i in range(len(predictions)):
                if missing_mask[i].any():
                    all_true.append(targets[i][missing_mask[i]])
                    all_pred.append(predictions[i][missing_mask[i]])

    if len(all_true) == 0:
        return {"RMSE": 0.0, "MAE": 0.0, "MAPE": 0.0}

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    # Clip predictions to [0, 1] range (normalized values)
    y_pred = np.clip(y_pred, 0.0, 1.0)

    metrics = compute_all_metrics(y_true, y_pred)
    return metrics


def evaluate_and_plot_sample(
    model: torch.nn.Module,
    test_windows: list,
    mechanism: str,
    missing_rate: float,
    model_name: str,
    plot_dir: str = config.PLOT_DIR,
) -> dict:
    """
    Evaluate model and save a sample imputation plot.
    Returns metrics dict.
    """
    metrics = evaluate_model(model, test_windows, mechanism, missing_rate)

    # Generate a sample plot from the first test window
    mask_fn = get_mask_fn(mechanism)
    w = test_windows[0]
    original = w["kWh_norm"]
    mask = mask_fn(original, missing_rate)
    masked_kWh = original * mask

    features = np.stack([
        masked_kWh, mask,
        w["hour_sin"], w["hour_cos"], w["dow_sin"],
    ], axis=-1).astype(np.float32)

    device = get_device()
    model.eval()
    with torch.no_grad():
        inp = torch.from_numpy(features).unsqueeze(0).to(device)
        pred = model(inp).cpu().numpy().squeeze()

    pred = np.clip(pred, 0.0, 1.0)

    plot_path = os.path.join(
        plot_dir, f"sample_{model_name}_{mechanism}_{int(missing_rate*100)}.png"
    )
    plot_imputation_sample(
        original, masked_kWh, pred, mask,
        model_name, mechanism, missing_rate, plot_path,
    )

    return metrics


def run_full_evaluation(
    test_windows: list,
    mechanisms: list = None,
    rates: list = None,
    model_names: list = None,
) -> pd.DataFrame:
    """
    Run evaluation across all models, mechanisms, and rates.
    Returns a DataFrame with all results.
    """
    if mechanisms is None:
        mechanisms = config.MISSING_MECHANISMS
    if rates is None:
        rates = config.MISSING_RATES
    if model_names is None:
        model_names = ["LSTM", "GRU", "ResNet"]

    device = get_device()
    results = []

    for model_name in model_names:
        for mechanism in mechanisms:
            for rate in rates:
                print(f"\nEvaluating {model_name} | {mechanism} @ {rate*100:.0f}%...")

                # Load trained model
                model = create_model(model_name)
                model_path = os.path.join(
                    config.MODEL_DIR,
                    f"{model_name}_{mechanism}_{int(rate*100)}.pt",
                )

                if not os.path.exists(model_path):
                    print(f"  WARNING: Model not found at {model_path}, skipping.")
                    continue

                model.load_state_dict(
                    torch.load(model_path, map_location=device, weights_only=True)
                )

                metrics = evaluate_and_plot_sample(
                    model, test_windows, mechanism, rate, model_name,
                )

                results.append({
                    "Model": model_name,
                    "Mechanism": mechanism,
                    "Missing Rate": f"{rate*100:.0f}%",
                    "RMSE": round(metrics["RMSE"], 6),
                    "MAE": round(metrics["MAE"], 6),
                    "MAPE (%)": round(metrics["MAPE"], 2),
                })

                print(
                    f"  RMSE={metrics['RMSE']:.6f} | "
                    f"MAE={metrics['MAE']:.6f} | "
                    f"MAPE={metrics['MAPE']:.2f}%"
                )

    df = pd.DataFrame(results)
    return df


def generate_overlay_comparisons(test_windows: list, plot_dir: str = config.PLOT_DIR):
    """
    Generate comparative overlay plots (Ground Truth vs LSTM vs GRU vs ResNet)
    for representative conditions (e.g. MCAR 20%, MAR 20%, MNAR 30%).
    """
    device = get_device()
    scenarios = [
        ("MCAR", 0.20),
        ("MAR", 0.20),
        ("MNAR", 0.30),
    ]

    for mechanism, rate in scenarios:
        models = {}
        for model_name in ["LSTM", "GRU", "ResNet"]:
            model_path = os.path.join(
                config.MODEL_DIR, f"{model_name}_{mechanism}_{int(rate*100)}.pt"
            )
            if not os.path.exists(model_path):
                continue
            m = create_model(model_name)
            m.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            m.to(device)
            m.eval()
            models[model_name] = m

        if len(models) < 2:
            continue

        # Choose a representative test window with variance
        w = test_windows[min(len(test_windows)-1, 5)]
        original = w["kWh_norm"]
        mask_fn = get_mask_fn(mechanism)
        mask = mask_fn(original, rate)
        masked_kWh = original * mask

        features = np.stack([
            masked_kWh, mask,
            w["hour_sin"], w["hour_cos"], w["dow_sin"],
        ], axis=-1).astype(np.float32)

        preds = {}
        with torch.no_grad():
            inp = torch.from_numpy(features).unsqueeze(0).to(device)
            for name, m in models.items():
                pred = m(inp).cpu().numpy().squeeze()
                preds[name] = np.clip(pred, 0.0, 1.0)

        save_path = os.path.join(
            plot_dir, f"comparison_overlay_{mechanism}_{int(rate*100)}.png"
        )
        plot_overlay_comparison(original, preds, mask, mechanism, rate, save_path)


def save_results_table(df: pd.DataFrame, path: str = None):
    """Save results to CSV and print a formatted table."""
    if path is None:
        path = os.path.join(config.RESULTS_DIR, "results_table.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\nResults saved to {path}")
    print("\n" + "=" * 80)
    print("RESULTS TABLE")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    from data_loader import prepare_data
    from utils import set_seed

    set_seed()
    _, _, _, _, test_w = prepare_data()

    df = run_full_evaluation(test_w)
    save_results_table(df)
