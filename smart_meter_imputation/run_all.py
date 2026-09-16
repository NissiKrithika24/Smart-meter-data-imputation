"""
Master script: train all 3 models × 3 mechanisms × 4 rates = 36 experiments,
then evaluate and produce the final results table and comparison plots.

Usage:
    python run_all.py
    python run_all.py --epochs 30         # override number of epochs
    python run_all.py --models LSTM GRU   # only run specific models
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import config
from utils import set_seed, get_device
from data_loader import prepare_data
from train import create_model, train_model
from evaluate import run_full_evaluation, save_results_table, generate_overlay_comparisons


def create_comparison_heatmaps(results_df: pd.DataFrame, save_dir: str = config.PLOT_DIR):
    """
    Create heatmap comparisons of RMSE, MAE, MAPE across models/mechanisms/rates.
    """
    os.makedirs(save_dir, exist_ok=True)

    for metric in ["RMSE", "MAE", "MAPE (%)"]:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
        fig.suptitle(f"{metric} Comparison Across Models", fontsize=14, fontweight="bold")

        for ax, model_name in zip(axes, ["LSTM", "GRU", "ResNet"]):
            model_data = results_df[results_df["Model"] == model_name]
            if model_data.empty:
                continue

            pivot = model_data.pivot_table(
                values=metric, index="Mechanism", columns="Missing Rate", aggfunc="first"
            )

            # Reorder columns
            rate_order = ["10%", "20%", "30%", "40%"]
            pivot = pivot.reindex(columns=[c for c in rate_order if c in pivot.columns])

            sns.heatmap(
                pivot, annot=True, fmt=".4f" if metric != "MAPE (%)" else ".2f",
                cmap="YlOrRd", ax=ax, cbar_kws={"shrink": 0.8},
            )
            ax.set_title(model_name, fontsize=12)
            ax.set_ylabel("" if ax != axes[0] else "Mechanism")
            ax.set_xlabel("Missing Rate")

        plt.tight_layout()
        safe_metric = metric.replace(" ", "_").replace("(%)", "pct")
        fig.savefig(os.path.join(save_dir, f"heatmap_{safe_metric}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Heatmap saved: heatmap_{safe_metric}.png")


def create_bar_comparison(results_df: pd.DataFrame, save_dir: str = config.PLOT_DIR):
    """
    Create grouped bar charts comparing models for each mechanism.
    """
    os.makedirs(save_dir, exist_ok=True)

    for mechanism in config.MISSING_MECHANISMS:
        mech_data = results_df[results_df["Mechanism"] == mechanism].copy()
        if mech_data.empty:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f"Model Comparison — {mechanism} Missingness", fontsize=14, fontweight="bold")

        for ax, metric in zip(axes, ["RMSE", "MAE", "MAPE (%)"]):
            pivot = mech_data.pivot_table(
                values=metric, index="Missing Rate", columns="Model", aggfunc="first"
            )
            rate_order = ["10%", "20%", "30%", "40%"]
            pivot = pivot.reindex([r for r in rate_order if r in pivot.index])

            pivot.plot(kind="bar", ax=ax, rot=0, color=["#2196F3", "#4CAF50", "#F44336"])
            ax.set_title(metric)
            ax.set_xlabel("Missing Rate")
            ax.set_ylabel(metric)
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, f"bars_{mechanism}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Bar chart saved: bars_{mechanism}.png")


def main():
    parser = argparse.ArgumentParser(description="Run all imputation experiments")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help="Training epochs per experiment")
    parser.add_argument(
        "--models", nargs="+", default=["LSTM", "GRU", "ResNet"],
        choices=["LSTM", "GRU", "ResNet"], help="Models to train"
    )
    parser.add_argument(
        "--mechanisms", nargs="+", default=config.MISSING_MECHANISMS,
        choices=["MCAR", "MAR", "MNAR"], help="Missingness mechanisms"
    )
    parser.add_argument(
        "--rates", nargs="+", type=float, default=config.MISSING_RATES,
        help="Missingness rates"
    )
    args = parser.parse_args()

    set_seed()
    device = get_device()
    print(f"Device: {device}")

    total_start = time.time()

    # ── Data Preparation ──
    print("\n" + "=" * 60)
    print("PHASE 0: DATA PREPARATION")
    print("=" * 60)
    df, scalers, train_windows, val_windows, test_windows = prepare_data()

    total_experiments = len(args.models) * len(args.mechanisms) * len(args.rates)
    print(f"\nTotal experiments to run: {total_experiments}")

    # ── Training ──
    print("\n" + "=" * 60)
    print("PHASE 2 & 3: TRAINING ALL MODELS")
    print("=" * 60)

    experiment_count = 0
    for model_name in args.models:
        for mechanism in args.mechanisms:
            for rate in args.rates:
                experiment_count += 1
                print(f"\n--- Experiment {experiment_count}/{total_experiments} ---")
                print(f"  Model: {model_name} | Mechanism: {mechanism} | Rate: {rate*100:.0f}%")

                model = create_model(model_name)
                param_count = sum(p.numel() for p in model.parameters())
                print(f"  Parameters: {param_count:,}")

                result = train_model(
                    model, train_windows, val_windows,
                    mechanism, rate, num_epochs=args.epochs,
                )

                print(
                    f"  Done: best_epoch={result['best_epoch']}, "
                    f"time={result['training_time']:.1f}s"
                )

    # ── Evaluation ──
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    results_df = run_full_evaluation(
        test_windows,
        mechanisms=args.mechanisms,
        rates=args.rates,
        model_names=args.models,
    )

    # Save results
    save_results_table(results_df)

    # ── Plots ──
    print("\n" + "=" * 60)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 60)

    create_comparison_heatmaps(results_df)
    create_bar_comparison(results_df)
    generate_overlay_comparisons(test_windows)

    total_time = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"ALL DONE! Total time: {total_time/60:.1f} minutes")
    print(f"Results: {config.RESULTS_DIR}")
    print(f"Plots:   {config.PLOT_DIR}")
    print(f"Models:  {config.MODEL_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
