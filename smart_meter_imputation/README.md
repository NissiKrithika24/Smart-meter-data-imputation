# Missing Value Imputation for Smart Meter Time-Series Data

A deep learning approach to imputing missing values in 15-minute interval smart
meter electricity consumption data, comparing **LSTM**, **GRU**, and **1D ResNet**
architectures under three missingness mechanisms (MCAR, MAR, MNAR) at four rates
(10%, 20%, 30%, 40%).

## Project Structure

```
smart_meter_imputation/
├── config.py                 # Central configuration (paths, hyperparameters)
├── data_loader.py            # Data loading, cleaning, normalization, windowing
├── missingness.py            # MCAR / MAR / MNAR missingness simulation
├── models/
│   ├── __init__.py
│   ├── lstm_imputer.py       # Bidirectional LSTM imputer
│   ├── gru_imputer.py        # Bidirectional GRU imputer
│   └── resnet_imputer.py     # 1D ResNet imputer (primary approach)
├── train.py                  # Unified training loop with masked MSE loss
├── evaluate.py               # Evaluation metrics (RMSE, MAE, MAPE) + plots
├── run_all.py                # Master script: all 36 experiments
├── utils.py                  # Metrics, plotting, reproducibility helpers
├── literature_review.md      # Phase 1 literature review
├── requirements.txt          # Python dependencies
└── outputs/
    ├── models/               # Saved model checkpoints (.pt)
    ├── plots/                # Imputation sample plots + comparison charts
    └── results/              # Results table (CSV)
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# If you have a CUDA GPU:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Quick Start

### Run all 36 experiments (full pipeline)

```bash
python run_all.py
```

This will:
1. Load and clean the smart meter dataset
2. Train LSTM, GRU, and ResNet models under all missingness conditions
3. Evaluate each on the test set
4. Generate results table and comparison plots

### Run a single experiment

```bash
# Train LSTM with MCAR missingness at 20%
python train.py --model LSTM --mechanism MCAR --rate 0.2

# Evaluate all trained models
python evaluate.py
```

### Customize experiments

```bash
# Only train ResNet with fewer epochs
python run_all.py --models ResNet --epochs 30

# Specific mechanisms and rates
python run_all.py --mechanisms MCAR MAR --rates 0.1 0.3
```

## Dataset

- **Source**: Indore Smart Meter data (November 2024)
- **Interval**: 15-minute readings (96 per day)
- **Target**: `kWh` (electricity consumption per interval)
- **Meters used**: ~100 complete meters (all 2,880 readings for November)
- **Split**: 70/15/15 by meter (train/val/test) — avoids temporal leakage

## Models

### LSTM Imputer (Baseline 1)
- 2-layer Bidirectional LSTM
- Input: [kWh, mask, hour_sin, hour_cos, dow_sin]
- Inspired by GRU-D (mask as input) and BRITS (bidirectional processing)

### GRU Imputer (Baseline 2)
- 2-layer Bidirectional GRU
- Same architecture as LSTM but with fewer parameters
- Direct comparison to evaluate LSTM vs GRU for this task

### 1D ResNet Imputer (Primary)
- Conv1D stem → 4 Residual Blocks → Conv1D projection
- Dilated convolutions (1, 2, 4, 1) for multi-scale temporal reception
- Skip connections prevent vanishing gradients in deep architectures
- Justified by DRes-CNN literature for energy data imputation

## Missingness Mechanisms

| Mechanism | Description |
|-----------|-------------|
| **MCAR** | Uniform random masking (each point has equal probability) |
| **MAR** | Time-dependent: higher probability during nighttime (22:00-06:00) |
| **MNAR** | Value-dependent: higher consumption → higher miss probability |

## Evaluation

- **RMSE** (Root Mean Squared Error) — overall reconstruction accuracy
- **MAE** (Mean Absolute Error) — robust to outliers
- **MAPE** (Mean Absolute Percentage Error) — on non-zero actuals only

All metrics are computed **only on the artificially masked positions** of the test set.

## Outputs

After running, check `outputs/`:
- `results/results_table.csv` — Full results table
- `plots/sample_*.png` — Per-experiment imputation visualizations
- `plots/heatmap_*.png` — Metric heatmaps across all conditions
- `plots/bars_*.png` — Grouped bar charts comparing models

## References

See `literature_review.md` for the full literature review covering GRU-D, BRITS,
M-RNN, SAITS, MIDA, and motivation for the ResNet approach.
