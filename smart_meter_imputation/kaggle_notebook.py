#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
==========================================================================
 Smart Meter Missing Value Imputation — Complete Kaggle Notebook
==========================================================================

 This single-file script implements the full pipeline:
   Phase 1: Literature review findings inform architecture choices
   Phase 2: LSTM and GRU baseline imputers
   Phase 3: 1D Dilated ResNet imputer (primary approach)

 Dataset : Indore Smart Meter data (Nov 2024, 15-min intervals)
 Models  : Bi-LSTM, Bi-GRU, 1D Dilated ResNet
 Missing : MCAR / MAR / MNAR @ 10%, 20%, 30%, 40%
 Metrics : RMSE, MAE, MAPE

 HOW TO RUN ON KAGGLE
 --------------------
 1. Upload "Nov_2024.csv" as a Kaggle Dataset.
 2. Create a new Kaggle Notebook with GPU accelerator enabled.
 3. Copy-paste this entire file into a single code cell (or multiple
    cells separated at the "# %%" markers).
 4. Update DATA_PATH below to match your Kaggle dataset path.
 5. Run All.

==========================================================================
"""

# %%  ===================================================================
#  CELL 1 — IMPORTS & CONFIGURATION
# ====================================================================

import os
import random
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU     : {torch.cuda.get_device_name(0)}")

# ── Configuration ──────────────────────────────────────────────────────
# >>> CHANGE THIS to your Kaggle dataset directory or full CSV path <<<
_KAGGLE_DATASET_DIR = "/kaggle/input/datasets/daranissikrithika/nov2024"

# Auto-detect: if user points to a directory, find the .csv file inside it
import glob as _glob
if os.path.isdir(_KAGGLE_DATASET_DIR):
    _csvs = _glob.glob(os.path.join(_KAGGLE_DATASET_DIR, "**", "*.csv"), recursive=True)
    if not _csvs:
        raise FileNotFoundError(f"No CSV files found in {_KAGGLE_DATASET_DIR}")
    DATA_PATH = _csvs[0]
    print(f"Auto-detected CSV: {DATA_PATH}")
    if len(_csvs) > 1:
        print(f"  (Found {len(_csvs)} CSVs, using the first one. Set DATA_PATH manually if wrong.)")
elif os.path.isfile(_KAGGLE_DATASET_DIR):
    DATA_PATH = _KAGGLE_DATASET_DIR
else:
    DATA_PATH = _KAGGLE_DATASET_DIR  # let pandas raise its own error

OUTPUT_DIR   = "/kaggle/working/outputs"
MODEL_DIR    = os.path.join(OUTPUT_DIR, "models")
PLOT_DIR     = os.path.join(OUTPUT_DIR, "plots")
RESULTS_DIR  = os.path.join(OUTPUT_DIR, "results")
for d in [OUTPUT_DIR, MODEL_DIR, PLOT_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# Data settings
TARGET_COLUMN     = "kWh"
TIMESTAMP_COLUMN  = "Date Time"
METER_ID_COLUMN   = "MSN"
PV_METER_IDS      = ["T0000576", "T0014122", "T0019208"]
EXPECTED_READINGS  = 2880       # 96 intervals/day × 30 days
INTERVAL_MINUTES   = 15
MAX_METERS         = 50         # number of meters for experiments

# Windowing
WINDOW_SIZE = 96                # 24 hours = 96 × 15-min
STRIDE      = 96                # non-overlapping daily windows

# Train / Val / Test
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# Missingness
MISSING_RATES      = [0.10, 0.20, 0.30, 0.40]
MISSING_MECHANISMS = ["MCAR", "MAR", "MNAR"]

# Model hyperparameters
INPUT_FEATURES      = 5
BATCH_SIZE          = 64
LEARNING_RATE       = 1e-3
NUM_EPOCHS          = 30
EARLY_STOP_PATIENCE = 6
WEIGHT_DECAY        = 1e-5

# LSTM / GRU
RNN_HIDDEN_SIZE = 128
RNN_NUM_LAYERS  = 2
RNN_DROPOUT     = 0.2

# ResNet
RESNET_CHANNELS    = [64, 128, 256, 128]
RESNET_KERNEL_SIZE = 3
RESNET_STEM_KERNEL = 7
RESNET_DILATIONS   = [1, 2, 4, 1]

RANDOM_SEED = 42

print(f"\nConfig  : {MAX_METERS} meters, {NUM_EPOCHS} epochs, window={WINDOW_SIZE}, stride={STRIDE}")
print(f"Rates   : {MISSING_RATES}")
print(f"Mechs   : {MISSING_MECHANISMS}")
print(f"Models  : LSTM, GRU, ResNet")


# ── Reproducibility ────────────────────────────────────────────────────
def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_seed()
DEVICE = get_device()
print(f"Device  : {DEVICE}")


# %%  ===================================================================
#  CELL 2 — DATA LOADING & PREPROCESSING
# ====================================================================

def load_and_clean(path=DATA_PATH):
    """Load CSV, drop PV meters & missing rows, select complete meters."""
    cache = os.path.join(OUTPUT_DIR, f"clean_{MAX_METERS}m.parquet")
    if os.path.exists(cache):
        print(f"Loading from cache: {cache}")
        df = pd.read_parquet(cache)
        print(f"  {len(df):,} rows, {df[METER_ID_COLUMN].nunique()} meters")
        return df

    print("Loading raw CSV …")
    df = pd.read_csv(path, usecols=[METER_ID_COLUMN, TIMESTAMP_COLUMN,
                                     TARGET_COLUMN, "kWh EXP"])
    print(f"  Raw rows: {len(df):,}")

    # Drop PV / solar export meters
    pv = (df["kWh EXP"] > 0) | (df[METER_ID_COLUMN].isin(PV_METER_IDS))
    df = df[~pv].drop(columns=["kWh EXP"]).reset_index(drop=True)
    print(f"  After PV drop: {len(df):,}")

    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN])
    df = df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)

    # Expected 15-min grid for November 2024
    grid = pd.date_range("2024-11-01", "2024-11-30 23:45:00",
                         freq=f"{INTERVAL_MINUTES}min")
    thresh = int(len(grid) * 0.97)
    print(f"  Expected readings: {len(grid)}, threshold (97%): {thresh}")

    counts = df.groupby(METER_ID_COLUMN)[TIMESTAMP_COLUMN].nunique()
    counts = counts.sort_values(ascending=False)
    complete = counts[counts >= thresh].index.tolist()
    print(f"  Near-complete meters: {len(complete)}")

    if len(complete) > MAX_METERS:
        rng = np.random.RandomState(RANDOM_SEED)
        complete = sorted(rng.choice(complete, MAX_METERS, replace=False))
    print(f"  Selected: {len(complete)}")

    df = df[df[METER_ID_COLUMN].isin(complete)].reset_index(drop=True)
    df = df.drop_duplicates(subset=[METER_ID_COLUMN, TIMESTAMP_COLUMN], keep="first")

    # Reindex each meter to the full grid
    dfs = []
    for msn in tqdm(complete, desc="Reindexing meters"):
        mdf = df[df[METER_ID_COLUMN] == msn].copy().set_index(TIMESTAMP_COLUMN)
        mdf = mdf.reindex(grid)
        mdf[TARGET_COLUMN] = mdf[TARGET_COLUMN].ffill().bfill()
        mdf[METER_ID_COLUMN] = msn
        mdf = mdf.reset_index().rename(columns={"index": TIMESTAMP_COLUMN})
        dfs.append(mdf)

    df = pd.concat(dfs, ignore_index=True)
    print(f"  Final: {len(df):,} rows, {df[METER_ID_COLUMN].nunique()} meters")
    df.to_parquet(cache, index=False)
    return df


def add_time_features(df):
    ts = df[TIMESTAMP_COLUMN]
    hour = ts.dt.hour + ts.dt.minute / 60.0
    dow  = ts.dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["dow_sin"]  = np.sin(2 * np.pi * dow  / 7.0)
    return df


def normalize_per_meter(df):
    scalers = {}
    df["kWh_norm"] = 0.0
    for msn in df[METER_ID_COLUMN].unique():
        mask = df[METER_ID_COLUMN] == msn
        vals = df.loc[mask, TARGET_COLUMN].values.reshape(-1, 1)
        sc = MinMaxScaler(feature_range=(0, 1)).fit(vals)
        scalers[msn] = sc
        df.loc[mask, "kWh_norm"] = sc.transform(vals).flatten()
    return df, scalers


def create_windows(df, window_size=WINDOW_SIZE, stride=STRIDE):
    windows = []
    for msn in df[METER_ID_COLUMN].unique():
        mdf = df[df[METER_ID_COLUMN] == msn].sort_values(TIMESTAMP_COLUMN)
        v = mdf["kWh_norm"].values
        hs = mdf["hour_sin"].values
        hc = mdf["hour_cos"].values
        ds = mdf["dow_sin"].values
        for s in range(0, len(v) - window_size + 1, stride):
            e = s + window_size
            windows.append({
                "kWh_norm": v[s:e].astype(np.float32),
                "hour_sin": hs[s:e].astype(np.float32),
                "hour_cos": hc[s:e].astype(np.float32),
                "dow_sin":  ds[s:e].astype(np.float32),
                "meter_id": msn,
            })
    return windows


def split_by_meter(df):
    meters = sorted(df[METER_ID_COLUMN].unique())
    rng = np.random.RandomState(RANDOM_SEED)
    rng.shuffle(meters)
    n = len(meters)
    nt = int(n * TRAIN_RATIO)
    nv = int(n * VAL_RATIO)
    train_m = meters[:nt]
    val_m   = meters[nt:nt+nv]
    test_m  = meters[nt+nv:]
    print(f"  Split: {len(train_m)} train / {len(val_m)} val / {len(test_m)} test meters")
    tw = create_windows(df[df[METER_ID_COLUMN].isin(train_m)])
    vw = create_windows(df[df[METER_ID_COLUMN].isin(val_m)])
    sw = create_windows(df[df[METER_ID_COLUMN].isin(test_m)])
    print(f"  Windows: {len(tw)} train / {len(vw)} val / {len(sw)} test")
    return tw, vw, sw


def prepare_data():
    df = load_and_clean()
    df = add_time_features(df)
    df, scalers = normalize_per_meter(df)
    tw, vw, sw = split_by_meter(df)
    return df, scalers, tw, vw, sw


# ── PyTorch Dataset ────────────────────────────────────────────────────
class ImputationDataset(Dataset):
    def __init__(self, windows, mask_fn, missing_rate):
        self.windows = windows
        self.mask_fn = mask_fn
        self.missing_rate = missing_rate

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        original = w["kWh_norm"].copy()
        mask = self.mask_fn(original, self.missing_rate)
        masked_kWh = original * mask
        features = np.stack([masked_kWh, mask,
                             w["hour_sin"], w["hour_cos"], w["dow_sin"]], axis=-1)
        return (torch.from_numpy(features).float(),
                torch.from_numpy(original).float(),
                torch.from_numpy(mask).float())


def create_dataloader(windows, mask_fn, missing_rate,
                      batch_size=BATCH_SIZE, shuffle=True):
    ds = ImputationDataset(windows, mask_fn, missing_rate)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


# ── Run data prep ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DATA PREPARATION")
print("=" * 60)
t0 = time.time()
df_full, scalers, train_windows, val_windows, test_windows = prepare_data()
print(f"Data ready in {time.time()-t0:.1f}s\n")


# %%  ===================================================================
#  CELL 3 — MISSINGNESS SIMULATION (MCAR / MAR / MNAR)
# ====================================================================

def simulate_mcar(values, rate):
    """Missing Completely At Random — uniform independent masking."""
    mask = np.ones_like(values, dtype=np.float32)
    n_mask = max(1, int(len(values) * rate))
    idx = np.random.choice(len(values), size=n_mask, replace=False)
    mask[idx] = 0.0
    return mask


def simulate_mar(values, rate):
    """Missing At Random — higher probability during night hours."""
    n = len(values)
    weights = np.ones(n, dtype=np.float32)
    if n == 96:
        night = list(range(0, 24)) + list(range(88, 96))
        weights[night] = 3.0
    else:
        q = n // 4
        weights[:q] = 3.0
        weights[-q:] = 3.0

    probs = weights / weights.sum() * (rate * n)
    probs = np.clip(probs, 0, 0.95)
    mask = (np.random.random(n) > probs).astype(np.float32)

    cur = (mask == 0).sum()
    tgt = int(rate * n)
    if cur < tgt:
        obs = np.where(mask == 1)[0]
        if len(obs) > 0:
            extra = min(tgt - cur, len(obs))
            w = weights[obs]; w = w / w.sum()
            mask[np.random.choice(obs, size=extra, replace=False, p=w)] = 0.0
    elif cur > tgt:
        mis = np.where(mask == 0)[0]
        if len(mis) > 0:
            rest = min(cur - tgt, len(mis))
            mask[np.random.choice(mis, size=rest, replace=False)] = 1.0
    return mask


def simulate_mnar(values, rate):
    """Missing Not At Random — higher values more likely to be missing."""
    n = len(values)
    av = np.abs(values)
    vr = av.max() - av.min()
    if vr < 1e-8:
        return simulate_mcar(values, rate)

    weights = (av - av.min()) / vr + 0.1
    probs = weights / weights.sum() * (rate * n)
    probs = np.clip(probs, 0, 0.95)
    mask = (np.random.random(n) > probs).astype(np.float32)

    cur = (mask == 0).sum()
    tgt = int(rate * n)
    if cur < tgt:
        obs = np.where(mask == 1)[0]
        if len(obs) > 0:
            extra = min(tgt - cur, len(obs))
            w = weights[obs]; w = w / w.sum()
            mask[np.random.choice(obs, size=extra, replace=False, p=w)] = 0.0
    elif cur > tgt:
        mis = np.where(mask == 0)[0]
        if len(mis) > 0:
            rest = min(cur - tgt, len(mis))
            mask[np.random.choice(mis, size=rest, replace=False)] = 1.0
    return mask


MASK_FUNCTIONS = {"MCAR": simulate_mcar, "MAR": simulate_mar, "MNAR": simulate_mnar}

def get_mask_fn(mechanism):
    if mechanism not in MASK_FUNCTIONS:
        raise ValueError(f"Unknown mechanism '{mechanism}'")
    return MASK_FUNCTIONS[mechanism]

# Quick sanity check
set_seed()
_tv = np.random.rand(96).astype(np.float32)
for _n, _fn in MASK_FUNCTIONS.items():
    for _r in [0.1, 0.2, 0.3, 0.4]:
        _m = _fn(_tv, _r)
        print(f"  {_n} @ {_r:.0%}: actual missing = {1-_m.mean():.2%}")
print()


# %%  ===================================================================
#  CELL 4 — MODEL DEFINITIONS
# ====================================================================

# ── Bidirectional LSTM Imputer (Baseline 1) ────────────────────────────
class LSTMImputer(nn.Module):
    """
    Bi-LSTM imputer.
    Design: Bidirectional processing (inspired by BRITS) with
    missingness mask as input feature (inspired by GRU-D).
    """
    def __init__(self, input_size=INPUT_FEATURES, hidden_size=RNN_HIDDEN_SIZE,
                 num_layers=RNN_NUM_LAYERS, dropout=RNN_DROPOUT):
        super().__init__()
        self.model_name = "LSTM"
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out.squeeze(-1)


# ── Bidirectional GRU Imputer (Baseline 2) ─────────────────────────────
class GRUImputer(nn.Module):
    """
    Bi-GRU imputer — mirrors LSTM but with fewer parameters.
    """
    def __init__(self, input_size=INPUT_FEATURES, hidden_size=RNN_HIDDEN_SIZE,
                 num_layers=RNN_NUM_LAYERS, dropout=RNN_DROPOUT):
        super().__init__()
        self.model_name = "GRU"
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,
                          bidirectional=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out)
        return out.squeeze(-1)


# ── 1D Residual Block ──────────────────────────────────────────────────
class ResBlock1D(nn.Module):
    """Residual block: two Conv1D layers + skip connection."""
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               padding=pad, dilation=dilation, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                               padding=pad, dilation=dilation, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.skip  = nn.Identity()
        if in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


# ── 1D Dilated ResNet Imputer (Primary) ────────────────────────────────
class ResNetImputer(nn.Module):
    """
    1D ResNet for time-series imputation.

    Architecture justification (Phase 1 literature review):
    - Residual connections solve vanishing gradients (He et al., 2016)
    - 1D convolutions capture local diurnal motifs efficiently
    - Dilated convolutions expand receptive field without extra parameters
    - Mask + time features as input channels (similar to GRU-D mask usage)

    Stem Conv1d(k=7) → 4 × ResBlock(dilated) → Conv1d(k=1) projection
    """
    def __init__(self, input_size=INPUT_FEATURES,
                 channels=None, kernel_size=RESNET_KERNEL_SIZE,
                 stem_kernel=RESNET_STEM_KERNEL, dilations=None):
        super().__init__()
        self.model_name = "ResNet"
        if channels is None:
            channels = RESNET_CHANNELS
        if dilations is None:
            dilations = RESNET_DILATIONS

        sp = (stem_kernel - 1) // 2
        self.stem = nn.Sequential(
            nn.Conv1d(input_size, channels[0], stem_kernel, padding=sp, bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
        )
        blocks = []
        in_ch = channels[0]
        for out_ch, dil in zip(channels, dilations):
            blocks.append(ResBlock1D(in_ch, out_ch, kernel_size, dil))
            in_ch = out_ch
        self.res_blocks = nn.Sequential(*blocks)
        self.projection = nn.Conv1d(channels[-1], 1, kernel_size=1)

    def forward(self, x):
        x = x.transpose(1, 2)       # (B, features, T)
        x = self.stem(x)
        x = self.res_blocks(x)
        x = self.projection(x)
        return x.squeeze(1)          # (B, T)


# ── Model factory ─────────────────────────────────────────────────────
MODEL_CLASSES = {"LSTM": LSTMImputer, "GRU": GRUImputer, "ResNet": ResNetImputer}

def create_model(name):
    return MODEL_CLASSES[name]()

# Print parameter counts
for n, cls in MODEL_CLASSES.items():
    m = cls()
    pc = sum(p.numel() for p in m.parameters())
    print(f"  {n:6s} parameters: {pc:>10,}")
print()


# %%  ===================================================================
#  CELL 5 — METRICS & PLOTTING UTILITIES
# ====================================================================

def compute_rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def compute_mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def compute_mape(y_true, y_pred, eps=1e-8):
    nz = np.abs(y_true) > eps
    if nz.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100)

def compute_all_metrics(y_true, y_pred):
    return {"RMSE": compute_rmse(y_true, y_pred),
            "MAE":  compute_mae(y_true, y_pred),
            "MAPE": compute_mape(y_true, y_pred)}


def plot_imputation_sample(original, masked, imputed, mask,
                           model_name, mechanism, rate, save_path=None):
    fig, ax = plt.subplots(figsize=(14, 5))
    t = np.arange(len(original))
    ax.plot(t, original, color="gray", alpha=0.5, lw=1, label="Original (true)")
    obs = mask == 1
    ax.scatter(t[obs], original[obs], color="blue", s=12, zorder=3, label="Observed")
    mis = mask == 0
    if mis.any():
        ax.scatter(t[mis], original[mis], color="green", marker="x", s=30, zorder=4, label="True (missing)")
        ax.scatter(t[mis], imputed[mis], color="red", marker="o", s=20, zorder=5, alpha=0.7, label="Imputed")
    ax.set_xlabel("Timestep (15-min intervals)")
    ax.set_ylabel("kWh (normalized)")
    ax.set_title(f"{model_name} Imputation — {mechanism} @ {rate*100:.0f}% missing")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_training_loss(train_losses, val_losses, label, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    ep = range(1, len(train_losses) + 1)
    ax.plot(ep, train_losses, label="Train", color="blue")
    ax.plot(ep, val_losses,   label="Val",   color="orange")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.set_title(f"{label} — Training Curves")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_overlay_comparison(original, preds_dict, mask,
                            mechanism, rate, save_path=None):
    """All models overlaid on the same ground-truth window."""
    fig, ax = plt.subplots(figsize=(14, 6))
    t = np.arange(len(original))
    ax.plot(t, original, color="black", lw=2.0, label="Ground Truth", zorder=3)
    miss_idx = np.where(mask == 0)[0]
    for idx in miss_idx:
        ax.axvspan(idx - 0.5, idx + 0.5, color="lightgray", alpha=0.35, zorder=1)
    colors = {"LSTM": "#1f77b4", "GRU": "#2ca02c", "ResNet": "#d62728"}
    lstyles = {"LSTM": "--", "GRU": "-.", "ResNet": "-"}
    for name, pred in preds_dict.items():
        ax.plot(t, pred, color=colors.get(name, "purple"),
                linestyle=lstyles.get(name, "-"),
                linewidth=2.2 if name == "ResNet" else 1.5,
                label=f"{name} Imputed", zorder=4)
    obs_idx = np.where(mask == 1)[0]
    ax.scatter(t[obs_idx], original[obs_idx], color="gray", s=10, alpha=0.5,
               zorder=2, label="Observed")
    ax.set_xlabel("Time Interval (15-min steps)", fontsize=11)
    ax.set_ylabel("Normalized kWh", fontsize=11)
    ax.set_title(f"Comparative Imputation ({mechanism} @ {int(rate*100)}%)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# %%  ===================================================================
#  CELL 6 — TRAINING LOOP
# ====================================================================

def masked_mse_loss(predictions, targets, masks):
    """MSE on missing positions (primary) + small aux on observed."""
    miss = 1.0 - masks
    mc = miss.sum()
    ml = ((predictions - targets)**2 * miss).sum() / mc if mc > 0 else torch.tensor(0.0, device=predictions.device)
    oc = masks.sum()
    ol = ((predictions - targets)**2 * masks).sum() / oc if oc > 0 else torch.tensor(0.0, device=predictions.device)
    return 0.8 * ml + 0.2 * ol


def train_model(model, train_wins, val_wins, mechanism, missing_rate,
                num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, patience=EARLY_STOP_PATIENCE):
    device = get_device()
    model = model.to(device)
    mask_fn = get_mask_fn(mechanism)

    train_loader = create_dataloader(train_wins, mask_fn, missing_rate, shuffle=True)
    val_loader   = create_dataloader(val_wins,   mask_fn, missing_rate, shuffle=False)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    mname = getattr(model, "model_name", "model")
    save_path = os.path.join(MODEL_DIR, f"{mname}_{mechanism}_{int(missing_rate*100)}.pt")

    train_losses, val_losses = [], []
    best_val, best_ep, pat_cnt = float("inf"), 0, 0
    t0 = time.time()

    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        ep_loss, nb = 0.0, 0
        for feat, tgt, msk in train_loader:
            feat, tgt, msk = feat.to(device), tgt.to(device), msk.to(device)
            optimizer.zero_grad()
            pred = model(feat)
            loss = masked_mse_loss(pred, tgt, msk)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ep_loss += loss.item(); nb += 1
        train_losses.append(ep_loss / max(nb, 1))

        # Validate
        model.eval()
        ep_vloss, nvb = 0.0, 0
        with torch.no_grad():
            for feat, tgt, msk in val_loader:
                feat, tgt, msk = feat.to(device), tgt.to(device), msk.to(device)
                pred = model(feat)
                loss = masked_mse_loss(pred, tgt, msk)
                ep_vloss += loss.item(); nvb += 1
        vl = ep_vloss / max(nvb, 1)
        val_losses.append(vl)
        scheduler.step(vl)

        if vl < best_val:
            best_val, best_ep, pat_cnt = vl, epoch, 0
            torch.save(model.state_dict(), save_path)
        else:
            pat_cnt += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"    Ep {epoch:3d}/{num_epochs} | Train {train_losses[-1]:.6f} | "
                  f"Val {vl:.6f} | Best {best_val:.6f} (ep {best_ep})")
        if pat_cnt >= patience:
            print(f"    Early stop at epoch {epoch}")
            break

    elapsed = time.time() - t0
    model.load_state_dict(torch.load(save_path, weights_only=True))
    return {"train_losses": train_losses, "val_losses": val_losses,
            "best_epoch": best_ep, "model_path": save_path,
            "training_time": elapsed}


# %%  ===================================================================
#  CELL 7 — EVALUATION
# ====================================================================

def evaluate_model(model, test_wins, mechanism, missing_rate):
    device = get_device()
    model = model.to(device); model.eval()
    mask_fn = get_mask_fn(mechanism)
    loader = create_dataloader(test_wins, mask_fn, missing_rate,
                               batch_size=BATCH_SIZE, shuffle=False)
    all_true, all_pred = [], []
    with torch.no_grad():
        for feat, tgt, msk in loader:
            feat = feat.to(device)
            pred = model(feat).cpu().numpy()
            tgt  = tgt.numpy(); msk = msk.numpy()
            mm = (msk == 0)
            for i in range(len(pred)):
                if mm[i].any():
                    all_true.append(tgt[i][mm[i]])
                    all_pred.append(pred[i][mm[i]])
    if not all_true:
        return {"RMSE": 0.0, "MAE": 0.0, "MAPE": 0.0}
    yt = np.concatenate(all_true)
    yp = np.clip(np.concatenate(all_pred), 0.0, 1.0)
    return compute_all_metrics(yt, yp)


def evaluate_and_plot_sample(model, test_wins, mechanism, rate, model_name):
    metrics = evaluate_model(model, test_wins, mechanism, rate)
    mask_fn = get_mask_fn(mechanism)
    w = test_wins[0]
    orig = w["kWh_norm"]
    mask = mask_fn(orig, rate)
    masked_v = orig * mask
    feat = np.stack([masked_v, mask, w["hour_sin"], w["hour_cos"], w["dow_sin"]],
                    axis=-1).astype(np.float32)
    device = get_device(); model.eval()
    with torch.no_grad():
        inp = torch.from_numpy(feat).unsqueeze(0).to(device)
        pred = model(inp).cpu().numpy().squeeze()
    pred = np.clip(pred, 0.0, 1.0)
    sp = os.path.join(PLOT_DIR, f"sample_{model_name}_{mechanism}_{int(rate*100)}.png")
    plot_imputation_sample(orig, masked_v, pred, mask, model_name, mechanism, rate, sp)
    return metrics


# %%  ===================================================================
#  CELL 8 — RUN ALL 36 EXPERIMENTS
# ====================================================================

print("\n" + "=" * 60)
print("TRAINING ALL MODELS (3 models × 3 mechanisms × 4 rates = 36)")
print("=" * 60)

set_seed()
total_start = time.time()
experiment_count = 0
total_experiments = len(MODEL_CLASSES) * len(MISSING_MECHANISMS) * len(MISSING_RATES)

for model_name in ["LSTM", "GRU", "ResNet"]:
    for mechanism in MISSING_MECHANISMS:
        for rate in MISSING_RATES:
            experiment_count += 1
            print(f"\n--- Experiment {experiment_count}/{total_experiments} ---")
            print(f"  Model: {model_name} | {mechanism} @ {rate*100:.0f}%")

            model = create_model(model_name)
            pc = sum(p.numel() for p in model.parameters())
            print(f"  Params: {pc:,}")

            result = train_model(model, train_windows, val_windows,
                                 mechanism, rate)
            print(f"  Done: best_ep={result['best_epoch']}, "
                  f"time={result['training_time']:.1f}s")

train_time = time.time() - total_start
print(f"\nTotal training time: {train_time/60:.1f} minutes")


# %%  ===================================================================
#  CELL 9 — EVALUATION & RESULTS TABLE
# ====================================================================

print("\n" + "=" * 60)
print("EVALUATION ON TEST SET")
print("=" * 60)

device = get_device()
results = []

for model_name in ["LSTM", "GRU", "ResNet"]:
    for mechanism in MISSING_MECHANISMS:
        for rate in MISSING_RATES:
            print(f"\nEval {model_name} | {mechanism} @ {rate*100:.0f}% …")
            mp = os.path.join(MODEL_DIR, f"{model_name}_{mechanism}_{int(rate*100)}.pt")
            if not os.path.exists(mp):
                print(f"  SKIP — model not found")
                continue
            model = create_model(model_name)
            model.load_state_dict(torch.load(mp, map_location=device, weights_only=True))
            metrics = evaluate_and_plot_sample(model, test_windows,
                                               mechanism, rate, model_name)
            results.append({
                "Model": model_name,
                "Mechanism": mechanism,
                "Missing Rate": f"{rate*100:.0f}%",
                "RMSE": round(metrics["RMSE"], 6),
                "MAE":  round(metrics["MAE"],  6),
                "MAPE (%)": round(metrics["MAPE"], 2),
            })
            print(f"  RMSE={metrics['RMSE']:.6f} | MAE={metrics['MAE']:.6f} | "
                  f"MAPE={metrics['MAPE']:.2f}%")

results_df = pd.DataFrame(results)

# Save CSV
csv_path = os.path.join(RESULTS_DIR, "results_table.csv")
results_df.to_csv(csv_path, index=False)
print(f"\nResults saved to {csv_path}")

# Print formatted table
print("\n" + "=" * 80)
print("COMPLETE RESULTS TABLE")
print("=" * 80)
print(results_df.to_string(index=False))
print("=" * 80)


# %%  ===================================================================
#  CELL 10 — COMPARISON VISUALIZATIONS
# ====================================================================

print("\n" + "=" * 60)
print("GENERATING COMPARISON PLOTS")
print("=" * 60)

# ── Heatmaps per metric ───────────────────────────────────────────────
for metric in ["RMSE", "MAE", "MAPE (%)"]:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle(f"{metric} Comparison Across Models", fontsize=14, fontweight="bold")
    for ax, mn in zip(axes, ["LSTM", "GRU", "ResNet"]):
        md = results_df[results_df["Model"] == mn]
        if md.empty:
            continue
        piv = md.pivot_table(values=metric, index="Mechanism",
                             columns="Missing Rate", aggfunc="first")
        ro = ["10%", "20%", "30%", "40%"]
        piv = piv.reindex(columns=[c for c in ro if c in piv.columns])
        sns.heatmap(piv, annot=True,
                    fmt=".4f" if metric != "MAPE (%)" else ".2f",
                    cmap="YlOrRd", ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title(mn, fontsize=12)
        ax.set_ylabel("" if ax != axes[0] else "Mechanism")
        ax.set_xlabel("Missing Rate")
    plt.tight_layout()
    safe = metric.replace(" ", "_").replace("(%)", "pct")
    fig.savefig(os.path.join(PLOT_DIR, f"heatmap_{safe}.png"), dpi=150, bbox_inches="tight")
    plt.show(); plt.close(fig)
    print(f"  Heatmap saved: heatmap_{safe}.png")


# ── Grouped bar charts per mechanism ──────────────────────────────────
for mechanism in MISSING_MECHANISMS:
    mech_data = results_df[results_df["Mechanism"] == mechanism]
    if mech_data.empty:
        continue
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Model Comparison — {mechanism} Missingness",
                 fontsize=14, fontweight="bold")
    for ax, metric in zip(axes, ["RMSE", "MAE", "MAPE (%)"]):
        piv = mech_data.pivot_table(values=metric, index="Missing Rate",
                                     columns="Model", aggfunc="first")
        ro = ["10%", "20%", "30%", "40%"]
        piv = piv.reindex([r for r in ro if r in piv.index])
        piv.plot(kind="bar", ax=ax, rot=0, color=["#2196F3", "#4CAF50", "#F44336"])
        ax.set_title(metric); ax.set_xlabel("Missing Rate")
        ax.set_ylabel(metric); ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, f"bars_{mechanism}.png"), dpi=150, bbox_inches="tight")
    plt.show(); plt.close(fig)
    print(f"  Bar chart saved: bars_{mechanism}.png")


# ── Overlay comparison plots ──────────────────────────────────────────
scenarios = [("MCAR", 0.20), ("MAR", 0.20), ("MNAR", 0.30)]
for mechanism, rate in scenarios:
    models = {}
    for mn in ["LSTM", "GRU", "ResNet"]:
        mp = os.path.join(MODEL_DIR, f"{mn}_{mechanism}_{int(rate*100)}.pt")
        if not os.path.exists(mp):
            continue
        m = create_model(mn)
        m.load_state_dict(torch.load(mp, map_location=device, weights_only=True))
        m.to(device); m.eval()
        models[mn] = m

    if len(models) < 2:
        continue

    w = test_windows[min(len(test_windows)-1, 5)]
    orig = w["kWh_norm"]
    mask_fn = get_mask_fn(mechanism)
    mask = mask_fn(orig, rate)
    masked_v = orig * mask
    feat = np.stack([masked_v, mask, w["hour_sin"], w["hour_cos"], w["dow_sin"]],
                    axis=-1).astype(np.float32)
    preds = {}
    with torch.no_grad():
        inp = torch.from_numpy(feat).unsqueeze(0).to(device)
        for nm, md in models.items():
            p = md(inp).cpu().numpy().squeeze()
            preds[nm] = np.clip(p, 0.0, 1.0)

    sp = os.path.join(PLOT_DIR, f"overlay_{mechanism}_{int(rate*100)}.png")
    plot_overlay_comparison(orig, preds, mask, mechanism, rate, sp)

print(f"\nAll plots saved to: {PLOT_DIR}")


# %%  ===================================================================
#  CELL 11 — FINAL SUMMARY
# ====================================================================

total_elapsed = time.time() - total_start

print("\n" + "=" * 60)
print("ALL DONE!")
print("=" * 60)
print(f"Total wall-clock time : {total_elapsed/60:.1f} minutes")
print(f"Models saved          : {MODEL_DIR}")
print(f"Plots saved           : {PLOT_DIR}")
print(f"Results CSV           : {csv_path}")
print(f"Experiments completed : {experiment_count}")
print()

# Final summary: best model per mechanism
print("BEST MODEL PER MECHANISM (by RMSE):")
print("-" * 50)
for mech in MISSING_MECHANISMS:
    sub = results_df[results_df["Mechanism"] == mech]
    if sub.empty:
        continue
    best = sub.loc[sub["RMSE"].idxmin()]
    print(f"  {mech:5s}: {best['Model']:6s} @ {best['Missing Rate']} "
          f"→ RMSE={best['RMSE']:.6f}, MAE={best['MAE']:.6f}, "
          f"MAPE={best['MAPE (%)']:.2f}%")
print()

# Display final table nicely
from IPython.display import display
try:
    display(results_df.style.highlight_min(
        subset=["RMSE", "MAE", "MAPE (%)"], color="lightgreen", axis=0
    ))
except Exception:
    print(results_df.to_string(index=False))
