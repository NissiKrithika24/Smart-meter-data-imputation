"""
Central configuration for the Smart Meter Missing Value Imputation project.
All hyperparameters, paths, and experimental settings in one place.
"""

import os

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
DATA_PATH = r"C:\Users\ratna\OneDrive - IIT Indore\Attachments\Indore SM missing data\Nov_2024.csv"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

# Create output directories
for d in [OUTPUT_DIR, MODEL_DIR, PLOT_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ──────────────────────────────────────────────
# Data settings
# ──────────────────────────────────────────────
TARGET_COLUMN = "kWh"
TIMESTAMP_COLUMN = "Date Time"
METER_ID_COLUMN = "MSN"

# PV meters to exclude (identified from dataset analysis)
PV_METER_IDS = ["T0000576", "T0014122", "T0019208"]

# Expected readings per meter for November 2024 (96 per day × 30 days)
EXPECTED_READINGS_PER_METER = 2880
INTERVAL_MINUTES = 15  # 15-minute intervals

# Number of complete meters to select for experiments (keeps training feasible)
MAX_METERS = 50

# ──────────────────────────────────────────────
# Windowing
# ──────────────────────────────────────────────
WINDOW_SIZE = 96  # 24 hours = 96 × 15-min intervals (one full diurnal cycle)
STRIDE = 96       # Non-overlapping full 24h diurnal days

# ──────────────────────────────────────────────
# Train / Val / Test split
# ──────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ──────────────────────────────────────────────
# Missingness experiment settings
# ──────────────────────────────────────────────
MISSING_RATES = [0.10, 0.20, 0.30, 0.40]
MISSING_MECHANISMS = ["MCAR", "MAR", "MNAR"]

# ──────────────────────────────────────────────
# Model hyperparameters
# ──────────────────────────────────────────────
# Common
INPUT_FEATURES = 5   # kWh_value + mask + hour_sin + hour_cos + dow_sin (see data_loader)
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
EARLY_STOP_PATIENCE = 5
WEIGHT_DECAY = 1e-5

# LSTM / GRU
RNN_HIDDEN_SIZE = 128
RNN_NUM_LAYERS = 2
RNN_DROPOUT = 0.2

# ResNet
RESNET_CHANNELS = [64, 128, 256, 128]
RESNET_KERNEL_SIZE = 3
RESNET_STEM_KERNEL = 7
RESNET_DILATIONS = [1, 2, 4, 1]

# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
RANDOM_SEED = 42
