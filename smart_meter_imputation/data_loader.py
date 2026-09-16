"""
Data loading, cleaning, normalization, windowing, and dataset creation.
Handles the full pipeline from raw CSV to PyTorch DataLoader.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

import config


# ──────────────────────────────────────────────
# Loading and Cleaning
# ──────────────────────────────────────────────
def load_and_clean(path: str = config.DATA_PATH) -> pd.DataFrame:
    """
    Load the smart meter CSV data and perform cleaning:
    1. Drop PV meter rows (meters with kWh EXP > 0)
    2. Parse timestamps
    3. Keep only meters with complete 15-min time grids (2880 readings for Nov)
    4. Select up to MAX_METERS complete meters
    Caches cleaned result to parquet for fast subsequent loads.
    """
    cache_path = os.path.join(config.OUTPUT_DIR, f"clean_smart_meters_{config.MAX_METERS}m.parquet")
    if os.path.exists(cache_path):
        print(f"Loading cleaned dataset from cache: {cache_path}")
        df = pd.read_parquet(cache_path)
        print(f"  Loaded {len(df):,} rows, {df[config.METER_ID_COLUMN].nunique()} meters from cache")
        return df

    print("Loading dataset from raw CSV...")
    df = pd.read_csv(
        path,
        usecols=[
            config.METER_ID_COLUMN, config.TIMESTAMP_COLUMN,
            config.TARGET_COLUMN, "kWh EXP",
        ],
    )
    print(f"  Raw rows: {len(df):,}")

    # Step 1: Drop PV meter rows
    pv_mask = (df["kWh EXP"] > 0) | (df[config.METER_ID_COLUMN].isin(config.PV_METER_IDS))
    df = df[~pv_mask].drop(columns=["kWh EXP"]).reset_index(drop=True)
    print(f"  After dropping PV meters: {len(df):,}")

    # Step 2: Parse timestamps
    df[config.TIMESTAMP_COLUMN] = pd.to_datetime(df[config.TIMESTAMP_COLUMN])

    # Step 3: Drop rows with NaN in kWh
    df = df.dropna(subset=[config.TARGET_COLUMN]).reset_index(drop=True)

    # Step 4: Keep near-complete meters (≥97% of expected readings)
    # No meters have all 2880 readings; max observed is ~2832. Use 97% threshold.
    expected_grid = pd.date_range(
        start="2024-11-01 00:00:00",
        end="2024-11-30 23:45:00",
        freq=f"{config.INTERVAL_MINUTES}min",
    )
    expected_count = len(expected_grid)  # 2880
    completeness_threshold = int(expected_count * 0.97)  # ~2794
    print(f"  Expected readings per meter: {expected_count}")
    print(f"  Completeness threshold (97%): {completeness_threshold}")

    meter_counts = df.groupby(config.METER_ID_COLUMN)[config.TIMESTAMP_COLUMN].nunique()
    # Sort by most complete first
    meter_counts = meter_counts.sort_values(ascending=False)
    complete_meters = meter_counts[meter_counts >= completeness_threshold].index.tolist()
    print(f"  Near-complete meters (>= {completeness_threshold} unique timestamps): {len(complete_meters)}")

    # Select up to MAX_METERS
    if len(complete_meters) > config.MAX_METERS:
        rng = np.random.RandomState(config.RANDOM_SEED)
        complete_meters = sorted(rng.choice(complete_meters, config.MAX_METERS, replace=False))
    print(f"  Selected meters: {len(complete_meters)}")

    df = df[df[config.METER_ID_COLUMN].isin(complete_meters)].reset_index(drop=True)

    # For each meter, keep only one reading per expected timestamp (dedup)
    df = df.drop_duplicates(
        subset=[config.METER_ID_COLUMN, config.TIMESTAMP_COLUMN], keep="first"
    )

    # Reindex each meter to the full grid, keeping only timestamps in the expected grid
    dfs = []
    for msn in tqdm(complete_meters, desc="Reindexing meters"):
        meter_df = df[df[config.METER_ID_COLUMN] == msn].copy()
        meter_df = meter_df.set_index(config.TIMESTAMP_COLUMN)
        # Reindex to the expected grid; any missing timestamps get NaN then forward-fill
        meter_df = meter_df.reindex(expected_grid)
        meter_df[config.TARGET_COLUMN] = meter_df[config.TARGET_COLUMN].ffill().bfill()
        meter_df[config.METER_ID_COLUMN] = msn
        meter_df = meter_df.reset_index().rename(columns={"index": config.TIMESTAMP_COLUMN})
        dfs.append(meter_df)

    df = pd.concat(dfs, ignore_index=True)
    print(f"  Final dataset: {len(df):,} rows, {df[config.METER_ID_COLUMN].nunique()} meters")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"  Cached cleaned dataset to {cache_path}")
    return df


# ──────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical time features: hour_sin, hour_cos, dow_sin, dow_cos."""
    ts = df[config.TIMESTAMP_COLUMN]
    hour = ts.dt.hour + ts.dt.minute / 60.0  # fractional hour
    dow = ts.dt.dayofweek  # 0=Mon, 6=Sun

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    return df


# ──────────────────────────────────────────────
# Normalization
# ──────────────────────────────────────────────
def normalize_per_meter(
    df: pd.DataFrame, fit_meters: list | None = None
) -> tuple[pd.DataFrame, dict]:
    """
    Apply MinMax scaling to kWh per meter.
    Returns the dataframe with a 'kWh_norm' column and a dict of scalers keyed by MSN.
    """
    scalers = {}
    df["kWh_norm"] = 0.0

    meters = df[config.METER_ID_COLUMN].unique()
    for msn in meters:
        mask = df[config.METER_ID_COLUMN] == msn
        values = df.loc[mask, config.TARGET_COLUMN].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))

        if fit_meters is None or msn in fit_meters:
            scaler.fit(values)
        scalers[msn] = scaler
        df.loc[mask, "kWh_norm"] = scaler.transform(values).flatten()

    return df, scalers


def inverse_normalize(values: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """Inverse transform normalized values back to original scale."""
    return scaler.inverse_transform(values.reshape(-1, 1)).flatten()


# ──────────────────────────────────────────────
# Windowing
# ──────────────────────────────────────────────
def create_windows(
    df: pd.DataFrame,
    window_size: int = config.WINDOW_SIZE,
    stride: int = config.STRIDE,
) -> list[dict]:
    """
    Slice each meter's time series into overlapping windows.

    Returns a list of dicts, each containing:
        - 'kWh_norm': np.ndarray of shape (window_size,)
        - 'hour_sin': np.ndarray of shape (window_size,)
        - 'hour_cos': np.ndarray of shape (window_size,)
        - 'dow_sin': np.ndarray of shape (window_size,)
        - 'meter_id': str
    """
    windows = []
    for msn in df[config.METER_ID_COLUMN].unique():
        meter_df = df[df[config.METER_ID_COLUMN] == msn].sort_values(config.TIMESTAMP_COLUMN)
        values = meter_df["kWh_norm"].values
        h_sin = meter_df["hour_sin"].values
        h_cos = meter_df["hour_cos"].values
        d_sin = meter_df["dow_sin"].values

        for start in range(0, len(values) - window_size + 1, stride):
            end = start + window_size
            windows.append({
                "kWh_norm": values[start:end].astype(np.float32),
                "hour_sin": h_sin[start:end].astype(np.float32),
                "hour_cos": h_cos[start:end].astype(np.float32),
                "dow_sin": d_sin[start:end].astype(np.float32),
                "meter_id": msn,
            })
    return windows


# ──────────────────────────────────────────────
# Train / Val / Test Split (by meter)
# ──────────────────────────────────────────────
def split_by_meter(
    df: pd.DataFrame,
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio: float = config.VAL_RATIO,
    seed: int = config.RANDOM_SEED,
) -> tuple[list, list, list]:
    """
    Split meters into train/val/test groups, then window each group.
    Returns (train_windows, val_windows, test_windows).
    """
    meters = sorted(df[config.METER_ID_COLUMN].unique())
    rng = np.random.RandomState(seed)
    rng.shuffle(meters)

    n = len(meters)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_meters = meters[:n_train]
    val_meters = meters[n_train : n_train + n_val]
    test_meters = meters[n_train + n_val :]

    print(f"  Split: {len(train_meters)} train / {len(val_meters)} val / {len(test_meters)} test meters")

    train_windows = create_windows(df[df[config.METER_ID_COLUMN].isin(train_meters)])
    val_windows = create_windows(df[df[config.METER_ID_COLUMN].isin(val_meters)])
    test_windows = create_windows(df[df[config.METER_ID_COLUMN].isin(test_meters)])

    print(f"  Windows: {len(train_windows)} train / {len(val_windows)} val / {len(test_windows)} test")
    return train_windows, val_windows, test_windows


# ──────────────────────────────────────────────
# PyTorch Dataset
# ──────────────────────────────────────────────
class ImputationDataset(Dataset):
    """
    PyTorch Dataset that applies missingness on-the-fly and returns
    (input_features, target, mask) tuples.

    input_features: (window_size, 5) — [masked_kWh, mask, hour_sin, hour_cos, dow_sin]
    target: (window_size,) — original kWh_norm values
    mask: (window_size,) — 1=observed, 0=missing
    """

    def __init__(self, windows: list, mask_fn, missing_rate: float):
        self.windows = windows
        self.mask_fn = mask_fn
        self.missing_rate = missing_rate

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        original = w["kWh_norm"].copy()  # (window_size,)

        # Generate missingness mask
        mask = self.mask_fn(original, self.missing_rate)  # 1=observed, 0=missing

        # Masked input: set missing positions to 0
        masked_kWh = original * mask

        # Build input feature tensor: (window_size, 5)
        features = np.stack([
            masked_kWh,
            mask,
            w["hour_sin"],
            w["hour_cos"],
            w["dow_sin"],
        ], axis=-1)  # (window_size, 5)

        return (
            torch.from_numpy(features).float(),       # (window_size, 5)
            torch.from_numpy(original).float(),        # (window_size,)
            torch.from_numpy(mask).float(),            # (window_size,)
        )


def create_dataloader(
    windows: list,
    mask_fn,
    missing_rate: float,
    batch_size: int = config.BATCH_SIZE,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader from windows with a given missingness function and rate."""
    dataset = ImputationDataset(windows, mask_fn, missing_rate)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


# ──────────────────────────────────────────────
# Main data preparation pipeline
# ──────────────────────────────────────────────
def prepare_data() -> tuple[pd.DataFrame, dict, list, list, list]:
    """
    Full data preparation pipeline.
    Returns: (df, scalers, train_windows, val_windows, test_windows)
    """
    df = load_and_clean()
    df = add_time_features(df)
    df, scalers = normalize_per_meter(df)
    train_windows, val_windows, test_windows = split_by_meter(df)
    return df, scalers, train_windows, val_windows, test_windows


if __name__ == "__main__":
    df, scalers, train_w, val_w, test_w = prepare_data()
    print(f"\nReady: {len(train_w)} train, {len(val_w)} val, {len(test_w)} test windows")
