"""
Missingness simulation: MCAR, MAR, MNAR.

Each function takes a 1D array of values and a missing rate, returning
a binary mask where 1 = observed, 0 = missing.
"""

import numpy as np


def simulate_mcar(values: np.ndarray, rate: float) -> np.ndarray:
    """
    Missing Completely At Random.
    Each timestep has an independent probability `rate` of being masked.

    Args:
        values: 1D array of shape (window_size,)
        rate: fraction of values to mask (0.0 to 1.0)

    Returns:
        mask: 1D binary array (1=observed, 0=missing)
    """
    mask = np.ones_like(values, dtype=np.float32)
    n_mask = max(1, int(len(values) * rate))
    idx = np.random.choice(len(values), size=n_mask, replace=False)
    mask[idx] = 0.0
    return mask


def simulate_mar(values: np.ndarray, rate: float) -> np.ndarray:
    """
    Missing At Random.
    Missingness probability depends on the position within the day (proxy for
    time-of-day). Higher miss probability during "night" hours (positions 0-23
    and 88-95 in a 96-step window correspond to 00:00-05:45 and 22:00-23:45).

    This simulates real-world communication outages that are more likely
    during nighttime.

    Args:
        values: 1D array of shape (window_size,)
        rate: overall target fraction of missing values

    Returns:
        mask: 1D binary array (1=observed, 0=missing)
    """
    n = len(values)
    # Create position-dependent probability weights
    # Night positions (first 24 and last 8 of a 96-step window) get higher weight
    weights = np.ones(n, dtype=np.float32)

    if n == 96:
        # 00:00-05:45 → positions 0-23, 22:00-23:45 → positions 88-95
        night_positions = list(range(0, 24)) + list(range(88, 96))
        weights[night_positions] = 3.0  # 3x more likely to be missing at night
    else:
        # Generic: first and last quarter get higher weight
        q = n // 4
        weights[:q] = 3.0
        weights[-q:] = 3.0

    # Normalize weights to probabilities that achieve the target rate
    probs = weights / weights.sum() * (rate * n)
    probs = np.clip(probs, 0, 0.95)  # cap individual probabilities

    mask = (np.random.random(n) > probs).astype(np.float32)

    # Adjust to hit the target rate approximately
    current_missing = (mask == 0).sum()
    target_missing = int(rate * n)

    if current_missing < target_missing:
        # Need to mask more — pick from observed positions preferring high-weight
        observed = np.where(mask == 1)[0]
        if len(observed) > 0:
            extra = min(target_missing - current_missing, len(observed))
            w = weights[observed]
            w = w / w.sum()
            extra_idx = np.random.choice(observed, size=extra, replace=False, p=w)
            mask[extra_idx] = 0.0
    elif current_missing > target_missing:
        # Need to unmask some
        missing = np.where(mask == 0)[0]
        if len(missing) > 0:
            restore = min(current_missing - target_missing, len(missing))
            restore_idx = np.random.choice(missing, size=restore, replace=False)
            mask[restore_idx] = 1.0

    return mask


def simulate_mnar(values: np.ndarray, rate: float) -> np.ndarray:
    """
    Missing Not At Random.
    Missingness probability is proportional to the value itself.
    Higher consumption readings are more likely to be missing — simulating
    meter overload / saturation scenarios.

    Args:
        values: 1D array of shape (window_size,)
        rate: overall target fraction of missing values

    Returns:
        mask: 1D binary array (1=observed, 0=missing)
    """
    n = len(values)

    # Weight by value magnitude (higher values → higher miss probability)
    abs_vals = np.abs(values)
    val_range = abs_vals.max() - abs_vals.min()

    if val_range < 1e-8:
        # All values are the same → fall back to MCAR
        return simulate_mcar(values, rate)

    # Normalize values to [0, 1] and use as relative weights
    weights = (abs_vals - abs_vals.min()) / val_range
    weights = weights + 0.1  # small baseline so zero-consumption can still be masked

    # Normalize to target rate
    probs = weights / weights.sum() * (rate * n)
    probs = np.clip(probs, 0, 0.95)

    mask = (np.random.random(n) > probs).astype(np.float32)

    # Adjust to hit target
    current_missing = (mask == 0).sum()
    target_missing = int(rate * n)

    if current_missing < target_missing:
        observed = np.where(mask == 1)[0]
        if len(observed) > 0:
            extra = min(target_missing - current_missing, len(observed))
            w = weights[observed]
            w = w / w.sum()
            extra_idx = np.random.choice(observed, size=extra, replace=False, p=w)
            mask[extra_idx] = 0.0
    elif current_missing > target_missing:
        missing = np.where(mask == 0)[0]
        if len(missing) > 0:
            restore = min(current_missing - target_missing, len(missing))
            restore_idx = np.random.choice(missing, size=restore, replace=False)
            mask[restore_idx] = 1.0

    return mask


# ──────────────────────────────────────────────
# Registry for easy lookup
# ──────────────────────────────────────────────
MASK_FUNCTIONS = {
    "MCAR": simulate_mcar,
    "MAR": simulate_mar,
    "MNAR": simulate_mnar,
}


def get_mask_fn(mechanism: str):
    """Return the mask function for the given mechanism name."""
    if mechanism not in MASK_FUNCTIONS:
        raise ValueError(f"Unknown mechanism '{mechanism}'. Choose from {list(MASK_FUNCTIONS.keys())}")
    return MASK_FUNCTIONS[mechanism]


if __name__ == "__main__":
    # Quick sanity check
    np.random.seed(42)
    test_vals = np.random.rand(96).astype(np.float32)

    for name, fn in MASK_FUNCTIONS.items():
        for rate in [0.1, 0.2, 0.3, 0.4]:
            mask = fn(test_vals, rate)
            actual_rate = 1.0 - mask.mean()
            print(f"  {name} @ {rate:.0%}: actual missing = {actual_rate:.2%}")
