"""
data_loader.py
--------------
FR arbitrage ML pipeline — data loading, feature engineering, splitting.

Feature set (9 total, FR-only, zero leakage):
    CORE (6):
        fr_abs, fr_roll_mean_M, fr_momentum, fr_roll_std_M,
        fr_decay_rate, n_consecutive_above
    EXTENDED (3, added if core model generalizes):
        fr_sign, cs_rank, fr_zscore_M

All historical features use strict shift(1) before any rolling operation.
n_consecutive_above uses shift(1) before streak counting.
fr_decay_rate uses epsilon + clip to prevent division instability.

Label horizon: 4 settlements (~32h, aligned with observed avg hold 3.9)
Splits: Train 2022-2023 | Val 2024 | Test 2025 (locked)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ── Constants ────────────────────────────────────────────────────────────────

SETTLEMENTS_PER_DAY: int = 3

# Rolling windows (settlements)
SHORT_WINDOW: int = 3    # 1 day
MED_WINDOW:   int = 21   # 7 days

# FR thresholds (percent units — data is stored as pct internally)
ENTRY_THRESHOLD_PCT: float = 0.05   # |FR| >= 0.05% → rule-based entry
EXIT_THRESHOLD_PCT:  float = 0.02   # |FR| < 0.02%  → rule-based exit

# fr_decay_rate stability
EPSILON_FR: float = 0.005           # below this mean_M → treat as flat regime
DECAY_CLIP: float = 5.0             # clip decay_rate to [-5, 5]

# Label horizon aligned with avg hold duration 3.9 settlements
LABEL_HORIZON: int = 4

# Temporal split boundaries
TRAIN_END: str = "2023-12-31"
VAL_END:   str = "2024-12-31"
# Test = 2025+ — touched ONCE at final evaluation only

# Purge + embargo (settlements)
PURGE_SETTLEMENTS:   int = 4   # match label horizon
EMBARGO_SETTLEMENTS: int = 6   # ~2 days buffer between splits

# Feature column lists
FEATURE_COLS_CORE: list[str] = [
    "fr_abs",
    "fr_roll_mean_M",
    "fr_momentum",
    "fr_roll_std_M",
    "fr_decay_rate",
    "n_consecutive_above",
]

FEATURE_COLS_EXTENDED: list[str] = FEATURE_COLS_CORE + [
    "fr_sign",
    "cs_rank",
    "fr_zscore_M",
]

# Default: start with extended, ablate to core if overfitting detected
FEATURE_COLS: list[str] = FEATURE_COLS_EXTENDED


# ── Feature Engineering ───────────────────────────────────────────────────────

def _shift_roll(series: pd.Series, window: int, func: str) -> pd.Series:
    """shift(1) first, then rolling — strict no-leakage contract."""
    return getattr(
        series.shift(1).rolling(window=window, min_periods=max(1, window // 2)),
        func
    )()


def _consecutive_above(fr_abs: pd.Series, threshold: float) -> pd.Series:
    """
    Count consecutive settlements where |FR| >= threshold, looking BACKWARD.
    shift(1) applied before streak counting — current settlement excluded.

    Implementation:
        1. shift(1): only look at t-1 and earlier
        2. Mark above/below threshold
        3. Cumsum on change-points to create group IDs
        4. cumcount within each group = streak length at each point
    """
    above = (fr_abs.shift(1) >= threshold).astype(int)
    # Group ID increments every time the above/below state changes
    group = (above != above.shift(1)).cumsum()
    streak = above.groupby(group).cumcount()
    # Zero out streaks that are currently in "below" state
    return streak * above


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct all features per-coin, then concatenate.
    All historical aggregates use shift(1). No future data in any feature.
    """
    result_frames: list[pd.DataFrame] = []

    for symbol, grp in df.groupby("symbol", sort=False):
        g = grp.copy().sort_values("timestamp")
        fr     = g["funding_rate_pct"]
        fr_abs = fr.abs()

        # ── Core features ────────────────────────────────────────────────────

        # Current period — known at decision time T (no shift needed)
        g["fr_abs"]  = fr_abs
        g["fr_sign"] = np.sign(fr).astype(np.float32)

        # 7-day rolling baseline (historical)
        g["fr_roll_mean_M"] = _shift_roll(fr_abs, MED_WINDOW, "mean")
        g["fr_roll_std_M"]  = _shift_roll(fr_abs, MED_WINDOW, "std")

        # 1-day rolling mean (historical) — used only inside decay_rate
        fr_roll_mean_S = _shift_roll(fr_abs, SHORT_WINDOW, "mean")

        # Momentum: how much current FR deviates from 7d baseline
        g["fr_momentum"] = g["fr_abs"] - g["fr_roll_mean_M"]

        # Decay rate: is FR accelerating or decelerating vs baseline?
        # Clip prevents extreme values when mean_M ≈ 0 (LOW regime)
        g["fr_decay_rate"] = (
            fr_roll_mean_S / (g["fr_roll_mean_M"].clip(lower=EPSILON_FR)
        )).clip(-DECAY_CLIP, DECAY_CLIP)

        # Consecutive settlements above entry threshold (shift inside function)
        g["n_consecutive_above"] = _consecutive_above(fr_abs, ENTRY_THRESHOLD_PCT)

        # ── Extended features ─────────────────────────────────────────────────

        # Z-score of current |FR| vs 7d window
        std_safe = g["fr_roll_std_M"].replace(0, np.nan)
        g["fr_zscore_M"] = (g["fr_abs"] - g["fr_roll_mean_M"]) / std_safe

        result_frames.append(g)

    combined = pd.concat(result_frames, ignore_index=True)

    # Cross-sectional percentile rank at each timestamp
    # Uses only data available at T — coins present at T ranked against each other
    combined["cs_rank"] = (
        combined.groupby("timestamp")["fr_abs"]
        .rank(pct=True, method="average")
    )

    return combined.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


# ── Label Construction ────────────────────────────────────────────────────────

def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary label: did |FR| stay above EXIT_THRESHOLD for next LABEL_HORIZON settlements?

    Label = 1 if mean(|FR|[T+1 : T+LABEL_HORIZON]) > EXIT_THRESHOLD_PCT

    Horizon = 4 settlements (~32h), aligned with observed avg hold 3.9.

    Note: fr_forward_mean is intentionally future data — it is ONLY used as label,
    never as a feature. It is never shifted to prevent leakage into features.
    """
    result_frames: list[pd.DataFrame] = []

    for symbol, grp in df.groupby("symbol", sort=False):
        g = grp.copy().sort_values("timestamp")
        fr_abs = g["funding_rate_pct"].abs()

        # Forward mean: T+1 through T+LABEL_HORIZON
        # shift(-1): start from T+1
        # rolling(LABEL_HORIZON): mean over next N
        # shift(-(LABEL_HORIZON-1)): align result back to T
        g["fr_forward_mean"] = (
            fr_abs
            .shift(-1)
            .rolling(LABEL_HORIZON, min_periods=max(1, LABEL_HORIZON // 2))
            .mean()
            .shift(-(LABEL_HORIZON - 1))
        )

        g["label"] = (g["fr_forward_mean"] > EXIT_THRESHOLD_PCT).astype(np.int8)
        g["label_continuous"] = g["fr_forward_mean"] - ENTRY_THRESHOLD_PCT

        result_frames.append(g)

    return (
        pd.concat(result_frames, ignore_index=True)
        .sort_values(["timestamp", "symbol"])
        .reset_index(drop=True)
    )


# ── Temporal Split ────────────────────────────────────────────────────────────

def temporal_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Train: 2022-01-01 → 2023-12-31 (minus purge tail)
    Val:   2024-01-01 → 2024-12-31 (minus embargo head)
    Test:  2025-01-01 → end        (minus embargo head)

    Purge: remove last PURGE_SETTLEMENTS from train end
           → prevents label overlap across boundary
    Embargo: skip first EMBARGO_SETTLEMENTS of val and test
           → prevents feature overlap (rolling windows span boundary)
    """
    timestamps = np.sort(df["timestamp"].unique())

    train_end_ts = pd.Timestamp(TRAIN_END, tz="UTC")
    val_end_ts   = pd.Timestamp(VAL_END,   tz="UTC")

    def nth_after(boundary: pd.Timestamp, n: int) -> pd.Timestamp:
        after = timestamps[timestamps > boundary]
        return after[min(n, len(after) - 1)]

    # Purge tail of train
    purge_cutoff = train_end_ts - pd.Timedelta(hours=PURGE_SETTLEMENTS * 8)

    # Embargo heads
    train_val_embargo_end = nth_after(train_end_ts, EMBARGO_SETTLEMENTS)
    val_test_embargo_end  = nth_after(val_end_ts,   EMBARGO_SETTLEMENTS)

    train_mask = (df["timestamp"] <= train_end_ts) & (df["timestamp"] < purge_cutoff)
    val_mask   = (df["timestamp"] > train_val_embargo_end) & (df["timestamp"] <= val_end_ts)
    test_mask  =  df["timestamp"] > val_test_embargo_end

    train_df = df[train_mask].copy()
    val_df   = df[val_mask].copy()
    test_df  = df[test_mask].copy()

    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"[split] {name:5s}: {len(split):>8,} rows | "
              f"{split['timestamp'].min().date()} → {split['timestamp'].max().date()}")

    return train_df, val_df, test_df


# ── Scaling ───────────────────────────────────────────────────────────────────

def fit_scaler(train_df: pd.DataFrame, feature_cols: list[str] = FEATURE_COLS) -> RobustScaler:
    """Fit RobustScaler on train only. Never fit on val/test."""
    scaler = RobustScaler()
    scaler.fit(train_df[feature_cols].fillna(0))
    return scaler


def apply_scaler(
    df: pd.DataFrame,
    scaler: RobustScaler,
    feature_cols: list[str] = FEATURE_COLS,
) -> np.ndarray:
    return scaler.transform(df[feature_cols].fillna(0))


# ── Full Pipeline ─────────────────────────────────────────────────────────────

def prepare_dataset(
    df: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLS,
    cost_tier: str = "mid",
    use_label_v3: bool = True,
) -> dict:
    """
    Full pipeline: features → labels → split → scale.

    Parameters
    ----------
    df           : output of load_real_fr_data() — columns:
                   timestamp, symbol, funding_rate, funding_rate_pct
    feature_cols : feature columns to use
    cost_tier    : cost tier for label v3 simulation ("low"|"mid"|"high")
    use_label_v3 : if True, use bot-simulation labels (recommended);
                   if False, fall back to label v1 (fr_forward_mean threshold)

    Returns dict with X_train/val/test, y arrays, raw DataFrames, scaler.
    """
    print("[pipeline] Building features...")
    with_features = build_features(df)

    # Split full universe BEFORE filtering (needed for simulation exit logic)
    print("[pipeline] Splitting full universe...")
    full_train_df, full_val_df, full_test_df = temporal_split(with_features)

    if use_label_v3:
        print("[pipeline] Building labels v3 (bot simulation)...")
        from label_builder import build_labels_v3, label_stats
        with_labels = build_labels_v3(with_features, cost_tier=cost_tier)
        # Only train on entry candidates with valid label
        with_labels = with_labels[with_labels["label_is_entry"]].copy()
        with_labels["label"] = with_labels["label_v3"]
        print(f"[pipeline] Entry candidates: {len(with_labels):,} rows")
    else:
        print("[pipeline] Building labels v1 (fr_forward_mean)...")
        with_labels = build_labels(with_features)

    # Drop rows where any feature couldn't be computed
    drop_cols = ["label"] + feature_cols
    before = len(with_labels)
    with_labels = with_labels.dropna(subset=drop_cols)
    print(f"[pipeline] Dropped {before - len(with_labels):,} rows with NaN features/labels")

    print("[pipeline] Splitting entry candidates...")
    train_df, val_df, test_df = temporal_split(with_labels)

    # Class balance check
    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        pos_rate = split["label"].mean()
        print(f"[balance] {name}: {pos_rate:.1%} positive", end="")
        if pos_rate < 0.15 or pos_rate > 0.85:
            print(" ⚠ SEVERE IMBALANCE", end="")
        print()

    scaler = fit_scaler(train_df, feature_cols)

    X_train = apply_scaler(train_df, scaler, feature_cols)
    X_val   = apply_scaler(val_df,   scaler, feature_cols)
    X_test  = apply_scaler(test_df,  scaler, feature_cols)

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": train_df["label"].values,
        "y_val":   val_df["label"].values,
        "y_test":  test_df["label"].values,
        "train_df": train_df, "val_df": val_df, "test_df": test_df,
        "full_train_df": full_train_df, "full_val_df": full_val_df, "full_test_df": full_test_df,
        "scaler": scaler,
        "feature_names": feature_cols,
    }

