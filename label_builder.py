"""
label_builder.py
----------------
Label v3: forward-simulate the bot from every entry candidate.

Objective: label = 1 if trade would be PROFITABLE (net > 0 after cost).
This directly aligns with the goal — filter noise trades, preserve profit trades.

Why v3 is better than v1/v2:
    v1 (fr_forward_mean > 0.02%): predicts FR persistence, not trade profitability.
       FR can stay above 0.02% but trade still loses if hold is 1 settlement.
    v2 (threshold at 0.04/0.05%): same structural problem + train/val regime gap.
    v3 (simulate): uses exact same exit logic as real bot → labels match reality.

Label construction:
    For each row where |FR| >= ENTRY_THRESHOLD (entry candidate):
        1. Simulate bot from t+1 (entry execution) using future FR
        2. Track gross collection until exit condition fires
        3. Subtract cost
        4. label = 1 if net_pct > 0 else 0

    Non-candidate rows: label = -1 (masked out — never used in training).

Note on leakage:
    Forward-simulated FR IS future data — this is intentional and correct.
    Labels are targets (ground truth), not features.
    Labels are NEVER used as input features. Strict separation enforced.
    Labels are built AFTER all features, independently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from simulation_v2 import (
    ACTUAL_COSTS,
    COST_TIERS,
    ENTRY_THRESHOLD,
    EXIT_THRESHOLD,
    _get_cost,
)

# Label sentinel for non-entry rows (excluded from training)
NON_ENTRY_LABEL: int = -1


def _simulate_single_trade(
    fr_sequence: np.ndarray,
    cost: float,
) -> tuple[int, float, int]:
    """
    Simulate one trade forward from entry execution (t+1).

    Parameters
    ----------
    fr_sequence : |FR| values from t+1 onwards (already shifted — no lookahead at t)
    cost        : round-trip cost for this symbol (%)

    Returns
    -------
    (label, net_pct, hold_settlements)
        label = 1 if net_pct > 0 else 0
        net_pct = gross_collected - cost
        hold_settlements = number of settlements held
    """
    gross = 0.0
    hold = 0

    for fr_abs in fr_sequence:
        # Collect this settlement's FR
        gross += fr_abs
        hold += 1

        # Check exit: |FR| < EXIT_THRESHOLD → exit after this collection
        # (exit signal fires at t, executes at t+1 — but in label context we
        # simulate forward deterministically, so we exit immediately on signal)
        if fr_abs < EXIT_THRESHOLD:
            break

    net_pct = gross - cost
    label = 1 if net_pct > 0 else 0
    return label, net_pct, hold


def build_labels_v3(
    df: pd.DataFrame,
    cost_tier: str = "mid",
    max_forward_settlements: int = 50,
) -> pd.DataFrame:
    """
    Attach label_v3 to each row.

    For entry candidate rows (|FR| >= ENTRY_THRESHOLD):
        label_v3 = 1 if simulated trade net > 0, else 0
    For non-entry rows:
        label_v3 = NON_ENTRY_LABEL (-1)

    Also attaches:
        label_net_pct    : simulated net % (for analysis, never as feature)
        label_hold       : simulated hold settlements
        label_is_entry   : bool mask (True for trainable rows)

    Parameters
    ----------
    df                     : must have columns: symbol, timestamp, funding_rate_pct, fr_abs
    cost_tier              : "low" | "mid" | "high" for coins without actual cost
    max_forward_settlements: cap on forward look to avoid runaway long holds

    Returns
    -------
    df with label_v3, label_net_pct, label_hold, label_is_entry columns appended
    """
    df = df.copy().sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    labels = np.full(len(df), NON_ENTRY_LABEL, dtype=np.int8)
    net_pcts = np.zeros(len(df), dtype=np.float32)
    holds = np.zeros(len(df), dtype=np.int16)

    for symbol, grp in df.groupby("symbol", sort=False):
        idx = grp.index.values                           # original df indices
        fr_abs_arr = grp["fr_abs"].values.astype(float)
        cost = _get_cost(symbol, cost_tier)

        for i_local, i_global in enumerate(idx):
            if fr_abs_arr[i_local] < ENTRY_THRESHOLD:
                continue  # Not an entry candidate — label stays -1

            # Entry executes at t+1 → forward sequence starts from t+1
            seq_start = i_local + 1
            seq_end = min(i_local + 1 + max_forward_settlements, len(fr_abs_arr))

            if seq_start >= len(fr_abs_arr):
                # No future data available — mark as label 0 (conservative)
                labels[i_global] = 0
                net_pcts[i_global] = -cost
                holds[i_global] = 0
                continue

            fr_seq = fr_abs_arr[seq_start:seq_end]
            label, net_pct, hold = _simulate_single_trade(fr_seq, cost)

            labels[i_global] = label
            net_pcts[i_global] = net_pct
            holds[i_global] = hold

    df["label_v3"] = labels
    df["label_net_pct"] = net_pcts
    df["label_hold"] = holds
    df["label_is_entry"] = (labels != NON_ENTRY_LABEL)

    return df


def label_stats(df: pd.DataFrame, split_name: str = "") -> dict:
    """
    Print and return label distribution stats.
    Only considers trainable rows (label_is_entry == True).
    """
    entry_rows = df[df["label_is_entry"]]
    n_entry = len(entry_rows)

    if n_entry == 0:
        print(f"[label_stats {split_name}] No entry candidates found")
        return {}

    pos_rate = (entry_rows["label_v3"] == 1).mean()
    neg_rate = (entry_rows["label_v3"] == 0).mean()
    avg_net  = entry_rows["label_net_pct"].mean()
    avg_hold_pos = entry_rows.loc[entry_rows["label_v3"] == 1, "label_hold"].mean()
    avg_hold_neg = entry_rows.loc[entry_rows["label_v3"] == 0, "label_hold"].mean()

    print(f"\n[label_stats {split_name}]")
    print(f"  Entry candidates:  {n_entry:,}")
    print(f"  Profitable (1):    {pos_rate:.1%}")
    print(f"  Noise (0):         {neg_rate:.1%}")
    print(f"  Avg net_pct:       {avg_net:.4f}%")
    print(f"  Avg hold (profit): {avg_hold_pos:.1f} settlements")
    print(f"  Avg hold (noise):  {avg_hold_neg:.1f} settlements")

    return {
        "split": split_name,
        "n_entry": n_entry,
        "pos_rate": float(pos_rate),
        "neg_rate": float(neg_rate),
        "avg_net_pct": float(avg_net),
        "avg_hold_profit": float(avg_hold_pos),
        "avg_hold_noise": float(avg_hold_neg),
    }
