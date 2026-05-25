"""
baseline_rule.py
----------------
Rule-based baseline — exact replication of existing bot logic.
Every ML model must beat this on val/test to be considered useful.

Entry : |FR| >= ENTRY_THRESHOLD_PCT (0.05%)
Exit  : |FR| <  EXIT_THRESHOLD_PCT  (0.02%) OR FR sign flips
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from data_loader import ENTRY_THRESHOLD_PCT, EXIT_THRESHOLD_PCT


def simulate_strategy(
    df: pd.DataFrame,
    entry_signals: np.ndarray,
    max_slots: int = 6,
    strategy_name: str = "unnamed",
) -> dict:
    df = df.copy().reset_index(drop=True)
    df["_sig"] = entry_signals.astype(bool)
    timestamps = sorted(df["timestamp"].unique())
    open_pos: dict[str, dict] = {}
    pnl: list[dict] = []

    for ts in timestamps:
        period = df[df["timestamp"] == ts].set_index("symbol")

        # Collect + check exits
        for sym in list(open_pos.keys()):
            if sym not in period.index:
                continue
            row = period.loc[sym]
            cur_fr = row["funding_rate_pct"]
            pnl.append({"timestamp": ts, "symbol": sym, "fr_collected": abs(cur_fr)})
            if (open_pos[sym]["entry_fr"] > 0) != (cur_fr > 0) or abs(cur_fr) < EXIT_THRESHOLD_PCT:
                del open_pos[sym]

        # Enter new
        slots = max_slots - len(open_pos)
        if slots <= 0:
            continue
        cands = period[period["_sig"] & ~period.index.isin(open_pos)].sort_values("fr_abs", ascending=False)
        for sym, row in cands.head(slots).iterrows():
            open_pos[sym] = {"entry_fr": row["funding_rate_pct"], "entry_ts": ts}

    if not pnl:
        return _empty(strategy_name)

    pnl_df = pd.DataFrame(pnl)
    n_periods = len(timestamps)
    total_fr  = pnl_df["fr_collected"].sum()
    spy = 1095  # settlements/year

    per_period = pnl_df.groupby("timestamp")["fr_collected"].sum()
    per_period = per_period.reindex(pd.Index(timestamps), fill_value=0)
    sharpe = (per_period.mean() / per_period.std() * np.sqrt(spy)) if per_period.std() > 0 else 0.0

    return {
        "strategy": strategy_name,
        "total_fr_collected_pct": round(total_fr, 4),
        "apy_proxy_pct":  round(total_fr / n_periods * spy, 4),
        "sharpe_proxy":   round(sharpe, 4),
        "win_rate":       round((pnl_df["fr_collected"] > EXIT_THRESHOLD_PCT).mean(), 4),
        "n_settlements_held": len(pnl_df),
        "utilization":    round(len(pnl_df) / max(n_periods * max_slots, 1), 4),
        "n_periods":      n_periods,
    }


def _empty(name: str) -> dict:
    return {"strategy": name, "total_fr_collected_pct": 0.0, "apy_proxy_pct": 0.0,
            "sharpe_proxy": 0.0, "win_rate": 0.0, "n_settlements_held": 0,
            "utilization": 0.0, "n_periods": 0}


def rule_based_signal(df: pd.DataFrame) -> np.ndarray:
    return (df["fr_abs"] >= ENTRY_THRESHOLD_PCT).values.astype(int)


def run_baseline(df: pd.DataFrame, split_name: str = "val") -> dict:
    return simulate_strategy(df, rule_based_signal(df), strategy_name=f"rule_based_{split_name}")

