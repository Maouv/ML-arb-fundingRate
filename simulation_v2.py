"""
simulation_v2.py
----------------
Corrected bot simulation — exact parity with phase3_backtest/engine/simulator.py.

Fixes vs original baseline_rule.py:
    1. t+1 causality: signal at t → entry executed at t+1
    2. Exit condition: ONLY |FR| < EXIT_THRESHOLD (removed sign-flip exit)
    3. Gross collection: starts from t+1 (first settlement AFTER entry)
    4. Priority: sort by (fr_abs - cost) descending, tie-break alphabetical
    5. Cost model: per-coin actual costs + tier fallback (matches phase3 config)
    6. APY: annualized net yield (gross - cost), not raw gross

Design:
    - Stateless: takes df + signal array → returns metrics dict
    - No external dependencies beyond numpy/pandas
    - All params explicit (no hidden globals)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Cost Model (mirrors phase3_backtest/config.py) ────────────────────────────

ACTUAL_COSTS: dict[str, float] = {
    "ETHUSDT":  0.0814,
    "ZECUSDT":  0.0882,
    "XRPUSDT":  0.1018,
    "DOGEUSDT": 0.1088,
    "LINKUSDT": 0.1113,
    "SUIUSDT":  0.1119,
    "AAVEUSDT": 0.1275,
    "INJUSDT":  0.1562,
    "UNIUSDT":  0.1773,
    "ADAUSDT":  0.2002,
}

COST_TIERS: dict[str, float] = {"low": 0.08, "mid": 0.12, "high": 0.20}

ENTRY_THRESHOLD: float = 0.05   # |FR| >= 0.05% → entry signal
EXIT_THRESHOLD:  float = 0.02   # |FR| < 0.02%  → exit signal (strict less-than)
SETTLEMENTS_PER_YEAR: int = 1095  # 3 per day × 365
MAX_SLOTS: int = 6
SIZE_PER_PAIR: float = 300.0    # $300 per position (matches phase3 config)


def _get_cost(symbol: str, tier: str = "mid") -> float:
    """Return round-trip cost for symbol. Actual cost takes priority over tier."""
    if symbol in ACTUAL_COSTS:
        return ACTUAL_COSTS[symbol]
    if tier not in COST_TIERS:
        raise ValueError(f"Invalid tier '{tier}'. Must be one of: {list(COST_TIERS.keys())}")
    return COST_TIERS[tier]


# ── Core Simulator ────────────────────────────────────────────────────────────

def simulate_bot(
    df: pd.DataFrame,
    entry_signals: np.ndarray,
    cost_tier: str = "mid",
    max_slots: int = MAX_SLOTS,
    strategy_name: str = "unnamed",
) -> dict:
    """
    Simulate the FR arbitrage bot with strict t+1 causality.

    Event loop order per settlement (matches phase3_backtest/engine/simulator.py):
        Step 1 — Execute pending exits (from t-1 signal)
        Step 2 — Execute pending entries (from t-1 signal)
        Step 3 — Collect FR[t] for ALL open positions (including just-opened)
        Step 4 — Generate new signals from FR[t]

    Parameters
    ----------
    df           : long-format DataFrame with columns:
                   timestamp, symbol, funding_rate_pct, fr_abs
    entry_signals: boolean/int array aligned with df rows (1 = candidate entry)
    cost_tier    : "low" | "mid" | "high" (for coins without actual cost)
    max_slots    : maximum concurrent positions
    strategy_name: label for output dict

    Returns
    -------
    dict with performance metrics (APY, Sharpe, win_rate, n_trades, avg_net_pct, ...)
    """
    df = df.copy().reset_index(drop=True)
    df["_sig"] = entry_signals.astype(bool)

    # Build per-timestamp FR lookup: {timestamp: {symbol: fr_pct}}
    # Using dict-of-dicts for O(1) lookup in hot loop
    ts_fr: dict[pd.Timestamp, dict[str, float]] = {}
    ts_sig: dict[pd.Timestamp, set[str]] = {}  # symbols with entry signal at each ts

    for row in df.itertuples(index=False):
        ts = row.timestamp
        sym = row.symbol
        fr = row.funding_rate_pct

        if ts not in ts_fr:
            ts_fr[ts] = {}
            ts_sig[ts] = set()

        ts_fr[ts][sym] = fr
        if row._sig:  # noqa: SLF001 — itertuples field name
            ts_sig[ts].add(sym)

    timestamps = sorted(ts_fr.keys())

    # State
    open_pos: dict[str, dict] = {}         # symbol → {entry_ts, entry_fr, gross, cost, hold}
    pending_entries: dict[str, float] = {} # symbol → trigger_fr (from t-1 signal)
    pending_exits: set[str] = set()        # symbols signalled for exit at t-1
    trades: list[dict] = []

    for ts in timestamps:
        fr_now = ts_fr[ts]

        # ── Step 1: Execute pending exits ────────────────────────────────────
        for sym in list(pending_exits):
            if sym in open_pos:
                pos = open_pos.pop(sym)
                net_pct = pos["gross"] - pos["cost"]
                trades.append({
                    "symbol":           sym,
                    "entry_ts":         pos["entry_ts"],
                    "exit_ts":          ts,
                    "hold_settlements": pos["hold"],
                    "gross_pct":        pos["gross"],
                    "cost_rt_pct":      pos["cost"],
                    "net_pct":          net_pct,
                    "entry_fr":         pos["entry_fr"],
                    "exit_fr":          fr_now.get(sym, 0.0),
                })
        pending_exits.clear()

        # ── Step 2: Execute pending entries ──────────────────────────────────
        available_slots = max_slots - len(open_pos)
        if available_slots > 0 and pending_entries:
            # Sort by (expected net = trigger_fr_abs - cost) desc, then symbol asc
            candidates = [
                (sym, fr_val)
                for sym, fr_val in pending_entries.items()
                if sym not in open_pos
            ]
            candidates.sort(
                key=lambda x: (-(abs(x[1]) - _get_cost(x[0], cost_tier)), x[0])
            )
            for sym, trigger_fr in candidates[:available_slots]:
                cost = _get_cost(sym, cost_tier)
                open_pos[sym] = {
                    "entry_ts":  ts,
                    "entry_fr":  trigger_fr,
                    "gross":     0.0,    # collection starts at Step 3 of THIS settlement
                    "cost":      cost,
                    "hold":      0,
                }
        pending_entries.clear()

        # ── Step 3: Collect FR[t] for ALL open positions ──────────────────────
        # Includes positions just opened in Step 2 — they collect FR at entry settlement
        for sym in list(open_pos.keys()):
            if sym in fr_now:
                open_pos[sym]["gross"] += abs(fr_now[sym])
                open_pos[sym]["hold"] += 1

        # ── Step 4: Generate new signals from FR[t] ───────────────────────────
        # Entry signal: |FR| >= threshold AND in candidate set AND no open position
        sig_syms = ts_sig[ts]
        for sym in sig_syms:
            if sym not in open_pos:
                pending_entries[sym] = fr_now[sym]

        # Exit signal: |FR| < exit threshold (strict)
        for sym in list(open_pos.keys()):
            if sym in fr_now and abs(fr_now[sym]) < EXIT_THRESHOLD:
                pending_exits.add(sym)

    # Close any positions still open at end of data (forced close)
    last_ts = timestamps[-1] if timestamps else None
    for sym, pos in open_pos.items():
        net_pct = pos["gross"] - pos["cost"]
        trades.append({
            "symbol":           sym,
            "entry_ts":         pos["entry_ts"],
            "exit_ts":          last_ts,
            "hold_settlements": pos["hold"],
            "gross_pct":        pos["gross"],
            "cost_rt_pct":      pos["cost"],
            "net_pct":          net_pct,
            "entry_fr":         pos["entry_fr"],
            "exit_fr":          0.0,
        })

    return _compute_metrics(trades, timestamps, strategy_name)


def _compute_metrics(
    trades: list[dict],
    timestamps: list[pd.Timestamp],
    strategy_name: str,
) -> dict:
    """Compute performance metrics from closed trade list."""
    if not trades:
        return _empty_metrics(strategy_name)

    trade_df = pd.DataFrame(trades)
    n_periods = len(timestamps)
    spy = SETTLEMENTS_PER_YEAR

    net_per_trade = trade_df["net_pct"].values
    total_net_pct = net_per_trade.sum()

    # APY: annualized net yield per capital unit
    # net_pct is already in % of SIZE_PER_PAIR, annualize over settlements
    apy = total_net_pct / n_periods * spy if n_periods > 0 else 0.0

    # Per-period net yield (for Sharpe)
    # Aggregate net_pct across trades that closed each period
    trade_df["exit_ts"] = pd.to_datetime(trade_df["exit_ts"], utc=True)
    per_period = (
        trade_df.groupby("exit_ts")["net_pct"]
        .sum()
        .reindex(pd.Index(timestamps), fill_value=0.0)
    )
    sharpe = (
        per_period.mean() / per_period.std() * np.sqrt(spy)
        if per_period.std() > 0 else 0.0
    )

    win_rate    = float((net_per_trade > 0).mean())
    n_trades    = len(trades)
    avg_net_pct = float(net_per_trade.mean())

    # Noise trades: net_pct <= 0
    noise_rate = float((net_per_trade <= 0).mean())

    return {
        "strategy":             strategy_name,
        "n_trades":             n_trades,
        "n_periods":            n_periods,
        "total_net_pct":        round(total_net_pct, 4),
        "apy_pct":              round(apy, 4),
        "sharpe":               round(sharpe, 4),
        "win_rate":             round(win_rate, 4),
        "avg_net_pct":          round(avg_net_pct, 4),
        "noise_rate":           round(noise_rate, 4),
        "utilization":          round(
            sum(t["hold_settlements"] for t in trades) / max(n_periods * MAX_SLOTS, 1), 4
        ),
        "trade_df":             trade_df,  # for downstream analysis
    }


def _empty_metrics(name: str) -> dict:
    return {
        "strategy": name, "n_trades": 0, "n_periods": 0,
        "total_net_pct": 0.0, "apy_pct": 0.0, "sharpe": 0.0,
        "win_rate": 0.0, "avg_net_pct": 0.0, "noise_rate": 0.0,
        "utilization": 0.0, "trade_df": pd.DataFrame(),
    }


# ── Baseline Signal Builder ───────────────────────────────────────────────────

def rule_signal(df: pd.DataFrame) -> np.ndarray:
    """Pure rule: |FR| >= entry threshold. Returns bool array aligned with df."""
    return (df["fr_abs"] >= ENTRY_THRESHOLD).values.astype(int)


def run_baseline_v2(df: pd.DataFrame, split_name: str = "val", cost_tier: str = "mid") -> dict:
    """Run corrected rule-based baseline."""
    return simulate_bot(df, rule_signal(df), cost_tier=cost_tier, strategy_name=f"baseline_v2_{split_name}")


# ── Comparison Helper ─────────────────────────────────────────────────────────

def compare_strategies(baseline: dict, lgbm: dict) -> pd.DataFrame:
    """
    Print side-by-side comparison table.
    Returns DataFrame for programmatic use.
    """
    metrics = ["apy_pct", "sharpe", "win_rate", "n_trades", "avg_net_pct", "noise_rate", "utilization"]
    rows = []
    for m in metrics:
        b_val = baseline.get(m, "n/a")
        l_val = lgbm.get(m, "n/a")
        delta = ""
        if isinstance(b_val, float) and isinstance(l_val, float):
            delta = f"{l_val - b_val:+.4f}"
        rows.append({"metric": m, "baseline": b_val, "lgbm": l_val, "delta": delta})

    cmp_df = pd.DataFrame(rows)
    print("\n" + "=" * 55)
    print(f"STRATEGY COMPARISON — baseline vs lgbm")
    print("=" * 55)
    print(cmp_df.to_string(index=False))
    return cmp_df
