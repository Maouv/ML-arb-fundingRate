"""
run_research.py
---------------
Orchestrator: load real data → baseline → LGBM → RF sanity → compare.

v2 changes (Fix #1, #2, #3):
    - Simulation uses simulation_v2.simulate_bot (t+1 causality, correct exit)
    - Labels use label_builder.build_labels_v3 (bot simulation labels)
    - Threshold tuned on train holdout, not val set

Usage (Kaggle):
    from run_research import run_full_comparison
    results = run_full_comparison("/kaggle/input/dataset/dafanaalfarizi/fr-arbitrage-data/funding_rate_data_8h_100/")
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import FEATURE_COLS, prepare_dataset, prepare_full_splits
from data_loader_real import align_universe, load_real_fr_data
from model_lgbm import FundingRateLGBM, evaluate_lgbm, run_rf_sanity_check
from simulation_v2 import (
    compare_strategies,
    rule_signal,
    run_baseline_v2,
    simulate_bot,
)


# ── Signal Builder (entry candidates → full df signal) ───────────────────────

def _build_full_signal(
    model: FundingRateLGBM,
    entry_df: pd.DataFrame,
    X_entry: np.ndarray,
    full_df: pd.DataFrame,
    feature_cols: list[str],
    scaler,
) -> np.ndarray:
    """
    Map LGBM predictions (on entry candidates) back to full df.
    Non-entry rows get signal=0. Entry rows get rule & lgbm signal.
    """
    from data_loader import ENTRY_THRESHOLD_PCT, apply_scaler
    signal = np.zeros(len(full_df), dtype=int)
    entry_mask = (full_df["fr_abs"] >= ENTRY_THRESHOLD_PCT).values
    if entry_mask.sum() == 0:
        return signal
    X_full_entries = apply_scaler(full_df[entry_mask], scaler, feature_cols)
    lgbm_preds = model.predict(X_full_entries).astype(bool)
    signal[entry_mask] = lgbm_preds.astype(int)
    return signal


# ── Regime Labeler ────────────────────────────────────────────────────────────

def label_market_regime(df: pd.DataFrame) -> pd.DataFrame:
    per_ts = df.groupby("timestamp")["fr_abs"].agg(["median", "std"]).reset_index()
    per_ts.columns = ["timestamp", "fr_med", "fr_std"]
    per_ts["fr_std_roll"] = per_ts["fr_std"].rolling(21, min_periods=3).mean().fillna(0)

    def assign(row) -> str:
        if row["fr_std"] > 2 * max(row["fr_std_roll"], 0.001):
            return "STRESS"
        if row["fr_med"] < 0.03:
            return "LOW"
        if row["fr_med"] >= 0.07:
            return "HIGH"
        return "NORMAL"

    per_ts["regime"] = per_ts.apply(assign, axis=1)
    return df.merge(per_ts[["timestamp", "regime"]], on="timestamp", how="left")


def regime_breakdown(df: pd.DataFrame, signal: np.ndarray, name: str) -> pd.DataFrame:
    rows = []
    for regime in ["LOW", "NORMAL", "HIGH", "STRESS"]:
        mask = (df["regime"] == regime).values
        if mask.sum() < 100:
            continue
        m = simulate_bot(
            df[mask].reset_index(drop=True),
            signal[mask],
            strategy_name=f"{name}_{regime}",
        )
        rows.append({
            "regime": regime, "n_periods": m["n_periods"],
            "apy_pct": m["apy_pct"], "sharpe": m["sharpe"],
            "utilization": m["utilization"],
        })
    return pd.DataFrame(rows)


# ── Sanity Checks ─────────────────────────────────────────────────────────────

def run_sanity_checks(train_sim: dict, val_sim: dict, name: str) -> list[str]:
    warnings_out: list[str] = []
    tr_sh, va_sh = train_sim.get("sharpe", 0), val_sim.get("sharpe", 0)
    if tr_sh > 0 and va_sh < tr_sh * 0.6:
        warnings_out.append(f"OVERFIT: Sharpe {tr_sh:.2f}→{va_sh:.2f} ({(1-va_sh/tr_sh)*100:.0f}% decay)")
    util = val_sim.get("utilization", 1)
    if util < 0.05:
        warnings_out.append("OVER-FILTER: <5% slots used")
    if util > 0.95:
        warnings_out.append("NO-FILTER: >95% slots used — same as baseline")
    tag = "✓ OK" if not warnings_out else "⚠ ISSUES"
    print(f"\n[sanity {name}] {tag}")
    for w in warnings_out:
        print(f"  → {w}")
    return warnings_out


# ── Main ──────────────────────────────────────────────────────────────────────

def run_full_comparison(
    data_dir:     str,
    output_dir:   str = "./research_output",
    feature_cols: list[str] = FEATURE_COLS,
    cost_tier:    str = "mid",
) -> dict:
    t0 = time.time()
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    print("=" * 60)
    print("FR Arbitrage ML Research Pipeline  [v2 — fixes #1 #2 #3]")
    print("=" * 60)

    # ── 1. Load data ──
    print("\n[1/5] Loading real FR data...")
    raw = load_real_fr_data(data_dir)
    raw = align_universe(raw, min_history_days=548)

    # ── 2. Prepare dataset ──
    print("\n[2/5] Feature engineering + label v3 + split...")
    ds = prepare_dataset(raw, feature_cols=feature_cols, cost_tier=cost_tier, use_label_v3=True)
    X_train, X_val   = ds["X_train"], ds["X_val"]
    y_train, y_val   = ds["y_train"], ds["y_val"]
    train_df, val_df = ds["train_df"], ds["val_df"]

    # Full splits for simulation (all rows, not entry-only)
    train_full_df, val_full_df, test_full_df = prepare_full_splits(raw, feature_cols)

    # Add regime labels on full val
    combined_df = label_market_regime(pd.concat([train_full_df, val_full_df], ignore_index=True))
    train_ts_set = set(train_full_df["timestamp"].unique())
    val_ts_set   = set(val_full_df["timestamp"].unique())
    train_full_df = combined_df[combined_df["timestamp"].isin(train_ts_set)].copy()
    val_full_df   = combined_df[combined_df["timestamp"].isin(val_ts_set)].copy()

    all_results: dict[str, dict] = {}

    # ── 3. Baseline v2 (corrected simulation) ──
    print("\n[3/5] Rule-based baseline (v2 — correct simulation)...")
    baseline_train = run_baseline_v2(train_full_df, "train", cost_tier=cost_tier)
    baseline_val   = run_baseline_v2(val_full_df,   "val",   cost_tier=cost_tier)
    all_results["baseline"] = {"sim_metrics": baseline_val, "auc": "n/a"}
    print(f"  Val APY:    {baseline_val['apy_pct']:.4f}%")
    print(f"  Val Sharpe: {baseline_val['sharpe']:.4f}")
    print(f"  Val trades: {baseline_val['n_trades']}")
    print(f"  Win rate:   {baseline_val['win_rate']:.1%}")
    print(f"  Noise rate: {baseline_val['noise_rate']:.1%}")

    # ── 4. LGBM (train only on entry candidates with label v3) ──
    print("\n[4/5] LightGBM (trained on label v3, threshold on train holdout)...")
    lgbm_model = FundingRateLGBM()
    lgbm_model.fit(X_train, y_train, ds["train_df"], X_val, y_val, run_cv=True)

    lgbm_train_sig = _build_full_signal(lgbm_model, ds["train_df"], X_train, train_full_df, feature_cols, ds["scaler"])
    lgbm_train_sim = simulate_bot(train_full_df, lgbm_train_sig, cost_tier=cost_tier, strategy_name="lgbm_train")

    lgbm_eval = evaluate_lgbm(
        lgbm_model, X_val, y_val, ds["val_df"],
        val_full_df=val_full_df,
        split_name="val",
        feature_names=ds["feature_names"],
        baseline_metrics=baseline_val,
        cost_tier=cost_tier,
        scaler=ds["scaler"],
    )
    all_results["lgbm"] = lgbm_eval
    lgbm_warnings = run_sanity_checks(lgbm_train_sim, lgbm_eval["sim_metrics"], "LGBM")

    # ── 4b. RF sanity ──
    print("\n[4b] RF sanity check...")
    rf_auc   = run_rf_sanity_check(X_train, y_train, X_val, y_val)
    lgbm_auc = lgbm_eval["auc"]
    auc_gap  = lgbm_auc - rf_auc if isinstance(lgbm_auc, float) else 0.0
    print(f"  LGBM AUC: {lgbm_auc:.4f} | RF AUC: {rf_auc:.4f} | Gap: {auc_gap:+.4f}")
    if auc_gap < 0.03:
        print("  ⚠ Gap < 0.03 — LGBM complexity not justified over RF")
    else:
        print("  ✓ LGBM justified")

    # ── 5. Side-by-side comparison ──
    print("\n[5/5] Results comparison...")
    compare_strategies(baseline_val, lgbm_eval["sim_metrics"])

    # Regime breakdown
    baseline_sig_val = rule_signal(val_full_df)
    lgbm_sig_val     = lgbm_eval["combined_signal"]
    regime_bl   = regime_breakdown(val_full_df, baseline_sig_val, "baseline")
    regime_lgbm = regime_breakdown(val_full_df, lgbm_sig_val,     "lgbm")
    print("\nBaseline by regime:\n", regime_bl.to_string(index=False))
    print("\nLGBM by regime:\n",     regime_lgbm.to_string(index=False))

    # Save
    regime_bl.to_csv(out   / "regime_baseline.csv", index=False)
    regime_lgbm.to_csv(out / "regime_lgbm.csv",     index=False)
    if lgbm_eval.get("feature_importance") is not None:
        lgbm_eval["feature_importance"].to_csv(out / "feature_importance.csv", index=False)

    elapsed = time.time() - t0
    summary = {
        "baseline_val_apy":  float(baseline_val["apy_pct"]),
        "baseline_val_n_trades": int(baseline_val["n_trades"]),
        "baseline_noise_rate": float(baseline_val["noise_rate"]),
        "lgbm_val_apy":      float(lgbm_eval["sim_metrics"]["apy_pct"]),
        "lgbm_val_n_trades": int(lgbm_eval["sim_metrics"]["n_trades"]),
        "lgbm_noise_rate":   float(lgbm_eval["sim_metrics"]["noise_rate"]),
        "lgbm_val_auc":      float(lgbm_auc) if isinstance(lgbm_auc, float) else 0.0,
        "rf_sanity_auc":     float(rf_auc),
        "auc_gap":           float(auc_gap),
        "lgbm_threshold":    float(lgbm_model.threshold),
        "lgbm_warnings":     lgbm_warnings,
        "elapsed_seconds":   round(elapsed, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[Done] {elapsed:.0f}s | Results → {out}/")
    print("\n=== SUMMARY JSON ===")
    print(json.dumps(summary, indent=2))

    return {
        "dataset":          ds,
        "baseline":         baseline_val,
        "lgbm_model":       lgbm_model,
        "regime_baseline":  regime_bl,
        "regime_lgbm":      regime_lgbm,
        "all_results":      all_results,
        "summary":          summary,
    }


def evaluate_on_test(results: dict, cost_tier: str = "mid") -> dict:
    """
    LOCKED — run ONCE after all decisions finalized.
    Do NOT use output to retune anything.
    """
    ds         = results["dataset"]
    lgbm_model = results["lgbm_model"]
    X_test, y_test, test_df = ds["X_test"], ds["y_test"], ds["test_df"]

    print("\n" + "!" * 55)
    print("FINAL TEST SET EVALUATION (2025) — DO NOT RETUNE")
    print("!" * 55)

    baseline_test = run_baseline_v2(test_df, "test", cost_tier=cost_tier)
    lgbm_sig      = lgbm_model.combined_signal(X_test, test_df)
    lgbm_test_sim = simulate_bot(test_df, lgbm_sig, cost_tier=cost_tier, strategy_name="lgbm_test")

    delta_apy    = lgbm_test_sim["apy_pct"] - baseline_test["apy_pct"]
    delta_sharpe = lgbm_test_sim["sharpe"]  - baseline_test["sharpe"]
    print(f"Baseline — APY: {baseline_test['apy_pct']:.4f}% | Sharpe: {baseline_test['sharpe']:.4f} | Trades: {baseline_test['n_trades']}")
    print(f"LGBM     — APY: {lgbm_test_sim['apy_pct']:.4f}% | Sharpe: {lgbm_test_sim['sharpe']:.4f} | Trades: {lgbm_test_sim['n_trades']}")
    print(f"Delta    — APY: {delta_apy:+.4f}% | Sharpe: {delta_sharpe:+.4f}")

    return {"baseline": baseline_test, "lgbm": lgbm_test_sim}

