"""
run_research.py
---------------
Orchestrator: load real data → baseline → LGBM → RF sanity → compare.

Usage (Kaggle):
    from run_research import run_full_comparison
    results = run_full_comparison("/kaggle/input/your-data/")
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_rule import run_baseline, rule_based_signal, simulate_strategy
from data_loader import FEATURE_COLS, prepare_dataset
from data_loader_real import align_universe, load_real_fr_data
from model_lgbm import FundingRateLGBM, evaluate_lgbm, run_rf_sanity_check
from statistical_validation import run_full_statistical_validation


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
        m = simulate_strategy(df[mask].reset_index(drop=True), signal[mask], strategy_name=f"{name}_{regime}")
        rows.append({"regime": regime, "n_periods": m["n_periods"],
                     "apy_proxy": m["apy_proxy_pct"], "sharpe": m["sharpe_proxy"],
                     "utilization": m["utilization"]})
    return pd.DataFrame(rows)


# ── Sanity Checks ─────────────────────────────────────────────────────────────

def run_sanity_checks(train_sim: dict, val_sim: dict, name: str) -> list[str]:
    warnings_out: list[str] = []
    tr_sh, va_sh = train_sim.get("sharpe_proxy", 0), val_sim.get("sharpe_proxy", 0)
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
    data_dir:   str,
    output_dir: str = "./research_output",
    feature_cols: list[str] = FEATURE_COLS,
) -> dict:
    t0 = time.time()
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    print("=" * 60)
    print("FR Arbitrage ML Research Pipeline")
    print("=" * 60)

    # ── 1. Load data ──
    print("\n[1/5] Loading real FR data...")
    raw = load_real_fr_data(data_dir)
    raw = align_universe(raw, min_history_days=548)  # 18 months

    # ── 2. Prepare dataset ──
    print("\n[2/5] Feature engineering + split...")
    ds = prepare_dataset(raw, feature_cols=feature_cols)
    X_train, X_val   = ds["X_train"], ds["X_val"]
    y_train, y_val   = ds["y_train"], ds["y_val"]
    train_df, val_df = ds["train_df"], ds["val_df"]

    # Add regime labels
    combined_df = label_market_regime(pd.concat([train_df, val_df], ignore_index=True))
    train_df = combined_df[combined_df["timestamp"].isin(train_df["timestamp"])].copy()
    val_df   = combined_df[combined_df["timestamp"].isin(val_df["timestamp"])].copy()

    all_results: dict[str, dict] = {}

    # ── 3. Baseline ──
    print("\n[3/5] Rule-based baseline...")
    baseline_train = run_baseline(train_df, "train")
    baseline_val   = run_baseline(val_df,   "val")
    all_results["baseline"] = {"sim_metrics": baseline_val, "auc": "n/a"}
    print(f"  Val APY: {baseline_val['apy_proxy_pct']:.4f}% | Sharpe: {baseline_val['sharpe_proxy']:.4f}")

    # ── 4. LGBM ──
    print("\n[4/5] LightGBM...")
    lgbm_model = FundingRateLGBM()
    lgbm_model.fit(X_train, y_train, train_df, X_val, y_val, run_cv=True)

    lgbm_train_sig = lgbm_model.combined_signal(X_train, train_df)
    lgbm_train_sim = simulate_strategy(train_df, lgbm_train_sig, strategy_name="lgbm_train")

    lgbm_eval = evaluate_lgbm(
        lgbm_model, X_val, y_val, val_df,
        split_name="val",
        feature_names=ds["feature_names"],
        baseline_metrics=baseline_val,
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
        print("  ⚠ Gap < 0.03 — consider using RF (simpler, lower overfit risk)")
    else:
        print("  ✓ LGBM justified over RF")

    # ── 5. Comparison ──
    print("\n[5/5] Results...")
    rows = []
    for name, res in all_results.items():
        sim = res["sim_metrics"]
        rows.append({
            "strategy":     name,
            "apy_proxy_%":  sim["apy_proxy_pct"],
            "sharpe":       sim["sharpe_proxy"],
            "win_rate":     sim["win_rate"],
            "utilization":  sim["utilization"],
            "auc":          res.get("auc", "n/a"),
        })
    comparison = pd.DataFrame(rows).sort_values("apy_proxy_%", ascending=False)
    print("\n" + "=" * 55)
    print("STRATEGY COMPARISON (VAL SET)")
    print("=" * 55)
    print(comparison.to_string(index=False))

    # Regime breakdown
    baseline_sig = rule_based_signal(val_df)
    lgbm_sig     = lgbm_eval["combined_signal"]
    regime_bl    = regime_breakdown(val_df, baseline_sig, "baseline")
    regime_lgbm  = regime_breakdown(val_df, lgbm_sig,    "lgbm")
    print("\nBaseline by regime:\n", regime_bl.to_string(index=False))
    print("\nLGBM by regime:\n",     regime_lgbm.to_string(index=False))

    # Save
    comparison.to_csv(out / "comparison.csv", index=False)
    regime_bl.to_csv(out   / "regime_baseline.csv", index=False)
    regime_lgbm.to_csv(out / "regime_lgbm.csv",     index=False)
    if lgbm_eval.get("feature_importance") is not None:
        lgbm_eval["feature_importance"].to_csv(out / "feature_importance.csv", index=False)

    elapsed = time.time() - t0
    summary = {
        "baseline_val_apy":  float(baseline_val["apy_proxy_pct"]),
        "lgbm_val_apy":      float(lgbm_eval["sim_metrics"]["apy_proxy_pct"]),
        "lgbm_val_auc":      float(lgbm_auc) if isinstance(lgbm_auc, float) else 0.0,
        "rf_sanity_auc":     float(rf_auc),
        "auc_gap":           float(auc_gap),
        "lgbm_warnings":     lgbm_warnings,
        "elapsed_seconds":   round(elapsed, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[Done] {elapsed:.0f}s | Results → {out}/")

    return {
        "dataset":        ds,
        "baseline":       baseline_val,
        "lgbm_model":     lgbm_model,
        "comparison":     comparison,
        "regime_baseline": regime_bl,
        "regime_lgbm":    regime_lgbm,
        "all_results":    all_results,
    }


def evaluate_on_test(results: dict) -> dict:
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

    baseline_test = run_baseline(test_df, "test")
    lgbm_sig      = lgbm_model.combined_signal(X_test, test_df)
    lgbm_test_sim = simulate_strategy(test_df, lgbm_sig, strategy_name="lgbm_test")

    delta = lgbm_test_sim["apy_proxy_pct"] - baseline_test["apy_proxy_pct"]
    print(f"Baseline APY: {baseline_test['apy_proxy_pct']:.4f}%")
    print(f"LGBM APY:     {lgbm_test_sim['apy_proxy_pct']:.4f}%")
    print(f"Delta:        {delta:+.4f}%")

    return {"baseline": baseline_test, "lgbm": lgbm_test_sim}

