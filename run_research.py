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

from data_loader import FEATURE_COLS, prepare_dataset
from data_loader_real import align_universe, load_real_fr_data
from model_lgbm import FundingRateLGBM, run_rf_sanity_check
from simulation_v2 import (
    compare_strategies,
    rule_signal,
    run_baseline_v2,
    simulate_bot,
)


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


# ── Charts ────────────────────────────────────────────────────────────────────

def _generate_charts(out, baseline, lgbm_sim, stat_val, regime_bl, regime_lgbm, fi):
    """Generate PNG charts for Kaggle display."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. Strategy comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics = ["apy_pct", "sharpe", "win_rate"]
    titles = ["APY (%)", "Sharpe Ratio", "Win Rate"]
    for ax, m, t in zip(axes, metrics, titles):
        vals = [baseline[m], lgbm_sim[m]]
        ax.bar(["Baseline", "LGBM"], vals, color=["#888", "#2196F3"])
        ax.set_title(t)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Baseline vs LGBM", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=120)
    plt.close(fig)

    # 2. Feature importance
    fig, ax = plt.subplots(figsize=(8, 5))
    top = fi.head(10)
    ax.barh(top["feature"][::-1], top["importance"][::-1], color="#4CAF50")
    ax.set_title("Top 10 Feature Importance")
    fig.tight_layout()
    fig.savefig(out / "feature_importance.png", dpi=120)
    plt.close(fig)

    # 3. Regime breakdown
    fig, ax = plt.subplots(figsize=(8, 4))
    regimes = regime_bl["regime"].values
    x = np.arange(len(regimes))
    w = 0.35
    ax.bar(x - w/2, regime_bl["apy_pct"].values, w, label="Baseline", color="#888")
    ax.bar(x + w/2, regime_lgbm["apy_pct"].values, w, label="LGBM", color="#2196F3")
    ax.set_xticks(x)
    ax.set_xticklabels(regimes)
    ax.set_ylabel("APY (%)")
    ax.set_title("APY by Market Regime")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "regime_apy.png", dpi=120)
    plt.close(fig)

    # 4. Cost Monte Carlo distribution
    if stat_val.get("cost_mc"):
        mc = stat_val["cost_mc"]
        fig, ax = plt.subplots(figsize=(6, 4))
        info = f"Mean: {mc['apy_mean']:.1f}%\n5th: {mc['apy_5th']:.1f}%\n95th: {mc['apy_95th']:.1f}%"
        ax.text(0.5, 0.5, info, transform=ax.transAxes, fontsize=14,
                va="center", ha="center", family="monospace",
                bbox=dict(boxstyle="round", facecolor="#e3f2fd"))
        ax.set_title("Cost Monte Carlo (±30%) — APY Range")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out / "cost_mc.png", dpi=120)
        plt.close(fig)

    # 5. Cumulative PnL curve
    trade_df = lgbm_sim.get("trade_df", pd.DataFrame())
    if not trade_df.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        cum_pnl = trade_df.sort_values("exit_ts")["net_pct"].cumsum()
        ax.plot(cum_pnl.values, color="#2196F3", linewidth=1.5)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Cumulative Net (%)")
        ax.set_title("LGBM Equity Curve (Val)")
        fig.tight_layout()
        fig.savefig(out / "equity_curve.png", dpi=120)
        plt.close(fig)

    print(f"[charts] Saved PNGs → {out}/")


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

    # Full universe splits (all rows, not just entry candidates) — for simulation
    full_train_df = ds["full_train_df"]
    full_val_df   = ds["full_val_df"]

    # Add regime labels to full dfs
    combined_full = label_market_regime(pd.concat([full_train_df, full_val_df], ignore_index=True))
    full_train_ts = set(full_train_df["timestamp"].unique())
    full_val_ts   = set(full_val_df["timestamp"].unique())
    full_train_df = combined_full[combined_full["timestamp"].isin(full_train_ts)].copy()
    full_val_df   = combined_full[combined_full["timestamp"].isin(full_val_ts)].copy()

    all_results: dict[str, dict] = {}

    # ── 3. Baseline v2 (corrected simulation) ──
    print("\n[3/5] Rule-based baseline (v2 — correct simulation)...")
    baseline_train = run_baseline_v2(full_train_df, "train", cost_tier=cost_tier)
    baseline_val   = run_baseline_v2(full_val_df,   "val",   cost_tier=cost_tier)
    all_results["baseline"] = {"sim_metrics": baseline_val, "auc": "n/a"}
    print(f"  Val APY:    {baseline_val['apy_pct']:.4f}%")
    print(f"  Val Sharpe: {baseline_val['sharpe']:.4f}")
    print(f"  Val trades: {baseline_val['n_trades']}")
    print(f"  Win rate:   {baseline_val['win_rate']:.1%}")
    print(f"  Noise rate: {baseline_val['noise_rate']:.1%}")

    # ── 4. LGBM (train only on entry candidates with label v3) ──
    print("\n[4/5] LightGBM (trained on label v3, threshold on train holdout)...")
    lgbm_model = FundingRateLGBM()
    lgbm_model.fit(X_train, y_train, train_df, X_val, y_val, run_cv=True)

    # Build LGBM signal on full universe: ML predicts on entry candidates, rest = 0
    def build_full_signal(model, X_entry, entry_df, full_df):
        """Map ML signal from entry-only rows back to full universe."""
        sig = np.zeros(len(full_df), dtype=int)
        # Get combined signal for entry candidates
        entry_sig = model.combined_signal(X_entry, entry_df)
        # Map back using (timestamp, symbol) as key
        entry_keys = set(zip(entry_df["timestamp"], entry_df["symbol"]))
        full_keys = list(zip(full_df["timestamp"], full_df["symbol"]))
        # Build lookup: (ts, sym) → signal value
        entry_lookup = {}
        for i, (ts, sym) in enumerate(zip(entry_df["timestamp"], entry_df["symbol"])):
            entry_lookup[(ts, sym)] = entry_sig[i]
        for i, key in enumerate(full_keys):
            if key in entry_lookup:
                sig[i] = entry_lookup[key]
        return sig

    lgbm_train_sig = build_full_signal(lgbm_model, X_train, train_df, full_train_df)
    lgbm_train_sim = simulate_bot(full_train_df, lgbm_train_sig, cost_tier=cost_tier, strategy_name="lgbm_train")

    lgbm_val_sig = build_full_signal(lgbm_model, X_val, val_df, full_val_df)
    lgbm_val_sim = simulate_bot(full_val_df, lgbm_val_sig, cost_tier=cost_tier, strategy_name="lgbm_val")

    # Compute AUC/AP on entry candidates only (where we have labels)
    from sklearn.metrics import average_precision_score, roc_auc_score
    proba = lgbm_model.predict_proba(X_val)
    lgbm_auc = roc_auc_score(y_val, proba) if len(np.unique(y_val)) > 1 else float("nan")
    lgbm_ap  = average_precision_score(y_val, proba)
    fi = lgbm_model.feature_importance_df(ds["feature_names"])

    print(f"\n{'='*55}")
    print(f"[LGBM VAL]")
    print(f"  AUC-ROC: {lgbm_auc:.4f} | Avg Precision: {lgbm_ap:.4f}")
    print(f"  Threshold: {lgbm_model.threshold:.4f}")
    print(f"  APY:       {lgbm_val_sim['apy_pct']:.4f}% | Sharpe: {lgbm_val_sim['sharpe']:.4f}")
    print(f"  Win rate:  {lgbm_val_sim['win_rate']:.1%} | Noise rate: {lgbm_val_sim['noise_rate']:.1%}")
    print(f"  Trades:    {lgbm_val_sim['n_trades']} | Utilization: {lgbm_val_sim['utilization']:.1%}")

    d_apy    = lgbm_val_sim["apy_pct"]    - baseline_val.get("apy_pct", 0)
    d_sharpe = lgbm_val_sim["sharpe"]      - baseline_val.get("sharpe", 0)
    d_noise  = lgbm_val_sim["noise_rate"]  - baseline_val.get("noise_rate", 0)
    improved = d_apy > 0 and d_sharpe > 0
    print(f"\n  vs Baseline → ΔAPY: {d_apy:+.4f}% | ΔSharpe: {d_sharpe:+.4f} | ΔNoise: {d_noise:+.4f}")
    print(f"  → {'✓ IMPROVEMENT' if improved else '✗ NO IMPROVEMENT'}")
    print(f"\n  Top features:")
    print(fi.head(6).to_string(index=False))

    lgbm_eval = {
        "sim_metrics":        lgbm_val_sim,
        "auc":                lgbm_auc,
        "avg_precision":      lgbm_ap,
        "feature_importance": fi,
        "combined_signal":    lgbm_val_sig,
    }
    all_results["lgbm"] = lgbm_eval
    lgbm_warnings = run_sanity_checks(lgbm_train_sim, lgbm_val_sim, "LGBM")

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
    baseline_sig_val = rule_signal(full_val_df)
    regime_bl   = regime_breakdown(full_val_df, baseline_sig_val, "baseline")
    regime_lgbm = regime_breakdown(full_val_df, lgbm_val_sig,     "lgbm")
    print("\nBaseline by regime:\n", regime_bl.to_string(index=False))
    print("\nLGBM by regime:\n",     regime_lgbm.to_string(index=False))

    # ── 6. Statistical validation ──
    from statistical_validation import run_full_statistical_validation
    stat_val = run_full_statistical_validation(
        df=full_val_df,
        signal=lgbm_val_sig,
        sim_metrics=lgbm_val_sim,
        cost_tier=cost_tier,
    )

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
        "statistical_validation": stat_val,
        "elapsed_seconds":   round(elapsed, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # ── Charts ──
    _generate_charts(out, baseline_val, lgbm_val_sim, stat_val, regime_bl, regime_lgbm, fi)

    # ── ONNX export ──
    lgbm_model.export_onnx(
        path=str(out / "lgbm_model.onnx"),
        feature_names=ds["feature_names"],
        scaler=ds["scaler"],
    )

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
    full_test_df = ds["full_test_df"]

    print("\n" + "!" * 55)
    print("FINAL TEST SET EVALUATION (2025) — DO NOT RETUNE")
    print("!" * 55)

    # Full universe for simulation
    full_test_df = label_market_regime(full_test_df)

    baseline_test = run_baseline_v2(full_test_df, "test", cost_tier=cost_tier)

    # Map LGBM signal to full universe
    entry_sig = lgbm_model.combined_signal(X_test, test_df)
    sig = np.zeros(len(full_test_df), dtype=int)
    entry_lookup = {}
    for i, (ts, sym) in enumerate(zip(test_df["timestamp"], test_df["symbol"])):
        entry_lookup[(ts, sym)] = entry_sig[i]
    for i, (ts, sym) in enumerate(zip(full_test_df["timestamp"], full_test_df["symbol"])):
        if (ts, sym) in entry_lookup:
            sig[i] = entry_lookup[(ts, sym)]

    lgbm_test_sim = simulate_bot(full_test_df, sig, cost_tier=cost_tier, strategy_name="lgbm_test")

    delta_apy    = lgbm_test_sim["apy_pct"] - baseline_test["apy_pct"]
    delta_sharpe = lgbm_test_sim["sharpe"]  - baseline_test["sharpe"]
    print(f"Baseline — APY: {baseline_test['apy_pct']:.4f}% | Sharpe: {baseline_test['sharpe']:.4f} | Trades: {baseline_test['n_trades']} | WR: {baseline_test['win_rate']:.1%}")
    print(f"LGBM     — APY: {lgbm_test_sim['apy_pct']:.4f}% | Sharpe: {lgbm_test_sim['sharpe']:.4f} | Trades: {lgbm_test_sim['n_trades']} | WR: {lgbm_test_sim['win_rate']:.1%}")
    print(f"Delta    — APY: {delta_apy:+.4f}% | Sharpe: {delta_sharpe:+.4f}")

    if delta_apy > 0 and delta_sharpe > 0:
        print("\n✓ TEST PASSED — improvement holds on unseen data. Safe to deploy.")
    else:
        print("\n✗ TEST FAILED — improvement does NOT generalize. Do NOT deploy.")

    return {"baseline": baseline_test, "lgbm": lgbm_test_sim}

