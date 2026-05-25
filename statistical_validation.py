"""
statistical_validation.py
--------------------------
Statistical validation suite:
    1. Bootstrap CI — resample trades, 95% CI for APY/Sharpe
    2. Permutation test — shuffle signal, p-value for improvement over random
    3. Cost Monte Carlo — vary costs ±30%, stress test profitability
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from simulation_v2 import simulate_bot, SETTLEMENTS_PER_YEAR


def bootstrap_ci(
    trade_df: pd.DataFrame,
    n_periods: int,
    n_boot: int = 1000,
    ci: float = 0.95,
) -> dict:
    """Bootstrap confidence intervals for APY and Sharpe from trade results."""
    if trade_df.empty:
        return {"apy_lo": 0, "apy_hi": 0, "sharpe_lo": 0, "sharpe_hi": 0}

    nets = trade_df["net_pct"].values
    n_trades = len(nets)
    alpha = (1 - ci) / 2

    apys, sharpes = [], []
    for _ in range(n_boot):
        sample = np.random.choice(nets, size=n_trades, replace=True)
        total = sample.sum()
        apy = total / n_periods * SETTLEMENTS_PER_YEAR
        std = sample.std()
        sharpe = (sample.mean() / std * np.sqrt(SETTLEMENTS_PER_YEAR)) if std > 0 else 0
        apys.append(apy)
        sharpes.append(sharpe)

    return {
        "apy_lo": float(np.percentile(apys, alpha * 100)),
        "apy_hi": float(np.percentile(apys, (1 - alpha) * 100)),
        "apy_median": float(np.median(apys)),
        "sharpe_lo": float(np.percentile(sharpes, alpha * 100)),
        "sharpe_hi": float(np.percentile(sharpes, (1 - alpha) * 100)),
        "sharpe_median": float(np.median(sharpes)),
    }


def permutation_test(
    df: pd.DataFrame,
    signal: np.ndarray,
    cost_tier: str = "mid",
    n_perms: int = 500,
    metric: str = "apy_pct",
) -> dict:
    """Shuffle signal among entry candidates, compute p-value."""
    real_result = simulate_bot(df, signal, cost_tier=cost_tier, strategy_name="perm_real")
    real_metric = real_result[metric]

    count_ge = 0
    for _ in range(n_perms):
        shuffled = np.random.permutation(signal)
        perm_result = simulate_bot(df, shuffled, cost_tier=cost_tier, strategy_name="perm_shuf")
        if perm_result[metric] >= real_metric:
            count_ge += 1

    p_value = (count_ge + 1) / (n_perms + 1)
    return {
        "real_metric": real_metric,
        "p_value": p_value,
        "n_perms": n_perms,
        "significant": p_value < 0.05,
    }


def cost_monte_carlo(
    df: pd.DataFrame,
    signal: np.ndarray,
    cost_tier: str = "mid",
    n_sims: int = 500,
    cost_noise_pct: float = 0.30,
) -> dict:
    """
    Vary costs ±cost_noise_pct per simulation.
    Each sim applies a random multiplier ~ U(1-noise, 1+noise) to all costs.
    """
    from simulation_v2 import ACTUAL_COSTS, COST_TIERS, _get_cost, simulate_bot as _sim_bot
    import simulation_v2 as sim_mod

    # Save originals
    orig_actual = sim_mod.ACTUAL_COSTS.copy()
    orig_tiers = sim_mod.COST_TIERS.copy()

    apys, sharpes, win_rates = [], [], []

    for _ in range(n_sims):
        mult = np.random.uniform(1 - cost_noise_pct, 1 + cost_noise_pct)
        # Monkey-patch costs
        sim_mod.ACTUAL_COSTS = {k: v * mult for k, v in orig_actual.items()}
        sim_mod.COST_TIERS = {k: v * mult for k, v in orig_tiers.items()}

        result = simulate_bot(df, signal, cost_tier=cost_tier, strategy_name="mc_cost")
        apys.append(result["apy_pct"])
        sharpes.append(result["sharpe"])
        win_rates.append(result["win_rate"])

    # Restore
    sim_mod.ACTUAL_COSTS = orig_actual
    sim_mod.COST_TIERS = orig_tiers

    return {
        "apy_mean": float(np.mean(apys)),
        "apy_5th": float(np.percentile(apys, 5)),
        "apy_95th": float(np.percentile(apys, 95)),
        "sharpe_mean": float(np.mean(sharpes)),
        "sharpe_5th": float(np.percentile(sharpes, 5)),
        "win_rate_5th": float(np.percentile(win_rates, 5)),
        "pct_profitable": float(np.mean(np.array(apys) > 0) * 100),
    }


def run_full_statistical_validation(
    df: pd.DataFrame,
    signal: np.ndarray,
    sim_metrics: dict,
    cost_tier: str = "mid",
) -> dict:
    """Run all validation tests and print summary."""
    print("\n" + "=" * 55)
    print("[STATISTICAL VALIDATION]")
    print("=" * 55)

    # 1. Bootstrap CI
    print("\n[1] Bootstrap CI (1000 resamples)...")
    trade_df = sim_metrics.get("trade_df", pd.DataFrame())
    n_periods = sim_metrics.get("n_periods", 1)
    boot = bootstrap_ci(trade_df, n_periods)
    print(f"  APY 95% CI:    [{boot['apy_lo']:.1f}%, {boot['apy_hi']:.1f}%]")
    print(f"  Sharpe 95% CI: [{boot['sharpe_lo']:.2f}, {boot['sharpe_hi']:.2f}]")

    # 2. Permutation test
    print("\n[2] Permutation test (500 shuffles)...")
    perm = permutation_test(df, signal, cost_tier=cost_tier)
    sig_str = "✓ SIGNIFICANT" if perm["significant"] else "✗ NOT SIGNIFICANT"
    print(f"  p-value: {perm['p_value']:.4f} — {sig_str}")

    # 3. Cost Monte Carlo
    print("\n[3] Cost Monte Carlo (±30%, 500 sims)...")
    mc = cost_monte_carlo(df, signal, cost_tier=cost_tier)
    print(f"  APY range:     [{mc['apy_5th']:.1f}%, {mc['apy_95th']:.1f}%]")
    print(f"  Sharpe (5th):  {mc['sharpe_5th']:.2f}")
    print(f"  Win rate (5th): {mc['win_rate_5th']:.1%}")
    print(f"  % sims profitable: {mc['pct_profitable']:.0f}%")

    return {"bootstrap": boot, "permutation": perm, "cost_mc": mc}
