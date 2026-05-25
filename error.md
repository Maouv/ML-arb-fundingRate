fix dari tadi ga ke fix2 aku ---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_57/719049693.py in <cell line: 0>()
     19 # Sekarang run seperti biasa
     20 from run_research import run_full_comparison
---> 21 results = run_full_comparison(
     22     data_dir=DATA_DIR,
     23     output_dir="/kaggle/working/research_output",

/kaggle/input/datasets/dafanaalfarizi/fr-arbitrage-src/run_research.py in run_full_comparison(data_dir, output_dir, feature_cols, cost_tier)
    131     # ── 3. Baseline v2 (corrected simulation) ──
    132     print("\n[3/5] Rule-based baseline (v2 — correct simulation)...")
--> 133     baseline_train = run_baseline_v2(train_df, "train", cost_tier=cost_tier)
    134     baseline_val   = run_baseline_v2(val_df,   "val",   cost_tier=cost_tier)
    135     all_results["baseline"] = {"sim_metrics": baseline_val, "auc": "n/a"}

/kaggle/input/datasets/dafanaalfarizi/fr-arbitrage-src/simulation_v2.py in run_baseline_v2(df, split_name, cost_tier)
    273 def run_baseline_v2(df: pd.DataFrame, split_name: str = "val", cost_tier: str = "mid") -> dict:
    274     """Run corrected rule-based baseline."""
--> 275     return simulate_bot(df, rule_signal(df), cost_tier=cost_tier, strategy_name=f"baseline_v2_{split_name}")
    276 
    277 

/kaggle/input/datasets/dafanaalfarizi/fr-arbitrage-src/simulation_v2.py in simulate_bot(df, entry_signals, cost_tier, max_slots, strategy_name)
    105 
    106         ts_fr[ts][sym] = fr
--> 107         if row._sig:  # noqa: SLF001 — itertuples field name
    108             ts_sig[ts].add(sym)
    109 

AttributeError: 'Pandas' object has no attribute '_sig'
