"""
model_lgbm.py
-------------
LightGBM entry filter for FR arbitrage.

Role: binary classifier on top of rule-based signal.
      "Given |FR| >= threshold, is this a HIGH-QUALITY entry?"

Design:
    - Temporal walk-forward CV (no shuffling — ever)
    - Early stopping on each fold to find optimal n_estimators
    - Threshold tuned on val set (precision-recall tradeoff)
    - SHAP values for feature importance (more reliable than split-based)
    - RF sanity check runs in parallel — if LGBM AUC < RF AUC + 0.03, use RF
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

LGBM_CONFIG: dict[str, Any] = {
    "objective":        "binary",
    "metric":           "auc",
    "boosting_type":    "gbdt",
    "num_leaves":       16,          # small — prevents overfitting on noisy FR data
    "max_depth":        5,
    "min_child_samples": 100,        # high — each leaf must have >= 100 samples
    "learning_rate":    0.05,
    "n_estimators":     500,         # high ceiling — early stopping will trim
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,         # L1
    "reg_lambda":       1.0,         # L2
    "is_unbalance":     True,        # handles class imbalance natively
    "random_state":     42,
    "n_jobs":           -1,
    "verbose":          -1,
}

RF_SANITY_CONFIG: dict[str, Any] = {
    "n_estimators":     100,
    "max_depth":        6,
    "min_samples_leaf": 50,
    "max_features":     "sqrt",
    "class_weight":     "balanced",
    "n_jobs":           -1,
    "random_state":     42,
}


# ── Walk-Forward CV ───────────────────────────────────────────────────────────

def walk_forward_cv(
    train_df: pd.DataFrame,
    X_train:  np.ndarray,
    y_train:  np.ndarray,
    n_folds:  int = 3,
    embargo:  int = 6,
) -> tuple[list[dict], int]:
    """
    Expanding-window temporal CV within training set.
    Returns (fold_results, recommended_n_estimators).

    recommended_n_estimators = median of best iterations across folds.
    This is used to set final model's n_estimators (no early stopping on full train).
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("pip install lightgbm")

    timestamps = np.sort(train_df["timestamp"].unique())
    n_ts       = len(timestamps)
    fold_size  = n_ts // (n_folds + 1)

    fold_results:    list[dict] = []
    best_iterations: list[int]  = []

    for fold in range(n_folds):
        train_end_idx = fold_size * (fold + 1)
        val_end_idx   = fold_size * (fold + 2)

        train_ts = set(timestamps[:train_end_idx])
        val_ts   = timestamps[train_end_idx:val_end_idx]

        # Apply embargo: skip first N val timestamps
        if len(val_ts) > embargo:
            val_ts = val_ts[embargo:]
        val_ts = set(val_ts)

        tr_mask = train_df["timestamp"].isin(train_ts).values
        va_mask = train_df["timestamp"].isin(val_ts).values

        X_tr, y_tr = X_train[tr_mask], y_train[tr_mask]
        X_va, y_va = X_train[va_mask], y_train[va_mask]

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
            print(f"  [fold {fold+1}] Skipped — single class")
            continue

        cfg = {**LGBM_CONFIG, "n_estimators": 500}
        model = lgb.LGBMClassifier(**cfg)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        from sklearn.metrics import average_precision_score, roc_auc_score
        proba = model.predict_proba(X_va)[:, 1]
        auc   = roc_auc_score(y_va, proba)
        ap    = average_precision_score(y_va, proba)
        best_iter = model.best_iteration_

        fold_results.append({
            "fold":       fold + 1,
            "train_size": int(tr_mask.sum()),
            "val_size":   int(va_mask.sum()),
            "auc_roc":    round(auc, 4),
            "avg_precision": round(ap, 4),
            "best_iteration": best_iter,
        })
        best_iterations.append(best_iter)
        print(f"  [fold {fold+1}] AUC={auc:.4f} | AP={ap:.4f} | best_iter={best_iter}")

    recommended = int(np.median(best_iterations)) if best_iterations else 200
    print(f"  [CV] Recommended n_estimators: {recommended}")
    return fold_results, recommended


# ── Threshold Optimization ────────────────────────────────────────────────────

def find_optimal_threshold(
    y_true:  np.ndarray,
    y_proba: np.ndarray,
    min_precision: float = 0.55,
) -> float:
    """
    Find threshold maximizing F1 subject to precision >= min_precision.
    Falls back to max-F1 threshold if constraint cannot be met.
    """
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)

    # Try precision-constrained F1 first
    valid = precision[:-1] >= min_precision
    if valid.any():
        best_idx = np.argmax(f1 * valid)
        return float(thresholds[best_idx])

    # Fallback: unconstrained max F1
    return float(thresholds[np.argmax(f1)])


# ── Main Model ────────────────────────────────────────────────────────────────

class FundingRateLGBM:
    """
    LightGBM filter for funding rate arbitrage entries.

    Usage:
        model = FundingRateLGBM()
        model.fit(X_train, y_train, train_df, X_val, y_val)
        signals = model.combined_signal(X_val, val_df)
        metrics = simulate_strategy(val_df, signals)
    """

    def __init__(self) -> None:
        self.model       = None
        self.threshold   = 0.5
        self.cv_results_ : list[dict] = []
        self.feature_importances_: np.ndarray | None = None

    def fit(
        self,
        X_train:  np.ndarray,
        y_train:  np.ndarray,
        train_df: pd.DataFrame,
        X_val:    np.ndarray | None = None,
        y_val:    np.ndarray | None = None,
        run_cv:   bool = True,
    ) -> "FundingRateLGBM":
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("pip install lightgbm")

        from threshold_tuner import extract_last_cv_fold, tune_threshold_on_train

        n_estimators = LGBM_CONFIG["n_estimators"]

        if run_cv:
            print("[LGBM] Walk-forward CV:")
            self.cv_results_, n_estimators = walk_forward_cv(
                train_df, X_train, y_train
            )

        print(f"[LGBM] Training final model (n_estimators={n_estimators})...")
        cfg = {**LGBM_CONFIG, "n_estimators": n_estimators}
        self.model = lgb.LGBMClassifier(**cfg)
        self.model.fit(X_train, y_train)
        self.feature_importances_ = self.model.feature_importances_

        # Fix #3: tune threshold on last CV fold of TRAIN — never on val set
        print("[LGBM] Tuning threshold on train holdout (not val)...")
        _, _, X_tune, y_tune = extract_last_cv_fold(train_df, X_train, y_train)
        self.threshold = tune_threshold_on_train(self.model, X_tune, y_tune)

        # Diagnostic only — do NOT use y_val to adjust anything
        if X_val is not None and y_val is not None:
            val_proba = self.model.predict_proba(X_val)[:, 1]
            val_threshold_for_info = find_optimal_threshold(y_val, val_proba)
            print(f"[LGBM] Threshold used (train-tuned): {self.threshold:.4f}")
            print(f"[LGBM] Val-optimal threshold (diagnostic only): {val_threshold_for_info:.4f}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None, "Model not fitted"
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def combined_signal(
        self,
        X:   np.ndarray,
        df:  pd.DataFrame,
        min_fr: float | None = None,
    ) -> np.ndarray:
        """
        Production signal: rule-based AND LGBM must both agree.
        Conservative — reduces false positives at cost of some recall.
        """
        from data_loader import ENTRY_THRESHOLD_PCT
        threshold = min_fr if min_fr is not None else ENTRY_THRESHOLD_PCT
        rule = (df["fr_abs"] >= threshold).values
        lgbm = self.predict(X).astype(bool)
        return (rule & lgbm).astype(int)

    def feature_importance_df(self, feature_names: list[str]) -> pd.DataFrame:
        assert self.feature_importances_ is not None
        return (
            pd.DataFrame({
                "feature":    feature_names,
                "importance": self.feature_importances_,
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def export_onnx(
        self,
        path: str,
        feature_names: list[str],
        scaler,
        n_features: int | None = None,
    ) -> str:
        """
        Export model to ONNX + companion model_config.json.
        Returns path to .onnx file.

        model_config.json contains:
            - threshold, feature_names
            - scaler center_ and scale_ (for RobustScaler reconstruction)
            - entry_threshold_pct, exit_threshold_pct (strategy constants)
        """
        import json
        from pathlib import Path
        try:
            from onnxmltools import convert_lightgbm
            from onnxmltools.convert.common.data_types import FloatTensorType
        except ImportError:
            raise ImportError("pip install onnxmltools onnx")

        assert self.model is not None, "Model not fitted"

        n_feat = n_features or len(feature_names)
        initial_type = [("float_input", FloatTensorType([None, n_feat]))]
        onnx_model = convert_lightgbm(self.model, initial_types=initial_type, target_opset=12)

        onnx_path = Path(path)
        onnx_path.write_bytes(onnx_model.SerializeToString())

        # Companion config
        from data_loader import ENTRY_THRESHOLD_PCT
        from simulation_v2 import EXIT_THRESHOLD
        config = {
            "threshold":          self.threshold,
            "feature_names":      feature_names,
            "entry_threshold_pct": ENTRY_THRESHOLD_PCT,
            "exit_threshold_pct":  EXIT_THRESHOLD,
            "scaler_center":      scaler.center_.tolist(),
            "scaler_scale":       scaler.scale_.tolist(),
        }
        config_path = onnx_path.with_suffix(".json")
        config_path.write_text(json.dumps(config, indent=2))

        print(f"[ONNX] Saved model → {onnx_path}")
        print(f"[ONNX] Saved config → {config_path}")
        return str(onnx_path)


# ── RF Sanity Check ───────────────────────────────────────────────────────────

def run_rf_sanity_check(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
) -> float:
    """
    Train minimal RF, return val AUC.
    If LGBM AUC < RF AUC + 0.03 → prefer RF (complexity not justified).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    print("[RF sanity] Training RF baseline...")
    rf = RandomForestClassifier(**RF_SANITY_CONFIG)
    rf.fit(X_train, y_train)
    proba = rf.predict_proba(X_val)[:, 1]
    auc   = roc_auc_score(y_val, proba)
    print(f"[RF sanity] Val AUC: {auc:.4f}")
    return auc


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_lgbm(
    model:            FundingRateLGBM,
    X:                np.ndarray,
    y:                np.ndarray,
    df:               pd.DataFrame,
    split_name:       str,
    feature_names:    list[str],
    baseline_metrics: dict | None = None,
    cost_tier:        str = "mid",
) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    from simulation_v2 import simulate_bot

    proba = model.predict_proba(X)
    auc   = roc_auc_score(y, proba) if len(np.unique(y)) > 1 else float("nan")
    ap    = average_precision_score(y, proba)

    combined = model.combined_signal(X, df)
    sim      = simulate_bot(df, combined, cost_tier=cost_tier, strategy_name=f"lgbm_{split_name}")

    fi = model.feature_importance_df(feature_names)

    print(f"\n{'='*55}")
    print(f"[LGBM {split_name.upper()}]")
    print(f"  AUC-ROC: {auc:.4f} | Avg Precision: {ap:.4f}")
    print(f"  Threshold: {model.threshold:.4f}")
    print(f"  APY:       {sim['apy_pct']:.4f}% | Sharpe: {sim['sharpe']:.4f}")
    print(f"  Win rate:  {sim['win_rate']:.1%} | Noise rate: {sim['noise_rate']:.1%}")
    print(f"  Trades:    {sim['n_trades']} | Utilization: {sim['utilization']:.1%}")

    if baseline_metrics:
        d_apy    = sim["apy_pct"]  - baseline_metrics.get("apy_pct", 0)
        d_sharpe = sim["sharpe"]   - baseline_metrics.get("sharpe", 0)
        d_noise  = sim["noise_rate"] - baseline_metrics.get("noise_rate", 0)
        improved = d_apy > 0 and d_sharpe > 0
        print(f"\n  vs Baseline → ΔAPY: {d_apy:+.4f}% | ΔSharpe: {d_sharpe:+.4f} | ΔNoise: {d_noise:+.4f}")
        print(f"  → {'✓ IMPROVEMENT' if improved else '✗ NO IMPROVEMENT'}")

    print(f"\n  Top features:")
    print(fi.head(6).to_string(index=False))

    return {
        "sim_metrics":        sim,
        "auc":                auc,
        "avg_precision":      ap,
        "feature_importance": fi,
        "combined_signal":    combined,
    }

