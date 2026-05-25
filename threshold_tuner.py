"""
threshold_tuner.py
------------------
Threshold tuning entirely within the training set via nested CV.

Problem with original approach:
    model.threshold = find_optimal_threshold(y_val, val_proba)
    → val set information leaked into model selection.
    → reported val metrics are optimistically biased.

Fix:
    Tune threshold on a held-out portion of TRAIN using the same
    walk-forward scheme. Val set is never touched during threshold selection.

Strategy:
    Use the last fold of walk-forward CV as the threshold-tuning holdout.
    This matches the temporal structure: tune on the most recent training data,
    which is most representative of near-future behavior.

Objective:
    Maximize recall (not F1) subject to precision >= min_precision.
    Rationale: user goal is "don't reject profitable trades" → high recall
    on label=1 (profit) is the priority. Accepting some noise is tolerable.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve


def tune_threshold_on_train(
    model,
    X_train_last_fold: np.ndarray,
    y_train_last_fold: np.ndarray,
    min_precision: float = 0.45,
    min_recall: float = 0.60,
) -> float:
    """
    Tune classification threshold on held-out train fold.

    Strategy:
        1. Among thresholds where precision >= min_precision,
           pick the one with highest recall.
        2. If no threshold meets precision constraint,
           fall back to threshold at min_recall.
        3. If no threshold meets recall constraint either,
           fall back to 0.3 (permissive — errs toward keeping trades).

    Parameters
    ----------
    model                  : fitted model with predict_proba(X)
    X_train_last_fold      : features for threshold-tuning holdout
    y_train_last_fold      : labels for threshold-tuning holdout
    min_precision          : minimum acceptable precision (default 0.45)
    min_recall             : minimum acceptable recall (default 0.60)

    Returns
    -------
    float threshold in [0, 1]
    """
    if len(np.unique(y_train_last_fold)) < 2:
        print("[threshold] Single class in holdout — defaulting to 0.4")
        return 0.4

    proba = model.predict_proba(X_train_last_fold)
    if proba.ndim == 2:
        proba = proba[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_train_last_fold, proba)
    # precision_recall_curve returns n+1 points; thresholds has n points
    precision = precision[:-1]
    recall = recall[:-1]

    # Objective: high recall on profitable trades (user goal: don't reject profit)
    # Constraint: precision >= min_precision (don't approve everything)
    precision_ok = precision >= min_precision

    if precision_ok.any():
        # Among precision-constrained thresholds, maximize recall
        recall_constrained = np.where(precision_ok, recall, -np.inf)
        best_idx = int(np.argmax(recall_constrained))
        chosen = float(thresholds[best_idx])
        achieved_precision = float(precision[best_idx])
        achieved_recall = float(recall[best_idx])
        print(f"[threshold] Precision-constrained tuning: "
              f"threshold={chosen:.4f} | precision={achieved_precision:.3f} | recall={achieved_recall:.3f}")
        return chosen

    # Fallback: find threshold closest to min_recall
    recall_gap = np.abs(recall - min_recall)
    best_idx = int(np.argmin(recall_gap))
    chosen = float(thresholds[best_idx])
    print(f"[threshold] Recall-target fallback: threshold={chosen:.4f} "
          f"| recall={recall[best_idx]:.3f} (precision={precision[best_idx]:.3f})")
    return chosen


def extract_last_cv_fold(
    train_df,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_folds: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract the last walk-forward fold as threshold-tuning holdout.

    Returns
    -------
    (X_tune_train, y_tune_train, X_tune_val, y_tune_val)
        tune_train: first (n_folds-1)/n_folds of timestamps
        tune_val  : last 1/n_folds of timestamps
    """
    import pandas as pd

    timestamps = np.sort(train_df["timestamp"].unique())
    n_ts = len(timestamps)
    fold_size = n_ts // (n_folds + 1)

    # Last fold: train on everything before last fold_size, val on last fold_size
    split_ts = timestamps[fold_size * n_folds]

    train_mask = (train_df["timestamp"] < split_ts).values
    val_mask   = (train_df["timestamp"] >= split_ts).values

    return X_train[train_mask], y_train[train_mask], X_train[val_mask], y_train[val_mask]
