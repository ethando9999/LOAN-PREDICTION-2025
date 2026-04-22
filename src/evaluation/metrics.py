"""Metrics cho credit scoring: PR-AUC, ROC-AUC, Gini, F-beta, KS, Brier."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    fbeta_score,
    roc_auc_score,
)
from scipy import stats


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(stats.ks_2samp(pos, neg).statistic)


def gini(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return 2 * roc_auc_score(y_true, y_score) - 1


def evaluate(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "gini": float(gini(y_true, y_score)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2)),
        "ks": ks_statistic(y_true, y_score),
        "brier": float(brier_score_loss(y_true, y_score)),
    }


def leaderboard(results: dict[str, dict]) -> pd.DataFrame:
    return pd.DataFrame(results).T.sort_values("pr_auc", ascending=False)
