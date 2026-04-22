"""Các mô hình baseline: Logistic Regression, Random Forest, XGBoost, CatBoost."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np


def make_logistic_regression(**kwargs) -> Any:
    from sklearn.linear_model import LogisticRegression

    defaults = dict(penalty="l2", class_weight="balanced", solver="lbfgs", max_iter=1000, n_jobs=-1)
    defaults.update(kwargs)
    return LogisticRegression(**defaults)


def make_random_forest(**kwargs) -> Any:
    from sklearn.ensemble import RandomForestClassifier

    defaults = dict(n_estimators=500, class_weight="balanced", n_jobs=-1, random_state=42)
    defaults.update(kwargs)
    return RandomForestClassifier(**defaults)


def make_xgboost(scale_pos_weight: float = 1.0, **kwargs) -> Any:
    import xgboost as xgb

    defaults = dict(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    defaults.update(kwargs)
    return xgb.XGBClassifier(**defaults)


def make_catboost(**kwargs) -> Any:
    from catboost import CatBoostClassifier

    defaults = dict(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        auto_class_weights="Balanced",
        eval_metric="PRAUC",
        random_seed=42,
        verbose=0,
    )
    defaults.update(kwargs)
    return CatBoostClassifier(**defaults)


def compute_scale_pos_weight(y: np.ndarray) -> float:
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    return float(neg / max(pos, 1))
