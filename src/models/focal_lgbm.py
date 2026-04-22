"""LightGBM với Focal Loss cho imbalanced classification."""
from __future__ import annotations

import numpy as np
import lightgbm as lgb


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def focal_loss_obj(alpha: float = 0.75, gamma: float = 2.0):
    """Focal loss custom objective cho LightGBM — trả về (grad, hess).

    FL = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def obj(preds: np.ndarray, dtrain: lgb.Dataset):
        y = dtrain.get_label()
        p = _sigmoid(preds)
        alpha_t = alpha * y + (1 - alpha) * (1 - y)
        p_t = p * y + (1 - p) * (1 - y)
        eps = 1e-8

        log_pt = np.log(np.clip(p_t, eps, 1.0))
        factor = (1 - p_t) ** gamma

        grad = alpha_t * factor * (gamma * p_t * log_pt + p_t - 1) * (2 * y - 1)
        hess = alpha_t * factor * (
            (1 - p_t) * (1 - p_t)
            - gamma * (1 - p_t) * p_t * log_pt
            + gamma * p_t * (gamma * p_t * log_pt + p_t - 1)
        )
        hess = np.abs(hess) + 1e-6
        return grad, hess

    return obj


def focal_eval(alpha: float = 0.75, gamma: float = 2.0):
    def feval(preds: np.ndarray, dtrain: lgb.Dataset):
        y = dtrain.get_label()
        p = _sigmoid(preds)
        alpha_t = alpha * y + (1 - alpha) * (1 - y)
        p_t = p * y + (1 - p) * (1 - y)
        eps = 1e-8
        loss = -alpha_t * (1 - p_t) ** gamma * np.log(np.clip(p_t, eps, 1.0))
        return "focal_loss", float(loss.mean()), False

    return feval


def train_focal_lgbm(
    X_train, y_train, X_val, y_val,
    params: dict | None = None,
    alpha: float = 0.75,
    gamma: float = 2.0,
    num_boost_round: int = 2000,
    early_stopping_rounds: int = 100,
) -> lgb.Booster:
    base = {
        "objective": focal_loss_obj(alpha, gamma),
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 20,
        "reg_lambda": 1.0,
        "verbose": -1,
    }
    if params:
        base.update(params)

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    booster = lgb.train(
        base,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        feval=focal_eval(alpha, gamma),
        callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(200)],
    )
    return booster


def predict_proba(booster: lgb.Booster, X) -> np.ndarray:
    raw = booster.predict(X, raw_score=True)
    return _sigmoid(raw)
