"""Optuna hyperparameter tuning cho Focal-LightGBM."""
from __future__ import annotations

import optuna
import numpy as np
from sklearn.metrics import average_precision_score

from src.models.focal_lgbm import predict_proba, train_focal_lgbm


def tune(X_tr, y_tr, X_va, y_va, n_trials: int = 40, seed: int = 42) -> dict:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 0, 10),
        }
        alpha = trial.suggest_float("alpha", 0.25, 0.9)
        gamma = trial.suggest_float("gamma", 0.5, 4.0)

        booster = train_focal_lgbm(
            X_tr, y_tr, X_va, y_va,
            params=params, alpha=alpha, gamma=gamma,
            num_boost_round=1500, early_stopping_rounds=80,
        )
        y_score = predict_proba(booster, X_va)
        return average_precision_score(y_va, y_score)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = dict(study.best_params)
    print(f"  Best PR-AUC (val): {study.best_value:.4f}")
    return best
