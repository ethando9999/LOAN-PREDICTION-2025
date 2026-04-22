"""Tiện ích SHAP cho tree-based models."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shap


def explain_tree(model, X, sample: int | None = 5000, seed: int = 42):
    if sample is not None and len(X) > sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=sample, replace=False)
        X_s = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
    else:
        X_s = X
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_s)
    return explainer, shap_values, X_s


def save_summary(shap_values, X, path: Path, plot_type: str = "dot") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(shap_values, X, plot_type=plot_type, show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_waterfall(explainer, shap_values, X, row: int, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exp = shap.Explanation(
        values=shap_values[row],
        base_values=explainer.expected_value
        if np.ndim(explainer.expected_value) == 0
        else explainer.expected_value[0],
        data=X.iloc[row].values if hasattr(X, "iloc") else X[row],
        feature_names=list(X.columns) if hasattr(X, "columns") else None,
    )
    shap.plots.waterfall(exp, show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
