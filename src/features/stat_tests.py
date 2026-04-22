"""Kiểm định thống kê: Chi-square, ANOVA F-test, VIF."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor


def chi_square_test(df: pd.DataFrame, cat_cols: list[str], target: str) -> pd.DataFrame:
    rows = []
    for c in cat_cols:
        tab = pd.crosstab(df[c], df[target])
        if tab.shape[0] < 2 or tab.shape[1] < 2:
            continue
        chi2, p, dof, _ = stats.chi2_contingency(tab)
        rows.append({"feature": c, "chi2": chi2, "p_value": p, "dof": dof})
    return pd.DataFrame(rows).sort_values("p_value")


def anova_f_test(df: pd.DataFrame, num_cols: list[str], target: str) -> pd.DataFrame:
    rows = []
    groups = df[target].unique()
    for c in num_cols:
        samples = [df.loc[df[target] == g, c].dropna() for g in groups]
        samples = [s for s in samples if len(s) > 1]
        if len(samples) < 2:
            continue
        f, p = stats.f_oneway(*samples)
        rows.append({"feature": c, "f_stat": f, "p_value": p})
    return pd.DataFrame(rows).sort_values("p_value")


def compute_vif(df: pd.DataFrame, num_cols: list[str]) -> pd.DataFrame:
    X = df[num_cols].dropna().astype(float).values
    rows = []
    for i, c in enumerate(num_cols):
        try:
            v = variance_inflation_factor(X, i)
        except Exception:
            v = np.nan
        rows.append({"feature": c, "vif": v})
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def drop_high_vif(df: pd.DataFrame, num_cols: list[str], threshold: float = 5.0) -> list[str]:
    cols = list(num_cols)
    while True:
        vif = compute_vif(df, cols)
        worst = vif.iloc[0]
        if worst["vif"] <= threshold or len(cols) <= 2:
            return cols
        cols.remove(worst["feature"])
