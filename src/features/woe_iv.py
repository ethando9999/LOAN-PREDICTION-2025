"""Weight of Evidence (WoE) và Information Value (IV)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class WOEBinning:
    feature: str
    is_numeric: bool
    bins: list
    woe_map: dict
    iv: float


def _bin_numeric(s: pd.Series, n_bins: int = 10) -> pd.Series:
    return pd.qcut(s, q=n_bins, duplicates="drop")


def fit_woe(df: pd.DataFrame, feature: str, target: str, n_bins: int = 10) -> WOEBinning:
    x = df[feature]
    y = df[target].astype(int)
    is_numeric = pd.api.types.is_numeric_dtype(x) and x.nunique() > n_bins
    binned = _bin_numeric(x, n_bins) if is_numeric else x.astype("object").fillna("__NA__")

    tab = pd.crosstab(binned, y)
    if 0 not in tab.columns:
        tab[0] = 0
    if 1 not in tab.columns:
        tab[1] = 0

    total_good = max(tab[0].sum(), 1)
    total_bad = max(tab[1].sum(), 1)
    eps = 0.5
    good_rate = (tab[0] + eps) / total_good
    bad_rate = (tab[1] + eps) / total_bad
    woe = np.log(good_rate / bad_rate)
    iv = ((good_rate - bad_rate) * woe).sum()

    return WOEBinning(
        feature=feature,
        is_numeric=is_numeric,
        bins=list(tab.index),
        woe_map={str(k): float(v) for k, v in woe.items()},
        iv=float(iv),
    )


def iv_table(df: pd.DataFrame, features: list[str], target: str, n_bins: int = 10) -> pd.DataFrame:
    rows = []
    for f in features:
        try:
            b = fit_woe(df, f, target, n_bins)
            rows.append({"feature": f, "iv": b.iv, "is_numeric": b.is_numeric})
        except Exception as e:
            rows.append({"feature": f, "iv": np.nan, "is_numeric": None, "error": str(e)})
    return pd.DataFrame(rows).sort_values("iv", ascending=False)


def select_by_iv(iv_df: pd.DataFrame, min_iv: float = 0.02) -> list[str]:
    return iv_df.loc[iv_df["iv"] >= min_iv, "feature"].tolist()
