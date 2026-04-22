"""Đọc và chia dataset thành train/val/test."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def load_raw(path: Path) -> pd.DataFrame:
    csvs = list(Path(path).glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {path}")
    if len(csvs) == 1:
        return pd.read_csv(csvs[0])
    return pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)


def stratified_split(
    df: pd.DataFrame,
    target: str,
    train_size: float = 0.70,
    val_size: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test_size = 1.0 - train_size - val_size
    train, tmp = train_test_split(
        df, test_size=(val_size + test_size), stratify=df[target], random_state=seed
    )
    rel = test_size / (val_size + test_size)
    val, test = train_test_split(tmp, test_size=rel, stratify=tmp[target], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
