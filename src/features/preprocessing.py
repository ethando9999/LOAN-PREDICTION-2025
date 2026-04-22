"""Pipeline tiền xử lý: encoding, split, flip target (default=1)."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def prepare_target(df: pd.DataFrame, target: str, positive_is_default: bool = False) -> pd.DataFrame:
    """Quy ước: 1 = default (bad), 0 = paid back (good)."""
    out = df.copy()
    if not positive_is_default:
        out[target] = 1 - out[target].astype(int)
    else:
        out[target] = out[target].astype(int)
    return out


def encode_categoricals(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    cat_cols: list[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, OrdinalEncoder]:
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    train = train.copy()
    val = val.copy()
    test = test.copy()
    train[cat_cols] = enc.fit_transform(train[cat_cols])
    val[cat_cols] = enc.transform(val[cat_cols])
    test[cat_cols] = enc.transform(test[cat_cols])
    return train, val, test, enc


def save_processed(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out_dir / "train.parquet", index=False)
    val.to_parquet(out_dir / "val.parquet", index=False)
    test.to_parquet(out_dir / "test.parquet", index=False)
