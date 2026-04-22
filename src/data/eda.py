"""EDA: phân phối class, thống kê mô tả, phân loại feature."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def class_balance(df: pd.DataFrame, target: str) -> dict:
    counts = df[target].value_counts().to_dict()
    total = len(df)
    return {
        "counts": {str(k): int(v) for k, v in counts.items()},
        "ratios": {str(k): round(v / total, 4) for k, v in counts.items()},
        "total": total,
    }


def split_feature_types(df: pd.DataFrame, target: str) -> dict[str, list[str]]:
    feats = [c for c in df.columns if c != target]
    numeric = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in feats if c not in numeric]
    low_card_numeric = [c for c in numeric if df[c].nunique() <= 10]
    return {
        "all": feats,
        "numeric": numeric,
        "categorical": categorical,
        "low_cardinality_numeric": low_card_numeric,
    }


def describe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols].describe().T.round(4)


def run_eda(csv_path: Path, target: str, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)

    report = {
        "shape": list(df.shape),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": df.isna().sum().to_dict(),
        "class_balance": class_balance(df, target),
        "feature_types": split_feature_types(df, target),
    }

    desc = describe_numeric(df, report["feature_types"]["numeric"])
    desc.to_csv(out_dir / "numeric_describe.csv")

    cat_summary = {}
    for c in report["feature_types"]["categorical"]:
        cat_summary[c] = df[c].value_counts().head(20).to_dict()
    (out_dir / "categorical_counts.json").write_text(json.dumps(cat_summary, indent=2, default=str))

    (out_dir / "eda_report.json").write_text(json.dumps(report, indent=2, default=str))
    return report


if __name__ == "__main__":
    import yaml

    cfg = yaml.safe_load(Path("configs/config.yaml").read_text())
    raw = next(Path(cfg["data"]["raw_dir"]).glob("*.csv"))
    report = run_eda(raw, cfg["data"]["target_col"], Path("reports"))
    print(json.dumps(report, indent=2, default=str))
