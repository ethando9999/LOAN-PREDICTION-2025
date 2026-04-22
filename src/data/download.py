"""Tải dataset Loan Prediction 2025 từ Kaggle về data/raw/."""
from __future__ import annotations

import os
import zipfile
from pathlib import Path


def download_dataset(slug: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(slug, path=str(dest), unzip=False, quiet=False)

    for z in dest.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest)
        z.unlink()
    return dest


if __name__ == "__main__":
    import yaml

    cfg = yaml.safe_load(Path("configs/config.yaml").read_text())
    out = download_dataset(cfg["data"]["kaggle_slug"], Path(cfg["data"]["raw_dir"]))
    files = sorted(out.iterdir())
    print(f"Downloaded {len(files)} files to {out}:")
    for f in files:
        print(f"  {f.name} ({f.stat().st_size/1024/1024:.2f} MB)")
