"""
indices/data_ingestion/cache_manager.py

Handles:
    1. CSV path management
    2. Existing data loading
    3. Incremental updates
    4. CSV persistence

CHANGE from indice_loader standalone:
    Import path updated to indices.config.index_metadata
    All logic identical.
"""

from pathlib import Path

import pandas as pd

from indices.config.index_metadata import (    # ← updated import path
    get_csv_path,
)


def csv_exists(index_name: str) -> bool:
    return get_csv_path(index_name).exists()


def load_existing_data(index_name: str) -> pd.DataFrame:
    csv_path = get_csv_path(index_name)
    if not csv_path.exists():
        return pd.DataFrame(columns=["Date", "TotalReturnsIndex"])
    return pd.read_csv(csv_path, parse_dates=["Date"])


def get_last_date(index_name: str):
    df = load_existing_data(index_name)
    if df.empty:
        return None
    return df["Date"].max()


def save_data(index_name: str, df: pd.DataFrame) -> Path:
    csv_path = get_csv_path(index_name)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return csv_path


def merge_with_cache(
    existing_df: pd.DataFrame,
    new_df:      pd.DataFrame,
) -> pd.DataFrame:
    if existing_df.empty:
        combined = new_df.copy()
    else:
        combined = pd.concat([existing_df, new_df], ignore_index=True)

    return (
        combined
        .drop_duplicates(subset=["Date"])
        .sort_values("Date")
        .reset_index(drop=True)
    )


def get_download_start_date(index_name: str, inception_date):
    last_date = get_last_date(index_name)
    if last_date is None:
        return inception_date
    return (
        pd.Timestamp(last_date) + pd.Timedelta(days=1)
    ).to_pydatetime()
