"""
indices/data_ingestion/nifty_tri_downloader.py

Core NSE TRI downloader.

Responsibilities:
    1. Download TRI history
    2. Handle chunking
    3. Retry failed requests
    4. Clean data
    5. Update local cache

CHANGE from indice_loader standalone:
    All five import blocks updated to indices.* namespace.
    Zero logic changes.
"""

import json
import time
from datetime import datetime

import pandas as pd
import requests
from dateutil.relativedelta import relativedelta

from indices.config.index_metadata import (          # ← updated
    NSE_TRI_ENDPOINT,
    HEADERS,
    CHUNK_DAYS,
    REQUEST_DELAY_SECONDS,
    MAX_RETRIES,
    REQUEST_TIMEOUT,
    get_inception_date,
    get_index_metadata,
)
from indices.data_ingestion.validators import (       # ← updated
    validate_index_name,
)
from indices.data_ingestion.session_manager import (  # ← updated
    create_session,
    refresh_session,
)
from indices.data_ingestion.cache_manager import (    # ← updated
    load_existing_data,
    save_data,
    merge_with_cache,
    get_download_start_date,
)
from indices.utils.logger import get_logger           # ← updated

logger = get_logger(__name__)


def fetch_chunk(
    session:    requests.Session,
    index_name: str,
    start_date: datetime,
    end_date:   datetime,
) -> pd.DataFrame:

    metadata = get_index_metadata(index_name)
    nse_name = metadata.get("nse_name", index_name)

    payload = {
        "cinfo": (
            f"{{'name':'{nse_name}',"
            f"'startDate':'{start_date:%d-%b-%Y}',"
            f"'endDate':'{end_date:%d-%b-%Y}',"
            f"'indexName':'{index_name}'}}"
        )
    }

    response = session.post(
        NSE_TRI_ENDPOINT,
        json=payload,
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    data = response.json()
    if not data:
        return pd.DataFrame()

    return pd.DataFrame(data)


def fetch_chunk_with_retry(
    session:    requests.Session,
    index_name: str,
    start_date: datetime,
    end_date:   datetime,
) -> pd.DataFrame:

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fetch_chunk(session, index_name, start_date, end_date)
        except Exception as exc:
            logger.warning(
                f"Attempt {attempt}/{MAX_RETRIES} failed for {index_name}: {exc}"
            )
            session = refresh_session(session)
            time.sleep(2)

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries.")


def clean_downloaded_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df["Date"] = pd.to_datetime(df["Date"], format="%d %b %Y")
    df["TotalReturnsIndex"] = pd.to_numeric(
        df["TotalReturnsIndex"], errors="coerce"
    )
    df = df[["Date", "TotalReturnsIndex"]]

    return (
        df
        .dropna()
        .sort_values("Date")
        .reset_index(drop=True)
    )


def download_index(index_name: str) -> pd.DataFrame:
    validate_index_name(index_name)

    inception_date = datetime.strptime(
        get_inception_date(index_name), "%d-%b-%Y"
    )

    existing_df = load_existing_data(index_name)
    start_date  = get_download_start_date(index_name, inception_date)
    end_date    = datetime.today()

    if start_date >= end_date:
        logger.info(f"{index_name} already current.")
        return existing_df

    session           = create_session()
    downloaded_chunks = []
    current_start     = start_date

    while current_start <= end_date:
        current_end = min(
            current_start + relativedelta(days=CHUNK_DAYS),
            end_date,
        )

        logger.info(
            f"{index_name}: {current_start:%d-%b-%Y} -> {current_end:%d-%b-%Y}"
        )

        chunk_df = fetch_chunk_with_retry(
            session, index_name, current_start, current_end
        )

        if not chunk_df.empty:
            downloaded_chunks.append(chunk_df)

        current_start = current_end + relativedelta(days=1)
        time.sleep(REQUEST_DELAY_SECONDS)

    if downloaded_chunks:
        new_df      = pd.concat(downloaded_chunks, ignore_index=True)
        new_df      = clean_downloaded_data(new_df)
        before_rows = len(existing_df)
        final_df    = merge_with_cache(existing_df, new_df)
        after_rows  = len(final_df)

        if after_rows == before_rows:
            logger.info(f"{index_name}: No new data available")
            return existing_df
    else:
        logger.info(f"{index_name}: No new data available from NSE")
        return existing_df

    save_data(index_name, final_df)
    logger.info(f"{index_name}: {len(final_df):,} rows saved.")
    return final_df


def update_index(index_name: str) -> pd.DataFrame:
    return download_index(index_name)
