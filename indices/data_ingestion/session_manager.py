"""
data_ingestion/session_manager.py

Creates and manages NSE sessions.

The NSE endpoint requires cookies from
the historical-data page before the
TRI API can be called.
"""
import time
import requests

from indices.config.index_metadata import (
    HEADERS,
    REQUEST_TIMEOUT,
)


NSE_HISTORICAL_PAGE = (
    "https://www.niftyindices.com/"
    "reports/historical-data"
)


def create_session() -> requests.Session:

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": HEADERS["User-Agent"]
        }
    )

    for attempt in range(3):

        try:

            response = session.get(
                NSE_HISTORICAL_PAGE,
                timeout=REQUEST_TIMEOUT,
            )

            response.raise_for_status()

            return session

        except Exception:

            if attempt == 2:
                raise

            time.sleep(5)

    return session

def refresh_session(
    session: requests.Session
) -> requests.Session:
    """
    Refresh cookies if NSE starts
    rejecting requests.

    Returns
    -------
    requests.Session
    """

    try:

        session.cookies.clear()

        response = session.get(
            NSE_HISTORICAL_PAGE,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        return session

    except Exception:

        return create_session()