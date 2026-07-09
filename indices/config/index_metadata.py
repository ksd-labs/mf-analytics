"""
indices/config/index_metadata.py

Central registry for all benchmark indices supported by the
NSE TRI downloader.

This file serves as the single source of truth for:
    1. NSE index names
    2. Local CSV filenames
    3. Inception dates
    4. Future metadata extensions

PATH CHANGE from indice_loader standalone project:
    PROJECT_ROOT now resolves 3 levels up (indices/config/ → indices/ → mf_analytics/)
    INDEX_DATA_DIR now points to data/tri/ (not data/raw/indices/)
"""

from pathlib import Path


# ============================================================================
# DATA DIRECTORIES
# ============================================================================

# indices/config/index_metadata.py lives 3 levels deep inside mf_analytics/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR       = PROJECT_ROOT / "data"
INDEX_DATA_DIR = DATA_DIR / "tri"          # ← changed from data/raw/indices/

LOG_DIR = DATA_DIR / "logs"


# ============================================================================
# NSE TRI ENDPOINT
# ============================================================================

#NSE_TRI_ENDPOINT = (
#    "https://www.niftyindices.com/Backpage.aspx/"
#    "getTotalReturnIndexString"
#)
#copied from indice_loader standalone project, but changed to match the new endpoint
NSE_TRI_ENDPOINT = (
    "https://www.niftyindices.com/"
    "BackPage/getTotalReturnIndexString"
)

# ============================================================================
# HTTP CONFIGURATION
# ============================================================================

HEADERS = {
    "Content-Type":    "application/json; charset=UTF-8",
    "X-Requested-With": "XMLHttpRequest",
    "Referer":         "https://www.niftyindices.com/reports/historical-data",
    "Origin":          "https://www.niftyindices.com",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/137.0.0.0 Safari/537.36"
    ),
}


# ============================================================================
# DOWNLOAD CONFIGURATION
# ============================================================================

CHUNK_DAYS             = 365
REQUEST_DELAY_SECONDS  = 1
MAX_RETRIES            = 3
REQUEST_TIMEOUT        = 60


# ============================================================================
# INDEX METADATA
# ============================================================================

INDEX_METADATA = {
    "NIFTY 500": {
        "filename":    "NIFTY_500_TRI.csv",
        "inception":   "01-Jan-1994",
        "category":    "Broad Market",
        "nse_name":    "NIFTY 500",
        "description": "Top 500 Indian companies by free-float market cap",
    },
    "NIFTY 50": {
        "filename":    "NIFTY_50_TRI.csv",
        "inception":   "30-Jun-1999",
        "category":    "Large Cap",
        "nse_name":    "NIFTY 50",
        "description": "Top 50 Indian large-cap companies",
    },
    "NIFTY NEXT 50": {
        "filename":    "NIFTY_NEXT_50_TRI.csv",
        "inception":   "01-Jan-1996",
        "category":    "Large Cap",
        "description": "Next 50 companies after NIFTY 50",
    },
    "NIFTY 100": {
        "filename":    "NIFTY_100_TRI.csv",
        "inception":   "01-Jan-2003",
        "category":    "Large Cap",
        "nse_name":    "NIFTY 100",
        "description": "Top 100 Indian companies",
    },
    "NIFTY 200": {
        "filename":    "NIFTY_200_TRI.csv",
        "inception":   "01-Jan-2004",
        "category":    "Broad Market",
        "description": "Top 200 Indian companies",
    },
    "NIFTY LARGE MIDCAP 250": {
        "filename":    "NIFTY_LARGE_MIDCAP_250_TRI.csv",
        "inception":   "01-Jan-2005",
        "category":    "Large & Mid Cap",
        "description": "Combination of large and mid-cap stocks",
    },
    "NIFTY MIDCAP 150": {
        "filename":    "NIFTY_MIDCAP_150_TRI.csv",
        "inception":   "01-Apr-2005",
        "category":    "Mid Cap",
        "nse_name":    "NIFTY MIDCAP 150",
        "description": "Top 150 mid-cap companies",
    },
    "NIFTY SMALLCAP 250": {
        "filename":    "NIFTY_SMALLCAP_250_TRI.csv",
        "inception":   "01-Apr-2005",
        "category":    "Small Cap",
        "nse_name":    "NIFTY SMALLCAP 250",
        "description": "Top 250 small-cap companies",
    },
    "NIFTY MICROCAP 250": {
        "filename":    "NIFTY_MICROCAP_250_TRI.csv",
        "inception":   "01-Jan-2005",
        "category":    "Micro Cap",
        "description": "Micro-cap equity universe",
    },
    "NIFTY200 MOMENTUM 30": {
        "filename":    "NIFTY_200_MOMENTUM_30_TRI.csv",
        "inception":   "01-Apr-2005",
        "category":    "Factor",
        "nse_name":    "NIFTY200 MOMENTUM 30",
        "description": "Momentum factor index",
    },
    "NIFTY ALPHA 50": {
        "filename":    "NIFTY_ALPHA_50_TRI.csv",
        "inception":   "01-Jan-2005",
        "category":    "Factor",
        "description": "Alpha factor index",
    },
    "NIFTY100 QUALITY 30": {
        "filename":    "NIFTY_100_QUALITY_30_TRI.csv",
        "inception":   "01-Oct-2009",
        "category":    "Factor",
        "nse_name":    "NIFTY100 QUALTY30",
        "description": "Quality factor index",
    },
    "NIFTY200 QUALITY 30": {
        "filename":    "NIFTY_200_QUALITY_30_TRI.csv",
        "inception":   "01-Apr-2005",
        "category":    "Factor",
        "nse_name":    "NIFTY200 QUALITY 30",
        "description": "Quality factor index",
    },
    "NIFTY500 VALUE 50": {
        "filename":    "NIFTY_500_VALUE_50_TRI.csv",
        "inception":   "01-Apr-2005",
        "category":    "Factor",
        "nse_name":    "NIFTY500 VALUE 50",
        "description": "Value factor index",
    },
    "NIFTY50 VALUE 20": {
        "filename":    "NIFTY_50_VALUE_20_TRI.csv",
        "inception":   "01-Apr-2009",
        "category":    "Factor",
        "nse_name":    "NIFTY50 VALUE 20",
        "description": "Value factor index",
    },
    "NIFTY100 LOW VOLATILITY 30": {
        "filename":    "NIFTY_100_LOW_VOLATILITY_30_TRI.csv",
        "inception":   "01-Apr-2005",
        "category":    "Factor",
        "nse_name":    "NIFTY100 LOWVOL30",
        "description": "Low volatility factor index",
    },
}


# ============================================================================
# HELPER FUNCTIONS  (unchanged from indice_loader)
# ============================================================================

def get_supported_indices() -> list[str]:
    return sorted(INDEX_METADATA.keys())


def is_supported_index(index_name: str) -> bool:
    return index_name in INDEX_METADATA


def get_index_metadata(index_name: str) -> dict:
    if index_name not in INDEX_METADATA:
        raise KeyError(f"Unsupported index: {index_name}")
    return INDEX_METADATA[index_name]


def get_csv_path(index_name: str) -> Path:
    metadata = get_index_metadata(index_name)
    return INDEX_DATA_DIR / metadata["filename"]


def get_inception_date(index_name: str) -> str:
    metadata = get_index_metadata(index_name)
    return metadata["inception"]


# ============================================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================================

INDEX_DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
