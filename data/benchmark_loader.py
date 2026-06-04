"""
data/benchmark_loader.py
=========================
Benchmark data layer for alpha generation features.

Strategy:
    We use Index Fund NAVs as benchmark proxies. SEBI-registered index funds
    tracking each benchmark are already in mftool — no external data source needed.

    Category → Benchmark Index → Index Fund NAV (proxy for TRI)

    This is a standard institutional practice when the Total Return Index (TRI)
    series itself is not available via the data source — index fund NAVs track
    the TRI very closely (tracking error < 0.1% per year for most index funds).

Benchmark Mapping (SEBI mandated benchmarks per category):
    Large Cap         → Nifty 100 TRI        (proxy: Nifty 100 Index Fund)
    Mid Cap           → Nifty Midcap 150 TRI  (proxy: Nifty Midcap 150 Index Fund)
    Small Cap         → Nifty Smallcap 250 TRI(proxy: Nifty Smallcap 250 Index Fund)
    Flexi Cap         → Nifty 500 TRI         (proxy: Nifty 500 Index Fund)
    Multi Cap         → Nifty 500 TRI         (proxy: Nifty 500 Index Fund)
    ELSS              → Nifty 500 TRI         (proxy: Nifty 500 Index Fund)
    Value             → Nifty 500 TRI         (proxy: Nifty 500 Index Fund)
    Contra            → Nifty 500 TRI         (proxy: Nifty 500 Index Fund)
    Focused           → Nifty 500 TRI         (proxy: Nifty 500 Index Fund)
    Aggressive Hybrid → Nifty 50 TRI          (proxy: Nifty 50 Index Fund)
    Balanced Advantage→ Nifty 50 TRI          (proxy: Nifty 50 Index Fund)
    Index Funds       → None (each fund is its own benchmark)
"""

import streamlit as st
import pandas as pd
from typing import Optional, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Human-readable benchmark names (for display in UI)
BENCHMARK_DISPLAY_NAMES: Dict[str, str] = {
    "Large Cap":          "Nifty 100 TRI",
    "Mid Cap":            "Nifty Midcap 150 TRI",
    "Small Cap":          "Nifty Smallcap 250 TRI",
    "Flexi Cap":          "Nifty 500 TRI",
    "Multi Cap":          "Nifty 500 TRI",
    "ELSS":               "Nifty 500 TRI",
    "Value":              "Nifty 500 TRI",
    "Contra":             "Nifty 500 TRI",
    "Focused":            "Nifty 500 TRI",
    "Aggressive Hybrid":  "Nifty 50 TRI",
    "Balanced Advantage": "Nifty 50 TRI",
    "Index Funds":        "N/A — each fund tracks its own index",
}

# Keywords to search for in scheme names when looking for the benchmark index fund.
# Listed in priority order — first keyword that produces a Direct Growth match wins.
# We deliberately search for multiple keywords to handle different AMC naming conventions.
BENCHMARK_SEARCH_KEYWORDS: Dict[str, List[str]] = {
    "Large Cap": [
        "nifty 100 index fund",
        "nifty100 index fund",
        "nifty 100 index",
    ],
    "Mid Cap": [
        "nifty midcap 150 index fund",
        "midcap 150 index fund",
        "nifty midcap 150",
    ],
    "Small Cap": [
        "nifty smallcap 250 index fund",
        "smallcap 250 index fund",
        "nifty smallcap 250",
    ],
    "Flexi Cap": [
        "nifty 500 index fund",
        "nifty500 index fund",
        "nifty 500 ",
    ],
    "Multi Cap": [
        "nifty 500 index fund",
        "nifty 500 ",
    ],
    "ELSS": [
        "nifty 500 index fund",
        "nifty 500 ",
    ],
    "Value": [
        "nifty 500 index fund",
        "nifty 500 ",
    ],
    "Contra": [
        "nifty 500 index fund",
        "nifty 500 ",
    ],
    "Focused": [
        "nifty 500 index fund",
        "nifty 500 ",
    ],
    "Aggressive Hybrid": [
        "nifty 50 index fund",
        "nifty50 index fund",
        "nifty 50 ",
    ],
    "Balanced Advantage": [
        "nifty 50 index fund",
        "nifty 50 ",
    ],
    "Index Funds": [],   # No single benchmark
}


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK SCHEME FINDER
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def find_benchmark_scheme(category: str) -> Optional[Dict]:
    """
    Find the best matching Direct Growth index fund for a category's benchmark.

    Search strategy:
        1. Load all available schemes from mftool
        2. Filter to Direct Growth plans only (index funds use Direct for clean tracking)
        3. Exclude ETFs (we want open-ended index funds, not exchange-traded)
        4. Search scheme names for benchmark keywords in priority order
        5. Return the first match as {code, name}

    Args:
        category: One of the 12 supported category strings

    Returns:
        {'code': str, 'name': str} for the benchmark scheme, or None if not found.
    """
    keywords = BENCHMARK_SEARCH_KEYWORDS.get(category, [])
    if not keywords:
        logger.info(f"No benchmark defined for category: {category}")
        return None

    from data.fund_loader import get_all_schemes

    all_schemes = get_all_schemes()
    if not all_schemes:
        logger.error("Cannot find benchmark — scheme list is empty")
        return None

    # Filter: must be Direct + Growth + NOT ETF
    exclusions = ["etf", "exchange traded", "idcw", "dividend", "regular"]

    candidates = {
        code: name
        for code, name in all_schemes.items()
        if "direct" in name.lower()
        and "growth" in name.lower()
        and not any(excl in name.lower() for excl in exclusions)
    }

    # Search by keyword priority
    for keyword in keywords:
        for code, name in candidates.items():
            if keyword in name.lower():
                logger.info(f"Benchmark for {category}: {name} ({code})")
                return {"code": code, "name": name}

    logger.warning(f"No benchmark scheme found for {category}")
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_benchmark_nav(category: str) -> Optional[pd.DataFrame]:
    """
    Fetch the NAV history for a category's benchmark index fund.

    Args:
        category: Category string

    Returns:
        DataFrame with DatetimeIndex and 'nav' float column, or None.
    """
    benchmark = find_benchmark_scheme(category)
    if benchmark is None:
        return None

    from data.fund_loader import get_nav_history
    nav_df = get_nav_history(benchmark["code"])

    if nav_df is None:
        logger.warning(f"Could not fetch benchmark NAV for {category} "
                       f"(scheme: {benchmark.get('name')})")
    return nav_df


def get_all_category_benchmarks() -> Dict[str, Optional[Dict]]:
    """
    Find benchmark schemes for all 12 categories.
    Used on the Dashboard page to show benchmark coverage.

    Returns:
        {category: {'code': str, 'name': str} or None}
    """
    from utils.constants import CATEGORIES
    return {cat: find_benchmark_scheme(cat) for cat in CATEGORIES}


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK AVAILABILITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def get_benchmark_info(category: str) -> Dict:
    """
    Return display-ready benchmark information for a category.
    Used in the Fund Analytics sidebar.

    Returns:
        {
          'display_name': str,       e.g. 'Nifty 100 TRI'
          'scheme_name':  str,       actual index fund name used as proxy
          'scheme_code':  str,       scheme code
          'available':    bool,      True if benchmark NAV can be loaded
        }
    """
    display = BENCHMARK_DISPLAY_NAMES.get(category, "N/A")
    scheme  = find_benchmark_scheme(category)

    return {
        "display_name": display,
        "scheme_name":  scheme["name"] if scheme else "Not found",
        "scheme_code":  scheme["code"] if scheme else None,
        "available":    scheme is not None,
    }
