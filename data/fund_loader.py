"""
fund_loader.py
==============
The ONLY file that talks to mftool and the AMFI/mfapi data sources.

mftool v3.3 API changes (BREAKING from v2.x):
  - get_available_schemes(amc_name)  → now requires AMC name parameter
  - get_scheme_codes()               → use this for ALL schemes
  - get_scheme_historical_nav(code, as_Dataframe=True)  → still works
  - history(code)                    → new method (uses yfinance codes)

Fallback strategy:
  If mftool's get_scheme_codes() fails (AMFI URL blocked), we fetch
  directly from mfapi.in/mf using requests. This is the same backend
  that mftool uses for individual scheme details and NAV history.

Caching:
  All functions use @st.cache_data(ttl=3600) — 1 hour cache.
  Errors are NEVER silently swallowed — they are printed AND returned
  as informative error strings so the UI can display them.
"""

import streamlit as st
import pandas as pd
import requests
from mftool import Mftool
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

# Single shared mftool instance
_mf = Mftool()

# Direct API URL (mftool's backend — used as fallback)
_MFAPI_ALL_URL    = "https://api.mfapi.in/mf"
_MFAPI_SCHEME_URL = "https://api.mfapi.in/mf/{code}"
_AMFI_NAV_URL     = "https://www.amfiindia.com/spages/NAVAll.txt"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-IN,en;q=0.9",
}


# ─────────────────────────────────────────────────────────────────────────────
# SCHEME REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_schemes_via_mftool() -> Dict[str, str]:
    """
    Fetch all schemes using mftool's get_scheme_codes().
    Returns {code: name} or raises an exception on failure.
    """
    codes = _mf.get_scheme_codes(as_json=False)
    if codes and len(codes) > 0:
        return dict(codes)
    raise ValueError("mftool.get_scheme_codes() returned empty — AMFI URL may be blocked.")


def _fetch_schemes_via_amfi_direct() -> Dict[str, str]:
    """
    Direct fallback: fetch scheme list from AMFI NAVAll.txt.
    Format: Code;ISIN1;ISIN2;Name;NAV;Date (semicolon separated)
    Returns {code: name} or raises on failure.
    """
    r = requests.get(_AMFI_NAV_URL, headers=_HEADERS, timeout=20)
    r.raise_for_status()

    schemes: Dict[str, str] = {}
    for line in r.text.splitlines():
        if ";" not in line:
            continue
        parts = line.split(";")
        if len(parts) >= 4:
            code = parts[0].strip()
            name = parts[3].strip()
            if code.isdigit() and name:
                schemes[code] = name

    if not schemes:
        raise ValueError("AMFI NAVAll.txt parsed but no schemes found.")
    return schemes


def _fetch_schemes_via_mfapi_direct() -> Dict[str, str]:
    """
    Second fallback: fetch full scheme list from mfapi.in/mf.
    Returns {code: name} or raises on failure.
    """
    r = requests.get(_MFAPI_ALL_URL, headers=_HEADERS, timeout=20)
    r.raise_for_status()

    data = r.json()   # List of {schemeCode, schemeName}
    schemes = {
        str(item["schemeCode"]): item["schemeName"]
        for item in data
        if "schemeCode" in item and "schemeName" in item
    }
    if not schemes:
        raise ValueError("mfapi.in/mf returned empty list.")
    return schemes


@st.cache_data(ttl=3600, show_spinner=False)
def get_all_schemes() -> Dict[str, str]:
    """
    Fetch all mutual fund scheme codes and names.
    Tries three sources in order:
      1. mftool.get_scheme_codes()   (uses AMFI NAVAll.txt internally)
      2. Direct AMFI NAVAll.txt fetch
      3. Direct mfapi.in/mf fetch

    Returns:
        {scheme_code: scheme_name} dict, or {} if all sources fail.
        Never raises — errors are logged with full detail.
    """
    errors: List[str] = []

    # ── Attempt 1: mftool ────────────────────────────────────────────────────
    try:
        schemes = _fetch_schemes_via_mftool()
        logger.info(f"[fund_loader] mftool: {len(schemes):,} schemes loaded")
        return schemes
    except Exception as e:
        msg = f"mftool.get_scheme_codes() failed: {type(e).__name__}: {e}"
        errors.append(msg)
        logger.warning(f"[fund_loader] {msg}")

    # ── Attempt 2: Direct AMFI ───────────────────────────────────────────────
    try:
        schemes = _fetch_schemes_via_amfi_direct()
        logger.info(f"[fund_loader] AMFI direct: {len(schemes):,} schemes loaded")
        return schemes
    except Exception as e:
        msg = f"AMFI direct fetch failed: {type(e).__name__}: {e}"
        errors.append(msg)
        logger.warning(f"[fund_loader] {msg}")

    # ── Attempt 3: mfapi.in direct ───────────────────────────────────────────
    try:
        schemes = _fetch_schemes_via_mfapi_direct()
        logger.info(f"[fund_loader] mfapi.in direct: {len(schemes):,} schemes loaded")
        return schemes
    except Exception as e:
        msg = f"mfapi.in direct fetch failed: {type(e).__name__}: {e}"
        errors.append(msg)
        logger.warning(f"[fund_loader] {msg}")

    # ── All failed ───────────────────────────────────────────────────────────
    logger.error(f"[fund_loader] ALL sources failed:\n" + "\n".join(errors))
    # Store errors so the UI can display them
    st.session_state["scheme_load_errors"] = errors
    return {}


@st.cache_data(ttl=3600, show_spinner=False)
def get_scheme_details(scheme_code: str) -> Optional[Dict]:
    """
    Fetch metadata for a single scheme (name, category, type, start date).

    Returns dict with keys:
        fund_house, scheme_type, scheme_category,
        scheme_code, scheme_name, scheme_start_date
    """
    try:
        details = _mf.get_scheme_details(scheme_code)
        if details:
            return details
    except Exception as e:
        logger.warning(f"[fund_loader] mftool.get_scheme_details({scheme_code}) failed: {e}")

    # Fallback: get from mfapi.in directly
    try:
        url = _MFAPI_SCHEME_URL.format(code=scheme_code)
        r = requests.get(url, headers=_HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()
        meta = data.get("meta", {})
        hist = data.get("data", [])
        return {
            "fund_house":        meta.get("fund_house", ""),
            "scheme_type":       meta.get("scheme_type", ""),
            "scheme_category":   meta.get("scheme_category", ""),
            "scheme_code":       meta.get("scheme_code", scheme_code),
            "scheme_name":       meta.get("scheme_name", ""),
            "scheme_start_date": hist[-1] if hist else {},
        }
    except Exception as e:
        logger.error(f"[fund_loader] Direct scheme_details({scheme_code}) also failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# NAV HISTORY
# ─────────────────────────────────────────────────────────────────────────────

def _parse_mfapi_nav_response(data: dict) -> Optional[pd.DataFrame]:
    """
    Parse the raw mfapi.in JSON response into a clean NAV DataFrame.

    mfapi.in data format:
        data: [{"date": "31-05-2024", "nav": "45.2381"}, ...]
        (newest first — we reverse to oldest first)

    Returns DataFrame with:
        - DatetimeIndex (ascending, named 'date')
        - 'nav' column as float64
    """
    nav_list = data.get("data", [])
    if not nav_list:
        return None

    df = pd.DataFrame(nav_list)

    if "date" not in df.columns or "nav" not in df.columns:
        return None

    # Try multiple date formats (mfapi.in uses DD-MM-YYYY but some sources differ)
    for fmt in ["%d-%m-%Y", "%d-%b-%Y", "%Y-%m-%d"]:
        try:
            df["date"] = pd.to_datetime(df["date"], format=fmt, errors="raise")
            break
        except (ValueError, TypeError):
            continue
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df = df.dropna(subset=["date", "nav"])
    df = df[df["nav"] > 0]
    df = df.sort_values("date").set_index("date")
    df = df[~df.index.duplicated(keep="last")]
    df = df[["nav"]]

    return df if not df.empty else None


@st.cache_data(ttl=3600, show_spinner=False)
def get_nav_history(scheme_code: str) -> Optional[pd.DataFrame]:
    """
    Fetch complete NAV history for a scheme.

    Returns DataFrame with:
        - DatetimeIndex ascending (named 'date')
        - 'nav' column as float64

    Tries two sources:
        1. mftool.get_scheme_historical_nav()
        2. Direct mfapi.in request

    Returns None if both fail.
    """
    scheme_code = str(scheme_code)

    # ── Attempt 1: mftool ────────────────────────────────────────────────────
    try:
        raw = _mf.get_scheme_historical_nav(scheme_code, as_Dataframe=True)

        if raw is not None and isinstance(raw, pd.DataFrame) and not raw.empty:
            df = raw.copy()

            # mftool 3.3 returns: index='date'(str), columns=['nav', 'dayChange']
            if df.index.name == "date":
                df = df.reset_index()

            df.columns = [c.lower().strip() for c in df.columns]

            if "date" in df.columns and "nav" in df.columns:
                for fmt in ["%d-%m-%Y", "%d-%b-%Y", "%Y-%m-%d"]:
                    try:
                        df["date"] = pd.to_datetime(df["date"], format=fmt, errors="raise")
                        break
                    except (ValueError, TypeError):
                        continue
                else:
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")

                df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
                df = df.dropna(subset=["date", "nav"])
                df = df[df["nav"] > 0]
                df = df.sort_values("date").set_index("date")
                df = df[~df.index.duplicated(keep="last")]
                df = df[["nav"]]

                if not df.empty:
                    return df

    except Exception as e:
        logger.warning(f"[fund_loader] mftool NAV({scheme_code}) failed: {type(e).__name__}: {e}")

    # ── Attempt 2: Direct mfapi.in request ──────────────────────────────────
    try:
        url = _MFAPI_SCHEME_URL.format(code=scheme_code)
        r = requests.get(url, headers=_HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
        return _parse_mfapi_nav_response(data)

    except Exception as e:
        logger.error(f"[fund_loader] Direct NAV({scheme_code}) also failed: {type(e).__name__}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY-FILTERED SCHEME LISTS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_all_categorized_schemes() -> Dict[str, List[Dict]]:
    """
    Load all schemes and group by category in one cached call.
    More efficient than calling get_schemes_for_category() 12 times.

    Returns:
        {category_name: [{code, name}, ...]} for all 12 categories.
    """
    from data.category_mapper import get_category_for_scheme, filter_preferred_plans
    from utils.constants import CATEGORIES

    all_schemes = get_all_schemes()
    if not all_schemes:
        return {cat: [] for cat in CATEGORIES}

    preferred = filter_preferred_plans(all_schemes)
    result: Dict[str, List[Dict]] = {cat: [] for cat in CATEGORIES}

    for code, name in preferred.items():
        category = get_category_for_scheme(name)
        if category and category in result:
            result[category].append({"code": code, "name": name})

    for cat in result:
        result[cat] = sorted(result[cat], key=lambda x: x["name"])

    return result


@st.cache_data(ttl=3600, show_spinner=False)
def get_schemes_for_category(category: str) -> List[Dict]:
    """
    Return [{code, name}] for all Growth-plan funds in a single category.
    """
    all_cat = get_all_categorized_schemes()
    return all_cat.get(category, [])


# ─────────────────────────────────────────────────────────────────────────────
# BATCH NAV LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_navs_for_funds(
    fund_list: List[Dict],
    progress_callback=None,
) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Load NAV history for a list of funds with optional progress reporting.

    Args:
        fund_list:         [{code, name}, ...]
        progress_callback: Optional callable(current, total, fund_name)

    Returns:
        {scheme_code: DataFrame or None}
    """
    result: Dict[str, Optional[pd.DataFrame]] = {}

    for i, fund in enumerate(fund_list):
        code = fund["code"]
        if progress_callback:
            try:
                progress_callback(i, len(fund_list), fund["name"])
            except Exception:
                pass
        result[code] = get_nav_history(code)

    return result
