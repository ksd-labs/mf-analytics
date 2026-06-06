"""
data/factor_loader.py
======================
Constructs Fama-French factor return series using Indian index fund NAVs.

Factor construction strategy:
    We use index fund NAVs as factor proxies — no external data needed.
    All proxy instruments are open-ended index funds available in mftool.

    Factor        Formula                         Proxy Instruments
    ─────────────────────────────────────────────────────────────────────────
    Market-Rf     Nifty 500 return − rf_daily      Motilal Oswal / UTI Nifty 500
    SMB           Smallcap250 − Nifty100           Nippon Smallcap 250 − UTI Nifty 100
    HML           Value50 − Nifty500               Nifty 500 Value 50 − Nifty 500
    WML           Momentum30 − Nifty500            Nifty 200 Momentum 30 − Nifty 500

    SMB approximation: True SMB requires sorting every listed stock into
    size buckets monthly. Using (Smallcap250 − Nifty100) as a proxy is
    a widely accepted simplification for India-specific research.

    HML approximation: Nifty 500 Value 50 tracks the 50 most value-oriented
    stocks in the Nifty 500. Subtracting the broad Nifty 500 isolates the
    value premium. This is a reasonable proxy though not exact HML.

Graceful degradation:
    If a proxy instrument is not found (e.g., Momentum 30 is a newer index),
    that factor is excluded and the model uses fewer factors.
    Minimum viable model = 1 factor (Market only = CAPM).
    Results always report which factors were included.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# FACTOR PROXY SEARCH KEYWORDS
# ─────────────────────────────────────────────────────────────────────────────
# Listed in priority order — first keyword that produces a Direct Growth match wins.

FACTOR_PROXY_KEYWORDS: Dict[str, List[str]] = {
    # Market factor — broad market index
    "market": [
        "nifty 500 index fund",
        "nifty500 index fund",
        "nifty 500 ",
    ],
    # Small cap proxy (for SMB numerator)
    "small": [
        "nifty smallcap 250 index fund",
        "smallcap 250 index fund",
        "nifty smallcap 250",
    ],
    # Large cap proxy (for SMB denominator)
    "large": [
        "nifty 100 index fund",
        "nifty100 index fund",
        "nifty 100 ",
    ],
    # Value proxy (for HML numerator)
    "value": [
        "nifty 500 value 50",
        "value 50 index",
        "nifty value 50",
        "nifty500 value",
    ],
    # Momentum proxy (for WML numerator)
    "momentum": [
        "nifty 200 momentum 30",
        "momentum 30 index",
        "nifty200 momentum",
        "nifty 200 momentum",
    ],
}

# Human-readable factor names for display
FACTOR_DISPLAY_NAMES: Dict[str, str] = {
    "market": "Market (Mkt-Rf)",
    "smb":    "Size (SMB)",
    "hml":    "Value (HML)",
    "wml":    "Momentum (WML)",
}


# ─────────────────────────────────────────────────────────────────────────────
# PROXY SCHEME FINDER
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _find_proxy_scheme(factor_role: str) -> Optional[Dict]:
    """
    Find the best matching Direct Growth index fund for a factor proxy role.

    Args:
        factor_role: One of 'market', 'small', 'large', 'value', 'momentum'

    Returns:
        {'code': str, 'name': str} or None if not found.
    """
    keywords = FACTOR_PROXY_KEYWORDS.get(factor_role, [])
    if not keywords:
        return None

    from data.fund_loader import get_all_schemes
    all_schemes = get_all_schemes()
    if not all_schemes:
        return None

    exclusions = ["etf", "exchange traded", "idcw", "dividend", "regular"]
    candidates = {
        code: name for code, name in all_schemes.items()
        if "direct" in name.lower()
        and "growth" in name.lower()
        and not any(e in name.lower() for e in exclusions)
    }

    for keyword in keywords:
        for code, name in candidates.items():
            if keyword in name.lower():
                logger.info(f"Factor proxy [{factor_role}]: {name} ({code})")
                return {"code": code, "name": name}

    logger.warning(f"No proxy found for factor role: {factor_role}")
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def _load_proxy_nav(factor_role: str) -> Optional[pd.Series]:
    """
    Load and process the NAV series for a factor proxy instrument.

    Returns:
        Clean NAV pd.Series with DatetimeIndex, or None.
    """
    scheme = _find_proxy_scheme(factor_role)
    if scheme is None:
        return None

    from data.fund_loader import get_nav_history
    from data.nav_processor import process_nav

    nav_df = get_nav_history(scheme["code"])
    if nav_df is None:
        return None

    return process_nav(nav_df)


# ─────────────────────────────────────────────────────────────────────────────
# FACTOR RETURN CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_factor_returns(
    rf_rate: float = 0.065,
) -> Tuple[Optional[pd.DataFrame], Dict[str, Optional[str]]]:
    """
    Construct daily factor return series for the 4-factor Fama-French-Carhart model.

    Returns a tuple of:
        (factor_df, proxy_names)

    factor_df:   pd.DataFrame with columns for available factors:
                 - 'market': Nifty500 excess return (return - rf_daily)
                 - 'smb':    Small cap premium (Smallcap250 - Nifty100)
                 - 'hml':    Value premium (Value50 - Nifty500)
                 - 'wml':    Momentum premium (Momentum30 - Nifty500)
                 Only columns where proxy data was found are included.
                 All series aligned to their common date range.

    proxy_names: Dict mapping factor name to the scheme name used as proxy,
                 or None if the factor could not be constructed.
                 Used for display in the UI to show what proxy was used.

    Args:
        rf_rate: Annual risk-free rate for Market factor construction.
    """
    from data.nav_processor import compute_daily_returns

    rf_daily = rf_rate / 252

    # ── Load all proxy NAVs ───────────────────────────────────────────────────
    market_nav  = _load_proxy_nav("market")
    small_nav   = _load_proxy_nav("small")
    large_nav   = _load_proxy_nav("large")
    value_nav   = _load_proxy_nav("value")
    momentum_nav = _load_proxy_nav("momentum")

    proxy_names: Dict[str, Optional[str]] = {}

    for role, nav in [("market", market_nav), ("small", small_nav),
                      ("large", large_nav), ("value", value_nav),
                      ("momentum", momentum_nav)]:
        scheme = _find_proxy_scheme(role)
        proxy_names[role] = scheme["name"] if scheme else None

    # ── Compute individual return series ──────────────────────────────────────
    mkt_ret  = compute_daily_returns(market_nav)   if market_nav  is not None else None
    small_ret = compute_daily_returns(small_nav)   if small_nav   is not None else None
    large_ret = compute_daily_returns(large_nav)   if large_nav   is not None else None
    val_ret  = compute_daily_returns(value_nav)    if value_nav   is not None else None
    mom_ret  = compute_daily_returns(momentum_nav) if momentum_nav is not None else None

    # ── Build factor series ───────────────────────────────────────────────────
    factor_series: Dict[str, pd.Series] = {}

    # Market factor = Nifty500 return - rf_daily
    if mkt_ret is not None:
        factor_series["market"] = mkt_ret - rf_daily
        proxy_names["market_factor"] = proxy_names.get("market")

    # SMB = Small return - Large return
    if small_ret is not None and large_ret is not None:
        common = small_ret.index.intersection(large_ret.index)
        if len(common) >= 60:
            factor_series["smb"] = (
                small_ret.reindex(common) - large_ret.reindex(common)
            )

    # HML = Value return - Market return (as broad market proxy)
    # Using Nifty500 as the "low book-to-market" proxy
    if val_ret is not None and mkt_ret is not None:
        common = val_ret.index.intersection(mkt_ret.index)
        if len(common) >= 60:
            factor_series["hml"] = (
                val_ret.reindex(common) - mkt_ret.reindex(common)
            )

    # WML = Momentum return - Market return
    if mom_ret is not None and mkt_ret is not None:
        common = mom_ret.index.intersection(mkt_ret.index)
        if len(common) >= 60:
            factor_series["wml"] = (
                mom_ret.reindex(common) - mkt_ret.reindex(common)
            )

    if not factor_series:
        return None, proxy_names

    # ── Align all factor series to common dates ───────────────────────────────
    factor_df = pd.DataFrame(factor_series)
    factor_df = factor_df.dropna(how="all")

    if len(factor_df) < 60:
        return None, proxy_names

    logger.info(
        f"Factor returns built: {list(factor_df.columns)}, "
        f"{len(factor_df)} daily observations"
    )

    return factor_df, proxy_names


# ─────────────────────────────────────────────────────────────────────────────
# AVAILABILITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def get_factor_availability() -> Dict[str, bool]:
    """
    Return which factors are available (proxy found in mftool).
    Used in the UI to explain which model variant will be run.

    Returns:
        {'market': True/False, 'smb': True/False, 'hml': True/False, 'wml': True/False}
    """
    factor_df, _ = get_factor_returns()
    if factor_df is None:
        return {f: False for f in ["market", "smb", "hml", "wml"]}
    return {f: f in factor_df.columns for f in ["market", "smb", "hml", "wml"]}
