"""
utils/session.py
================
Centralised session state key management.

All analytics cache keys include ANALYTICS_VERSION — when new metrics are
added and the version is bumped in constants.py, every cached result is
automatically invalidated on the next page load without the user needing
to do anything.

Usage in pages:
    from utils.session import (
        fund_key, category_key, alpha_key,
        clear_analytics_cache, render_refresh_button,
    )
"""

import streamlit as st
from utils.constants import ANALYTICS_VERSION


# ─────────────────────────────────────────────────────────────────────────────
# KEY BUILDERS
# Each key includes ANALYTICS_VERSION so bumping the version auto-invalidates.
# ─────────────────────────────────────────────────────────────────────────────

def fund_key(scheme_code: str, rf_pct: float) -> str:
    """Session state key for a single-fund metrics dict."""
    return f"fund_metrics_{scheme_code}_{rf_pct}_{ANALYTICS_VERSION}"


def alpha_key(scheme_code: str, rf_pct: float, category: str) -> str:
    """Session state key for alpha+Phase B metrics of a single fund."""
    return f"alpha_{scheme_code}_{rf_pct}_{category}_{ANALYTICS_VERSION}"


def category_analytics_key(category: str) -> str:
    """Session state key for 'has category analytics been computed' flag."""
    return f"analytics_done_{category}_{ANALYTICS_VERSION}"


def category_full_df_key(category: str) -> str:
    """Session state key for the full category metrics+quartile DataFrame."""
    return f"full_df_{category}_{ANALYTICS_VERSION}"


def category_fund_metrics_key(category: str) -> str:
    """Session state key for the {fund_name: metrics_dict} category dict."""
    return f"fund_metrics_{category}_{ANALYTICS_VERSION}"


def rankings_done_key(category: str) -> str:
    """Session state key for 'have rankings been computed' flag."""
    return f"rankings_done_{category}_{ANALYTICS_VERSION}"


def dq_scan_key(category: str) -> str:
    """Session state key for data quality scan results."""
    return f"dq_scan_{category}_{ANALYTICS_VERSION}"


def dq_reports_key(category: str) -> str:
    """Session state key for data quality reports dict."""
    return f"dq_reports_{category}_{ANALYTICS_VERSION}"


# ─────────────────────────────────────────────────────────────────────────────
# CACHE MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def clear_analytics_cache() -> int:
    """
    Clear ALL analytics-related session state keys.

    Called when the user clicks the Refresh NAV Data button.
    Also clears st.cache_data (the mftool / NAV API cache).

    Returns:
        Number of session state keys cleared.
    """
    ANALYTICS_PREFIXES = (
        "fund_metrics_", "full_df_", "analytics_done_",
        "rankings_done_", "alpha_", "dq_scan_", "dq_reports_",
    )
    keys_to_remove = [
        k for k in list(st.session_state.keys())
        if any(k.startswith(p) for p in ANALYTICS_PREFIXES)
    ]
    for key in keys_to_remove:
        del st.session_state[key]

    # Also clear the Streamlit function cache (NAV API calls)
    st.cache_data.clear()

    return len(keys_to_remove)


def render_refresh_button(location=None) -> None:
    """
    Render the standard Refresh NAV Data button.
    When clicked, clears both API cache and analytics session state.

    Args:
        location: Optional Streamlit container (defaults to current context).
    """
    ctx = location or st
    if ctx.button("🔄 Refresh NAV Data", use_container_width=True,
                  help="Clears cached NAV data and all computed analytics. "
                       "Everything will be recomputed fresh."):
        n_cleared = clear_analytics_cache()
        st.success(
            f"✅ Cache cleared — {n_cleared} analytics results removed. "
            "Data will reload on next action."
        )
        st.rerun()
