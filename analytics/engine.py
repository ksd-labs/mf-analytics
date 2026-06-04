"""
analytics/engine.py
===================
The analytics orchestrator — the single entry point for all metric calculations.

Pages NEVER call individual analytics functions directly. Instead they call:

    compute_fund_metrics(nav_df, rf_rate)      → metrics dict for one fund
    compute_category_metrics(fund_nav_dict, rf_rate) → metrics for all funds in category
    compute_category_quartiles(fund_metrics)   → adds quartile labels

This design means:
    - Pages contain zero mathematical logic
    - All metric computation is testable in isolation
    - The engine is the only place that knows the computation order
    - Caching can be applied at the engine level

Computation order (matters for Calmar ratio):
    1. Process raw NAV → clean nav, daily returns, log returns, monthly returns
    2. Performance (CAGR) — needed by Calmar
    3. Risk (Drawdown) — needed by Calmar
    4. Volatility
    5. Risk-Adjusted (Sharpe, Sortino, Calmar)
    6. Consistency (Rolling Returns) — also needed by Persistence
    7. Distribution (Skewness, Kurtosis)
    8. Stability (Win Rate)
    9. Persistence (% positive rolling periods, streaks)
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import logging

from data.nav_processor import (
    process_nav,
    compute_daily_returns,
    compute_log_returns,
    compute_monthly_returns,
    get_series_summary,
)
from analytics.performance    import calc_all_cagr
from analytics.risk           import calc_all_risk
from analytics.volatility     import calc_all_volatility
from analytics.risk_adjusted  import calc_all_risk_adjusted
from analytics.consistency    import calc_all_consistency
from analytics.distribution   import calc_all_distribution
from analytics.stability      import calc_all_stability
from analytics.persistence    import calc_all_persistence
from analytics.alpha          import calc_all_alpha
from analytics.quartile       import build_full_quartile_table
from utils.constants          import DEFAULT_RISK_FREE_RATE, MAR
from utils.validators         import check_nav_series

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE FUND METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_fund_metrics(
    nav_df: Optional[pd.DataFrame],
    rf_rate: float = DEFAULT_RISK_FREE_RATE,
    fund_name: str = "",
    benchmark_nav_df: Optional[pd.DataFrame] = None,
    benchmark_name: str = "",
) -> Dict:
    """
    Compute all quantitative metrics for a single mutual fund.

    This is the primary engine function. It processes the raw NAV data,
    runs all analytics modules in the correct order, and returns a
    comprehensive dict of results.

    Args:
        nav_df:    Raw NAV DataFrame from fund_loader.get_nav_history()
                   (DatetimeIndex, 'nav' column as float)
        rf_rate:   Annual risk-free rate for Sharpe/Sortino (default 6.5%)
        fund_name: Display name (for logging and summary)

    Returns:
        Dict containing:
          ── Processed series (for charts) ─────────────────────────────────
          'nav'              → clean pd.Series of NAV values
          'returns'          → daily simple returns pd.Series
          'log_returns'      → daily log returns pd.Series
          'monthly_returns'  → monthly simple returns pd.Series
          'drawdown_series'  → drawdown pd.Series (always ≤ 0)
          '_series_1y'       → 1Y rolling return pd.Series
          '_series_3y'       → 3Y rolling return pd.Series

          ── Data quality ──────────────────────────────────────────────────
          'summary'          → dict from get_series_summary()
          'is_valid'         → bool (False = insufficient data for any metric)
          'warnings'         → List[str]

          ── Scalar metrics (float or None) ────────────────────────────────
          See METRIC_LABELS in utils/constants.py for the full list.
          Every key from that dict is present here.

        On total failure, returns a dict with is_valid=False and empty metrics.
    """
    empty_result = {
        "nav": None, "returns": None, "log_returns": None,
        "monthly_returns": None, "drawdown_series": None,
        "_series_1y": None, "_series_3y": None,
        "summary": {}, "is_valid": False, "warnings": [],
    }
    # Add all scalar metrics as None
    for key in _ALL_METRIC_KEYS:
        empty_result[key] = None
    # Alpha metric fields
    empty_result["_rolling_alpha"]   = None
    empty_result["_benchmark_nav"]   = None
    empty_result["_benchmark_name"]  = ""

    # ── Step 1: Process NAV ───────────────────────────────────────────────────
    if nav_df is None:
        empty_result["warnings"] = [f"No NAV data available for {fund_name}."]
        return empty_result

    nav = process_nav(nav_df)
    if nav is None:
        empty_result["warnings"] = [f"NAV processing failed for {fund_name} — data may be corrupt."]
        return empty_result

    is_valid, nav_warnings = check_nav_series(nav)
    if not is_valid:
        empty_result["warnings"] = nav_warnings
        return empty_result

    # ── Step 2: Compute return series ─────────────────────────────────────────
    returns         = compute_daily_returns(nav)
    log_returns     = compute_log_returns(nav)
    monthly_returns = compute_monthly_returns(nav)
    summary         = get_series_summary(nav, fund_name=fund_name)

    result: Dict = {
        "nav":             nav,
        "returns":         returns,
        "log_returns":     log_returns,
        "monthly_returns": monthly_returns,
        "drawdown_series": None,   # Will be filled by risk module
        "_series_1y":      None,
        "_series_3y":      None,
        "summary":         summary,
        "is_valid":        True,
        "warnings":        nav_warnings,
    }

    # ── Step 3: Performance (CAGR) ────────────────────────────────────────────
    perf = calc_all_cagr(nav)
    result.update(perf)

    # ── Step 4: Risk (Drawdown) ───────────────────────────────────────────────
    risk = calc_all_risk(nav)
    result["drawdown_series"] = risk.pop("drawdown_series", None)
    result.update(risk)

    # ── Step 5: Volatility ────────────────────────────────────────────────────
    vol = calc_all_volatility(returns, mar=MAR)
    result.update(vol)

    # ── Step 6: Risk-Adjusted ─────────────────────────────────────────────────
    # For Calmar: prefer 3Y CAGR; fall back to inception CAGR
    cagr_for_calmar = result.get("cagr_3y") or result.get("cagr_inception")
    radj = calc_all_risk_adjusted(
        returns=returns,
        cagr_for_calmar=cagr_for_calmar,
        max_drawdown=result.get("max_drawdown"),
        rf_rate=rf_rate,
    )
    result.update(radj)

    # ── Step 7: Consistency (Rolling Returns) ─────────────────────────────────
    consistency = calc_all_consistency(nav)
    result["_series_1y"] = consistency.pop("_series_1y", None)
    result["_series_3y"] = consistency.pop("_series_3y", None)
    result.update(consistency)

    # ── Step 8: Distribution ─────────────────────────────────────────────────
    dist = calc_all_distribution(log_returns)
    result.update(dist)

    # ── Step 9: Stability ─────────────────────────────────────────────────────
    stab = calc_all_stability(returns, monthly_returns)
    result.update(stab)

    # ── Step 10: Persistence ──────────────────────────────────────────────────
    pers = calc_all_persistence(
        returns=returns,
        rolling_1y=result.get("_series_1y"),
        rolling_3y=result.get("_series_3y"),
    )
    result.update(pers)

    # ── Step 11: Alpha (Benchmark-Relative) — only if benchmark provided ──────
    result["_benchmark_nav"]  = None
    result["_benchmark_name"] = benchmark_name

    if benchmark_nav_df is not None:
        from data.nav_processor import align_nav_series

        benchmark_nav = process_nav(benchmark_nav_df)
        if benchmark_nav is not None and nav is not None:
            aligned = align_nav_series({"fund": nav, "benchmark": benchmark_nav})
            if len(aligned) == 2 and aligned.get("fund") is not None:
                b_aligned   = aligned["benchmark"]
                f_aligned   = aligned["fund"]
                b_returns   = compute_daily_returns(b_aligned)
                f_returns_b = compute_daily_returns(f_aligned)

                alpha_metrics = calc_all_alpha(f_returns_b, b_returns, rf_rate)
                result["_rolling_alpha"]  = alpha_metrics.pop("_rolling_alpha", None)
                result["_benchmark_nav"]  = benchmark_nav
                result.update(alpha_metrics)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY-LEVEL METRICS (multiple funds)
# ─────────────────────────────────────────────────────────────────────────────

def compute_category_metrics(
    fund_nav_dict: Dict[str, Optional[pd.DataFrame]],
    rf_rate: float = DEFAULT_RISK_FREE_RATE,
    progress_bar=None,
) -> Dict[str, Dict]:
    """
    Compute metrics for all funds in a category.

    Args:
        fund_nav_dict: {fund_name: nav_df_or_None, ...}
                       fund_name is used as the display label and dict key.
        rf_rate:       Annual risk-free rate for ratio calculations.
        progress_bar:  Optional Streamlit progress bar object.
                       Will be updated as each fund is processed.

    Returns:
        {fund_name: metrics_dict, ...}
        where metrics_dict is the output of compute_fund_metrics().

    Usage:
        nav_data = load_navs_for_funds(fund_list)
        all_metrics = compute_category_metrics(nav_data)
        full_table = compute_category_quartiles(all_metrics)
    """
    results: Dict[str, Dict] = {}
    total = len(fund_nav_dict)

    for i, (fund_name, nav_df) in enumerate(fund_nav_dict.items()):
        if progress_bar is not None:
            try:
                pct = (i + 1) / max(total, 1)
                progress_bar.progress(
                    pct,
                    text=f"Computing metrics: {fund_name[:50]} ({i+1}/{total})",
                )
            except Exception:
                pass

        try:
            metrics = compute_fund_metrics(nav_df, rf_rate=rf_rate, fund_name=fund_name)
            results[fund_name] = metrics
        except Exception as e:
            logger.error(f"engine: compute_fund_metrics failed for '{fund_name}': {e}")
            results[fund_name] = {"is_valid": False, "warnings": [str(e)]}

    return results


def compute_category_quartiles(
    fund_metrics: Dict[str, Dict],
) -> pd.DataFrame:
    """
    Build the quartile rankings table for a full category.

    Extracts scalar metrics from each fund's metrics dict, builds a wide
    DataFrame, and appends quartile label columns.

    Args:
        fund_metrics: Output of compute_category_metrics()

    Returns:
        DataFrame where:
            - Rows = fund names
            - Columns = metric values + '{metric}_quartile' columns
        Ready for display in Streamlit st.dataframe() or export to CSV.
    """
    return build_full_quartile_table(fund_metrics)


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT CACHED WRAPPERS
# ─────────────────────────────────────────────────────────────────────────────
# These are called from pages. st.cache_data memoises based on arguments.
# nav_df is not directly cacheable, so we cache at the scheme_code level.

@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_fund_metrics(
    scheme_code: str,
    rf_rate: float = DEFAULT_RISK_FREE_RATE,
) -> Dict:
    """
    Cached wrapper for compute_fund_metrics. Called from Fund Analytics page.

    Args:
        scheme_code: AMFI scheme code (used as cache key)
        rf_rate:     Annual risk-free rate

    Returns:
        Metrics dict for the fund (same as compute_fund_metrics output).
    """
    from data.fund_loader import get_nav_history
    nav_df = get_nav_history(scheme_code)
    return compute_fund_metrics(nav_df, rf_rate=rf_rate, fund_name=scheme_code)


@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_category_metrics(
    category: str,
    rf_rate: float = DEFAULT_RISK_FREE_RATE,
) -> Dict[str, Dict]:
    """
    Cached wrapper for computing metrics for ALL funds in a category.
    This is the most expensive operation — can take 30–120 seconds for
    large categories (Mid Cap has 25+ funds × ~2s each).

    Results are cached for 1 hour so subsequent page navigations are instant.

    Args:
        category: Category name (e.g. 'Large Cap')
        rf_rate:  Annual risk-free rate

    Returns:
        {fund_name: metrics_dict} for all funds in the category.
    """
    from data.fund_loader import get_schemes_for_category, load_navs_for_funds

    fund_list = get_schemes_for_category(category)
    if not fund_list:
        return {}

    # Load NAVs (each individual NAV is cached by get_nav_history)
    nav_dict = {
        fund["name"]: load_navs_for_funds([fund]).get(fund["code"])
        for fund in fund_list
    }

    return compute_category_metrics(nav_dict, rf_rate=rf_rate)


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL: Complete list of scalar metric keys produced by the engine
# ─────────────────────────────────────────────────────────────────────────────

_ALL_METRIC_KEYS: List[str] = [
    # Performance
    "cagr_1y", "cagr_3y", "cagr_5y", "cagr_inception",
    # Volatility
    "annualized_volatility", "downside_volatility",
    # Risk
    "max_drawdown", "avg_drawdown", "drawdown_duration",
    # Risk-adjusted
    "sharpe", "sortino", "calmar",
    # Consistency
    "avg_rolling_1y", "median_rolling_1y", "std_rolling_1y",
    "best_rolling_1y", "worst_rolling_1y",
    "avg_rolling_3y", "median_rolling_3y", "std_rolling_3y",
    "best_rolling_3y", "worst_rolling_3y",
    # Distribution
    "skewness", "kurtosis",
    # Stability
    "positive_freq", "negative_freq", "win_rate",
    # Persistence
    "pct_positive_rolling_1y", "pct_positive_rolling_3y",
    "max_consec_positive", "max_consec_negative",
    # Alpha Generation
    "excess_return", "beta", "r_squared", "tracking_error",
    "information_ratio", "jensens_alpha", "alpha_tstat",
    "up_capture", "down_capture", "capture_ratio",
]
