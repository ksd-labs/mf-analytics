"""
analytics/engine.py
===================
The analytics orchestrator — single entry point for all metric calculations.

Pages NEVER call individual analytics functions directly. Instead they call:
    compute_fund_metrics(nav_df, rf_rate)      → metrics dict for one fund
    compute_category_metrics(fund_nav_dict, ..) → metrics for all funds in category
    compute_category_quartiles(fund_metrics)   → adds quartile labels

Computation order:
    1.  Process raw NAV → clean nav, daily returns, log returns, monthly returns
    2.  Performance (CAGR)
    3.  Risk (Drawdown)
    4.  Volatility
    5.  Risk-Adjusted (Sharpe, Sortino, Calmar)
    6.  Consistency (Rolling Returns)
    7.  Distribution (Skewness, Kurtosis)
    8.  Stability (Win Rate)
    9.  Persistence (% positive rolling periods, streaks)
    10. Alpha (Benchmark-Relative) — Phase A
    11. Momentum — Phase B
    12. Alpha Persistence + Bull/Bear — Phase B
    13. Factor Model (Fama-French-Carhart 4-Factor) — Phase C

Note (Phase D): Step 14 (Active Share Proxies) removed.
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
from analytics.alpha             import calc_all_alpha
from analytics.momentum          import calc_all_momentum
from analytics.alpha_persistence import calc_all_alpha_persistence
from analytics.factor_model      import calc_all_factor_model
from analytics.quartile       import build_full_quartile_table
from utils.constants          import DEFAULT_RISK_FREE_RATE, MAR
from utils.validators         import check_nav_series

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE FUND METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_fund_metrics(
    nav_df:            Optional[pd.DataFrame],
    rf_rate:           float = DEFAULT_RISK_FREE_RATE,
    fund_name:         str   = "",
    benchmark_nav_df:  Optional[pd.DataFrame] = None,
    benchmark_name:    str   = "",
    factor_returns_df: Optional[pd.DataFrame] = None,
) -> Dict:
    empty_result = {
        "nav": None, "returns": None, "log_returns": None,
        "monthly_returns": None, "drawdown_series": None,
        "_series_1y": None, "_series_3y": None,
        "summary": {}, "is_valid": False, "warnings": [],
    }
    for key in _ALL_METRIC_KEYS:
        empty_result[key] = None
    empty_result["_rolling_alpha"]     = None
    empty_result["_benchmark_nav"]     = None
    empty_result["_benchmark_name"]    = ""
    empty_result["_benchmark_returns"] = None
    empty_result["_rolling_alpha_4f"]  = None
    empty_result["_factor_names_used"] = []

    # ── Step 1: Process NAV ───────────────────────────────────────────────────
    if nav_df is None:
        empty_result["warnings"] = [f"No NAV data available for {fund_name}."]
        return empty_result

    nav = process_nav(nav_df)
    if nav is None:
        empty_result["warnings"] = [f"NAV processing failed for {fund_name}."]
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
        "drawdown_series": None,
        "_series_1y":      None,
        "_series_3y":      None,
        "summary":         summary,
        "is_valid":        True,
        "warnings":        nav_warnings,
    }

    # ── Step 3: Performance (CAGR) ────────────────────────────────────────────
    result.update(calc_all_cagr(nav))

    # ── Step 4: Risk (Drawdown) ───────────────────────────────────────────────
    risk = calc_all_risk(nav)
    result["drawdown_series"] = risk.pop("drawdown_series", None)
    result.update(risk)

    # ── Step 5: Volatility ────────────────────────────────────────────────────
    result.update(calc_all_volatility(returns, mar=MAR))

    # ── Step 6: Risk-Adjusted ─────────────────────────────────────────────────
    cagr_for_calmar = result.get("cagr_3y") or result.get("cagr_inception")
    result.update(calc_all_risk_adjusted(
        returns         = returns,
        cagr_for_calmar = cagr_for_calmar,
        max_drawdown    = result.get("max_drawdown"),
        rf_rate         = rf_rate,
    ))

    # ── Step 7: Consistency (Rolling Returns) ─────────────────────────────────
    consistency = calc_all_consistency(nav)
    result["_series_1y"] = consistency.pop("_series_1y", None)
    result["_series_3y"] = consistency.pop("_series_3y", None)
    result.update(consistency)

    # ── Step 8: Distribution ─────────────────────────────────────────────────
    result.update(calc_all_distribution(log_returns))

    # ── Step 9: Stability ─────────────────────────────────────────────────────
    result.update(calc_all_stability(returns, monthly_returns))

    # ── Step 10: Persistence ──────────────────────────────────────────────────
    result.update(calc_all_persistence(
        returns    = returns,
        rolling_1y = result.get("_series_1y"),
        rolling_3y = result.get("_series_3y"),
    ))

    # ── Step 11: Alpha (Benchmark-Relative) ───────────────────────────────────
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
                result["_rolling_alpha"]     = alpha_metrics.pop("_rolling_alpha", None)
                result["_benchmark_nav"]     = benchmark_nav
                result["_benchmark_returns"] = b_returns
                result.update(alpha_metrics)

    # ── Step 12: Momentum ─────────────────────────────────────────────────────
    bm_returns_for_momentum = result.get("_benchmark_returns")
    result.update(calc_all_momentum(
        nav               = result.get("nav"),
        fund_returns      = result.get("returns"),
        benchmark_returns = bm_returns_for_momentum,
        rf_rate           = rf_rate,
    ))

    # ── Step 13: Alpha Persistence + Bull/Bear ────────────────────────────────
    result.update(calc_all_alpha_persistence(
        rolling_alpha     = result.get("_rolling_alpha"),
        fund_returns      = result.get("returns"),
        benchmark_returns = bm_returns_for_momentum,
        nav               = result.get("nav"),
        rf_rate           = rf_rate,
    ))

    # ── Step 14: Factor Model (Fama-French-Carhart 4-Factor) ──────────────────
    result["_rolling_alpha_4f"]  = None
    result["_factor_names_used"] = []

    if factor_returns_df is not None and not factor_returns_df.empty:
        fund_ret_for_factors = result.get("returns")
        if fund_ret_for_factors is not None:
            factor_metrics = calc_all_factor_model(
                fund_returns = fund_ret_for_factors,
                factor_df    = factor_returns_df,
                rf_rate      = rf_rate,
            )
            result["_rolling_alpha_4f"]  = factor_metrics.pop("_rolling_alpha_4f", None)
            result["_factor_names_used"] = factor_metrics.pop("factors_used", [])
            factor_metrics.pop("n_factors", None)
            result.update(factor_metrics)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY-LEVEL METRICS (multiple funds)
# ─────────────────────────────────────────────────────────────────────────────

def compute_category_metrics(
    fund_nav_dict:     Dict[str, Optional[pd.DataFrame]],
    rf_rate:           float = DEFAULT_RISK_FREE_RATE,
    progress_bar       = None,
    benchmark_nav_df:  Optional[pd.DataFrame] = None,
    benchmark_name:    str   = "",
    factor_returns_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Dict]:
    results: Dict[str, Dict] = {}
    total = len(fund_nav_dict)

    for i, (fund_name, nav_df) in enumerate(fund_nav_dict.items()):
        if progress_bar is not None:
            try:
                progress_bar.progress(
                    (i + 1) / max(total, 1),
                    text=f"Computing metrics: {fund_name[:50]} ({i+1}/{total})",
                )
            except Exception:
                pass

        try:
            results[fund_name] = compute_fund_metrics(
                nav_df,
                rf_rate           = rf_rate,
                fund_name         = fund_name,
                benchmark_nav_df  = benchmark_nav_df,
                benchmark_name    = benchmark_name,
                factor_returns_df = factor_returns_df,
            )
        except Exception as e:
            logger.error(f"engine: compute_fund_metrics failed for '{fund_name}': {e}")
            results[fund_name] = {"is_valid": False, "warnings": [str(e)]}

    return results


def compute_category_quartiles(
    fund_metrics: Dict[str, Dict],
) -> pd.DataFrame:
    return build_full_quartile_table(fund_metrics)


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT CACHED WRAPPERS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_fund_metrics(
    scheme_code: str,
    rf_rate:     float = DEFAULT_RISK_FREE_RATE,
) -> Dict:
    from data.fund_loader import get_nav_history
    nav_df = get_nav_history(scheme_code)
    return compute_fund_metrics(nav_df, rf_rate=rf_rate, fund_name=scheme_code)


@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_category_metrics(
    category:  str,
    rf_rate:   float = DEFAULT_RISK_FREE_RATE,
    plan_type: str   = "Direct",
) -> Dict[str, Dict]:
    from data.fund_loader      import get_schemes_for_category, load_navs_for_funds
    from data.benchmark_loader import get_benchmark_nav, get_benchmark_info

    fund_list = get_schemes_for_category(category, plan_type=plan_type)
    if not fund_list:
        return {}

    bm_info   = get_benchmark_info(category)
    bm_nav_df = get_benchmark_nav(category) if bm_info["available"] else None
    bm_name   = bm_info["display_name"]

    from data.factor_loader import get_factor_returns
    factor_df, _ = get_factor_returns(rf_rate=rf_rate)

    nav_dict = {
        fund["name"]: load_navs_for_funds([fund]).get(fund["code"])
        for fund in fund_list
    }

    return compute_category_metrics(
        nav_dict,
        rf_rate           = rf_rate,
        benchmark_nav_df  = bm_nav_df,
        benchmark_name    = bm_name,
        factor_returns_df = factor_df,
    )


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
    # Risk-Adjusted
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
    # Alpha Generation (Phase A)
    "excess_return", "beta", "r_squared", "tracking_error",
    "information_ratio", "jensens_alpha", "alpha_tstat",
    "up_capture", "down_capture", "capture_ratio",
    # Momentum (Phase B)
    "momentum_1m", "momentum_3m", "momentum_6m", "momentum_12m",
    "alpha_momentum", "momentum_sharpe",
    # Alpha Persistence & Regime (Phase B)
    "alpha_persistence", "bull_alpha", "bear_alpha",
    "alpha_regime_ratio", "drawdown_recovery_rate",
    # Factor Model (Phase C)
    "alpha_4f", "alpha_4f_tstat", "beta_market_4f",
    "beta_smb", "beta_hml", "beta_wml", "r_squared_4f",
    "contrib_market", "contrib_smb", "contrib_hml",
    "contrib_wml", "contrib_alpha",
]
