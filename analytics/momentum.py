"""
analytics/momentum.py
======================
Momentum metrics — short and medium-term return predictors.

Academic research (Jegadeesh & Titman 1993, Carhart 1997) consistently shows
that past 3–12 month returns predict future 3–6 month returns. Applied to
mutual funds, momentum is one of the strongest short-term alpha predictors.

Metrics computed:
    momentum_3m         Simple return over last 3 months
    momentum_6m         Simple return over last 6 months
    momentum_12m        Simple return over last 12 months (strongest predictor)
    alpha_momentum      Jensen's alpha computed over the last 12 months only
                        (benchmark-relative momentum — removes market noise)
    momentum_sharpe     12-month return / 12-month annualized volatility
                        (momentum quality — rewards smooth, sustained gains)

Implementation note:
    All momentum metrics are POINT-IN-TIME lookbacks from the most recent
    available NAV date. They answer: "What has this fund returned over the
    last N months?" — not rolling averages.

    The 1-month reversal effect (short-term mean reversion) means we should
    NOT use 1-month momentum as a forward predictor. We start from 3 months.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict
from utils.constants import TRADING_DAYS_PER_YEAR, DEFAULT_RISK_FREE_RATE
from data.nav_processor import slice_nav_for_years, compute_daily_returns


# ─────────────────────────────────────────────────────────────────────────────
# POINT-IN-TIME MOMENTUM
# ─────────────────────────────────────────────────────────────────────────────

def _momentum_for_months(
    nav:    Optional[pd.Series],
    months: int,
) -> Optional[float]:
    """
    Simple return from N months ago to most recent date.

    Formula:
        Momentum_N = (NAV_today / NAV_{N months ago}) - 1

    Args:
        nav:    Clean daily NAV series (DatetimeIndex, ascending)
        months: Lookback in calendar months (e.g. 3, 6, 12)

    Returns:
        Simple return as decimal (e.g. 0.12 = 12%), or None.
    """
    if nav is None or len(nav) < 20:
        return None

    end_date   = nav.index[-1]
    start_date = end_date - pd.DateOffset(months=months)

    # Get NAV at start date (backward fill — last available on or before that date)
    past_nav = nav[nav.index <= start_date]
    if len(past_nav) == 0:
        return None

    start_val = float(past_nav.iloc[-1])
    end_val   = float(nav.iloc[-1])

    if start_val <= 0:
        return None

    return float((end_val / start_val) - 1)


def calc_momentum_3m(nav: Optional[pd.Series]) -> Optional[float]:
    """
    3-Month Momentum — return over the last 3 calendar months.

    Interpretation:
        Positive → fund gained over last quarter
        Negative → fund lost over last quarter

    Used as a short-term trend indicator.
    Less reliable as a forward predictor than 6M or 12M.
    """
    return _momentum_for_months(nav, months=3)


def calc_momentum_6m(nav: Optional[pd.Series]) -> Optional[float]:
    """
    6-Month Momentum — return over the last 6 calendar months.

    One of the most studied momentum windows in academic literature.
    Funds in the top quartile of 6M momentum tend to outperform over
    the following 3–6 months (cross-sectional momentum effect).
    """
    return _momentum_for_months(nav, months=6)


def calc_momentum_12m(nav: Optional[pd.Series]) -> Optional[float]:
    """
    12-Month Momentum — return over the last 12 calendar months.

    The single strongest momentum predictor in mutual fund research.
    Top-quartile 12M momentum funds have historically outperformed
    bottom-quartile funds by 4–8% annually over the following year.

    Note: We do NOT exclude the most recent 1 month (as is standard in
    stock momentum research) because mutual funds have lower turnover
    and the 1-month reversal effect is weaker at the fund level.
    """
    return _momentum_for_months(nav, months=12)


# ─────────────────────────────────────────────────────────────────────────────
# ALPHA MOMENTUM (BENCHMARK-RELATIVE)
# ─────────────────────────────────────────────────────────────────────────────

def calc_alpha_momentum(
    fund_returns:      Optional[pd.Series],
    benchmark_returns: Optional[pd.Series],
    rf_rate:           float = DEFAULT_RISK_FREE_RATE,
    months:            int = 12,
) -> Optional[float]:
    """
    Alpha Momentum — Jensen's alpha computed over the last N months only.

    This is a purer momentum signal than raw return momentum because it
    removes the market component. A fund can have high 12M momentum simply
    because the market rallied — alpha momentum asks: "Did the fund beat
    its benchmark over the last year, and by how much?"

    Formula:
        Slice fund and benchmark returns to last N months
        Run CAPM regression on the slice
        Return annualized alpha from that regression

    Args:
        fund_returns:      Full daily simple return series
        benchmark_returns: Full daily simple return series
        rf_rate:           Annual risk-free rate
        months:            Lookback window in months (default 12)

    Returns:
        Annualized alpha over the lookback period, or None.
    """
    if fund_returns is None or benchmark_returns is None:
        return None

    end_date   = min(fund_returns.index[-1], benchmark_returns.index[-1])
    start_date = end_date - pd.DateOffset(months=months)

    f = fund_returns[(fund_returns.index >= start_date) & (fund_returns.index <= end_date)]
    b = benchmark_returns[(benchmark_returns.index >= start_date) & (benchmark_returns.index <= end_date)]

    if len(f) < 30 or len(b) < 30:
        return None

    from analytics.alpha import calc_jensens_alpha
    return calc_jensens_alpha(f, b, rf_rate=rf_rate)


# ─────────────────────────────────────────────────────────────────────────────
# RISK-ADJUSTED MOMENTUM
# ─────────────────────────────────────────────────────────────────────────────

def calc_momentum_sharpe(
    nav:    Optional[pd.Series],
    months: int = 12,
) -> Optional[float]:
    """
    Momentum Sharpe — 12-month return divided by 12-month annualized volatility.

    Formula:
        MomSharpe = 12M_Return / 12M_Ann_Volatility

    This is effectively a short-horizon Sharpe ratio and answers:
    "How much return per unit of risk did the fund generate recently?"

    A fund with strong momentum AND low recent volatility has a high
    Momentum Sharpe — a higher quality momentum signal than raw return alone.

    Interpretation:
        > 1.5  → Strong quality momentum
        1.0–1.5 → Good
        0.5–1.0 → Moderate
        < 0    → Negative recent momentum

    Args:
        nav:    Clean daily NAV series
        months: Lookback window in months (default 12)

    Returns:
        Momentum Sharpe ratio as float, or None.
    """
    if nav is None or len(nav) < 60:
        return None

    end_date   = nav.index[-1]
    start_date = end_date - pd.DateOffset(months=months)

    sliced = nav[nav.index >= start_date]
    if len(sliced) < 30:
        return None

    # Return over the period
    momentum = float((sliced.iloc[-1] / sliced.iloc[0]) - 1)

    # Annualized volatility over the period
    returns = compute_daily_returns(sliced)
    if returns is None or len(returns) < 20:
        return None

    vol = float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    if vol == 0 or np.isnan(vol):
        return None

    result = momentum / vol
    return float(result) if np.isfinite(result) else None


# ─────────────────────────────────────────────────────────────────────────────
# BATCH COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def calc_all_momentum(
    nav:               Optional[pd.Series],
    fund_returns:      Optional[pd.Series] = None,
    benchmark_returns: Optional[pd.Series] = None,
    rf_rate:           float = DEFAULT_RISK_FREE_RATE,
) -> Dict[str, Optional[float]]:
    """
    Compute all momentum metrics in one call.

    Args:
        nav:               Clean daily NAV series
        fund_returns:      Daily simple return series (for alpha momentum)
        benchmark_returns: Daily simple return series of benchmark
        rf_rate:           Annual risk-free rate

    Returns:
        Dict with keys:
            momentum_3m, momentum_6m, momentum_12m,
            alpha_momentum, momentum_sharpe
    """
    return {
        "momentum_3m":     calc_momentum_3m(nav),
        "momentum_6m":     calc_momentum_6m(nav),
        "momentum_12m":    calc_momentum_12m(nav),
        "alpha_momentum":  calc_alpha_momentum(
            fund_returns, benchmark_returns, rf_rate
        ) if fund_returns is not None and benchmark_returns is not None else None,
        "momentum_sharpe": calc_momentum_sharpe(nav),
    }
