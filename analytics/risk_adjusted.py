"""
analytics/risk_adjusted.py
==========================
Risk-adjusted performance metrics — return per unit of risk.

These ratios answer: "Is this fund compensating investors adequately
for the risk they are taking?"

Sharpe Ratio:
    Excess return per unit of TOTAL volatility.
    Formula: (Annualized Mean Excess Return) / (Annualized Volatility)
    where Excess Return = Daily Return − (RF Rate / 252)

    Positive Sharpe = fund outperforms risk-free rate per unit of risk.
    Sharpe > 1.0 is generally considered good.

Sortino Ratio:
    Excess return per unit of DOWNSIDE volatility only.
    Same as Sharpe but penalises only downside risk (bad days).
    Formula: (Annualized Mean Excess Return) / (Downside Volatility)

    Sortino is always ≥ Sharpe (because downside_vol ≤ total_vol).
    A large Sortino vs Sharpe gap means the fund's gains are more
    volatile than its losses — a positive asymmetry.

Calmar Ratio:
    Return relative to worst historical drawdown.
    Formula: CAGR (3Y or inception) / |Max Drawdown|

    Calmar > 1.0 means the fund's annual return exceeds its worst
    historical loss — a high bar to clear.

All ratios use the configurable risk-free rate from the sidebar.
Indian context: set RF = 6.5% (91-day T-bill yield).
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict
from utils.constants import TRADING_DAYS_PER_YEAR, DEFAULT_RISK_FREE_RATE


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _clean(returns: Optional[pd.Series], min_obs: int = 30) -> Optional[pd.Series]:
    if returns is None or len(returns) == 0:
        return None
    c = returns.replace([np.inf, -np.inf], np.nan).dropna()
    return c if len(c) >= min_obs else None


def _daily_rf(annual_rf: float) -> float:
    """Convert an annual risk-free rate to a daily rate."""
    return annual_rf / TRADING_DAYS_PER_YEAR


# ─────────────────────────────────────────────────────────────────────────────
# SHARPE RATIO
# ─────────────────────────────────────────────────────────────────────────────

def calc_sharpe(
    returns: Optional[pd.Series],
    rf_rate: float = DEFAULT_RISK_FREE_RATE,
) -> Optional[float]:
    """
    Sharpe Ratio — annualized excess return per unit of total volatility.

    Implementation:
        daily_excess = daily_return − (rf_rate / 252)
        Sharpe = mean(daily_excess) / std(daily_excess) × √252

    We use std of EXCESS returns (not raw returns) in the denominator.
    This is the standard Sharpe formulation (Sharpe 1994).

    Args:
        returns: Daily simple return series
        rf_rate: Annual risk-free rate as decimal (e.g. 0.065 for 6.5%)

    Returns:
        Sharpe Ratio as a float, or None if insufficient data.
    """
    clean = _clean(returns)
    if clean is None:
        return None

    daily_excess = clean - _daily_rf(rf_rate)
    std = float(daily_excess.std(ddof=1))

    if std == 0.0 or np.isnan(std):
        return None

    sharpe = float(daily_excess.mean() / std) * np.sqrt(TRADING_DAYS_PER_YEAR)
    return sharpe if np.isfinite(sharpe) else None


# ─────────────────────────────────────────────────────────────────────────────
# SORTINO RATIO
# ─────────────────────────────────────────────────────────────────────────────

def calc_sortino(
    returns: Optional[pd.Series],
    rf_rate: float = DEFAULT_RISK_FREE_RATE,
) -> Optional[float]:
    """
    Sortino Ratio — annualized excess return per unit of DOWNSIDE volatility.

    Implementation:
        daily_excess   = daily_return − (rf_rate / 252)
        downside_returns = excess returns where daily_return < rf_daily
        downside_vol   = std(downside_returns) × √252
        Sortino        = mean(daily_excess) × 252 / downside_vol

    Args:
        returns: Daily simple return series
        rf_rate: Annual risk-free rate (e.g. 0.065)

    Returns:
        Sortino Ratio as a float, or None.
    """
    clean = _clean(returns)
    if clean is None:
        return None

    rf_daily = _daily_rf(rf_rate)
    daily_excess = clean - rf_daily

    # Downside: only days where return is below rf (i.e. daily_return < rf_daily)
    downside = clean[clean < rf_daily]

    if len(downside) < 5:
        return None

    downside_std = float(downside.std(ddof=1))
    if downside_std == 0.0 or np.isnan(downside_std):
        return None

    # Annualize numerator and denominator separately
    ann_excess_return = float(daily_excess.mean()) * TRADING_DAYS_PER_YEAR
    ann_downside_vol  = downside_std * np.sqrt(TRADING_DAYS_PER_YEAR)

    sortino = ann_excess_return / ann_downside_vol
    return float(sortino) if np.isfinite(sortino) else None


# ─────────────────────────────────────────────────────────────────────────────
# CALMAR RATIO
# ─────────────────────────────────────────────────────────────────────────────

def calc_calmar(
    cagr_value: Optional[float],
    max_drawdown_value: Optional[float],
) -> Optional[float]:
    """
    Calmar Ratio — annualized CAGR divided by the absolute Max Drawdown.

    Formula: Calmar = CAGR / |Max Drawdown|

    Convention: we use the 3-year CAGR where available, falling back to
    the inception CAGR. The engine.py selects the appropriate CAGR.

    Args:
        cagr_value:          CAGR as a decimal (e.g. 0.15)
        max_drawdown_value:  Max drawdown as a negative decimal (e.g. -0.35)

    Returns:
        Calmar Ratio as a float, or None.
    """
    if cagr_value is None or max_drawdown_value is None:
        return None

    if max_drawdown_value >= 0.0:
        # Fund has never had a drawdown — Calmar is undefined (infinite)
        return None

    calmar = cagr_value / abs(max_drawdown_value)
    return float(calmar) if np.isfinite(calmar) else None


# ─────────────────────────────────────────────────────────────────────────────
# BATCH COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def calc_all_risk_adjusted(
    returns: Optional[pd.Series],
    cagr_for_calmar: Optional[float],
    max_drawdown: Optional[float],
    rf_rate: float = DEFAULT_RISK_FREE_RATE,
) -> Dict[str, Optional[float]]:
    """
    Compute Sharpe, Sortino, and Calmar in one call.

    Args:
        returns:          Daily simple return series
        cagr_for_calmar:  CAGR to use in Calmar (3Y preferred, else inception)
        max_drawdown:     Max drawdown value (negative decimal)
        rf_rate:          Annual risk-free rate

    Returns:
        Dict with keys: sharpe, sortino, calmar
    """
    return {
        "sharpe":  calc_sharpe(returns, rf_rate=rf_rate),
        "sortino": calc_sortino(returns, rf_rate=rf_rate),
        "calmar":  calc_calmar(cagr_for_calmar, max_drawdown),
    }
