"""
analytics/volatility.py
=======================
Volatility metrics — how much a fund's returns fluctuate.

Annualized Volatility:
    The standard deviation of daily returns, scaled to annual frequency.
    Formula: σ_annual = σ_daily × √252

    This is the most widely-used risk measure. Higher = more volatile.
    We use simple returns (not log returns) to match industry convention
    for mutual fund reporting.

Downside Volatility:
    Standard deviation of returns BELOW the Minimum Acceptable Return (MAR).
    Only "bad" days count — positive days are ignored entirely.
    Formula: σ_down = std(r_t where r_t < MAR_daily) × √252

    Used as the denominator in the Sortino Ratio. Penalises funds that
    have large negative days, while ignoring upside volatility.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict
from utils.constants import TRADING_DAYS_PER_YEAR, MAR


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _clean_returns(returns: Optional[pd.Series], min_obs: int = 10) -> Optional[pd.Series]:
    """Remove NaN and inf from a returns series; return None if too few left."""
    if returns is None or len(returns) == 0:
        return None
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    return clean if len(clean) >= min_obs else None


# ─────────────────────────────────────────────────────────────────────────────
# ANNUALIZED VOLATILITY
# ─────────────────────────────────────────────────────────────────────────────

def calc_annualized_volatility(returns: Optional[pd.Series]) -> Optional[float]:
    """
    Annualized standard deviation of daily simple returns.

    Formula:
        σ_annual = std(daily_returns) × √252

    Args:
        returns: Daily simple return series (from compute_daily_returns)

    Returns:
        Annualized volatility as a decimal (e.g. 0.18 = 18%), or None.
    """
    clean = _clean_returns(returns)
    if clean is None:
        return None

    std_daily = float(clean.std(ddof=1))   # ddof=1 → unbiased estimator
    return std_daily * np.sqrt(TRADING_DAYS_PER_YEAR)


# ─────────────────────────────────────────────────────────────────────────────
# DOWNSIDE VOLATILITY
# ─────────────────────────────────────────────────────────────────────────────

def calc_downside_volatility(
    returns: Optional[pd.Series],
    mar: float = MAR,
) -> Optional[float]:
    """
    Annualized standard deviation of returns BELOW the Minimum Acceptable Return.

    The daily MAR threshold is: daily_mar = mar / TRADING_DAYS_PER_YEAR

    Only observations where r_t < daily_mar contribute to the calculation.
    If there are fewer than 5 such observations, returns None (unstable estimate).

    Args:
        returns: Daily simple return series
        mar:     Minimum Acceptable Return (annual, decimal). Default=0 (any loss).
                 Set to rf_rate to use risk-free rate as MAR.

    Returns:
        Annualized downside volatility as a decimal, or None.
    """
    clean = _clean_returns(returns)
    if clean is None:
        return None

    # Convert annual MAR to daily threshold
    daily_mar = mar / TRADING_DAYS_PER_YEAR

    # Only retain returns below the threshold
    downside = clean[clean < daily_mar]

    if len(downside) < 5:
        return None

    # Std dev of downside observations × √252
    # ddof=1 for unbiased; if only a few obs, this may be noisy
    std_down = float(downside.std(ddof=1))
    return std_down * np.sqrt(TRADING_DAYS_PER_YEAR)


# ─────────────────────────────────────────────────────────────────────────────
# BATCH COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def calc_all_volatility(
    returns: Optional[pd.Series],
    mar: float = MAR,
) -> Dict[str, Optional[float]]:
    """
    Compute both volatility metrics in one call.

    Args:
        returns: Daily simple return series
        mar:     Annual MAR for downside volatility (default 0.0)

    Returns:
        Dict with keys: annualized_volatility, downside_volatility
    """
    return {
        "annualized_volatility": calc_annualized_volatility(returns),
        "downside_volatility":   calc_downside_volatility(returns, mar=mar),
    }
