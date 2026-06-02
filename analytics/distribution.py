"""
analytics/distribution.py
=========================
Return distribution shape metrics — skewness and excess kurtosis.

These metrics describe the SHAPE of the daily return distribution,
which a normal distribution (bell curve) cannot capture.

Skewness:
    Measures asymmetry of the return distribution.
    Formula: E[(r - μ)³] / σ³

    Negative skew (< 0): Left tail is longer → fund has occasional
        large losses but small gains on most days.
        Most equity funds have slight negative skew.

    Positive skew (> 0): Right tail is longer → fund has occasional
        large gains but small losses on most days.
        Value funds often show positive skew over long horizons.

    Interpretation: Negative skew is worse for investors because the
    fund "crashes more than it rallies."

Excess Kurtosis (Fisher definition):
    Measures the heaviness of the distribution's tails vs a normal distribution.
    Normal distribution has kurtosis = 3; excess kurtosis = kurtosis - 3.
    Formula: E[(r - μ)⁴] / σ⁴ − 3

    Positive excess kurtosis (> 0): Heavy tails → more extreme events
        (both large gains and large losses) than a normal distribution.
        Called "leptokurtic" or "fat-tailed."

    Negative excess kurtosis (< 0): Thin tails → fewer extreme events.
        Called "platykurtic."

    Most financial return series are leptokurtic (fat-tailed).
    High kurtosis means risk models based on normal distribution
    will underestimate the probability of extreme events.

Both metrics use LOG returns (more normally distributed than simple returns).
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Dict


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _clean_log_returns(log_returns: Optional[pd.Series], min_obs: int = 30) -> Optional[pd.Series]:
    """Remove NaN/inf, return None if too few observations."""
    if log_returns is None or len(log_returns) == 0:
        return None
    clean = log_returns.replace([np.inf, -np.inf], np.nan).dropna()
    return clean if len(clean) >= min_obs else None


# ─────────────────────────────────────────────────────────────────────────────
# SKEWNESS
# ─────────────────────────────────────────────────────────────────────────────

def calc_skewness(log_returns: Optional[pd.Series]) -> Optional[float]:
    """
    Compute the skewness of the daily log return distribution.

    Uses scipy.stats.skew with bias correction (Fisher-Pearson).
    A fund with significantly negative skew has a higher chance of
    occasional large losses than a symmetric fund.

    Args:
        log_returns: Daily log return series (from compute_log_returns)

    Returns:
        Skewness as a float, or None if insufficient data.
        - Negative: left-skewed (loss asymmetry)
        - Zero: symmetric
        - Positive: right-skewed (gain asymmetry)
    """
    clean = _clean_log_returns(log_returns)
    if clean is None:
        return None

    result = stats.skew(clean.values, bias=False)  # bias=False → adjusted skewness
    return float(result) if np.isfinite(result) else None


# ─────────────────────────────────────────────────────────────────────────────
# EXCESS KURTOSIS
# ─────────────────────────────────────────────────────────────────────────────

def calc_kurtosis(log_returns: Optional[pd.Series]) -> Optional[float]:
    """
    Compute excess kurtosis of the daily log return distribution.

    Uses scipy.stats.kurtosis with fisher=True (subtracts 3, so
    a normal distribution has kurtosis = 0).

    Mutual fund returns typically have kurtosis = 2–8 (fat tails).
    Values > 3 indicate very heavy tails — more crash risk than
    a normal distribution would suggest.

    Args:
        log_returns: Daily log return series

    Returns:
        Excess kurtosis as a float, or None if insufficient data.
        - > 0: fat-tailed (more extreme events than normal)
        - = 0: normal distribution tails
        - < 0: thin-tailed (fewer extreme events)
    """
    clean = _clean_log_returns(log_returns)
    if clean is None:
        return None

    # fisher=True → returns excess kurtosis (normal = 0, not 3)
    result = stats.kurtosis(clean.values, fisher=True, bias=False)
    return float(result) if np.isfinite(result) else None


# ─────────────────────────────────────────────────────────────────────────────
# BATCH COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def calc_all_distribution(log_returns: Optional[pd.Series]) -> Dict[str, Optional[float]]:
    """
    Compute both distribution metrics in one call.

    Args:
        log_returns: Daily log return series (from compute_log_returns)

    Returns:
        Dict with keys: skewness, kurtosis
    """
    return {
        "skewness": calc_skewness(log_returns),
        "kurtosis": calc_kurtosis(log_returns),
    }
