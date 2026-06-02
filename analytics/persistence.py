"""
analytics/persistence.py
========================
Persistence metrics — whether a fund's positive performance is sustained
or just sporadic.

These metrics go beyond "average return" to ask:
"How reliable is this fund at staying positive over time?"

% Positive Rolling Periods:
    Of all the 1-year (or 3-year) rolling windows, what fraction ended
    with a positive annualized return?

    Formula: count(rolling_return > 0) / total_rolling_windows

    100% means the fund has NEVER had a negative 1-year (or 3-year) return,
    no matter when you entered. This is a very strong consistency signal.

    95% = 1 in 20 random 1-year periods was negative.
    80% = 1 in 5 was negative.

Consecutive Positive Return Streak:
    Longest run of consecutive trading days with positive returns.
    Measures the fund's ability to sustain momentum.

Consecutive Negative Return Streak:
    Longest run of consecutive trading days with negative returns.
    A high value indicates the fund can go "stuck" in a loss-making
    phase for an extended period.

    Note: It is normal for equity funds to have runs of 5-10 consecutive
    negative days during market downturns. Runs > 20 days deserve scrutiny.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict


# ─────────────────────────────────────────────────────────────────────────────
# PERCENTAGE POSITIVE ROLLING PERIODS
# ─────────────────────────────────────────────────────────────────────────────

def calc_pct_positive_rolling(
    rolling_series: Optional[pd.Series],
) -> Optional[float]:
    """
    Fraction of rolling periods (1Y or 3Y) where the annualized return was positive.

    Args:
        rolling_series: Series of annualized rolling returns
                        (from compute_rolling_returns in nav_processor).
                        Each element = CAGR of one rolling window.

    Returns:
        Float between 0 and 1, or None if no valid data.
        e.g. 0.95 means 95% of all rolling periods were positive.
    """
    if rolling_series is None or len(rolling_series) == 0:
        return None

    clean = rolling_series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) == 0:
        return None

    return float((clean > 0).sum() / len(clean))


# ─────────────────────────────────────────────────────────────────────────────
# CONSECUTIVE STREAKS
# ─────────────────────────────────────────────────────────────────────────────

def _max_consecutive(boolean_series: pd.Series) -> int:
    """
    Find the maximum length of consecutive True values in a boolean series.

    Algorithm uses pandas groupby + cumcount for efficiency — no Python loop
    over individual elements.

    Args:
        boolean_series: A boolean pd.Series

    Returns:
        Maximum consecutive True count as int (0 if no True values).
    """
    if len(boolean_series) == 0:
        return 0

    # Create groups: every time the value changes, start a new group
    # Then within each group, count the streak length
    # This is fully vectorized
    groups = (boolean_series != boolean_series.shift()).cumsum()
    streak_lengths = boolean_series.groupby(groups).transform("sum")

    # Only count True streaks
    positive_streaks = streak_lengths[boolean_series]
    if len(positive_streaks) == 0:
        return 0

    return int(positive_streaks.max())


def calc_max_consecutive_positive(returns: Optional[pd.Series]) -> Optional[int]:
    """
    Longest consecutive streak of positive daily returns.

    Args:
        returns: Daily simple return series

    Returns:
        Integer count of trading days in the longest positive streak, or None.
    """
    if returns is None or len(returns) < 5:
        return None

    clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) == 0:
        return None

    positive = clean > 0
    return _max_consecutive(positive)


def calc_max_consecutive_negative(returns: Optional[pd.Series]) -> Optional[int]:
    """
    Longest consecutive streak of negative daily returns.

    Args:
        returns: Daily simple return series

    Returns:
        Integer count of trading days in the longest negative streak, or None.
    """
    if returns is None or len(returns) < 5:
        return None

    clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) == 0:
        return None

    negative = clean < 0
    return _max_consecutive(negative)


# ─────────────────────────────────────────────────────────────────────────────
# BATCH COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def calc_all_persistence(
    returns: Optional[pd.Series],
    rolling_1y: Optional[pd.Series],
    rolling_3y: Optional[pd.Series],
) -> Dict[str, Optional[object]]:
    """
    Compute all four persistence metrics in one call.

    Args:
        returns:    Daily simple return series
        rolling_1y: 1-year annualized rolling return series
        rolling_3y: 3-year annualized rolling return series

    Returns:
        Dict with keys:
            pct_positive_rolling_1y → float or None
            pct_positive_rolling_3y → float or None
            max_consec_positive     → int or None
            max_consec_negative     → int or None
    """
    return {
        "pct_positive_rolling_1y": calc_pct_positive_rolling(rolling_1y),
        "pct_positive_rolling_3y": calc_pct_positive_rolling(rolling_3y),
        "max_consec_positive":     calc_max_consecutive_positive(returns),
        "max_consec_negative":     calc_max_consecutive_negative(returns),
    }
