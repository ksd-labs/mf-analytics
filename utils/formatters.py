"""
formatters.py
=============
Display formatting utilities for the MF Analytics Platform.

All functions are pure — they take a raw numeric value and return a
formatted string safe for display in Streamlit tables and charts.
They NEVER raise exceptions — N/A is returned for any bad input.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union


# ─────────────────────────────────────────────────────────────────────────────
# CORE FORMATTERS
# ─────────────────────────────────────────────────────────────────────────────

def fmt_pct(value: Optional[float], decimals: int = 2) -> str:
    """
    Format a float (0.15 → '15.00%').
    Handles None, NaN, and inf gracefully.

    Args:
        value:    Raw float fraction (e.g. 0.15 for 15%)
        decimals: Decimal places in output

    Returns:
        Formatted string like '15.23%' or 'N/A'
    """
    if value is None:
        return "N/A"
    try:
        if np.isnan(value) or np.isinf(value):
            return "N/A"
        return f"{value * 100:.{decimals}f}%"
    except (TypeError, ValueError):
        return "N/A"


def fmt_num(value: Optional[float], decimals: int = 2) -> str:
    """
    Format a plain float number.

    Args:
        value:    Raw float
        decimals: Decimal places

    Returns:
        Formatted string like '3.14' or 'N/A'
    """
    if value is None:
        return "N/A"
    try:
        if np.isnan(value) or np.isinf(value):
            return "N/A"
        return f"{value:.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def fmt_ratio(value: Optional[float], decimals: int = 3) -> str:
    """
    Format a financial ratio (Sharpe, Sortino, Calmar, etc.).
    Uses 3 decimal places by default for precision.

    Args:
        value:    Raw float ratio
        decimals: Decimal places (default 3)

    Returns:
        Formatted string like '1.234' or 'N/A'
    """
    return fmt_num(value, decimals)


def fmt_nav(value: Optional[float]) -> str:
    """
    Format a NAV value with Indian Rupee symbol and 4 decimal places.
    e.g. 45.2381 → '₹45.2381'

    Args:
        value: NAV as float

    Returns:
        Formatted string like '₹45.2381' or 'N/A'
    """
    if value is None:
        return "N/A"
    try:
        if np.isnan(value) or np.isinf(value):
            return "N/A"
        return f"₹{value:,.4f}"
    except (TypeError, ValueError):
        return "N/A"


def fmt_days(days: Optional[Union[int, float]]) -> str:
    """
    Format a number of calendar days into a human-readable duration string.
    e.g. 400 → '1y 1m', 45 → '1m 15d', 10 → '10d'

    Args:
        days: Number of calendar days

    Returns:
        Human-readable string or 'N/A'
    """
    if days is None:
        return "N/A"
    try:
        if np.isnan(days) or np.isinf(days):
            return "N/A"
        days = int(days)
        if days <= 0:
            return "0d"
        if days < 30:
            return f"{days}d"
        elif days < 365:
            months = days // 30
            remaining_days = days % 30
            return f"{months}m {remaining_days}d"
        else:
            years = days // 365
            remaining = days % 365
            months = remaining // 30
            return f"{years}y {months}m"
    except (TypeError, ValueError):
        return "N/A"


def fmt_date(dt) -> str:
    """
    Format a date or Timestamp to 'DD-Mon-YYYY'.
    e.g. 2020-03-15 → '15-Mar-2020'

    Args:
        dt: datetime, Timestamp, or date string

    Returns:
        Formatted date string or 'N/A'
    """
    try:
        if dt is None or (isinstance(dt, float) and np.isnan(dt)):
            return "N/A"
        return pd.Timestamp(dt).strftime("%d-%b-%Y")
    except Exception:
        return "N/A"


def fmt_large_num(value: Optional[float]) -> str:
    """
    Format large numbers with K/M/B suffixes.
    e.g. 1_500_000 → '1.5M', 2_300 → '2.3K'
    """
    if value is None:
        return "N/A"
    try:
        if np.isnan(value) or np.isinf(value):
            return "N/A"
        if abs(value) >= 1e9:
            return f"{value / 1e9:.1f}B"
        elif abs(value) >= 1e6:
            return f"{value / 1e6:.1f}M"
        elif abs(value) >= 1e3:
            return f"{value / 1e3:.1f}K"
        return f"{value:.0f}"
    except (TypeError, ValueError):
        return "N/A"


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT STYLING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def style_quartile(val: str) -> str:
    """
    Return a CSS style string for a quartile badge in a Streamlit dataframe.
    Used with df.style.applymap(style_quartile).

    Args:
        val: One of 'Q1', 'Q2', 'Q3', 'Q4', 'N/A'

    Returns:
        CSS string for background + text color
    """
    styles = {
        "Q1": "background-color: #1b5e20; color: #a5d6a7; font-weight: bold",
        "Q2": "background-color: #33691e; color: #c5e1a5; font-weight: bold",
        "Q3": "background-color: #e65100; color: #ffe0b2; font-weight: bold",
        "Q4": "background-color: #b71c1c; color: #ffcdd2; font-weight: bold",
        "N/A": "background-color: #263238; color: #78909c",
    }
    return styles.get(str(val), "")


def style_positive_negative(val: str) -> str:
    """
    Return CSS style for a value that should be green if positive, red if negative.
    Works on strings like '12.34%' or '-5.67%'.

    Used with df.style.applymap(style_positive_negative).
    """
    try:
        # Strip % and spaces to get the raw number
        raw = str(val).replace("%", "").replace("₹", "").strip()
        num = float(raw)
        if num > 0:
            return "color: #4CAF50; font-weight: bold"
        elif num < 0:
            return "color: #F44336; font-weight: bold"
        return ""
    except ValueError:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE DISPLAY BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def format_metrics_for_display(metrics: dict) -> dict:
    """
    Take a raw metrics dict (floats) and return a display-ready dict (strings).
    Applies correct formatter to each known metric key.

    Args:
        metrics: Dict with raw float values keyed by metric name

    Returns:
        Dict with same keys but formatted string values
    """
    # Define which formatter to use per metric
    PCT_METRICS = {
        "cagr_1y", "cagr_3y", "cagr_5y", "cagr_inception",
        "annualized_volatility", "downside_volatility",
        "max_drawdown", "avg_drawdown",
        "avg_rolling_1y", "median_rolling_1y", "std_rolling_1y",
        "best_rolling_1y", "worst_rolling_1y",
        "avg_rolling_3y", "median_rolling_3y", "std_rolling_3y",
        "best_rolling_3y", "worst_rolling_3y",
        "positive_freq", "negative_freq", "win_rate",
        "pct_positive_rolling_1y", "pct_positive_rolling_3y",
    }
    RATIO_METRICS = {
        "sharpe", "sortino", "calmar",
        "skewness", "kurtosis",
    }
    DAYS_METRICS = {
        "drawdown_duration",
    }
    INT_METRICS = {
        "max_consec_positive", "max_consec_negative",
    }

    display = {}
    for key, value in metrics.items():
        if key in PCT_METRICS:
            display[key] = fmt_pct(value)
        elif key in RATIO_METRICS:
            display[key] = fmt_ratio(value)
        elif key in DAYS_METRICS:
            display[key] = fmt_days(value)
        elif key in INT_METRICS:
            display[key] = str(int(value)) if value is not None and not np.isnan(value) else "N/A"
        else:
            display[key] = fmt_num(value)

    return display
