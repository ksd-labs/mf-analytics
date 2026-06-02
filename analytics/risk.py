"""
analytics/risk.py
=================
Drawdown risk metrics — how far a fund falls from its peak, and for how long.

Drawdown Series:
    At any point t, the drawdown is how far NAV has fallen from its
    all-time peak up to that date.

    Formula: DD_t = (NAV_t - Peak_t) / Peak_t

    where Peak_t = max(NAV_0, NAV_1, ..., NAV_t)

    DD_t is always ≤ 0. A value of -0.30 means the fund is 30% below its peak.

Maximum Drawdown:
    The worst single trough relative to any prior peak.
    MDD = min(DD_t) for all t

Average Drawdown:
    The average depth of all "underwater" periods.
    AVG_DD = mean(DD_t where DD_t < 0)

Drawdown Duration:
    The length (in calendar days) of the longest continuous period where
    the fund has not recovered to its previous peak.
    A fund is "in drawdown" on any day where NAV < its previous all-time high.

All metrics are most meaningful when computed over long histories (3+ years).
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# DRAWDOWN SERIES
# ─────────────────────────────────────────────────────────────────────────────

def calc_drawdown_series(nav: Optional[pd.Series]) -> Optional[pd.Series]:
    """
    Compute the full drawdown series from a NAV series.

    The drawdown at each point is the percentage decline from the
    running maximum NAV up to that date.

    Args:
        nav: Clean daily NAV series (DatetimeIndex, ascending)

    Returns:
        Series of drawdown values (≤ 0), same index as input.
        Returns None if input is invalid.

    Example:
        NAV:      100, 110, 105, 95,  115, 120
        Peak:     100, 110, 110, 110, 115, 120
        Drawdown: 0,   0,  -4.5%, -13.6%, 0,  0
    """
    if nav is None or len(nav) < 2:
        return None

    # Running maximum up to each date (expanding window)
    running_peak = nav.expanding(min_periods=1).max()

    # Drawdown relative to peak
    # Vectorized: no Python loops needed
    drawdown = (nav - running_peak) / running_peak

    return drawdown


# ─────────────────────────────────────────────────────────────────────────────
# MAXIMUM DRAWDOWN
# ─────────────────────────────────────────────────────────────────────────────

def calc_max_drawdown(nav: Optional[pd.Series]) -> Optional[float]:
    """
    Maximum Drawdown — the worst peak-to-trough decline in the NAV history.

    A value of -0.35 means the fund once fell 35% from its peak.
    This is the most commonly reported risk metric in the industry.

    Args:
        nav: Clean daily NAV series

    Returns:
        Maximum drawdown as a negative decimal (e.g. -0.35), or None.
        Returns 0.0 if the fund never declined below its starting NAV.
    """
    dd_series = calc_drawdown_series(nav)
    if dd_series is None:
        return None

    return float(dd_series.min())


# ─────────────────────────────────────────────────────────────────────────────
# AVERAGE DRAWDOWN
# ─────────────────────────────────────────────────────────────────────────────

def calc_avg_drawdown(nav: Optional[pd.Series]) -> Optional[float]:
    """
    Average Drawdown — mean depth across all days spent below a prior peak.

    Unlike Max Drawdown (which only captures the worst event), Average
    Drawdown captures the typical severity of drawdown periods.

    A fund with a moderate Max Drawdown but high Average Drawdown has
    persistent losses — it regularly spends time underwater.

    Args:
        nav: Clean daily NAV series

    Returns:
        Average drawdown as a negative decimal, or None.
        Returns 0.0 if the fund was always at or above its peak.
    """
    dd_series = calc_drawdown_series(nav)
    if dd_series is None:
        return None

    # Only count days actually in drawdown (DD < 0)
    in_drawdown = dd_series[dd_series < 0]

    if len(in_drawdown) == 0:
        return 0.0   # Fund never had a drawdown

    return float(in_drawdown.mean())


# ─────────────────────────────────────────────────────────────────────────────
# DRAWDOWN DURATION
# ─────────────────────────────────────────────────────────────────────────────

def calc_drawdown_duration(nav: Optional[pd.Series]) -> Optional[int]:
    """
    Drawdown Duration — the longest continuous period (calendar days) where
    the fund's NAV has not recovered to its previous all-time high.

    A long drawdown duration means the fund took a long time to recover
    from a loss event — this is psychologically important for investors.

    Algorithm:
        1. Build a boolean series: in_drawdown_t = (NAV_t < peak_t)
        2. Find the longest consecutive True run
        3. Return its length in calendar days

    Args:
        nav: Clean daily NAV series (DatetimeIndex required)

    Returns:
        Maximum drawdown duration in calendar days (int), or None.
        Returns 0 if the fund was never in drawdown.
    """
    if nav is None or len(nav) < 2:
        return None

    if not isinstance(nav.index, pd.DatetimeIndex):
        return None

    running_peak = nav.expanding(min_periods=1).max()

    # True where NAV is strictly below its all-time high (= in drawdown)
    in_drawdown = nav < running_peak

    max_days = 0
    streak_start: Optional[pd.Timestamp] = None

    for date, is_dd in in_drawdown.items():
        if is_dd:
            if streak_start is None:
                streak_start = date
        else:
            if streak_start is not None:
                days = (date - streak_start).days
                max_days = max(max_days, days)
                streak_start = None

    # Handle the case where the series ends while still in drawdown
    if streak_start is not None:
        days = (in_drawdown.index[-1] - streak_start).days
        max_days = max(max_days, days)

    return max_days


# ─────────────────────────────────────────────────────────────────────────────
# DRAWDOWN PERIODS TABLE (for the Drawdown Chart)
# ─────────────────────────────────────────────────────────────────────────────

def calc_drawdown_periods(
    nav: Optional[pd.Series],
    top_n: int = 5,
) -> Optional[pd.DataFrame]:
    """
    Identify the top N worst drawdown events with start/end/recovery dates.

    Used to generate the annotated Drawdown Chart in visualizations.

    Args:
        nav:   Clean daily NAV series
        top_n: Number of worst events to return

    Returns:
        DataFrame with columns:
            start_date, trough_date, end_date (recovery or None if ongoing),
            max_drawdown (negative decimal),
            duration_days
        Sorted by max_drawdown (worst first).
    """
    dd_series = calc_drawdown_series(nav)
    if dd_series is None:
        return None

    records = []
    in_dd = False
    start_date = None
    trough_dd = 0.0
    trough_date = None

    for date, val in dd_series.items():
        if val < 0 and not in_dd:
            # Drawdown starts
            in_dd = True
            start_date = date
            trough_dd = val
            trough_date = date

        elif val < 0 and in_dd:
            # Update trough
            if val < trough_dd:
                trough_dd = val
                trough_date = date

        elif val >= 0 and in_dd:
            # Recovery
            in_dd = False
            records.append({
                "start_date":    start_date,
                "trough_date":   trough_date,
                "end_date":      date,
                "max_drawdown":  trough_dd,
                "duration_days": (date - start_date).days,
            })

    # Still in drawdown at series end
    if in_dd and start_date is not None:
        records.append({
            "start_date":    start_date,
            "trough_date":   trough_date,
            "end_date":      None,
            "max_drawdown":  trough_dd,
            "duration_days": (dd_series.index[-1] - start_date).days,
        })

    if not records:
        return None

    df = pd.DataFrame(records)
    df = df.sort_values("max_drawdown").head(top_n)
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# BATCH COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def calc_all_risk(nav: Optional[pd.Series]) -> Dict[str, Optional[float]]:
    """
    Compute all three risk metrics and the drawdown series in a single call.

    The drawdown series is computed once and reused — efficient for
    page rendering where multiple charts need it simultaneously.

    Args:
        nav: Clean daily NAV series

    Returns:
        Dict with keys:
            max_drawdown       → negative decimal or None
            avg_drawdown       → negative decimal or None
            drawdown_duration  → int (calendar days) or None
            drawdown_series    → pd.Series or None (for charts)
    """
    dd_series = calc_drawdown_series(nav)

    if dd_series is None:
        return {
            "max_drawdown":      None,
            "avg_drawdown":      None,
            "drawdown_duration": None,
            "drawdown_series":   None,
        }

    # Reuse dd_series for all three metrics — no recomputation
    mdd = float(dd_series.min())

    in_dd = dd_series[dd_series < 0]
    avg_dd = float(in_dd.mean()) if len(in_dd) > 0 else 0.0

    # Duration requires the original nav for the DatetimeIndex
    duration = calc_drawdown_duration(nav)

    return {
        "max_drawdown":      mdd,
        "avg_drawdown":      avg_dd,
        "drawdown_duration": duration,
        "drawdown_series":   dd_series,    # pd.Series, for charting
    }
