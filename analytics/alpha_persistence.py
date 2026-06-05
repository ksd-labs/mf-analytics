"""
analytics/alpha_persistence.py
================================
Alpha persistence and market-regime decomposition metrics.

These metrics answer three distinct questions:

1. Is the alpha sustained over time?
   → Alpha Persistence Score

2. Is the manager skilled in BOTH bull and bear markets?
   → Bull Market Alpha / Bear Market Alpha

3. How quickly does the fund recover from losses?
   → Drawdown Recovery Rate

Why these matter:
    A fund can generate positive Jensen's Alpha over a full history while
    actually only outperforming in bull markets — the bear market drag is
    hidden in the aggregate. Bull/Bear decomposition exposes this.

    Alpha persistence answers whether the skill is repeatable or was it
    concentrated in one or two exceptional years.

    Drawdown recovery rate is a behavioural alpha signal — managers who
    cut losses quickly and rotate into recovering positions show up as
    faster recovery rates, which predicts better future drawdowns.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Dict, List
from utils.constants import TRADING_DAYS_PER_YEAR, DEFAULT_RISK_FREE_RATE


# ─────────────────────────────────────────────────────────────────────────────
# ALPHA PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

def calc_alpha_persistence(
    rolling_alpha: Optional[pd.Series],
) -> Optional[float]:
    """
    Alpha Persistence Score — fraction of rolling 1-year windows where
    the fund generated positive Jensen's Alpha vs its benchmark.

    Formula:
        Alpha Persistence = count(rolling_alpha > 0) / total_windows

    Interpretation:
        1.00 → Fund ALWAYS beats benchmark on any 1-year window (exceptional)
        0.75 → Beats benchmark in 75% of 1-year windows (strong)
        0.50 → Beats benchmark half the time (coin flip — no persistent skill)
        < 0.50 → More often below benchmark than above (negative skill signal)

    Args:
        rolling_alpha: pd.Series of annualized rolling alpha values
                       (output of calc_rolling_alpha in analytics/alpha.py)

    Returns:
        Persistence score as float between 0 and 1, or None.
    """
    if rolling_alpha is None or len(rolling_alpha) == 0:
        return None

    clean = rolling_alpha.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 10:
        return None

    return float((clean > 0).sum() / len(clean))


# ─────────────────────────────────────────────────────────────────────────────
# BULL / BEAR MARKET ALPHA DECOMPOSITION
# ─────────────────────────────────────────────────────────────────────────────

def _regime_alpha(
    fund_returns:      pd.Series,
    benchmark_returns: pd.Series,
    rf_rate:           float,
    is_bull:           bool,
    min_obs:           int = 60,
) -> Optional[float]:
    """
    Internal helper — Jensen's alpha computed on bull OR bear market days only.

    A bull day is defined as any day the benchmark had a positive return.
    A bear day is any day the benchmark had a negative return.

    Flat days (benchmark return = 0) are excluded from both regimes.

    Args:
        fund_returns:      Aligned daily simple return series
        benchmark_returns: Aligned daily simple return series
        rf_rate:           Annual risk-free rate
        is_bull:           True → use bull days; False → use bear days
        min_obs:           Minimum observations required for regression

    Returns:
        Annualized regime alpha as decimal, or None.
    """
    from analytics.alpha import align_returns

    aligned = align_returns(fund_returns, benchmark_returns)
    if aligned is None:
        return None

    f, b = aligned

    if is_bull:
        mask = b > 0   # Bull: benchmark positive days
    else:
        mask = b < 0   # Bear: benchmark negative days

    f_regime = f[mask]
    b_regime = b[mask]

    if len(f_regime) < min_obs:
        return None

    rf_daily     = rf_rate / TRADING_DAYS_PER_YEAR
    excess_fund  = f_regime - rf_daily
    excess_bench = b_regime - rf_daily

    try:
        slope, intercept, _, _, _ = stats.linregress(
            excess_bench.values, excess_fund.values
        )
        # Annualize the daily intercept
        # Use number of regime days / total days to scale annualization correctly
        regime_fraction = len(f_regime) / len(f)
        alpha_ann = float(intercept * TRADING_DAYS_PER_YEAR * regime_fraction)
        return alpha_ann if np.isfinite(alpha_ann) else None
    except Exception:
        return None


def calc_bull_alpha(
    fund_returns:      Optional[pd.Series],
    benchmark_returns: Optional[pd.Series],
    rf_rate:           float = DEFAULT_RISK_FREE_RATE,
) -> Optional[float]:
    """
    Bull Market Alpha — Jensen's alpha computed on days the benchmark rose.

    Answers: "Does this manager outperform during market rallies?"

    A fund with positive bull alpha AND positive bear alpha is genuinely
    skilled in all market conditions. A fund with only bull alpha is
    effectively a high-beta fund that looks skilled in bull markets.

    Args:
        fund_returns:      Daily simple return series (aligned with benchmark)
        benchmark_returns: Daily simple return series
        rf_rate:           Annual risk-free rate

    Returns:
        Annualized bull market alpha as decimal, or None.
    """
    if fund_returns is None or benchmark_returns is None:
        return None
    return _regime_alpha(fund_returns, benchmark_returns, rf_rate, is_bull=True)


def calc_bear_alpha(
    fund_returns:      Optional[pd.Series],
    benchmark_returns: Optional[pd.Series],
    rf_rate:           float = DEFAULT_RISK_FREE_RATE,
) -> Optional[float]:
    """
    Bear Market Alpha — Jensen's alpha computed on days the benchmark fell.

    Answers: "Does this manager protect capital during market downturns?"

    This is the more valuable form of alpha — generating positive returns
    (or smaller losses) when the market is falling is harder and rarer
    than simply riding a bull market.

    High bear alpha = defensive manager with genuine downside protection skill.
    Negative bear alpha = manager loses more than expected in downturns.

    Args:
        fund_returns:      Daily simple return series
        benchmark_returns: Daily simple return series
        rf_rate:           Annual risk-free rate

    Returns:
        Annualized bear market alpha as decimal, or None.
    """
    if fund_returns is None or benchmark_returns is None:
        return None
    return _regime_alpha(fund_returns, benchmark_returns, rf_rate, is_bull=False)


def calc_alpha_regime_ratio(
    bull_alpha: Optional[float],
    bear_alpha: Optional[float],
) -> Optional[float]:
    """
    Alpha Regime Ratio — how much more alpha does the fund generate in
    bear markets relative to bull markets?

    Formula:
        Regime Ratio = Bear Alpha / Bull Alpha

    Interpretation:
        > 1.0  → Manager is a better bear-market protector than bull-market chaser
                  (the more valuable skill for most investors)
        ≈ 1.0  → Balanced — equally skilled in both regimes
        < 1.0  → Manager generates alpha mostly in bull markets
        Negative → One of the alphas is negative

    This ratio is only meaningful when both alphas are positive.

    Args:
        bull_alpha: Annualized bull market alpha
        bear_alpha: Annualized bear market alpha

    Returns:
        Regime ratio as float, or None.
    """
    if bull_alpha is None or bear_alpha is None:
        return None
    if bull_alpha == 0 or np.isnan(bull_alpha):
        return None

    ratio = bear_alpha / bull_alpha
    return float(ratio) if np.isfinite(ratio) else None


# ─────────────────────────────────────────────────────────────────────────────
# DRAWDOWN RECOVERY RATE
# ─────────────────────────────────────────────────────────────────────────────

def calc_drawdown_recovery_rate(
    nav: Optional[pd.Series],
    min_drawdown_pct: float = 0.02,
) -> Optional[float]:
    """
    Drawdown Recovery Rate — average calendar days to recover from a drawdown.

    For each completed drawdown period (fund falls below peak, then recovers):
        1. Record the date the drawdown began
        2. Record the date the fund recovered to its previous peak
        3. Compute recovery_days = recovery_date - drawdown_start_date

    Average across all completed drawdowns (ignoring trivial dips < min_drawdown_pct).

    Interpretation:
        Lower = faster recovery = better manager responsiveness
        Very long recovery durations indicate structural portfolio problems

    Note: Ongoing (incomplete) drawdowns at the end of the series are excluded
    because we cannot know their ultimate duration.

    Args:
        nav:              Clean daily NAV series (DatetimeIndex, ascending)
        min_drawdown_pct: Minimum drawdown depth to include (default 2%).
                          Filters out trivial noise dips.

    Returns:
        Average recovery duration in calendar days as float, or None.
        Returns 0.0 if the fund never had a qualifying drawdown.
    """
    if nav is None or len(nav) < 60:
        return None

    if not isinstance(nav.index, pd.DatetimeIndex):
        return None

    running_peak = nav.expanding(min_periods=1).max()
    drawdown     = (nav - running_peak) / running_peak   # Always ≤ 0

    recovery_durations: List[int] = []
    drawdown_start:      Optional[pd.Timestamp] = None
    peak_at_start:       float = 0.0

    for date, val in drawdown.items():
        nav_val = float(nav[date])

        if val < 0 and drawdown_start is None:
            # Drawdown begins
            drawdown_start = date
            peak_at_start  = float(running_peak[date])

        elif val >= 0 and drawdown_start is not None:
            # Recovery complete — fund back at or above previous peak
            depth = (peak_at_start - float(nav[drawdown_start])) / peak_at_start \
                    if peak_at_start > 0 else 0.0

            # Only count drawdowns deeper than min threshold
            if depth >= min_drawdown_pct:
                days = (date - drawdown_start).days
                if days > 0:
                    recovery_durations.append(days)

            drawdown_start = None
            peak_at_start  = 0.0

    if not recovery_durations:
        return 0.0

    return float(np.mean(recovery_durations))


# ─────────────────────────────────────────────────────────────────────────────
# BATCH COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def calc_all_alpha_persistence(
    rolling_alpha:     Optional[pd.Series],
    fund_returns:      Optional[pd.Series],
    benchmark_returns: Optional[pd.Series],
    nav:               Optional[pd.Series],
    rf_rate:           float = DEFAULT_RISK_FREE_RATE,
) -> Dict[str, Optional[float]]:
    """
    Compute all Phase B alpha persistence and regime metrics.

    Args:
        rolling_alpha:     Rolling alpha series from calc_rolling_alpha()
        fund_returns:      Daily simple return series of the fund
        benchmark_returns: Daily simple return series of the benchmark
        nav:               Clean daily NAV series
        rf_rate:           Annual risk-free rate

    Returns:
        Dict with keys:
            alpha_persistence, bull_alpha, bear_alpha,
            alpha_regime_ratio, drawdown_recovery_rate
    """
    bull  = calc_bull_alpha(fund_returns, benchmark_returns, rf_rate)
    bear  = calc_bear_alpha(fund_returns, benchmark_returns, rf_rate)

    return {
        "alpha_persistence":     calc_alpha_persistence(rolling_alpha),
        "bull_alpha":            bull,
        "bear_alpha":            bear,
        "alpha_regime_ratio":    calc_alpha_regime_ratio(bull, bear),
        "drawdown_recovery_rate":calc_drawdown_recovery_rate(nav),
    }
