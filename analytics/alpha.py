"""
analytics/alpha.py
==================
Alpha generation metrics — all benchmark-relative calculations.

Every function takes aligned daily return series:
    fund_returns:      pd.Series  (daily simple returns of the fund)
    benchmark_returns: pd.Series  (daily simple returns of the benchmark)

Both series MUST share the same DatetimeIndex (use align_nav_series
from nav_processor.py before calling these functions).

Metrics computed:
    ── Layer 1: Return Attribution ─────────────────────────────
    excess_return_ann     Annualized mean excess return over benchmark
    beta                  Sensitivity of fund to benchmark movements
    r_squared             % of fund variance explained by benchmark
    tracking_error        Annualized std dev of daily excess returns
    information_ratio     Excess return per unit of tracking error

    ── Layer 2: Manager Skill ───────────────────────────────────
    jensens_alpha         Intercept of CAPM regression (annualized)
    alpha_tstat           t-statistic of Jensen's alpha (significance test)
    up_capture            Fund return / Benchmark return in up markets
    down_capture          Fund return / Benchmark return in down markets
    capture_ratio         Up-capture / Down-capture (key alpha signal)

    ── Layer 3: Rolling Alpha (Persistence) ─────────────────────
    rolling_alpha_series  Jensen's alpha computed over rolling windows
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Dict, Tuple
from utils.constants import TRADING_DAYS_PER_YEAR, DEFAULT_RISK_FREE_RATE


# ─────────────────────────────────────────────────────────────────────────────
# ALIGNMENT HELPER
# ─────────────────────────────────────────────────────────────────────────────

def align_returns(
    fund_returns:      pd.Series,
    benchmark_returns: pd.Series,
) -> Optional[Tuple[pd.Series, pd.Series]]:
    """
    Align two return series to their common date range.

    Returns (fund_aligned, benchmark_aligned) or None if fewer than
    60 common trading days exist (not enough for meaningful regression).
    """
    common_idx = fund_returns.index.intersection(benchmark_returns.index)
    if len(common_idx) < 60:
        return None

    f = fund_returns.reindex(common_idx).replace([np.inf, -np.inf], np.nan).dropna()
    b = benchmark_returns.reindex(common_idx).replace([np.inf, -np.inf], np.nan).dropna()

    # Keep only rows where both have valid values
    common_valid = f.index.intersection(b.index)
    if len(common_valid) < 60:
        return None

    return f.reindex(common_valid), b.reindex(common_valid)


def _geometric_mean_return(returns: pd.Series) -> float:
    """Annualized geometric mean return from a daily return series."""
    gross = (1 + returns).prod()
    n_days = len(returns)
    return float(gross ** (TRADING_DAYS_PER_YEAR / n_days) - 1)


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1 — RETURN ATTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def calc_excess_return(
    fund_returns:      pd.Series,
    benchmark_returns: pd.Series,
) -> Optional[float]:
    """
    Annualized mean excess return of the fund over the benchmark.

    Formula:
        excess_daily = fund_return_t - benchmark_return_t
        excess_ann   = mean(excess_daily) × 252

    This is the arithmetic excess return (not geometric). For most
    practical purposes arithmetic and geometric excess returns are
    very similar over horizons under 5 years.

    Args:
        fund_returns:      Daily simple return series (aligned)
        benchmark_returns: Daily simple return series (aligned)

    Returns:
        Annualized excess return as decimal (e.g. 0.03 = 3%), or None.
    """
    aligned = align_returns(fund_returns, benchmark_returns)
    if aligned is None:
        return None

    f, b = aligned
    excess_daily = f - b
    return float(excess_daily.mean() * TRADING_DAYS_PER_YEAR)


def calc_beta(
    fund_returns:      pd.Series,
    benchmark_returns: pd.Series,
) -> Optional[float]:
    """
    Beta — sensitivity of fund returns to benchmark movements.

    Formula:
        β = Cov(fund, benchmark) / Var(benchmark)

    Interpretation:
        β = 1.0  → fund moves in line with benchmark
        β > 1.0  → fund amplifies benchmark moves (aggressive)
        β < 1.0  → fund mutes benchmark moves (defensive)
        β < 0    → fund moves opposite to benchmark (rare)

    Args:
        fund_returns:      Daily simple return series
        benchmark_returns: Daily simple return series

    Returns:
        Beta as float, or None.
    """
    aligned = align_returns(fund_returns, benchmark_returns)
    if aligned is None:
        return None

    f, b = aligned
    bench_var = float(b.var(ddof=1))
    if bench_var == 0 or np.isnan(bench_var):
        return None

    covariance = float(np.cov(f.values, b.values, ddof=1)[0][1])
    return covariance / bench_var


def calc_r_squared(
    fund_returns:      pd.Series,
    benchmark_returns: pd.Series,
) -> Optional[float]:
    """
    R-Squared — percentage of fund's return variance explained by the benchmark.

    Formula:
        R² = Correlation(fund, benchmark)²

    Interpretation:
        R² = 1.0  → fund returns are perfectly explained by benchmark (index fund)
        R² = 0.7  → 70% of return variation is explained by benchmark
        R² < 0.5  → fund is highly differentiated from its benchmark (active bets)

    A fund with high alpha AND low R² is the most compelling — it generates
    returns from manager skill, not just benchmark exposure.

    Args:
        fund_returns:      Daily simple return series
        benchmark_returns: Daily simple return series

    Returns:
        R² as float between 0 and 1, or None.
    """
    aligned = align_returns(fund_returns, benchmark_returns)
    if aligned is None:
        return None

    f, b = aligned
    correlation = float(np.corrcoef(f.values, b.values)[0][1])
    return float(correlation ** 2) if np.isfinite(correlation) else None


def calc_tracking_error(
    fund_returns:      pd.Series,
    benchmark_returns: pd.Series,
) -> Optional[float]:
    """
    Tracking Error — annualized standard deviation of daily excess returns.

    Formula:
        TE = Std(fund_return - benchmark_return) × √252

    Interpretation:
        Low TE (< 2%) → fund closely mirrors benchmark (closet indexer risk)
        High TE (> 8%) → fund takes large active bets vs benchmark

    Tracking Error is used as the denominator in the Information Ratio.

    Args:
        fund_returns:      Daily simple return series
        benchmark_returns: Daily simple return series

    Returns:
        Annualized tracking error as decimal, or None.
    """
    aligned = align_returns(fund_returns, benchmark_returns)
    if aligned is None:
        return None

    f, b = aligned
    excess = f - b
    return float(excess.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def calc_information_ratio(
    fund_returns:      pd.Series,
    benchmark_returns: pd.Series,
) -> Optional[float]:
    """
    Information Ratio — annualized excess return per unit of tracking error.

    Formula:
        IR = (Mean Daily Excess Return × 252) / Tracking Error
           = Annualized Excess Return / Tracking Error

    Interpretation:
        IR > 0.5   → good — fund consistently beats benchmark
        IR > 1.0   → excellent — strong and consistent outperformance
        IR < 0     → fund consistently underperforms benchmark

    The IR is more meaningful than excess return alone because it accounts
    for consistency — a fund that beats by 3% every year (low TE) has a
    higher IR than one that beats by 3% on average but swings wildly.

    Args:
        fund_returns:      Daily simple return series
        benchmark_returns: Daily simple return series

    Returns:
        Information Ratio as float, or None.
    """
    aligned = align_returns(fund_returns, benchmark_returns)
    if aligned is None:
        return None

    f, b = aligned
    excess = f - b
    te = float(excess.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))

    if te == 0 or np.isnan(te):
        return None

    ann_excess = float(excess.mean() * TRADING_DAYS_PER_YEAR)
    ir = ann_excess / te
    return float(ir) if np.isfinite(ir) else None


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2 — MANAGER SKILL
# ─────────────────────────────────────────────────────────────────────────────

def calc_jensens_alpha(
    fund_returns:      pd.Series,
    benchmark_returns: pd.Series,
    rf_rate:           float = DEFAULT_RISK_FREE_RATE,
) -> Optional[float]:
    """
    Jensen's Alpha — return attributable to manager skill, not benchmark exposure.

    CAPM regression:
        (fund_return - rf_daily) = α + β × (benchmark_return - rf_daily) + ε

    The intercept α is Jensen's Alpha (daily).
    We annualize by multiplying by 252.

    Interpretation:
        α > 0  → manager generated returns beyond what CAPM predicts
        α = 0  → manager generated exactly what CAPM predicts (no skill)
        α < 0  → manager destroyed value relative to risk taken

    Args:
        fund_returns:      Daily simple return series
        benchmark_returns: Daily simple return series
        rf_rate:           Annual risk-free rate (e.g. 0.065)

    Returns:
        Annualized Jensen's Alpha as decimal, or None.
    """
    aligned = align_returns(fund_returns, benchmark_returns)
    if aligned is None:
        return None

    f, b = aligned
    if len(f) < 63:   # Minimum 3 months for a meaningful regression
        return None

    rf_daily = rf_rate / TRADING_DAYS_PER_YEAR
    excess_fund  = f - rf_daily
    excess_bench = b - rf_daily

    # OLS regression: excess_fund = alpha + beta * excess_bench
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        excess_bench.values, excess_fund.values
    )

    # Annualize the daily alpha intercept
    alpha_ann = float(intercept * TRADING_DAYS_PER_YEAR)
    return alpha_ann if np.isfinite(alpha_ann) else None


def calc_alpha_tstat(
    fund_returns:      pd.Series,
    benchmark_returns: pd.Series,
    rf_rate:           float = DEFAULT_RISK_FREE_RATE,
) -> Optional[float]:
    """
    t-Statistic of Jensen's Alpha — is the alpha statistically significant?

    Formula:
        t = alpha_daily / std_error_of_alpha

    Interpretation:
        |t| > 2.0  → alpha is statistically significant at 95% confidence
        |t| < 2.0  → alpha could be due to random variation (noise)

    This is critical: most funds that appear to have positive alpha fail
    this test when examined over a long enough history.

    Args:
        fund_returns:      Daily simple return series
        benchmark_returns: Daily simple return series
        rf_rate:           Annual risk-free rate

    Returns:
        t-statistic as float, or None.
    """
    aligned = align_returns(fund_returns, benchmark_returns)
    if aligned is None:
        return None

    f, b = aligned
    if len(f) < 63:
        return None

    rf_daily     = rf_rate / TRADING_DAYS_PER_YEAR
    excess_fund  = f - rf_daily
    excess_bench = b - rf_daily

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        excess_bench.values, excess_fund.values
    )

    if std_err == 0 or np.isnan(std_err):
        return None

    t_stat = float(intercept / std_err)
    return t_stat if np.isfinite(t_stat) else None


def calc_up_capture(
    fund_returns:      pd.Series,
    benchmark_returns: pd.Series,
) -> Optional[float]:
    """
    Up-Capture Ratio — how much of the benchmark's gains the fund captures.

    Formula:
        Up periods = days where benchmark_return > 0
        Up-Capture = Geometric Mean Return of fund (up periods)
                   / Geometric Mean Return of benchmark (up periods) × 100

    Interpretation:
        110% → fund gains 10% MORE than benchmark during rallies
        90%  → fund captures only 90% of benchmark gains
        > 100% is desirable for growth-oriented funds

    Args:
        fund_returns:      Daily simple return series
        benchmark_returns: Daily simple return series

    Returns:
        Up-capture ratio as percentage (e.g. 108.5), or None.
    """
    aligned = align_returns(fund_returns, benchmark_returns)
    if aligned is None:
        return None

    f, b = aligned
    up_mask = b > 0

    if up_mask.sum() < 20:   # Need at least 20 up days for meaningful estimate
        return None

    fund_up  = f[up_mask]
    bench_up = b[up_mask]

    fund_geo  = _geometric_mean_return(fund_up)
    bench_geo = _geometric_mean_return(bench_up)

    if bench_geo == 0 or np.isnan(bench_geo):
        return None

    capture = (fund_geo / bench_geo) * 100
    return float(capture) if np.isfinite(capture) else None


def calc_down_capture(
    fund_returns:      pd.Series,
    benchmark_returns: pd.Series,
) -> Optional[float]:
    """
    Down-Capture Ratio — how much of the benchmark's losses the fund suffers.

    Formula:
        Down periods = days where benchmark_return < 0
        Down-Capture = Geometric Mean Return of fund (down periods)
                     / Geometric Mean Return of benchmark (down periods) × 100

    Interpretation:
        80%  → fund loses only 80% as much as benchmark during downturns
        110% → fund loses 10% MORE than benchmark during downturns
        < 100% is desirable — shows defensive characteristics

    Args:
        fund_returns:      Daily simple return series
        benchmark_returns: Daily simple return series

    Returns:
        Down-capture ratio as percentage (e.g. 85.2), or None.
    """
    aligned = align_returns(fund_returns, benchmark_returns)
    if aligned is None:
        return None

    f, b = aligned
    down_mask = b < 0

    if down_mask.sum() < 20:
        return None

    fund_down  = f[down_mask]
    bench_down = b[down_mask]

    fund_geo  = _geometric_mean_return(fund_down)
    bench_geo = _geometric_mean_return(bench_down)

    if bench_geo == 0 or np.isnan(bench_geo):
        return None

    capture = (fund_geo / bench_geo) * 100
    return float(capture) if np.isfinite(capture) else None


def calc_capture_ratio(
    up_capture:   Optional[float],
    down_capture: Optional[float],
) -> Optional[float]:
    """
    Capture Ratio — Up-Capture divided by Down-Capture.

    This is one of the most powerful alpha signals available from NAV data.

    Formula:
        Capture Ratio = Up-Capture / Down-Capture

    Interpretation:
        > 1.20  → Excellent: fund captures significantly more upside than downside
        1.0–1.2 → Good
        0.8–1.0 → Average
        < 0.80  → Poor: fund loses more than it gains relative to benchmark

    Example: Up-Capture=110%, Down-Capture=80% → Capture Ratio=1.375
    This fund is genuinely skilled at asymmetric market participation.

    Args:
        up_capture:   Up-Capture Ratio (percentage, e.g. 110.0)
        down_capture: Down-Capture Ratio (percentage, e.g. 80.0)

    Returns:
        Capture Ratio as float, or None.
    """
    if up_capture is None or down_capture is None:
        return None
    if down_capture == 0 or np.isnan(down_capture):
        return None

    ratio = up_capture / down_capture
    return float(ratio) if np.isfinite(ratio) else None


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 3 — ROLLING ALPHA (PERSISTENCE)
# ─────────────────────────────────────────────────────────────────────────────

def calc_rolling_alpha(
    fund_returns:      pd.Series,
    benchmark_returns: pd.Series,
    rf_rate:           float = DEFAULT_RISK_FREE_RATE,
    window_days:       int = TRADING_DAYS_PER_YEAR,
) -> Optional[pd.Series]:
    """
    Rolling Jensen's Alpha — how alpha evolves over time.

    Computes Jensen's Alpha using a rolling window of `window_days` trading days.
    Each point on the series answers: "What was the manager's annualized alpha
    over the N-day window ending on this date?"

    A fund with persistently positive rolling alpha demonstrates genuine
    and sustained manager skill, not just a one-off lucky year.

    Args:
        fund_returns:      Daily simple return series
        benchmark_returns: Daily simple return series
        rf_rate:           Annual risk-free rate
        window_days:       Rolling window length in trading days (default 252 = 1Y)

    Returns:
        pd.Series of annualized rolling alpha values, or None.
    """
    aligned = align_returns(fund_returns, benchmark_returns)
    if aligned is None:
        return None

    f, b = aligned
    if len(f) < window_days + 30:
        return None

    rf_daily     = rf_rate / TRADING_DAYS_PER_YEAR
    excess_fund  = f - rf_daily
    excess_bench = b - rf_daily

    rolling_alphas = []
    rolling_dates  = []

    # Slide the window one day at a time
    # Note: this is intentionally not vectorized — OLS per window requires a loop.
    # For 10-year daily data (2,520 rows) this takes ~0.5 seconds — acceptable.
    for end in range(window_days, len(excess_fund)):
        start = end - window_days
        ef = excess_fund.iloc[start:end].values
        eb = excess_bench.iloc[start:end].values

        # Skip if too many NaN in window
        valid = (~np.isnan(ef)) & (~np.isnan(eb))
        if valid.sum() < window_days * 0.8:   # Require 80% data completeness
            continue

        try:
            slope, intercept, _, _, _ = stats.linregress(eb[valid], ef[valid])
            alpha_ann = intercept * TRADING_DAYS_PER_YEAR
            if np.isfinite(alpha_ann):
                rolling_alphas.append(alpha_ann)
                rolling_dates.append(excess_fund.index[end])
        except Exception:
            continue

    if len(rolling_alphas) < 10:
        return None

    return pd.Series(rolling_alphas, index=rolling_dates, name="rolling_alpha")


# ─────────────────────────────────────────────────────────────────────────────
# BATCH COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def calc_all_alpha(
    fund_returns:      Optional[pd.Series],
    benchmark_returns: Optional[pd.Series],
    rf_rate:           float = DEFAULT_RISK_FREE_RATE,
) -> Dict:
    """
    Compute all alpha metrics in one call.

    Args:
        fund_returns:      Daily simple return series of the fund
        benchmark_returns: Daily simple return series of the benchmark
        rf_rate:           Annual risk-free rate

    Returns:
        Dict with all alpha metric keys.
        Metrics that cannot be computed return None.
        Rolling alpha series is stored under '_rolling_alpha' key.
    """
    empty = {
        "excess_return":    None,
        "beta":             None,
        "r_squared":        None,
        "tracking_error":   None,
        "information_ratio":None,
        "jensens_alpha":    None,
        "alpha_tstat":      None,
        "up_capture":       None,
        "down_capture":     None,
        "capture_ratio":    None,
        "_rolling_alpha":   None,
    }

    if fund_returns is None or benchmark_returns is None:
        return empty

    # Layer 1
    excess = calc_excess_return(fund_returns, benchmark_returns)
    beta   = calc_beta(fund_returns, benchmark_returns)
    rsq    = calc_r_squared(fund_returns, benchmark_returns)
    te     = calc_tracking_error(fund_returns, benchmark_returns)
    ir     = calc_information_ratio(fund_returns, benchmark_returns)

    # Layer 2
    j_alpha  = calc_jensens_alpha(fund_returns, benchmark_returns, rf_rate)
    t_stat   = calc_alpha_tstat(fund_returns, benchmark_returns, rf_rate)
    up_cap   = calc_up_capture(fund_returns, benchmark_returns)
    down_cap = calc_down_capture(fund_returns, benchmark_returns)
    cap_rat  = calc_capture_ratio(up_cap, down_cap)

    # Layer 3
    roll_alpha = calc_rolling_alpha(fund_returns, benchmark_returns, rf_rate)

    return {
        "excess_return":    excess,
        "beta":             beta,
        "r_squared":        rsq,
        "tracking_error":   te,
        "information_ratio":ir,
        "jensens_alpha":    j_alpha,
        "alpha_tstat":      t_stat,
        "up_capture":       up_cap,
        "down_capture":     down_cap,
        "capture_ratio":    cap_rat,
        "_rolling_alpha":   roll_alpha,   # pd.Series — for charts
    }
