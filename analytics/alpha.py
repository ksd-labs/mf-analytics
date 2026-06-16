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

Note (Phase D): Active Share Proxies removed — NAV-based proxies
(TE-proxy, 1-R² proxy, Active Bet Score) systematically mislabel
Indian equity funds as closet indexers due to structural market
concentration. True Active Share requires portfolio holdings data.
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
    common_idx = fund_returns.index.intersection(benchmark_returns.index)
    if len(common_idx) < 60:
        return None

    f = fund_returns.reindex(common_idx).replace([np.inf, -np.inf], np.nan).dropna()
    b = benchmark_returns.reindex(common_idx).replace([np.inf, -np.inf], np.nan).dropna()

    common_valid = f.index.intersection(b.index)
    if len(common_valid) < 60:
        return None

    return f.reindex(common_valid), b.reindex(common_valid)


def _geometric_mean_return(returns: pd.Series) -> float:
    gross  = (1 + returns).prod()
    n_days = len(returns)
    return float(gross ** (TRADING_DAYS_PER_YEAR / n_days) - 1)


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1 — RETURN ATTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def calc_excess_return(
    fund_returns:      pd.Series,
    benchmark_returns: pd.Series,
) -> Optional[float]:
    aligned = align_returns(fund_returns, benchmark_returns)
    if aligned is None:
        return None
    f, b = aligned
    return float((f - b).mean() * TRADING_DAYS_PER_YEAR)


def calc_beta(
    fund_returns:      pd.Series,
    benchmark_returns: pd.Series,
) -> Optional[float]:
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
    aligned = align_returns(fund_returns, benchmark_returns)
    if aligned is None:
        return None
    f, b        = aligned
    correlation = float(np.corrcoef(f.values, b.values)[0][1])
    return float(correlation ** 2) if np.isfinite(correlation) else None


def calc_tracking_error(
    fund_returns:      pd.Series,
    benchmark_returns: pd.Series,
) -> Optional[float]:
    aligned = align_returns(fund_returns, benchmark_returns)
    if aligned is None:
        return None
    f, b   = aligned
    excess = f - b
    return float(excess.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def calc_information_ratio(
    fund_returns:      pd.Series,
    benchmark_returns: pd.Series,
) -> Optional[float]:
    aligned = align_returns(fund_returns, benchmark_returns)
    if aligned is None:
        return None
    f, b   = aligned
    excess = f - b
    te     = float(excess.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    if te == 0 or np.isnan(te):
        return None
    ann_excess = float(excess.mean() * TRADING_DAYS_PER_YEAR)
    ir         = ann_excess / te
    return float(ir) if np.isfinite(ir) else None


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2 — MANAGER SKILL
# ─────────────────────────────────────────────────────────────────────────────

def calc_jensens_alpha(
    fund_returns:      pd.Series,
    benchmark_returns: pd.Series,
    rf_rate:           float = DEFAULT_RISK_FREE_RATE,
) -> Optional[float]:
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
    alpha_ann = float(intercept * TRADING_DAYS_PER_YEAR)
    return alpha_ann if np.isfinite(alpha_ann) else None


def calc_alpha_tstat(
    fund_returns:      pd.Series,
    benchmark_returns: pd.Series,
    rf_rate:           float = DEFAULT_RISK_FREE_RATE,
) -> Optional[float]:
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
    aligned = align_returns(fund_returns, benchmark_returns)
    if aligned is None:
        return None
    f, b    = aligned
    up_mask = b > 0
    if up_mask.sum() < 20:
        return None
    fund_geo  = _geometric_mean_return(f[up_mask])
    bench_geo = _geometric_mean_return(b[up_mask])
    if bench_geo == 0 or np.isnan(bench_geo):
        return None
    capture = (fund_geo / bench_geo) * 100
    return float(capture) if np.isfinite(capture) else None


def calc_down_capture(
    fund_returns:      pd.Series,
    benchmark_returns: pd.Series,
) -> Optional[float]:
    aligned = align_returns(fund_returns, benchmark_returns)
    if aligned is None:
        return None
    f, b      = aligned
    down_mask = b < 0
    if down_mask.sum() < 20:
        return None
    fund_geo  = _geometric_mean_return(f[down_mask])
    bench_geo = _geometric_mean_return(b[down_mask])
    if bench_geo == 0 or np.isnan(bench_geo):
        return None
    capture = (fund_geo / bench_geo) * 100
    return float(capture) if np.isfinite(capture) else None


def calc_capture_ratio(
    up_capture:   Optional[float],
    down_capture: Optional[float],
) -> Optional[float]:
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
    window_days:       int   = TRADING_DAYS_PER_YEAR,
) -> Optional[pd.Series]:
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

    for end in range(window_days, len(excess_fund)):
        start = end - window_days
        ef    = excess_fund.iloc[start:end].values
        eb    = excess_bench.iloc[start:end].values
        valid = (~np.isnan(ef)) & (~np.isnan(eb))
        if valid.sum() < window_days * 0.8:
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

    excess   = calc_excess_return(fund_returns, benchmark_returns)
    beta     = calc_beta(fund_returns, benchmark_returns)
    rsq      = calc_r_squared(fund_returns, benchmark_returns)
    te       = calc_tracking_error(fund_returns, benchmark_returns)
    ir       = calc_information_ratio(fund_returns, benchmark_returns)
    j_alpha  = calc_jensens_alpha(fund_returns, benchmark_returns, rf_rate)
    t_stat   = calc_alpha_tstat(fund_returns, benchmark_returns, rf_rate)
    up_cap   = calc_up_capture(fund_returns, benchmark_returns)
    down_cap = calc_down_capture(fund_returns, benchmark_returns)
    cap_rat  = calc_capture_ratio(up_cap, down_cap)
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
        "_rolling_alpha":   roll_alpha,
    }
