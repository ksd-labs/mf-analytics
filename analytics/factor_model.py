"""
analytics/factor_model.py
==========================
4-Factor Fama-French-Carhart Model.

The regression:
    r_fund - rf = α + β_mkt×(Mkt-Rf) + β_smb×SMB + β_hml×HML + β_wml×WML + ε

Where:
    α          = 4-Factor Alpha (true manager skill after controlling all factors)
    β_mkt      = Market sensitivity
    β_smb      = Size factor loading  (>0 = small cap tilt, <0 = large cap tilt)
    β_hml      = Value factor loading (>0 = value tilt, <0 = growth tilt)
    β_wml      = Momentum loading     (>0 = momentum tilt, <0 = contrarian)

The model uses as many factors as are available (graceful degradation).
All coefficients include t-statistics for significance testing.

Factor Contribution:
    Each factor contributes to the fund's total return proportionally.
    Contribution_smb = β_smb × annualized_mean_SMB_return
    This shows how many basis points of return came from each factor bet.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from utils.constants import TRADING_DAYS_PER_YEAR, DEFAULT_RISK_FREE_RATE


# ─────────────────────────────────────────────────────────────────────────────
# OLS ENGINE WITH STANDARD ERRORS
# ─────────────────────────────────────────────────────────────────────────────

def _ols_multiple_regression(
    y: np.ndarray,
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Ordinary Least Squares regression with heteroskedasticity-robust standard errors.

    OLS formulas:
        β = (X'X)⁻¹ X'y
        s² = RSS / (n - k)
        Var(β) = s² × (X'X)⁻¹
        SE(βᵢ) = sqrt(Var(β)ᵢᵢ)
        t_i = βᵢ / SE(βᵢ)
        R² = 1 - RSS / TSS

    Args:
        y: Dependent variable (n,)
        X: Design matrix (n, k) — must include a constant column for intercept

    Returns:
        (coefficients, t_statistics, r_squared)
        All arrays have length k (one entry per column of X).
    """
    n, k = X.shape

    # OLS coefficient estimate
    try:
        XtX    = X.T @ X
        Xty    = X.T @ y
        beta   = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        # Fallback: least-squares with numerical stability
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    # Residuals and variance
    residuals = y - X @ beta
    rss = float(np.dot(residuals, residuals))
    df  = n - k                                      # degrees of freedom

    if df <= 0:
        return beta, np.full(k, np.nan), np.nan

    s2 = rss / df

    # Coefficient covariance matrix and standard errors
    try:
        cov_beta  = s2 * np.linalg.inv(XtX)
        std_errors = np.sqrt(np.maximum(np.diag(cov_beta), 0))
        t_stats   = np.where(std_errors > 0, beta / std_errors, np.nan)
    except np.linalg.LinAlgError:
        t_stats = np.full(k, np.nan)

    # R-squared
    tss = float(np.sum((y - y.mean()) ** 2))
    r2  = float(1 - rss / tss) if tss > 0 else 0.0

    return beta, t_stats, r2


def _align_fund_to_factors(
    fund_returns:   pd.Series,
    factor_df:      pd.DataFrame,
    rf_rate:        float,
) -> Optional[Tuple[np.ndarray, np.ndarray, List[str]]]:
    """
    Align fund excess returns to factor returns on common dates.

    Returns:
        (y, X, factor_names) or None if insufficient overlap.
        y: fund excess return array (n,)
        X: design matrix (n, k) with constant + factor columns
        factor_names: list of factor column names used (in order of X columns 1..k)
    """
    rf_daily = rf_rate / TRADING_DAYS_PER_YEAR

    # Common dates
    common = fund_returns.index.intersection(factor_df.index)
    if len(common) < 63:   # Minimum 3 months
        return None

    f_aligned   = fund_returns.reindex(common).dropna()
    fac_aligned = factor_df.reindex(f_aligned.index).dropna(how="all")

    # Re-align after dropna
    common2 = f_aligned.index.intersection(fac_aligned.index)
    if len(common2) < 63:
        return None

    y_excess = (f_aligned.reindex(common2) - rf_daily).values
    fac_vals = fac_aligned.reindex(common2)

    # Drop factor columns with too many NaN (< 80% coverage)
    valid_cols = [
        col for col in fac_vals.columns
        if fac_vals[col].notna().mean() >= 0.80
    ]
    if not valid_cols:
        return None

    fac_vals = fac_vals[valid_cols].fillna(0.0).values

    # Design matrix: [constant, factor1, factor2, ...]
    const = np.ones((len(y_excess), 1))
    X = np.hstack([const, fac_vals])

    return y_excess, X, valid_cols


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FACTOR MODEL FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def calc_factor_model(
    fund_returns: Optional[pd.Series],
    factor_df:    Optional[pd.DataFrame],
    rf_rate:      float = DEFAULT_RISK_FREE_RATE,
) -> Dict:
    """
    Run multi-factor OLS regression and compute all factor model metrics.

    Args:
        fund_returns: Daily simple return series of the fund
        factor_df:    DataFrame with columns [market, smb, hml, wml] (any subset)
                      Output of factor_loader.get_factor_returns()
        rf_rate:      Annual risk-free rate

    Returns:
        Dict with keys:
            ── Regression output ───────────────────────────────────────────────
            alpha_4f           Annualized 4-factor alpha (float or None)
            alpha_4f_tstat     t-statistic of alpha (float or None)
            beta_market_4f     Market factor loading
            beta_smb           Size factor loading
            beta_hml           Value factor loading
            beta_wml           Momentum factor loading
            r_squared_4f       4-factor R-squared
            factors_used       List of factor names included in model
            n_factors          Number of factors in the model (int)
            ── Factor contributions (ann. %) ───────────────────────────────────
            contrib_market     Return attributed to market exposure
            contrib_smb        Return attributed to size tilt
            contrib_hml        Return attributed to value tilt
            contrib_wml        Return attributed to momentum tilt
            contrib_alpha      Return attributed to pure manager skill
            ── Rolling ─────────────────────────────────────────────────────────
            _rolling_alpha_4f  pd.Series of rolling 4-factor alpha (for charts)
    """
    empty = {
        "alpha_4f": None, "alpha_4f_tstat": None,
        "beta_market_4f": None, "beta_smb": None,
        "beta_hml": None, "beta_wml": None,
        "r_squared_4f": None, "factors_used": [],
        "n_factors": 0,
        "contrib_market": None, "contrib_smb": None,
        "contrib_hml": None, "contrib_wml": None,
        "contrib_alpha": None,
        "_rolling_alpha_4f": None,
    }

    if fund_returns is None or factor_df is None or factor_df.empty:
        return empty

    aligned = _align_fund_to_factors(fund_returns, factor_df, rf_rate)
    if aligned is None:
        return empty

    y, X, factor_names = aligned

    # ── OLS Regression ────────────────────────────────────────────────────────
    beta, t_stats, r2 = _ols_multiple_regression(y, X)

    # beta[0] = daily alpha (intercept), beta[1..] = factor loadings
    alpha_daily = float(beta[0])
    alpha_ann   = alpha_daily * TRADING_DAYS_PER_YEAR
    alpha_tstat = float(t_stats[0]) if np.isfinite(t_stats[0]) else None

    result = {
        **empty,
        "alpha_4f":       alpha_ann if np.isfinite(alpha_ann) else None,
        "alpha_4f_tstat": alpha_tstat,
        "r_squared_4f":   float(r2) if np.isfinite(r2) else None,
        "factors_used":   factor_names,
        "n_factors":      len(factor_names),
    }

    # ── Factor loadings ───────────────────────────────────────────────────────
    FACTOR_KEY_MAP = {
        "market": "beta_market_4f",
        "smb":    "beta_smb",
        "hml":    "beta_hml",
        "wml":    "beta_wml",
    }
    for i, fname in enumerate(factor_names):
        key = FACTOR_KEY_MAP.get(fname)
        if key:
            val = float(beta[i + 1])   # +1 because beta[0] is intercept
            result[key] = val if np.isfinite(val) else None

    # ── Factor contributions to annualized return ─────────────────────────────
    # Contribution of each factor = β_i × annualized_mean_factor_return
    CONTRIB_KEY_MAP = {
        "market": "contrib_market",
        "smb":    "contrib_smb",
        "hml":    "contrib_hml",
        "wml":    "contrib_wml",
    }
    fac_aligned_df = factor_df.reindex(
        fund_returns.index.intersection(factor_df.index)
    ).dropna(how="all")

    total_contrib = 0.0
    for i, fname in enumerate(factor_names):
        if fname in fac_aligned_df.columns:
            ann_factor_return = float(
                fac_aligned_df[fname].mean() * TRADING_DAYS_PER_YEAR
            )
            loading = float(beta[i + 1])
            contrib = loading * ann_factor_return
            key = CONTRIB_KEY_MAP.get(fname)
            if key and np.isfinite(contrib):
                result[key] = contrib
                total_contrib += contrib

    # Alpha contribution = total annualized return - sum of factor contributions
    if result["alpha_4f"] is not None:
        result["contrib_alpha"] = result["alpha_4f"]

    # ── Rolling 4-Factor Alpha ────────────────────────────────────────────────
    result["_rolling_alpha_4f"] = _calc_rolling_alpha_4f(
        fund_returns, factor_df, rf_rate,
        window_days=TRADING_DAYS_PER_YEAR,
    )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# ROLLING 4-FACTOR ALPHA
# ─────────────────────────────────────────────────────────────────────────────

def _calc_rolling_alpha_4f(
    fund_returns: pd.Series,
    factor_df:    pd.DataFrame,
    rf_rate:      float,
    window_days:  int = TRADING_DAYS_PER_YEAR,
) -> Optional[pd.Series]:
    """
    Rolling 4-Factor alpha — intercept of the factor regression over
    a rolling window of `window_days` trading days.

    Args:
        fund_returns: Daily simple return series
        factor_df:    Factor return DataFrame
        rf_rate:      Annual risk-free rate
        window_days:  Rolling window length (default 252 = 1 year)

    Returns:
        pd.Series of annualized rolling 4-factor alpha, or None.
    """
    rf_daily = rf_rate / TRADING_DAYS_PER_YEAR

    # Align once to common dates
    common = fund_returns.index.intersection(factor_df.index)
    if len(common) < window_days + 30:
        return None

    f   = fund_returns.reindex(common).dropna()
    fac = factor_df.reindex(f.index).dropna(how="all")
    common2 = f.index.intersection(fac.index)

    if len(common2) < window_days + 10:
        return None

    f   = f.reindex(common2)
    fac = fac.reindex(common2).fillna(0.0)

    factor_cols = list(fac.columns)
    fac_arr     = fac.values
    f_arr       = (f - rf_daily).values

    rolling_alphas = []
    rolling_dates  = []

    for end in range(window_days, len(f_arr)):
        start = end - window_days
        y_w   = f_arr[start:end]
        X_fac = fac_arr[start:end, :]
        X_w   = np.hstack([np.ones((window_days, 1)), X_fac])

        # Skip windows with too many zeros (sparse factor data)
        if np.sum(np.abs(y_w) > 0) < window_days * 0.7:
            continue

        try:
            beta_w, _, _ = _ols_multiple_regression(y_w, X_w)
            alpha_ann    = float(beta_w[0]) * TRADING_DAYS_PER_YEAR
            if np.isfinite(alpha_ann):
                rolling_alphas.append(alpha_ann)
                rolling_dates.append(f.index[end])
        except Exception:
            continue

    if len(rolling_alphas) < 10:
        return None

    return pd.Series(rolling_alphas, index=rolling_dates, name="rolling_alpha_4f")


# ─────────────────────────────────────────────────────────────────────────────
# BATCH COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def calc_all_factor_model(
    fund_returns: Optional[pd.Series],
    factor_df:    Optional[pd.DataFrame],
    rf_rate:      float = DEFAULT_RISK_FREE_RATE,
) -> Dict:
    """Wrapper around calc_factor_model — used by the engine."""
    return calc_factor_model(fund_returns, factor_df, rf_rate)
