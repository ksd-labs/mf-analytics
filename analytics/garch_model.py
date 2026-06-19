"""
analytics/garch_model.py
========================
GARCH(1,1) volatility model for daily mutual fund return series.

Architecture:
  - get_garch_summary() is the single entry point called by pages
  - All other functions are internal helpers
  - Returns a flat dict consistent with the rest of the analytics layer

Why GARCH?
  Volatility clustering — high-volatility days tend to cluster — is one of
  the most robustly documented phenomena in financial returns. GARCH captures
  this by modelling today's variance as a function of yesterday's variance and
  yesterday's squared return. This produces genuinely informative short-horizon
  volatility forecasts.

What this does NOT do:
  - Forecast future returns (expected return is set to the historical mean)
  - Predict the direction of next price move
  - Provide investment advice

Distribution assumption:
  Normal (Gaussian) residuals — simplest, most transparent.
  Limitation: normal distribution underestimates tail risk.
  Upgrade path: change dist='normal' to dist='t' for heavier tails.

Dependencies:
  arch >= 6.0.0  (pip install arch)
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict
from utils.constants import TRADING_DAYS_PER_YEAR


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL: Model fitting
# ─────────────────────────────────────────────────────────────────────────────

def _fit_garch(returns_pct: pd.Series):
    """
    Fit GARCH(1,1) with normal residuals to a PERCENTAGE-scaled return series.

    Scaling to % (multiply by 100) before fitting is standard practice for
    GARCH on daily returns — it avoids numerical precision issues when
    optimising over very small variance values (e.g. 0.0001).

    Returns arch ARCHModelResult or None on failure.
    """
    try:
        from arch import arch_model

        if len(returns_pct) < 252:
            return None

        model  = arch_model(returns_pct, vol='GARCH', p=1, q=1, dist='normal', mean='Constant')
        result = model.fit(disp='off', options={'maxiter': 2000, 'ftol': 1e-9})

        # Reject non-convergent fits
        if result.convergence_flag != 0:
            return None

        # Sanity check: GARCH parameters should be positive and sum < 1
        params = result.params
        alpha = params.get('alpha[1]', None)
        beta  = params.get('beta[1]',  None)
        if alpha is None or beta is None:
            return None
        if alpha < 0 or beta < 0 or (alpha + beta) >= 1.0:
            return None

        return result
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL: Metric extraction
# ─────────────────────────────────────────────────────────────────────────────

def _get_conditional_volatility(result) -> Optional[pd.Series]:
    """
    Extract historical conditional volatility as annualised decimal series.

    result.conditional_volatility is in % daily units (because we scaled input).
    Convert: cond_vol_ann = (cond_vol_pct_daily / 100) * sqrt(252)
    """
    try:
        cond_vol_pct_daily  = result.conditional_volatility  # pd.Series, % units daily
        cond_vol_decimal_ann = (cond_vol_pct_daily / 100.0) * np.sqrt(TRADING_DAYS_PER_YEAR)
        return cond_vol_decimal_ann
    except Exception:
        return None


def _forecast_volatility(result, horizons: list = [30, 60, 90]) -> Optional[Dict]:
    """
    Forecast annualised volatility for N-day horizons.

    Uses the expected variance over the horizon window:
        avg_var_h = mean(E[σ²_t+1], E[σ²_t+2], ..., E[σ²_t+h])
        vol_h_ann = sqrt(avg_var_h) / 100 * sqrt(252)

    This gives "if current conditions persist, what is the expected
    annualised volatility over the next h days?"
    """
    try:
        max_h    = max(horizons)
        forecast = result.forecast(horizon=max_h, reindex=False)
        # forecast.variance has shape (1, max_h), values in pct-squared units
        var_array = forecast.variance.values[-1]   # shape: (max_h,)

        result_dict = {}
        for h in horizons:
            avg_var_pct_sq  = var_array[:h].mean()
            vol_decimal_ann = np.sqrt(avg_var_pct_sq) / 100.0 * np.sqrt(TRADING_DAYS_PER_YEAR)
            result_dict[h]  = float(vol_decimal_ann)

        return result_dict
    except Exception:
        return None


def _compute_var_cvar(
    result,
    confidence_levels: list = [0.95, 0.99],
) -> Optional[Dict]:
    """
    Compute 1-day VaR and CVaR from the GARCH conditional distribution.

    Parametric formulas under normal distribution assumption:
        VaR_α  = -(μ + σ_current × Φ⁻¹(1-α))
        CVaR_α = -(μ - σ_current × φ(Φ⁻¹(α)) / α)

    where μ and σ_current are in decimal (not %) units.
    VaR and CVaR returned as positive decimals (loss magnitudes).
    """
    try:
        from scipy.stats import norm

        # Current conditional vol (most recent day), converted to decimal daily
        current_vol_decimal = float(result.conditional_volatility.iloc[-1]) / 100.0

        # Mean daily return in decimal
        mean_return_decimal = float(result.resid.mean()) / 100.0

        output = {}
        for cl in confidence_levels:
            alpha = 1.0 - cl
            z     = norm.ppf(alpha)                             # negative value

            # VaR: how much could we lose with probability (1-cl)?
            var  = -(mean_return_decimal + current_vol_decimal * z)

            # CVaR: expected loss given we exceed VaR
            cvar = -(mean_return_decimal - current_vol_decimal * norm.pdf(z) / alpha)

            output[cl] = {
                "var":  max(var,  0.0),   # ensure non-negative
                "cvar": max(cvar, 0.0),
            }

        return output
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def get_garch_summary(
    returns:   pd.Series,
    rf_rate:   float = 0.065,
    horizons:  list  = [30, 60, 90],
    cl_levels: list  = [0.95, 0.99],
) -> Dict:
    """
    Full GARCH(1,1) analytics summary for a daily return series.

    Args:
        returns:   Daily simple return series (decimal, e.g. 0.01 = 1%)
        rf_rate:   Annual risk-free rate (not used in GARCH itself, reserved)
        horizons:  N-day ahead volatility forecast horizons
        cl_levels: Confidence levels for VaR/CVaR

    Returns dict with keys:
        is_valid           — bool, False if model failed to fit
        cond_vol           — pd.Series, historical conditional vol (annualised)
        forecasts          — {horizon_days: annualised vol forecast}
        var_cvar           — {confidence: {var, cvar}} as positive decimals
        current_ann_vol    — float, today's conditional vol (annualised)
        historical_avg_vol — float, long-run average conditional vol (annualised)
        current_vol_regime — str, "Low" / "Normal" / "High" vs long-run average
        omega, alpha, beta — GARCH parameters
        persistence        — alpha + beta (near 1 = long-memory vol)
        half_life_days     — days for shock to decay to half its initial size
    """
    empty = {
        "is_valid": False, "error": "Insufficient data or model failed to converge.",
        "cond_vol": None, "forecasts": None, "var_cvar": None,
        "current_ann_vol": None, "historical_avg_vol": None,
        "current_vol_regime": None, "omega": None, "alpha": None,
        "beta": None, "persistence": None, "half_life_days": None,
    }

    clean = returns.dropna()
    if len(clean) < 252:
        empty["error"] = "Minimum 252 trading days required for GARCH estimation."
        return empty

    # Scale to % for numerical stability
    returns_pct = clean * 100.0
    result      = _fit_garch(returns_pct)

    if result is None:
        return empty

    # Extract components
    cond_vol    = _get_conditional_volatility(result)
    forecasts   = _forecast_volatility(result, horizons)
    var_cvar    = _compute_var_cvar(result, cl_levels)

    if cond_vol is None or forecasts is None:
        return empty

    # Model parameters
    params      = result.params
    omega       = float(params.get('omega',   np.nan))
    alpha       = float(params.get('alpha[1]',np.nan))
    beta        = float(params.get('beta[1]', np.nan))
    persistence = alpha + beta
    half_life   = np.log(0.5) / np.log(persistence) if 0 < persistence < 1 else np.nan

    # Current vol regime classification
    current_vol = float(cond_vol.iloc[-1])
    hist_avg    = float(cond_vol.mean())
    hist_std    = float(cond_vol.std())
    if current_vol > hist_avg + hist_std:
        regime = "High"
    elif current_vol < hist_avg - hist_std:
        regime = "Low"
    else:
        regime = "Normal"

    return {
        "is_valid":           True,
        "cond_vol":           cond_vol,
        "forecasts":          forecasts,
        "var_cvar":           var_cvar,
        "current_ann_vol":    current_vol,
        "historical_avg_vol": hist_avg,
        "current_vol_regime": regime,
        "omega":              omega,
        "alpha":              alpha,
        "beta":               beta,
        "persistence":        persistence,
        "half_life_days":     float(half_life) if np.isfinite(half_life) else None,
    }
