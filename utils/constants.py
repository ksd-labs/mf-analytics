"""
constants.py
============
Central configuration file for the MF Analytics Platform.

All magic numbers, category definitions, keyword mappings, color palettes,
and data sufficiency thresholds live here. Edit this file to:
  - Add/remove categories
  - Change the default risk-free rate
  - Adjust minimum data requirements
  - Update keyword mappings as SEBI renames categories
"""

from typing import Dict, List

# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION METADATA
# ─────────────────────────────────────────────────────────────────────────────

APP_TITLE: str       = "MF Quantitative Analytics"
APP_ICON: str        = "📊"
APP_SUBTITLE: str    = "Institutional-Grade Mutual Fund Analysis · India"
APP_VERSION: str       = "1.0.0"

# Bump this string every time new metrics are added to the engine.
# All session_state analytics cache keys include this value — changing it
# forces every cached result to be recomputed automatically.
ANALYTICS_VERSION: str = "phase_b"

# ─────────────────────────────────────────────────────────────────────────────
# FINANCIAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Approximate Indian 91-day T-bill yield (annualized).
# This is the default value shown in the sidebar slider.
# Users can override it interactively.
DEFAULT_RISK_FREE_RATE: float = 0.065   # 6.5%

# Number of trading days assumed in one year for annualization.
# Indian equity markets trade ~252 days/year.
TRADING_DAYS_PER_YEAR: int = 252

# Minimum Acceptable Return used in Sortino / Downside Volatility.
# Set to 0 → any negative return counts as downside.
MAR: float = 0.0

# ─────────────────────────────────────────────────────────────────────────────
# SUPPORTED FUND CATEGORIES
# ─────────────────────────────────────────────────────────────────────────────

CATEGORIES: List[str] = [
    "Large Cap",
    "Mid Cap",
    "Small Cap",
    "Flexi Cap",
    "Multi Cap",
    "ELSS",
    "Value",
    "Contra",
    "Focused",
    "Aggressive Hybrid",
    "Balanced Advantage",
    "Index Funds",
]

# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY → KEYWORD MAPPING
# ─────────────────────────────────────────────────────────────────────────────
# mftool returns full scheme names like:
#   "Axis Bluechip Fund - Direct Plan - Growth"
#   "HDFC Mid-Cap Opportunities Fund - Growth"
# We detect category by scanning the lowercase name for these keywords.
# Lists are ordered: first match wins, so put MORE SPECIFIC keywords first.

CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "Large Cap": [
        "large cap",
        "bluechip",
        "blue chip",
        "large-cap",
        "largecap",
    ],
    "Mid Cap": [
        "mid cap",
        "midcap",
        "mid-cap",
    ],
    "Small Cap": [
        "small cap",
        "smallcap",
        "small-cap",
    ],
    "Flexi Cap": [
        "flexi cap",
        "flexicap",
        "flexi-cap",
        "flexible cap",
    ],
    "Multi Cap": [
        "multi cap",
        "multicap",
        "multi-cap",
    ],
    "ELSS": [
        "elss",
        "long term equity",
        "tax saver",
        "taxsaver",
        "tax saving",
        "tax relief",
    ],
    "Value": [
        "value discovery",
        "value fund",
        " value ",          # space-padded to avoid false matches
    ],
    "Contra": [
        "contra",
    ],
    "Focused": [
        "focused",
        "focus fund",
        "focussed",
        "focus 25",
        "focus 30",
    ],
    "Aggressive Hybrid": [
        "aggressive hybrid",
        "hybrid equity",
        "equity hybrid",
        "equity & debt",
        "equity and debt",
    ],
    "Balanced Advantage": [
        "balanced advantage",
        "dynamic asset allocation",
        "baf",
        "dynamic equity",
    ],
    "Index Funds": [
        "index fund",
        "nifty 50 ",            # trailing space avoids "nifty 500"
        "nifty next 50",
        "nifty 100 ",
        "sensex fund",
        "nifty midcap 150",
        "nifty smallcap",
    ],
}

# These strings in a scheme name EXCLUDE it from the Index Funds category
# (ETFs trade on exchange — they are not open-ended index funds)
INDEX_EXCLUSIONS: List[str] = [
    "etf",
    "exchange traded",
]

# ─────────────────────────────────────────────────────────────────────────────
# PLAN / OPTION FILTERS
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# PLAN TYPE SELECTION
# ─────────────────────────────────────────────────────────────────────────────
# SEBI mandated Direct/Regular split in 2013. Direct plans have no distributor
# commission → lower expense ratio → higher returns. Comparing Direct vs
# Regular of the same fund distorts every metric — always analyse within
# one universe (Direct-only or Regular-only).

PLAN_TYPES: List[str] = ["Direct", "Regular"]

# Default plan type shown on app startup
DEFAULT_PLAN_TYPE: str = "Direct"

# Keywords that identify a Direct plan in the scheme name
DIRECT_KEYWORDS: List[str] = ["direct"]

# Keywords that identify a Regular plan explicitly
# Note: pre-2013 schemes have neither keyword — treated as Regular
REGULAR_KEYWORDS: List[str] = ["regular"]

# ─────────────────────────────────────────────────────────────────────────────

# We only want Growth option schemes (not Dividend / IDCW).
# A scheme name MUST contain at least one of these to pass.
PREFERRED_OPTIONS: List[str] = [
    "growth",
]

# If ANY of these appear in a scheme name, it is excluded.
# Catches Dividend, IDCW, Payout, Weekly/Monthly Dividend, Bonus options.
EXCLUDED_PLAN_KEYWORDS: List[str] = [
    "idcw",
    "dividend",
    "bonus",
    "weekly",
    "monthly dividend",
    "quarterly dividend",
    "annual dividend",
    "payout",
    "reinvestment",
    "segregated",
]

# Also exclude these fund structures entirely (they are not comparable to
# standard open-ended equity/hybrid funds)
EXCLUDED_STRUCTURE_KEYWORDS: List[str] = [
    "etf",
    "exchange traded",
    "fund of fund",
    " fof ",
    "interval fund",
    "fixed maturity",
    "fmp",
    "close ended",
    "liquid fund",
    "overnight fund",
    "arbitrage",
    "gilt",
    "debt fund",
]

# ─────────────────────────────────────────────────────────────────────────────
# MINIMUM DATA REQUIREMENTS (calendar days)
# ─────────────────────────────────────────────────────────────────────────────
# If a fund's NAV history is shorter than these thresholds,
# that metric will be reported as N/A instead of potentially misleading.

MIN_DAYS: Dict[str, int] = {
    "1y_cagr":              365,
    "3y_cagr":              365 * 3,
    "5y_cagr":              365 * 5,
    "inception_cagr":       30,
    "volatility":           30,
    "downside_volatility":  30,
    "max_drawdown":         30,
    "avg_drawdown":         30,
    "drawdown_duration":    30,
    "sharpe":               252,        # need at least 1 year for stable estimate
    "sortino":              252,
    "calmar":               252,
    "rolling_1y":           365 * 2,    # need 2 years to compute 1-year rolling
    "rolling_3y":           365 * 4,    # need 4 years to compute 3-year rolling
    "skewness":             30,
    "kurtosis":             30,
    "win_rate":             30,
    "streaks":              30,
}

# ─────────────────────────────────────────────────────────────────────────────
# CHART COLORS
# ─────────────────────────────────────────────────────────────────────────────

# Ordered list used for multi-fund comparison charts
CHART_COLORS: List[str] = [
    "#2196F3",   # Blue
    "#F44336",   # Red
    "#4CAF50",   # Green
    "#FF9800",   # Orange
    "#9C27B0",   # Purple
    "#00BCD4",   # Cyan
    "#FF5722",   # Deep Orange
    "#607D8B",   # Blue Grey
    "#E91E63",   # Pink
    "#009688",   # Teal
    "#FFC107",   # Amber
    "#3F51B5",   # Indigo
]

# Quartile color map (used in heatmaps and badges)
# Q1 = Best, Q4 = Worst
QUARTILE_COLORS: Dict[str, str] = {
    "Q1": "#4CAF50",    # Green
    "Q2": "#8BC34A",    # Light Green
    "Q3": "#FF9800",    # Orange
    "Q4": "#F44336",    # Red
    "N/A": "#9E9E9E",   # Grey — insufficient data
}

# ─────────────────────────────────────────────────────────────────────────────
# QUARTILE DIRECTION
# ─────────────────────────────────────────────────────────────────────────────
# For most metrics: higher = better → Q1 is highest values.
# For the metrics below: lower = better → Q1 is the LOWEST values.

LOWER_IS_BETTER: List[str] = [
    "annualized_volatility",
    "downside_volatility",
    "max_drawdown",
    "avg_drawdown",
    "drawdown_duration",
    "kurtosis",
    "std_rolling_1y",
    "std_rolling_3y",
    "worst_rolling_1y",
    "worst_rolling_3y",
    "down_capture",      # lower down-capture = better downside protection
    "drawdown_recovery_rate",  # faster recovery = better
]

# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY LABELS
# ─────────────────────────────────────────────────────────────────────────────
# Human-readable names for each internal metric key.
# Used in tables, charts, and export headers.

METRIC_LABELS: Dict[str, str] = {
    # Performance
    "cagr_1y":                 "1Y CAGR",
    "cagr_3y":                 "3Y CAGR",
    "cagr_5y":                 "5Y CAGR",
    "cagr_inception":          "Since Inception CAGR",
    # Volatility
    "annualized_volatility":   "Annualized Volatility",
    "downside_volatility":     "Downside Volatility",
    # Risk
    "max_drawdown":            "Max Drawdown",
    "avg_drawdown":            "Avg Drawdown",
    "drawdown_duration":       "Drawdown Duration (days)",
    # Risk-Adjusted
    "sharpe":                  "Sharpe Ratio",
    "sortino":                 "Sortino Ratio",
    "calmar":                  "Calmar Ratio",
    # Consistency — Rolling 1Y
    "avg_rolling_1y":          "Avg 1Y Rolling Return",
    "median_rolling_1y":       "Median 1Y Rolling Return",
    "std_rolling_1y":          "Std Dev 1Y Rolling Return",
    "best_rolling_1y":         "Best 1Y Rolling Return",
    "worst_rolling_1y":        "Worst 1Y Rolling Return",
    # Consistency — Rolling 3Y
    "avg_rolling_3y":          "Avg 3Y Rolling Return",
    "median_rolling_3y":       "Median 3Y Rolling Return",
    "std_rolling_3y":          "Std Dev 3Y Rolling Return",
    "best_rolling_3y":         "Best 3Y Rolling Return",
    "worst_rolling_3y":        "Worst 3Y Rolling Return",
    # Distribution
    "skewness":                "Skewness",
    "kurtosis":                "Kurtosis (Excess)",
    # Stability
    "positive_freq":           "Positive Day Frequency",
    "negative_freq":           "Negative Day Frequency",
    "win_rate":                "Win Rate (Monthly)",
    # Persistence
    "pct_positive_rolling_1y": "% Positive 1Y Rolling Periods",
    "pct_positive_rolling_3y": "% Positive 3Y Rolling Periods",
    "max_consec_positive":     "Max Consecutive Positive Days",
    "max_consec_negative":     "Max Consecutive Negative Days",
    # ── Alpha Generation ──────────────────────────────────────────────────────
    "excess_return":     "Excess Return (Ann.)",
    "beta":              "Beta",
    "r_squared":         "R-Squared",
    "tracking_error":    "Tracking Error",
    "information_ratio": "Information Ratio",
    "jensens_alpha":     "Jensen's Alpha (Ann.)",
    "alpha_tstat":       "Alpha t-Statistic",
    "up_capture":        "Up-Capture Ratio (%)",
    "down_capture":      "Down-Capture Ratio (%)",
    "capture_ratio":     "Capture Ratio",
    # ── Phase B: Momentum ─────────────────────────────────────────────────────
    "momentum_3m":       "3M Momentum",
    "momentum_6m":       "6M Momentum",
    "momentum_12m":      "12M Momentum",
    "alpha_momentum":    "Alpha Momentum (12M)",
    "momentum_sharpe":   "Momentum Sharpe",
    # ── Phase B: Alpha Persistence & Regime ───────────────────────────────────
    "alpha_persistence":      "Alpha Persistence Score",
    "bull_alpha":             "Bull Market Alpha",
    "bear_alpha":             "Bear Market Alpha",
    "alpha_regime_ratio":     "Alpha Regime Ratio",
    "drawdown_recovery_rate": "Drawdown Recovery (days)",
}
