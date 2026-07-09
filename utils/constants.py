"""
constants.py — only change from phase_d1: ANALYTICS_VERSION bumped to phase_f1
Full file reproduced to avoid partial-edit errors.
"""
from typing import Dict, List

APP_TITLE: str    = "MF Quantitative Analytics"
APP_ICON: str     = "📊"
APP_SUBTITLE: str = "Institutional-Grade Mutual Fund Analysis · India"
APP_VERSION: str  = "1.0.0"

ANALYTICS_VERSION: str = "phase_f1"   # ← bumped from phase_d1

DEFAULT_RISK_FREE_RATE: float = 0.065
TRADING_DAYS_PER_YEAR: int   = 252
MAR: float = 0.0

CATEGORIES: List[str] = [
    "Large Cap","Mid Cap","Small Cap","Flexi Cap","Multi Cap","ELSS",
    "Value","Contra","Focused","Aggressive Hybrid","Balanced Advantage","Index Funds",
]

CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "Large Cap":  ["large cap","bluechip","blue chip","large-cap","largecap"],
    "Mid Cap":    ["mid cap","midcap","mid-cap"],
    "Small Cap":  ["small cap","smallcap","small-cap"],
    "Flexi Cap":  ["flexi cap","flexicap","flexi-cap","flexible cap"],
    "Multi Cap":  ["multi cap","multicap","multi-cap"],
    "ELSS":       ["elss","long term equity","tax saver","taxsaver","tax saving","tax relief"],
    "Value":      ["value discovery","value fund"," value "],
    "Contra":     ["contra"],
    "Focused":    ["focused","focus fund","focussed","focus 25","focus 30"],
    "Aggressive Hybrid": ["aggressive hybrid","hybrid equity","equity hybrid","equity & debt","equity and debt"],
    "Balanced Advantage": ["balanced advantage","dynamic asset allocation","baf","dynamic equity"],
    "Index Funds": ["index fund","nifty 50 ","nifty next 50","nifty 100 ","sensex fund","nifty midcap 150","nifty smallcap"],
}
INDEX_EXCLUSIONS: List[str] = ["etf","exchange traded"]

PLAN_TYPES: List[str]      = ["Direct","Regular"]
DEFAULT_PLAN_TYPE: str     = "Direct"
DIRECT_KEYWORDS: List[str] = ["direct"]
REGULAR_KEYWORDS: List[str]= ["regular"]

PREFERRED_OPTIONS: List[str] = ["growth"]
EXCLUDED_PLAN_KEYWORDS: List[str] = [
    "idcw","dividend","bonus","weekly","monthly dividend",
    "quarterly dividend","annual dividend","payout","reinvestment","segregated",
]
EXCLUDED_STRUCTURE_KEYWORDS: List[str] = [
    "etf","exchange traded","fund of fund"," fof ","interval fund",
    "fixed maturity","fmp","close ended","liquid fund",
    "overnight fund","arbitrage","gilt","debt fund",
]

MIN_DAYS: Dict[str, int] = {
    "1y_cagr":365,"3y_cagr":365*3,"5y_cagr":365*5,"inception_cagr":30,
    "volatility":30,"downside_volatility":30,"max_drawdown":30,
    "avg_drawdown":30,"drawdown_duration":30,"sharpe":252,"sortino":252,
    "calmar":252,"rolling_1y":365*2,"rolling_3y":365*4,
    "skewness":30,"kurtosis":30,"win_rate":30,"streaks":30,
}

CHART_COLORS: List[str] = [
    "#2196F3","#F44336","#4CAF50","#FF9800","#9C27B0",
    "#00BCD4","#FF5722","#607D8B","#E91E63","#009688","#FFC107","#3F51B5",
]
QUARTILE_COLORS: Dict[str, str] = {
    "Q1":"#4CAF50","Q2":"#8BC34A","Q3":"#FF9800","Q4":"#F44336","N/A":"#9E9E9E",
}

LOWER_IS_BETTER: List[str] = [
    "annualized_volatility","downside_volatility","max_drawdown","avg_drawdown",
    "drawdown_duration","kurtosis","std_rolling_1y","std_rolling_3y",
    "worst_rolling_1y","worst_rolling_3y","down_capture","drawdown_recovery_rate",
]

METRIC_LABELS: Dict[str, str] = {
    "cagr_1y":"1Y CAGR","cagr_3y":"3Y CAGR","cagr_5y":"5Y CAGR",
    "cagr_inception":"Since Inception CAGR",
    "annualized_volatility":"Annualized Volatility","downside_volatility":"Downside Volatility",
    "max_drawdown":"Max Drawdown","avg_drawdown":"Avg Drawdown",
    "drawdown_duration":"Drawdown Duration (days)",
    "sharpe":"Sharpe Ratio","sortino":"Sortino Ratio","calmar":"Calmar Ratio",
    "avg_rolling_1y":"Avg 1Y Rolling Return","median_rolling_1y":"Median 1Y Rolling Return",
    "std_rolling_1y":"Std Dev 1Y Rolling Return","best_rolling_1y":"Best 1Y Rolling Return",
    "worst_rolling_1y":"Worst 1Y Rolling Return",
    "avg_rolling_3y":"Avg 3Y Rolling Return","median_rolling_3y":"Median 3Y Rolling Return",
    "std_rolling_3y":"Std Dev 3Y Rolling Return","best_rolling_3y":"Best 3Y Rolling Return",
    "worst_rolling_3y":"Worst 3Y Rolling Return",
    "skewness":"Skewness","kurtosis":"Kurtosis (Excess)",
    "positive_freq":"Positive Day Frequency","negative_freq":"Negative Day Frequency",
    "win_rate":"Win Rate (Monthly)",
    "pct_positive_rolling_1y":"% Positive 1Y Rolling Periods",
    "pct_positive_rolling_3y":"% Positive 3Y Rolling Periods",
    "max_consec_positive":"Max Consecutive Positive Days",
    "max_consec_negative":"Max Consecutive Negative Days",
    "excess_return":"Excess Return (Ann.)","beta":"Beta","r_squared":"R-Squared",
    "tracking_error":"Tracking Error","information_ratio":"Information Ratio",
    "jensens_alpha":"Jensen's Alpha (Ann.)","alpha_tstat":"Alpha t-Statistic",
    "up_capture":"Up-Capture Ratio (%)","down_capture":"Down-Capture Ratio (%)",
    "capture_ratio":"Capture Ratio",
    "momentum_1m":"1M Return","momentum_3m":"3M Momentum","momentum_6m":"6M Momentum",
    "momentum_12m":"12M Momentum","alpha_momentum":"Alpha Momentum (12M)",
    "momentum_sharpe":"Momentum Sharpe",
    "alpha_persistence":"Alpha Persistence Score","bull_alpha":"Bull Market Alpha",
    "bear_alpha":"Bear Market Alpha","alpha_regime_ratio":"Alpha Regime Ratio",
    "drawdown_recovery_rate":"Drawdown Recovery (days)",
    "alpha_4f":"4-Factor Alpha (Ann.)","alpha_4f_tstat":"4-Factor Alpha t-Stat",
    "beta_market_4f":"Market Beta (4F)","beta_smb":"Size Loading (SMB)",
    "beta_hml":"Value Loading (HML)","beta_wml":"Momentum Loading (WML)",
    "r_squared_4f":"4-Factor R-Squared","contrib_market":"Market Contribution (%)",
    "contrib_smb":"Size Contribution (%)","contrib_hml":"Value Contribution (%)",
    "contrib_wml":"Momentum Contribution (%)","contrib_alpha":"Pure Alpha Contribution (%)",
}
