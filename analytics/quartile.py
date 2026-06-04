"""
analytics/quartile.py
=====================
Quartile assignment and category-level ranking tables.

Quartile System:
    For any given metric, funds in a category are ranked and divided into 4 groups:
        Q1 = Best 25%    (top performers)
        Q2 = Next 25%
        Q3 = Next 25%
        Q4 = Worst 25%  (bottom performers)

    For most metrics (Sharpe, CAGR, Win Rate, etc.): higher = better → Q1 is highest.
    For risk metrics (Volatility, Drawdown, etc.):    lower = better → Q1 is lowest.

    This is determined by the LOWER_IS_BETTER constant in utils/constants.py.

Key design choices:
  - Quartiles are ALWAYS computed within a single category.
    Never compare funds across categories.
  - Funds with None/NaN values receive 'N/A' quartile (not Q4).
    A fund with insufficient history is not "bad" — it's just unmeasurable.
  - With fewer than 4 funds in a category, quartiles become approximate.
    A warning is issued but computation still proceeds.
  - The quartile table is built as a flat DataFrame, not nested dicts,
    so it can be directly used in Streamlit st.dataframe() calls.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List
from utils.constants import LOWER_IS_BETTER, METRIC_LABELS, QUARTILE_COLORS


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE METRIC QUARTILE ASSIGNMENT
# ─────────────────────────────────────────────────────────────────────────────

def assign_quartile_labels(
    series: pd.Series,
    lower_is_better: bool = False,
) -> pd.Series:
    """
    Assign Q1–Q4 labels to a series of metric values.

    Args:
        series:          pd.Series of float values (one per fund in category).
                         Index should be fund names or codes.
        lower_is_better: If True, the lowest values get Q1 (e.g. for Volatility).
                         If False, highest values get Q1 (e.g. for Sharpe).

    Returns:
        pd.Series of strings ('Q1', 'Q2', 'Q3', 'Q4', 'N/A'),
        same index as input.

    Algorithm:
        1. Separate valid (non-NaN) values from missing ones
        2. Compute 25th, 50th, 75th percentiles of valid values
        3. Assign labels based on which quartile each value falls in
        4. NaN values always → 'N/A'
    """
    result = pd.Series("N/A", index=series.index, dtype=str)

    valid_mask = series.notna()
    valid = series[valid_mask]

    if len(valid) == 0:
        return result

    if len(valid) < 4:
        # Too few funds for proper quartile analysis
        # Assign based on rank instead (rough approximation)
        ranks = valid.rank(ascending=not lower_is_better, method="average")
        n = len(valid)
        for idx, rank in ranks.items():
            pct = rank / n
            if pct <= 0.25:
                result[idx] = "Q1"
            elif pct <= 0.50:
                result[idx] = "Q2"
            elif pct <= 0.75:
                result[idx] = "Q3"
            else:
                result[idx] = "Q4"
        return result

    # Compute quartile boundaries from valid values
    q25 = float(valid.quantile(0.25))
    q50 = float(valid.quantile(0.50))
    q75 = float(valid.quantile(0.75))

    def _label(val: float) -> str:
        if lower_is_better:
            # Lower values → better → Q1
            if val <= q25:
                return "Q1"
            elif val <= q50:
                return "Q2"
            elif val <= q75:
                return "Q3"
            else:
                return "Q4"
        else:
            # Higher values → better → Q1
            if val >= q75:
                return "Q1"
            elif val >= q50:
                return "Q2"
            elif val >= q25:
                return "Q3"
            else:
                return "Q4"

    for idx, val in valid.items():
        result[idx] = _label(val)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY-LEVEL QUARTILE TABLE
# ─────────────────────────────────────────────────────────────────────────────

# Scalar metric keys that get quartile assignment (excludes series keys like _series_1y)
QUARTILE_METRICS: List[str] = [
    "cagr_1y", "cagr_3y", "cagr_5y", "cagr_inception",
    "annualized_volatility", "downside_volatility",
    "max_drawdown", "avg_drawdown", "drawdown_duration",
    "sharpe", "sortino", "calmar",
    "avg_rolling_1y", "median_rolling_1y", "std_rolling_1y",
    "best_rolling_1y", "worst_rolling_1y",
    "avg_rolling_3y", "median_rolling_3y", "std_rolling_3y",
    "best_rolling_3y", "worst_rolling_3y",
    "skewness", "kurtosis",
    "positive_freq", "negative_freq", "win_rate",
    "pct_positive_rolling_1y", "pct_positive_rolling_3y",
    # Alpha Generation
    "excess_return", "beta", "r_squared", "tracking_error",
    "information_ratio", "jensens_alpha", "alpha_tstat",
    "up_capture", "down_capture", "capture_ratio",
]


def build_metrics_dataframe(
    fund_metrics: Dict[str, Dict],
) -> pd.DataFrame:
    """
    Convert the per-fund metrics dict into a wide DataFrame.

    Args:
        fund_metrics: {fund_name: {metric_key: value, ...}, ...}
                      As returned by engine.compute_category_metrics()

    Returns:
        DataFrame where:
            - Rows = funds (indexed by fund_name)
            - Columns = metric keys (only scalar metrics, no series)
    """
    rows = {}
    for fund_name, metrics in fund_metrics.items():
        rows[fund_name] = {
            key: metrics.get(key)
            for key in QUARTILE_METRICS
            if not key.startswith("_")   # Exclude series keys
        }

    df = pd.DataFrame.from_dict(rows, orient="index")
    return df


def add_quartile_columns(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a quartile label column for every metric in the DataFrame.

    For each metric column, a new column named '{metric}_quartile' is added.

    Args:
        metrics_df: Wide DataFrame from build_metrics_dataframe()
                    Rows = funds, Columns = metric keys

    Returns:
        Same DataFrame with additional '{metric}_quartile' columns.
    """
    df = metrics_df.copy()

    for col in QUARTILE_METRICS:
        if col not in df.columns:
            continue

        is_lower_better = col in LOWER_IS_BETTER
        quartile_col = f"{col}_quartile"

        # Convert column to numeric, coerce non-numeric to NaN
        numeric_series = pd.to_numeric(df[col], errors="coerce")
        df[quartile_col] = assign_quartile_labels(
            numeric_series,
            lower_is_better=is_lower_better,
        )

    return df


def build_full_quartile_table(
    fund_metrics: Dict[str, Dict],
) -> pd.DataFrame:
    """
    One-shot function: build metrics DataFrame + add quartile columns.

    Args:
        fund_metrics: {fund_name: metrics_dict, ...}

    Returns:
        DataFrame with all metric values AND all quartile columns.
        This is the primary output consumed by the Rankings and
        Category Explorer pages.
    """
    metrics_df = build_metrics_dataframe(fund_metrics)
    return add_quartile_columns(metrics_df)


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY / DISPLAY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_quartile_summary_for_fund(
    fund_name: str,
    full_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract a display-ready quartile summary for a single fund.

    Args:
        fund_name: Row label in full_df
        full_df:   Output of build_full_quartile_table()

    Returns:
        2-column DataFrame: Metric | Quartile
        Used on the Fund Analytics page's quartile badge section.
    """
    if fund_name not in full_df.index:
        return pd.DataFrame(columns=["Metric", "Quartile"])

    quartile_cols = [c for c in full_df.columns if c.endswith("_quartile")]
    rows = []
    for col in quartile_cols:
        metric_key = col.replace("_quartile", "")
        label = METRIC_LABELS.get(metric_key, metric_key)
        quartile = full_df.loc[fund_name, col]
        rows.append({"Metric": label, "Quartile": quartile})

    return pd.DataFrame(rows)


def get_rankings_for_metric(
    full_df: pd.DataFrame,
    metric_key: str,
    top_n: int = 10,
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Get the top N (or bottom N) funds ranked by a single metric.

    Args:
        full_df:    Full metrics + quartile DataFrame
        metric_key: Column to rank by (e.g. 'sharpe', 'max_drawdown')
        top_n:      Number of funds to return
        ascending:  True → show lowest values first (for risk metrics)

    Returns:
        DataFrame with fund name, metric value, and quartile label.
        Used on the Rankings page.
    """
    if metric_key not in full_df.columns:
        return pd.DataFrame()

    quartile_col = f"{metric_key}_quartile"
    label = METRIC_LABELS.get(metric_key, metric_key)

    sub = full_df[[metric_key]].copy()
    if quartile_col in full_df.columns:
        sub[quartile_col] = full_df[quartile_col]

    sub = sub.dropna(subset=[metric_key])
    sub = sub.sort_values(metric_key, ascending=ascending).head(top_n)
    sub = sub.reset_index()
    sub.columns = ["Fund Name", label] + (
        ["Quartile"] if quartile_col in full_df.columns else []
    )

    return sub
