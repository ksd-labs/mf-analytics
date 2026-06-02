"""
pages/4_Fund_Comparison.py
===========================
Fund Comparison

Compare 2–5 funds from the same category side by side.

Shows:
  - Normalised NAV chart (all funds rebased to 100)
  - Drawdown comparison chart
  - 1-Year rolling return comparison
  - Side-by-side metrics table

All comparisons are within the same category — cross-category comparison
is not supported by design.
"""

import streamlit as st
import pandas as pd
import numpy as np

from data.fund_loader   import get_all_categorized_schemes, get_nav_history
from analytics.engine   import compute_fund_metrics
from visualizations     import (
    plot_nav_history,
    plot_drawdown,
    plot_rolling_timeseries,
    plot_rolling_distribution,
)
from utils.constants    import CATEGORIES, APP_TITLE, APP_ICON
from utils.formatters   import fmt_pct, fmt_ratio, fmt_days

st.set_page_config(
    page_title = "Fund Comparison — MF Analytics",
    page_icon  = "⚖️",
    layout     = "wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.divider()

    category = st.selectbox(
        "📂 Category",
        CATEGORIES,
        index = CATEGORIES.index(st.session_state.get("selected_category", "Large Cap")),
    )
    st.session_state["selected_category"] = category

    with st.spinner("Loading fund list…"):
        all_cat   = get_all_categorized_schemes()
        fund_list = all_cat.get(category, [])

    if not fund_list:
        st.warning("No funds found for this category.")
        st.stop()

    fund_names = [f["name"] for f in fund_list]
    fund_codes = {f["name"]: f["code"] for f in fund_list}

    selected_funds = st.multiselect(
        "🏦 Select Funds (2–5)",
        fund_names,
        default   = fund_names[:2],
        max_selections = 5,
        help      = "Select 2 to 5 funds from the same category to compare.",
    )

    st.divider()
    rf_pct = st.slider(
        "Risk-Free Rate (%)", 4.0, 9.0,
        st.session_state.get("rf_rate", 6.5), 0.1,
    )
    rf_rate = rf_pct / 100
    st.session_state["rf_rate"] = rf_pct

    st.divider()
    if st.button("🔄 Refresh NAV Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

st.title("⚖️ Fund Comparison")

if len(selected_funds) < 2:
    st.info("👈 Select at least **2 funds** from the sidebar to begin comparison.")
    st.stop()

st.subheader(f"Comparing {len(selected_funds)} funds — {category}")
st.caption(
    "All charts use a common date range where all selected funds have NAV data. "
    "Newer funds may have a shorter common period."
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# LOAD NAV + COMPUTE METRICS
# ─────────────────────────────────────────────────────────────────────────────

all_metrics: dict = {}
nav_dict:    dict = {}
dd_dict:     dict = {}
roll1y_dict: dict = {}
roll3y_dict: dict = {}

load_bar = st.progress(0, text="Loading fund data…")
for i, name in enumerate(selected_funds):
    code = fund_codes[name]
    load_bar.progress((i + 1) / len(selected_funds), text=f"Loading: {name[:55]}…")

    cache_key = f"fund_metrics_{code}_{rf_pct}"
    if cache_key not in st.session_state:
        nav_df  = get_nav_history(code)
        metrics = compute_fund_metrics(nav_df, rf_rate=rf_rate, fund_name=name)
        st.session_state[cache_key] = metrics
    else:
        metrics = st.session_state[cache_key]

    all_metrics[name] = metrics

    if metrics.get("is_valid"):
        nav_dict[name]    = metrics.get("nav")
        dd_dict[name]     = metrics.get("drawdown_series")
        roll1y_dict[name] = metrics.get("_series_1y")
        roll3y_dict[name] = metrics.get("_series_3y")

load_bar.empty()

valid_count = sum(1 for m in all_metrics.values() if m.get("is_valid"))
if valid_count == 0:
    st.error("None of the selected funds have valid NAV data.")
    st.stop()
if valid_count < len(selected_funds):
    invalid = [n for n, m in all_metrics.items() if not m.get("is_valid")]
    st.warning(f"⚠️ {', '.join(invalid)} could not be analysed due to insufficient data.")

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────

# ── Row 1: NAV comparison + Drawdown ──────────────────────────────────────────
r1c1, r1c2 = st.columns(2, gap="medium")
with r1c1:
    st.plotly_chart(
        plot_nav_history(nav_dict, normalize=True,
                         title="NAV Comparison (Rebased to 100)"),
        use_container_width=True,
    )
with r1c2:
    st.plotly_chart(
        plot_drawdown(dd_dict, title="Drawdown Comparison"),
        use_container_width=True,
    )

# ── Row 2: Rolling returns ────────────────────────────────────────────────────
valid_1y = {k: v for k, v in roll1y_dict.items() if v is not None}
valid_3y = {k: v for k, v in roll3y_dict.items() if v is not None}

if valid_1y:
    r2c1, r2c2 = st.columns(2, gap="medium")
    with r2c1:
        st.plotly_chart(
            plot_rolling_timeseries(valid_1y, "1-Year"),
            use_container_width=True,
        )
    with r2c2:
        st.plotly_chart(
            plot_rolling_distribution(valid_1y, "1-Year"),
            use_container_width=True,
        )

if valid_3y:
    r3c1, r3c2 = st.columns(2, gap="medium")
    with r3c1:
        st.plotly_chart(
            plot_rolling_timeseries(valid_3y, "3-Year"),
            use_container_width=True,
        )
    with r3c2:
        st.plotly_chart(
            plot_rolling_distribution(valid_3y, "3-Year"),
            use_container_width=True,
        )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SIDE-BY-SIDE METRICS TABLE
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("📋 Side-by-Side Metric Comparison")
st.caption(f"Risk-free rate used: {rf_pct:.1f}%")

COMPARE_METRICS = [
    ("cagr_1y",               "1Y CAGR",                "pct"),
    ("cagr_3y",               "3Y CAGR",                "pct"),
    ("cagr_5y",               "5Y CAGR",                "pct"),
    ("cagr_inception",        "Since Inception CAGR",   "pct"),
    ("annualized_volatility", "Annualized Volatility",  "pct"),
    ("downside_volatility",   "Downside Volatility",    "pct"),
    ("max_drawdown",          "Max Drawdown",           "pct"),
    ("avg_drawdown",          "Avg Drawdown",           "pct"),
    ("drawdown_duration",     "Drawdown Duration",      "days"),
    ("sharpe",                "Sharpe Ratio",           "ratio"),
    ("sortino",               "Sortino Ratio",          "ratio"),
    ("calmar",                "Calmar Ratio",           "ratio"),
    ("avg_rolling_1y",        "Avg 1Y Rolling Return",  "pct"),
    ("worst_rolling_1y",      "Worst 1Y Rolling Return","pct"),
    ("avg_rolling_3y",        "Avg 3Y Rolling Return",  "pct"),
    ("worst_rolling_3y",      "Worst 3Y Rolling Return","pct"),
    ("skewness",              "Skewness",               "ratio"),
    ("kurtosis",              "Excess Kurtosis",        "ratio"),
    ("win_rate",              "Monthly Win Rate",       "pct"),
    ("positive_freq",         "Positive Day Freq",      "pct"),
    ("pct_positive_rolling_1y","% Positive 1Y Periods", "pct"),
    ("pct_positive_rolling_3y","% Positive 3Y Periods", "pct"),
]

def _fmt(val, kind):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    if kind == "pct":   return fmt_pct(val)
    if kind == "ratio": return fmt_ratio(val)
    if kind == "days":  return fmt_days(val)
    return str(val)

rows = []
for key, label, kind in COMPARE_METRICS:
    row = {"Metric": label}
    for name in selected_funds:
        m = all_metrics.get(name, {})
        row[name[:30]] = _fmt(m.get(key), kind)
    rows.append(row)

compare_df = pd.DataFrame(rows).set_index("Metric")
st.dataframe(compare_df, use_container_width=True)

# Export
csv = compare_df.reset_index().to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download Comparison (CSV)",
    data      = csv,
    file_name = f"{category.replace(' ','_')}_comparison.csv",
    mime      = "text/csv",
)
