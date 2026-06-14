"""
pages/4_Fund_Comparison.py
===========================
Fund Comparison — Trailing Returns style (Value Research).

All funds start at 0% at the common start of the selected period.
Period selector: 1M / 3M / 6M / 1Y / 3Y / 5Y / All
"""

import streamlit as st
import pandas as pd
import numpy as np

from data.fund_loader      import get_all_categorized_schemes, get_nav_history
from analytics.engine      import compute_fund_metrics
from visualizations.nav_chart import plot_trailing_returns, plot_single_nav
from visualizations        import plot_drawdown, plot_rolling_timeseries, plot_rolling_distribution
from utils.constants       import CATEGORIES, APP_TITLE, APP_ICON
from utils.formatters      import fmt_pct, fmt_ratio, fmt_days
from utils.session         import render_refresh_button, fund_key as _fund_key

st.set_page_config(page_title="Fund Comparison — MF Analytics", page_icon="⚖️", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.divider()

    category = st.selectbox(
        "📂 Category", CATEGORIES,
        index=CATEGORIES.index(st.session_state.get("selected_category", "Large Cap")),
    )
    st.session_state["selected_category"] = category

    with st.spinner("Loading fund list…"):
        all_cat   = get_all_categorized_schemes(
            plan_type=st.session_state.get("plan_type", "Direct"))
        fund_list = all_cat.get(category, [])

    if not fund_list:
        st.warning("No funds found."); st.stop()

    fund_names = [f["name"] for f in fund_list]
    fund_codes = {f["name"]: f["code"] for f in fund_list}

    plan_type = st.radio(
        "Plan Universe", ["Direct", "Regular"],
        index=0 if st.session_state.get("plan_type", "Direct") == "Direct" else 1,
        horizontal=True,
    )
    st.session_state["plan_type"] = plan_type

    selected_funds = st.multiselect(
        "🏦 Select Funds (2–5)", fund_names,
        default=fund_names[:2], max_selections=5,
    )

    st.divider()
    rf_pct = st.slider("Risk-Free Rate (%)", 4.0, 9.0,
                       st.session_state.get("rf_rate", 6.5), 0.1)
    rf_rate = rf_pct / 100
    st.session_state["rf_rate"] = rf_pct

    st.divider()
    render_refresh_button()

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("⚖️ Fund Comparison")

if len(selected_funds) < 2:
    st.info("👈 Select at least **2 funds** from the sidebar to begin."); st.stop()

st.subheader(f"Comparing {len(selected_funds)} {plan_type} funds — {category}")
st.caption(
    "Each fund starts at **0%** at the common start of the selected period. "
    "Differences show true relative performance."
)
st.divider()

# ── Load NAV + Compute metrics ────────────────────────────────────────────────
nav_dict:    dict = {}
dd_dict:     dict = {}
roll1y_dict: dict = {}
roll3y_dict: dict = {}
all_metrics: dict = {}

bar = st.progress(0, text="Loading fund data…")
for i, name in enumerate(selected_funds):
    bar.progress((i + 1) / len(selected_funds), text=f"Loading: {name[:55]}…")
    code = fund_codes[name]
    ck   = _fund_key(code, rf_pct)

    if ck not in st.session_state:
        nav_df  = get_nav_history(code)
        metrics = compute_fund_metrics(nav_df, rf_rate=rf_rate, fund_name=name)
        st.session_state[ck] = metrics
    else:
        metrics = st.session_state[ck]

    all_metrics[name] = metrics
    if metrics.get("is_valid"):
        nav_dict[name]    = metrics.get("nav")
        dd_dict[name]     = metrics.get("drawdown_series")
        roll1y_dict[name] = metrics.get("_series_1y")
        roll3y_dict[name] = metrics.get("_series_3y")

bar.empty()

valid_count = sum(1 for m in all_metrics.values() if m.get("is_valid"))
if valid_count == 0:
    st.error("None of the selected funds have valid NAV data."); st.stop()

# ── Period Selector + Trailing Returns Chart ───────────────────────────────────
period = st.radio(
    "Select Period",
    options    = ["1M", "3M", "6M", "1Y", "3Y", "5Y", "All"],
    index      = 3,
    horizontal = True,
    key        = "comparison_period",
)

st.plotly_chart(
    plot_trailing_returns(
        nav_dict,
        period_label = period,
        title = (
            f"Trailing Returns ({period}) — "
            f"{len(selected_funds)} {plan_type} funds, {category}"
        ),
        height = 500,
    ),
    use_container_width=True,
)

st.divider()

# ── Drawdown + 1Y Rolling ──────────────────────────────────────────────────────
r1, r2 = st.columns(2, gap="medium")
with r1:
    st.subheader("📉 Drawdown Comparison")
    if dd_dict:
        st.plotly_chart(
            plot_drawdown(dd_dict, title="Drawdown Comparison"),
            use_container_width=True,
        )

with r2:
    st.subheader("🔁 1-Year Rolling Returns")
    valid_1y = {k: v for k, v in roll1y_dict.items() if v is not None}
    if valid_1y:
        st.plotly_chart(
            plot_rolling_timeseries(valid_1y, "1-Year"),
            use_container_width=True,
        )
    else:
        st.caption("Insufficient history for rolling returns.")

# ── Rolling distribution ───────────────────────────────────────────────────────
valid_1y = {k: v for k, v in roll1y_dict.items() if v is not None}
valid_3y = {k: v for k, v in roll3y_dict.items() if v is not None}

if valid_1y or valid_3y:
    rd1, rd2 = st.columns(2, gap="medium")
    with rd1:
        if valid_1y:
            st.plotly_chart(
                plot_rolling_distribution(valid_1y, "1-Year"),
                use_container_width=True,
            )
    with rd2:
        if valid_3y:
            st.plotly_chart(
                plot_rolling_distribution(valid_3y, "3-Year"),
                use_container_width=True,
            )

st.divider()

# ── Side-by-side Metrics Table ─────────────────────────────────────────────────
st.subheader("📋 Side-by-Side Metrics")
st.caption(f"Risk-free rate: {rf_pct:.1f}%")

COMPARE = [
    ("cagr_1y",               "1Y CAGR",                "pct"),
    ("cagr_3y",               "3Y CAGR",                "pct"),
    ("cagr_5y",               "5Y CAGR",                "pct"),
    ("cagr_inception",        "Since Inception CAGR",   "pct"),
    ("annualized_volatility", "Annualized Volatility",  "pct"),
    ("max_drawdown",          "Max Drawdown",           "pct"),
    ("sharpe",                "Sharpe Ratio",           "ratio"),
    ("sortino",               "Sortino Ratio",          "ratio"),
    ("calmar",                "Calmar Ratio",           "ratio"),
    ("avg_rolling_1y",        "Avg 1Y Rolling Return",  "pct"),
    ("worst_rolling_1y",      "Worst 1Y Rolling Return","pct"),
    ("avg_rolling_3y",        "Avg 3Y Rolling Return",  "pct"),
    ("win_rate",              "Monthly Win Rate",       "pct"),
    ("capture_ratio",         "Capture Ratio",          "ratio"),
    ("jensens_alpha",         "Jensen's Alpha",         "pct"),
    ("active_bet_score",      "Active Bet Score",       "ratio"),
    ("momentum_1m",           "1M Return",              "pct"),
    ("momentum_3m",           "3M Return",              "pct"),
    ("momentum_6m",           "6M Return",              "pct"),
    ("momentum_12m",          "12M Return",             "pct"),
]

def _fmt(val, kind):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    try:
        v = float(val)
        if kind == "pct":   return fmt_pct(v)
        if kind == "ratio": return fmt_ratio(v)
        if kind == "days":  return fmt_days(v)
    except Exception:
        return "N/A"
    return str(val)

rows = []
for key, label, kind in COMPARE:
    row = {"Metric": label}
    for name in selected_funds:
        m = all_metrics.get(name, {})
        row[name[:28]] = _fmt(m.get(key), kind)
    rows.append(row)

cdf = pd.DataFrame(rows).set_index("Metric")
st.dataframe(cdf, use_container_width=True)

csv = cdf.reset_index().to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download Comparison (CSV)",
    data=csv,
    file_name=f"{category.replace(' ','_')}_comparison.csv",
    mime="text/csv",
)
