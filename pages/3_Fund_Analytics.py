"""
pages/3_Fund_Analytics.py
==========================
Fund Analytics — Deep Dive

The most detailed view in the platform. For a single selected fund it shows:
  - 4 KPI metric cards at the top
  - Tab 1 — Charts:  All 6 charts (NAV, Drawdown, Rolling 1Y, Rolling 3Y)
  - Tab 2 — Metrics: All 31 scalar metrics in a formatted table
  - Tab 3 — Quartiles: Q1–Q4 badges per metric (requires category analytics)
  - Tab 4 — Data Quality: Coverage report for this fund
"""

import streamlit as st
import pandas as pd
import numpy as np

from data.fund_loader      import get_nav_history, get_all_categorized_schemes
from data.benchmark_loader import get_benchmark_nav, get_benchmark_info
from analytics.engine      import compute_fund_metrics
from visualizations.alpha_charts import (
    plot_fund_vs_benchmark, plot_rolling_alpha,
)
from visualizations.momentum_charts import (
    plot_momentum_bars, plot_bull_bear_alpha, plot_alpha_persistence_timeline,
)
from visualizations        import (
    plot_single_nav,
    plot_drawdown,
    plot_rolling_combined,
)
from utils.constants       import CATEGORIES, APP_TITLE, APP_ICON, METRIC_LABELS
from utils.formatters      import fmt_pct, fmt_ratio, fmt_days, fmt_nav, fmt_date
from utils.validators      import build_quality_report, get_data_coverage

st.set_page_config(
    page_title = "Fund Analytics — MF Analytics",
    page_icon  = "📋",
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

    plan_type = st.session_state.get("plan_type", "Direct")

    # Load fund list for selected category
    with st.spinner("Loading funds…"):
        all_cat   = get_all_categorized_schemes(plan_type=plan_type)
        fund_list = all_cat.get(category, [])

    if not fund_list:
        st.warning("No funds found.")
        st.stop()

    fund_names = [f["name"] for f in fund_list]
    fund_codes = {f["name"]: f["code"] for f in fund_list}

    prev_fund = st.session_state.get("selected_fund", fund_names[0])
    default_idx = fund_names.index(prev_fund) if prev_fund in fund_names else 0

    selected_name = st.selectbox(
        "🏦 Select Fund",
        fund_names,
        index   = default_idx,
        help    = "Select a fund to analyse.",
    )
    st.session_state["selected_fund"] = selected_name
    selected_code = fund_codes[selected_name]

    st.divider()
    rf_pct = st.slider(
        "Risk-Free Rate (%)", 4.0, 9.0,
        st.session_state.get("rf_rate", 6.5), 0.1,
    )
    rf_rate = rf_pct / 100
    st.session_state["rf_rate"] = rf_pct

    plan_type = st.radio(
        "Plan Universe",
        options    = ["Direct", "Regular"],
        index      = 0 if st.session_state.get("plan_type", "Direct") == "Direct" else 1,
        horizontal = True,
        help       = "Direct: no distributor commission. Regular: distributor-advised.",
    )
    st.session_state["plan_type"] = plan_type

    st.divider()
    if st.button("🔄 Refresh NAV Data", use_container_width=True):
        from utils.session import clear_analytics_cache
        clear_analytics_cache()
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# LOAD + COMPUTE
# ─────────────────────────────────────────────────────────────────────────────

st.title(f"📋 Fund Analytics")

cache_key = _fund_key(selected_code, rf_pct)
if cache_key not in st.session_state:
    with st.spinner(f"Loading NAV history for {selected_name[:60]}…"):
        nav_df = get_nav_history(selected_code)
    with st.spinner("Computing metrics…"):
        metrics = compute_fund_metrics(nav_df, rf_rate=rf_rate, fund_name=selected_name)
    st.session_state[cache_key] = metrics
else:
    metrics = st.session_state[cache_key]

if not metrics.get("is_valid"):
    st.error(f"Could not compute metrics for **{selected_name}**.")
    for w in metrics.get("warnings", []):
        st.warning(w)
    st.stop()

# Show any warnings (e.g. minor missing data)
for w in metrics.get("warnings", []):
    st.warning(w)

summary = metrics.get("summary", {})

# ─────────────────────────────────────────────────────────────────────────────
# FUND HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.subheader(selected_name)
st.caption(
    f"Universe: **{plan_type} plans**  |  Category: **{category}**  |  "
    f"Scheme Code: `{selected_code}`  |  "
    f"Inception: {fmt_date(summary.get('start_date'))}  |  "
    f"History: {summary.get('history_years', 'N/A')} years  |  "
    f"Latest NAV: {fmt_nav(summary.get('current_nav'))}  "
    f"({fmt_date(summary.get('end_date'))})"
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────────────────────────────────────

k1, k2, k3, k4, k5, k6 = st.columns(6)

def _kpi(col, label, value, is_pct=True, is_ratio=False, invert_color=False):
    """Render a metric card with conditional colouring."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        col.metric(label, "N/A")
        return
    if is_pct:
        disp = fmt_pct(value)
        delta_str = f"{value*100:+.2f}%" if not invert_color else None
    elif is_ratio:
        disp = fmt_ratio(value)
        delta_str = f"{value:+.3f}" if not invert_color else None
    else:
        disp = str(value)
        delta_str = None
    col.metric(label, disp)

_kpi(k1, "3Y CAGR",       metrics.get("cagr_3y"),               is_pct=True)
_kpi(k2, "1Y CAGR",       metrics.get("cagr_1y"),               is_pct=True)
_kpi(k3, "Ann. Volatility",metrics.get("annualized_volatility"), is_pct=True)
_kpi(k4, "Max Drawdown",   metrics.get("max_drawdown"),          is_pct=True)
_kpi(k5, "Sharpe Ratio",   metrics.get("sharpe"),                is_pct=False, is_ratio=True)
_kpi(k6, "Win Rate",       metrics.get("win_rate"),              is_pct=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab_charts, tab_alpha, tab_metrics, tab_quality = st.tabs([
    "📈 Charts",
    "⚡ Alpha Analytics",
    "📊 All Metrics",
    "🔬 Data Quality",
])

# ── TAB 1: CHARTS ─────────────────────────────────────────────────────────────
with tab_charts:

    nav = metrics.get("nav")
    dd  = metrics.get("drawdown_series")
    s1  = metrics.get("_series_1y")
    s3  = metrics.get("_series_3y")

    # NAV + Drawdown side by side
    row1_l, row1_r = st.columns(2, gap="medium")
    with row1_l:
        if nav is not None:
            st.plotly_chart(
                plot_single_nav(nav, selected_name),
                use_container_width=True,
            )
        else:
            st.warning("NAV chart not available.")

    with row1_r:
        if dd is not None:
            st.plotly_chart(
                plot_drawdown({selected_name: dd}),
                use_container_width=True,
            )
        else:
            st.warning("Drawdown chart not available.")

    # Rolling returns
    if s1 is not None:
        st.plotly_chart(
            plot_rolling_combined({selected_name: s1}, window_label="1-Year", height=650),
            use_container_width=True,
        )
    else:
        st.info("1-Year rolling returns require at least 2 years of NAV history.")

    if s3 is not None:
        st.plotly_chart(
            plot_rolling_combined({selected_name: s3}, window_label="3-Year", height=650),
            use_container_width=True,
        )
    else:
        st.info("3-Year rolling returns require at least 4 years of NAV history.")


# ── TAB ALPHA: ALPHA ANALYTICS ───────────────────────────────────────────────
with tab_alpha:
    st.subheader("⚡ Alpha Analytics")

    bm_info = get_benchmark_info(category)
    st.info(
        f"**Benchmark:** {bm_info['display_name']}  |  "
        f"**Proxy used:** {bm_info['scheme_name'][:70]}  |  "
        f"**Available:** {'✅' if bm_info['available'] else '❌ Not found'}",
        icon="📊",
    )

    if not bm_info["available"]:
        st.warning(
            "No benchmark index fund found for this category. "
            "Alpha metrics require a benchmark. "
            "Check your internet connection and try refreshing."
        )
    else:
        # Load benchmark and compute alpha if not already done
        alpha_cache_key = _alpha_key(selected_code, rf_pct, category)
        if alpha_cache_key not in st.session_state:
            with st.spinner("Loading benchmark NAV and computing alpha metrics…"):
                bm_nav_df = get_benchmark_nav(category)
                from analytics.engine import compute_fund_metrics
                from data.fund_loader import get_nav_history as _gnav
                nav_df_fresh = _gnav(selected_code)
                full_metrics = compute_fund_metrics(
                    nav_df_fresh, rf_rate=rf_rate,
                    fund_name=selected_name,
                    benchmark_nav_df=bm_nav_df,
                    benchmark_name=bm_info["display_name"],
                )
            st.session_state[alpha_cache_key] = full_metrics
        else:
            full_metrics = st.session_state[alpha_cache_key]

        # ── Alpha KPI cards ───────────────────────────────────────────────────
        a1, a2, a3, a4, a5 = st.columns(5)
        def _akpi(col, label, val, fmt="ratio"):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                col.metric(label, "N/A"); return
            col.metric(label, f"{val:.3f}" if fmt == "ratio" else f"{val*100:.2f}%")

        _akpi(a1, "Jensen's Alpha",    full_metrics.get("jensens_alpha"),    "pct")
        _akpi(a2, "Alpha t-Stat",      full_metrics.get("alpha_tstat"),      "ratio")
        _akpi(a3, "Information Ratio", full_metrics.get("information_ratio"),"ratio")
        _akpi(a4, "Capture Ratio",     full_metrics.get("capture_ratio"),    "ratio")
        _akpi(a5, "Beta",              full_metrics.get("beta"),             "ratio")

        st.divider()

        # ── Charts ────────────────────────────────────────────────────────────
        bm_nav = full_metrics.get("_benchmark_nav")
        fund_nav_for_alpha = full_metrics.get("nav")

        if fund_nav_for_alpha is not None and bm_nav is not None:
            st.plotly_chart(
                plot_fund_vs_benchmark(
                    fund_nav_for_alpha, bm_nav,
                    selected_name, bm_info["display_name"]
                ),
                use_container_width=True,
            )

        roll_alpha = full_metrics.get("_rolling_alpha")
        if roll_alpha is not None:
            st.plotly_chart(
                plot_rolling_alpha({selected_name: roll_alpha}, "1-Year"),
                use_container_width=True,
            )
        else:
            st.info("Rolling alpha requires 2+ years of overlapping fund and benchmark history.")

        # ── Alpha metrics table ───────────────────────────────────────────────
        st.subheader("All Alpha Metrics")
        ALPHA_METRICS = [
            ("excess_return",    "Excess Return (Annualized)",  "pct"),
            ("beta",             "Beta",                         "ratio"),
            ("r_squared",        "R-Squared",                   "ratio"),
            ("tracking_error",   "Tracking Error (Annualized)", "pct"),
            ("information_ratio","Information Ratio",           "ratio"),
            ("jensens_alpha",    "Jensen's Alpha (Annualized)","pct"),
            ("alpha_tstat",      "Alpha t-Statistic",           "ratio"),
            ("up_capture",       "Up-Capture Ratio",            "num"),
            ("down_capture",     "Down-Capture Ratio",          "num"),
            ("capture_ratio",    "Capture Ratio",               "ratio"),
        ]
        alpha_rows = []
        for key, label, kind in ALPHA_METRICS:
            val = full_metrics.get(key)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                disp = "N/A"
            elif kind == "pct":   disp = fmt_pct(val)
            elif kind == "ratio": disp = fmt_ratio(val)
            else:                 disp = f"{val:.2f}%"
            alpha_rows.append({"Metric": label, "Value": disp})

        st.dataframe(pd.DataFrame(alpha_rows), use_container_width=True, hide_index=True)

        sig = full_metrics.get("alpha_tstat")
        if sig is not None:
            if abs(sig) >= 2.0:
                st.success(f"✅ Alpha is **statistically significant** (|t| = {sig:.2f} ≥ 2.0) — manager skill is likely real.")
            else:
                st.warning(f"⚠️ Alpha is **not statistically significant** (|t| = {sig:.2f} < 2.0) — performance may be noise.")

        st.divider()

        # ── Phase B: Momentum ─────────────────────────────────────────────────
        st.subheader("📈 Return Momentum")
        m1, m2, m3, m4, m5 = st.columns(5)
        def _mkpi(col, label, val, pct=True):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                col.metric(label, "N/A"); return
            col.metric(label, f"{val*100:.2f}%" if pct else f"{val:.3f}")

        _mkpi(m1, "3M Momentum",    full_metrics.get("momentum_3m"))
        _mkpi(m2, "6M Momentum",    full_metrics.get("momentum_6m"))
        _mkpi(m3, "12M Momentum",   full_metrics.get("momentum_12m"))
        _mkpi(m4, "Alpha Momentum", full_metrics.get("alpha_momentum"))
        _mkpi(m5, "Mom. Sharpe",    full_metrics.get("momentum_sharpe"), pct=False)

        st.divider()

        # ── Phase B: Persistence timeline ─────────────────────────────────────
        st.subheader("🔁 Alpha Persistence")
        p1, p2, p3, p4 = st.columns(4)
        _mkpi(p1, "Persistence Score",   full_metrics.get("alpha_persistence"))
        _mkpi(p2, "Bull Market Alpha",   full_metrics.get("bull_alpha"))
        _mkpi(p3, "Bear Market Alpha",   full_metrics.get("bear_alpha"))
        _mkpi(p4, "Alpha Regime Ratio",  full_metrics.get("alpha_regime_ratio"), pct=False)

        if roll_alpha is not None:
            st.plotly_chart(
                plot_alpha_persistence_timeline(roll_alpha, selected_name),
                use_container_width=True,
            )

        # ── Phase B: Bull / Bear alpha chart ─────────────────────────────────
        if full_metrics.get("bull_alpha") is not None or full_metrics.get("bear_alpha") is not None:
            st.plotly_chart(
                plot_bull_bear_alpha({selected_name: full_metrics}),
                use_container_width=True,
            )

        # ── Drawdown recovery ─────────────────────────────────────────────────
        rec = full_metrics.get("drawdown_recovery_rate")
        if rec is not None:
            st.metric("Avg Drawdown Recovery", fmt_days(int(rec)))


# ── TAB 2: ALL METRICS ────────────────────────────────────────────────────────
with tab_metrics:
    st.caption("All 31 quantitative metrics computed for this fund.")

    SECTIONS = {
        "📈 Performance": [
            ("cagr_1y",        "1-Year CAGR",           "pct"),
            ("cagr_3y",        "3-Year CAGR",           "pct"),
            ("cagr_5y",        "5-Year CAGR",           "pct"),
            ("cagr_inception", "Since Inception CAGR",  "pct"),
        ],
        "🌊 Volatility": [
            ("annualized_volatility", "Annualized Volatility", "pct"),
            ("downside_volatility",   "Downside Volatility",   "pct"),
        ],
        "⚠️ Risk": [
            ("max_drawdown",      "Maximum Drawdown",       "pct"),
            ("avg_drawdown",      "Average Drawdown",       "pct"),
            ("drawdown_duration", "Max Drawdown Duration",  "days"),
        ],
        "⚖️ Risk-Adjusted": [
            ("sharpe",  "Sharpe Ratio",  "ratio"),
            ("sortino", "Sortino Ratio", "ratio"),
            ("calmar",  "Calmar Ratio",  "ratio"),
        ],
        "⚡ Alpha (vs Benchmark)": [
            ("excess_return",    "Excess Return (Ann.)",     "pct"),
            ("beta",             "Beta",                     "ratio"),
            ("r_squared",        "R-Squared",                "ratio"),
            ("tracking_error",   "Tracking Error",           "pct"),
            ("information_ratio","Information Ratio",        "ratio"),
            ("jensens_alpha",    "Jensen's Alpha (Ann.)",   "pct"),
            ("alpha_tstat",      "Alpha t-Statistic",        "ratio"),
            ("up_capture",       "Up-Capture Ratio",         "num"),
            ("down_capture",     "Down-Capture Ratio",       "num"),
            ("capture_ratio",    "Capture Ratio",            "ratio"),
        ],
        "📈 Momentum": [
            ("momentum_3m",    "3M Momentum",         "pct"),
            ("momentum_6m",    "6M Momentum",         "pct"),
            ("momentum_12m",   "12M Momentum",        "pct"),
            ("alpha_momentum", "Alpha Momentum (12M)","pct"),
            ("momentum_sharpe","Momentum Sharpe",     "ratio"),
        ],
        "🔁 Alpha Persistence": [
            ("alpha_persistence",     "Alpha Persistence Score", "pct"),
            ("bull_alpha",            "Bull Market Alpha",       "pct"),
            ("bear_alpha",            "Bear Market Alpha",       "pct"),
            ("alpha_regime_ratio",    "Alpha Regime Ratio",      "ratio"),
            ("drawdown_recovery_rate","Drawdown Recovery (days)","days"),
        ],
        "🔁 Rolling Returns — 1 Year": [
            ("avg_rolling_1y",    "Average 1Y Rolling Return",    "pct"),
            ("median_rolling_1y", "Median 1Y Rolling Return",     "pct"),
            ("std_rolling_1y",    "Std Dev 1Y Rolling Return",    "pct"),
            ("best_rolling_1y",   "Best 1Y Rolling Return",       "pct"),
            ("worst_rolling_1y",  "Worst 1Y Rolling Return",      "pct"),
        ],
        "🔁 Rolling Returns — 3 Year": [
            ("avg_rolling_3y",    "Average 3Y Rolling Return",    "pct"),
            ("median_rolling_3y", "Median 3Y Rolling Return",     "pct"),
            ("std_rolling_3y",    "Std Dev 3Y Rolling Return",    "pct"),
            ("best_rolling_3y",   "Best 3Y Rolling Return",       "pct"),
            ("worst_rolling_3y",  "Worst 3Y Rolling Return",      "pct"),
        ],
        "📐 Distribution": [
            ("skewness", "Skewness",         "ratio"),
            ("kurtosis", "Excess Kurtosis",  "ratio"),
        ],
        "📅 Stability": [
            ("positive_freq", "Positive Day Frequency", "pct"),
            ("negative_freq", "Negative Day Frequency", "pct"),
            ("win_rate",      "Monthly Win Rate",       "pct"),
        ],
        "🔗 Persistence": [
            ("pct_positive_rolling_1y", "% Positive 1Y Rolling Periods", "pct"),
            ("pct_positive_rolling_3y", "% Positive 3Y Rolling Periods", "pct"),
            ("max_consec_positive",     "Max Consecutive Positive Days",  "int"),
            ("max_consec_negative",     "Max Consecutive Negative Days",  "int"),
        ],
    }

    def _fmt(val, kind):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/A"
        if kind == "pct":   return fmt_pct(val)
        if kind == "ratio": return fmt_ratio(val)
        if kind == "days":  return fmt_days(val)
        if kind == "int":   return str(int(val))
        return str(val)

    for section_title, metric_list in SECTIONS.items():
        with st.expander(section_title, expanded=True):
            rows = [
                {"Metric": label, "Value": _fmt(metrics.get(key), kind)}
                for key, label, kind in metric_list
            ]
            sec_df = pd.DataFrame(rows)
            st.dataframe(sec_df, use_container_width=True, hide_index=True)


# ── TAB 3: DATA QUALITY ───────────────────────────────────────────────────────
with tab_quality:
    nav = metrics.get("nav")
    report = build_quality_report(selected_name, nav)

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("History",      f"{report.get('history_years', 0)} yrs")
    q2.metric("Data Points",  f"{report.get('data_points', 0):,}")
    q3.metric("Missing Data", f"{report.get('missing_pct', 0):.1f}%")
    q4.metric("Start Date",   fmt_date(report.get("start_date") if report.get("start_date") else None))

    if report.get("warnings"):
        for w in report["warnings"]:
            st.warning(w)

    st.subheader("Metric Coverage")
    st.caption("Shows which metrics can be computed given this fund's data history.")
    coverage = report.get("coverage", {})
    cov_rows = [
        {
            "Metric":    METRIC_LABELS.get(key, key),
            "Available": "✅ Yes" if avail else "❌ No (insufficient history)",
        }
        for key, avail in coverage.items()
    ]
    cov_df = pd.DataFrame(cov_rows)
    yes_count = sum(1 for r in cov_rows if "Yes" in r["Available"])
    st.caption(f"{yes_count} of {len(cov_rows)} metrics available for this fund.")
    st.dataframe(cov_df, use_container_width=True, hide_index=True)
