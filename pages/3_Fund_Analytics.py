"""
pages/3_Fund_Analytics.py
==========================
Fund Analytics — Deep Dive
"""

import streamlit as st
import pandas as pd
import numpy as np

from data.fund_loader        import get_nav_history, get_all_categorized_schemes
from data.benchmark_loader   import get_benchmark_nav, get_benchmark_info
from analytics.engine        import compute_fund_metrics
from visualizations.nav_chart        import plot_single_nav, plot_trailing_returns
from visualizations.drawdown_chart   import plot_drawdown
from visualizations.rolling_returns  import plot_rolling_combined
from visualizations.alpha_charts     import (
    plot_fund_vs_benchmark, plot_rolling_alpha,
)
from visualizations.momentum_charts  import (
    plot_momentum_bars, plot_bull_bear_alpha, plot_alpha_persistence_timeline,
)
from visualizations.factor_charts    import plot_rolling_alpha_4f
from utils.constants  import CATEGORIES, APP_TITLE, APP_ICON, METRIC_LABELS
from utils.formatters import fmt_pct, fmt_ratio, fmt_days, fmt_nav, fmt_date
from utils.validators import build_quality_report, get_data_coverage
from utils.session    import (
    alpha_key as _alpha_key, fund_key as _fund_key,
    render_refresh_button,
)

st.set_page_config(page_title="Fund Analytics — MF Analytics", page_icon="📋", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.divider()

    category = st.selectbox(
        "📂 Category", CATEGORIES,
        index=CATEGORIES.index(st.session_state.get("selected_category", "Large Cap")),
    )
    st.session_state["selected_category"] = category

    plan_type = st.radio(
        "Plan Universe", ["Direct", "Regular"],
        index=0 if st.session_state.get("plan_type", "Direct") == "Direct" else 1,
        horizontal=True,
    )
    st.session_state["plan_type"] = plan_type

    with st.spinner("Loading funds…"):
        all_cat   = get_all_categorized_schemes(plan_type=plan_type)
        fund_list = all_cat.get(category, [])

    if not fund_list:
        st.warning("No funds found."); st.stop()

    fund_names = [f["name"] for f in fund_list]
    fund_codes = {f["name"]: f["code"] for f in fund_list}

    prev = st.session_state.get("selected_fund", fund_names[0])
    idx  = fund_names.index(prev) if prev in fund_names else 0
    selected_name = st.selectbox("🏦 Select Fund", fund_names, index=idx)
    st.session_state["selected_fund"] = selected_name
    selected_code = fund_codes[selected_name]

    st.divider()
    col_rf, col_down, col_up = st.columns([3, 1, 1])
    rf_pct = col_rf.slider("Risk-Free Rate (%)", 4.0, 9.0,
                        st.session_state.get("rf_rate", 7.0), 0.1)
    if col_down.button("−", key=f"rf_down_{__file__}"):
        rf_pct = max(4.0, round(rf_pct - 0.1, 1))
        st.session_state["rf_rate"] = rf_pct
        st.rerun()
    if col_up.button("+", key=f"rf_up_{__file__}"):
        rf_pct = min(9.0, round(rf_pct + 0.1, 1))
        st.session_state["rf_rate"] = rf_pct
        st.rerun()

    rf_rate = rf_pct / 100
    st.session_state["rf_rate"] = rf_pct

    st.divider()
    render_refresh_button()
    from data.tri_loader import get_tri_nav, get_tri_staleness_warning, is_tri_available
    st.divider()
    st.markdown("**📡 Benchmark Data**")
    for idx_name, label in [
        ("NIFTY 500",        "Nifty 500"),
        ("NIFTY 100",        "Nifty 100"),
        ("NIFTY MIDCAP 150", "Midcap 150"),
        ("NIFTY SMALLCAP 250", "Smallcap 250"),
        ("NIFTY 50",         "Nifty 50"),
    ]:
        if is_tri_available(idx_name):
             nav = get_tri_nav(idx_name)
             warning = get_tri_staleness_warning(idx_name)
             last_date = nav.index[-1].strftime("%d %b %Y") if nav is not None else "?"
             if warning:
                 st.warning(f"{label}: {last_date} ⚠️", icon="⚠️")
             else:
                st.caption(f"✅ {label}: {last_date}")
        else:
            st.caption(f"🔄 {label}: proxy")
        

# ── Load + Compute ─────────────────────────────────────────────────────────────
st.title("📋 Fund Analytics")

ck = _fund_key(selected_code, rf_pct)
if ck not in st.session_state:
    with st.spinner(f"Loading NAV for {selected_name[:60]}…"):
        nav_df = get_nav_history(selected_code)
    with st.spinner("Computing metrics…"):
        metrics = compute_fund_metrics(nav_df, rf_rate=rf_rate, fund_name=selected_name)
    st.session_state[ck] = metrics
else:
    metrics = st.session_state[ck]

if not metrics.get("is_valid"):
    st.error(f"Could not compute metrics for **{selected_name}**.")
    for w in metrics.get("warnings", []): st.warning(w)
    st.stop()

for w in metrics.get("warnings", []): st.warning(w)
summary = metrics.get("summary", {})

# ── Header ────────────────────────────────────────────────────────────────────
st.subheader(selected_name)
st.caption(
    f"Universe: **{plan_type} plans** | Category: **{category}** | "
    f"Scheme Code: `{selected_code}` | "
    f"Inception: {fmt_date(summary.get('start_date'))} | "
    f"History: {summary.get('history_years', 'N/A')} years | "
    f"Latest NAV: {fmt_nav(summary.get('current_nav'))} "
    f"({fmt_date(summary.get('end_date'))})"
)
st.divider()

# ── KPI cards ─────────────────────────────────────────────────────────────────
k1, k2, k3 = st.columns(3)
def _kpi(col, label, val, pct=True):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        col.metric(label, "N/A"); return
    col.metric(label, fmt_pct(val) if pct else fmt_ratio(val))

_kpi(k1, "3Y CAGR",        metrics.get("cagr_3y"))
_kpi(k2, "1Y CAGR",        metrics.get("cagr_1y"))
_kpi(k3, "Ann. Volatility", metrics.get("annualized_volatility"))
st.divider()

# ── TABS ──────────────────────────────────────────────────────────────────────
tab_charts, tab_alpha, tab_factor, tab_metrics, tab_quality = st.tabs([
    "📈 Charts",
    "⚡ Alpha Analytics",
    "🔬 Factor Model",
    "📊 All Metrics",
    "🔬 Data Quality",
])

# ── TAB 1: CHARTS ──────────────────────────────────────────────────────────────
with tab_charts:
    nav = metrics.get("nav")
    dd  = metrics.get("drawdown_series")
    s1  = metrics.get("_series_1y")
    s3  = metrics.get("_series_3y")

    r1l, r1r = st.columns(2, gap="medium")
    with r1l:
        if nav is not None:
            st.plotly_chart(plot_single_nav(nav, selected_name), use_container_width=True)
        else:
            st.warning("NAV chart not available.")
    with r1r:
        if dd is not None:
            st.plotly_chart(plot_drawdown({selected_name: dd}), use_container_width=True)
        else:
            st.warning("Drawdown chart not available.")

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


# ── TAB 2: ALPHA ANALYTICS ─────────────────────────────────────────────────────
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
        st.warning("No benchmark index fund found. Check connectivity.")
    else:
        alpha_cache_key = _alpha_key(selected_code, rf_pct, category)
        if alpha_cache_key not in st.session_state:
            with st.spinner("Loading benchmark NAV and computing alpha metrics…"):
                bm_nav_df    = get_benchmark_nav(category)
                nav_df_fresh = get_nav_history(selected_code)
                full_metrics = compute_fund_metrics(
                    nav_df_fresh, rf_rate=rf_rate,
                    fund_name=selected_name,
                    benchmark_nav_df=bm_nav_df,
                    benchmark_name=bm_info["display_name"],
                )
            st.session_state[alpha_cache_key] = full_metrics
        else:
            full_metrics = st.session_state[alpha_cache_key]

        # KPI cards
        a1, a2, a3, a4, a5 = st.columns(5)
        def _akpi(col, label, val, pct=False):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                col.metric(label, "N/A"); return
            col.metric(label, fmt_pct(val) if pct else fmt_ratio(val))

        _akpi(a1, "Jensen's Alpha",    full_metrics.get("jensens_alpha"),    pct=True)
        _akpi(a2, "Alpha t-Stat",      full_metrics.get("alpha_tstat"))
        _akpi(a3, "Information Ratio", full_metrics.get("information_ratio"))
        _akpi(a4, "Capture Ratio",     full_metrics.get("capture_ratio"))
        _akpi(a5, "Beta",              full_metrics.get("beta"))

        st.divider()

        # ── Fund vs Benchmark — period selector ───────────────────────────────
        st.subheader("📈 Fund vs Benchmark — Trailing Returns")
        period_alpha = st.radio(
            "Period",
            options    = ["1M", "3M", "6M", "1Y", "3Y", "5Y", "All"],
            index      = 3,
            horizontal = True,
            key        = "alpha_period",
        )

        bm_nav_obj   = full_metrics.get("_benchmark_nav")
        fund_nav_obj = full_metrics.get("nav")

        if fund_nav_obj is not None and bm_nav_obj is not None:
            st.plotly_chart(
                plot_fund_vs_benchmark(
                    fund_nav_obj, bm_nav_obj,
                    selected_name, bm_info["display_name"],
                    period_label=period_alpha,
                    height=460,
                ),
                use_container_width=True,
            )
        elif fund_nav_obj is not None:
            st.plotly_chart(
                plot_trailing_returns(
                    {selected_name: fund_nav_obj},
                    period_label=period_alpha,
                ),
                use_container_width=True,
            )

        # Rolling alpha
        roll_alpha = full_metrics.get("_rolling_alpha")
        if roll_alpha is not None:
            st.plotly_chart(
                plot_rolling_alpha({selected_name: roll_alpha}, "1-Year"),
                use_container_width=True,
            )

        sig = full_metrics.get("alpha_tstat")
        if sig is not None:
            if abs(sig) >= 2.0:
                st.success(f"✅ Alpha is **statistically significant** (|t| = {sig:.2f} ≥ 2.0) — manager skill likely real.")
            else:
                st.warning(f"⚠️ Alpha is **not statistically significant** (|t| = {sig:.2f} < 2.0) — may be noise.")

        st.divider()

        # ── Momentum & Persistence ────────────────────────────────────────────
        st.subheader("📈 Return Momentum")
        m1, m2, m3, m4, m5 = st.columns(5)
        def _mkpi(col, label, val, pct=True):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                col.metric(label, "N/A"); return
            col.metric(label, fmt_pct(val) if pct else fmt_ratio(val))
        _mkpi(m1, "1M Return",   full_metrics.get("momentum_1m"))
        _mkpi(m2, "3M Return",   full_metrics.get("momentum_3m"))
        _mkpi(m3, "6M Return",   full_metrics.get("momentum_6m"))
        _mkpi(m4, "12M Return",  full_metrics.get("momentum_12m"))
        _mkpi(m5, "Mom. Sharpe", full_metrics.get("momentum_sharpe"), pct=False)

        st.divider()
        st.subheader("🔁 Alpha Persistence")
        p1, p2, p3, p4 = st.columns(4)
        _mkpi(p1, "Persistence Score", full_metrics.get("alpha_persistence"))
        _mkpi(p2, "Bull Alpha",        full_metrics.get("bull_alpha"))
        _mkpi(p3, "Bear Alpha",        full_metrics.get("bear_alpha"))
        _mkpi(p4, "Regime Ratio",      full_metrics.get("alpha_regime_ratio"), pct=False)

        roll_alpha_obj = full_metrics.get("_rolling_alpha")
        if roll_alpha_obj is not None:
            st.plotly_chart(
                plot_alpha_persistence_timeline(roll_alpha_obj, selected_name),
                use_container_width=True,
            )
        if full_metrics.get("bull_alpha") is not None:
            st.plotly_chart(
                plot_bull_bear_alpha({selected_name: full_metrics}),
                use_container_width=True,
            )


# ── TAB 3: FACTOR MODEL ────────────────────────────────────────────────────────
with tab_factor:
    st.subheader("🔬 Factor Model (Fama-French-Carhart)")

    from data.factor_loader import get_factor_returns, get_factor_availability, FACTOR_DISPLAY_NAMES
    avail     = get_factor_availability()
    n_factors = sum(avail.values())

    st.info(
        f"**Model:** {n_factors}-Factor  |  " +
        "  |  ".join([
            f"{'✅' if avail.get(f) else '❌'} {FACTOR_DISPLAY_NAMES.get(f, f)}"
            for f in ['market', 'smb', 'hml', 'wml']
        ]),
        icon="📐",
    )

    if n_factors == 0:
        st.warning("No factor proxy index funds found. Check internet connection.")
    else:
        factor_key = f"factor_{selected_code}_{rf_pct}"
        if factor_key not in st.session_state:
            with st.spinner("Loading factor data and computing 4-factor model…"):
                factor_df, _ = get_factor_returns(rf_rate=rf_rate)
                bm_info_f    = get_benchmark_info(category)
                bm_nav_f     = get_benchmark_nav(category) if bm_info_f["available"] else None
                nav_df_f     = get_nav_history(selected_code)
                factor_metrics = compute_fund_metrics(
                    nav_df_f, rf_rate=rf_rate,
                    fund_name=selected_name,
                    benchmark_nav_df=bm_nav_f,
                    benchmark_name=bm_info_f["display_name"],
                    factor_returns_df=factor_df,
                )
            st.session_state[factor_key] = factor_metrics
        else:
            factor_metrics = st.session_state[factor_key]

        f1, f2, f3, f4, f5 = st.columns(5)
        def _fkpi(col, label, val, pct=False):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                col.metric(label, "N/A"); return
            col.metric(label, fmt_pct(val) if pct else fmt_ratio(val))

        _fkpi(f1, "4F Alpha (Ann.)", factor_metrics.get("alpha_4f"),       pct=True)
        _fkpi(f2, "4F t-Stat",       factor_metrics.get("alpha_4f_tstat"))
        _fkpi(f3, "Market β",        factor_metrics.get("beta_market_4f"))
        _fkpi(f4, "Size β (SMB)",    factor_metrics.get("beta_smb"))
        _fkpi(f5, "4F R²",           factor_metrics.get("r_squared_4f"))

        f6, f7, f8, f9 = st.columns(4)
        _fkpi(f6, "Value β (HML)",    factor_metrics.get("beta_hml"))
        _fkpi(f7, "Momentum β (WML)", factor_metrics.get("beta_wml"))
        _fkpi(f8, "Pure Alpha",       factor_metrics.get("contrib_alpha"),  pct=True)
        _fkpi(f9, "Market Contrib",   factor_metrics.get("contrib_market"), pct=True)

        roll_4f = factor_metrics.get("_rolling_alpha_4f")
        if roll_4f is not None:
            st.plotly_chart(
                plot_rolling_alpha_4f({selected_name: roll_4f}),
                use_container_width=True,
            )

        tstat = factor_metrics.get("alpha_4f_tstat")
        if tstat is not None:
            if abs(tstat) >= 2.0:
                st.success(f"✅ 4-Factor Alpha **statistically significant** (|t| = {tstat:.2f}) — true skill confirmed.")
            else:
                st.warning(f"⚠️ 4-Factor Alpha **not significant** (|t| = {tstat:.2f}) — may be factor tilts.")


# ── TAB 4: ALL METRICS ─────────────────────────────────────────────────────────
with tab_metrics:
    st.caption("All quantitative metrics computed for this fund.")

    SECTIONS = {
        "📈 Performance": [
            ("cagr_1y",        "1-Year CAGR",         "pct"),
            ("cagr_3y",        "3-Year CAGR",         "pct"),
            ("cagr_5y",        "5-Year CAGR",         "pct"),
            ("cagr_inception", "Since Inception CAGR","pct"),
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
            ("excess_return",    "Excess Return (Ann.)",    "pct"),
            ("beta",             "Beta",                    "ratio"),
            ("r_squared",        "R-Squared",               "ratio"),
            ("tracking_error",   "Tracking Error",          "pct"),
            ("information_ratio","Information Ratio",        "ratio"),
            ("jensens_alpha",    "Jensen's Alpha (Ann.)",   "pct"),
            ("alpha_tstat",      "Alpha t-Statistic",       "ratio"),
            ("up_capture",       "Up-Capture Ratio",        "num"),
            ("down_capture",     "Down-Capture Ratio",      "num"),
            ("capture_ratio",    "Capture Ratio",           "ratio"),
        ],
        "📈 Momentum": [
            ("momentum_1m",   "1M Return",            "pct"),
            ("momentum_3m",   "3M Return",            "pct"),
            ("momentum_6m",   "6M Return",            "pct"),
            ("momentum_12m",  "12M Return",           "pct"),
            ("alpha_momentum","Alpha Momentum (12M)", "pct"),
            ("momentum_sharpe","Momentum Sharpe",     "ratio"),
        ],
        "🔁 Alpha Persistence": [
            ("alpha_persistence",     "Alpha Persistence Score", "pct"),
            ("bull_alpha",            "Bull Market Alpha",       "pct"),
            ("bear_alpha",            "Bear Market Alpha",       "pct"),
            ("alpha_regime_ratio",    "Alpha Regime Ratio",      "ratio"),
            ("drawdown_recovery_rate","Drawdown Recovery (days)","days"),
        ],
        "🔬 Factor Model": [
            ("alpha_4f",        "4-Factor Alpha (Ann.)",   "pct"),
            ("alpha_4f_tstat",  "4-Factor t-Stat",         "ratio"),
            ("beta_market_4f",  "Market Beta",             "ratio"),
            ("beta_smb",        "Size Loading (SMB)",      "ratio"),
            ("beta_hml",        "Value Loading (HML)",     "ratio"),
            ("beta_wml",        "Momentum Loading (WML)",  "ratio"),
            ("r_squared_4f",    "4-Factor R-Squared",      "ratio"),
            ("contrib_alpha",   "Pure Alpha Contribution", "pct"),
        ],
        "🔁 Consistency (1Y Rolling)": [
            ("avg_rolling_1y",    "Avg 1Y Rolling",    "pct"),
            ("median_rolling_1y", "Median 1Y Rolling", "pct"),
            ("std_rolling_1y",    "Std Dev 1Y Rolling","pct"),
            ("best_rolling_1y",   "Best 1Y Rolling",   "pct"),
            ("worst_rolling_1y",  "Worst 1Y Rolling",  "pct"),
        ],
        "📅 Stability": [
            ("positive_freq", "Positive Day Frequency", "pct"),
            ("negative_freq", "Negative Day Frequency", "pct"),
            ("win_rate",      "Monthly Win Rate",        "pct"),
        ],
    }

    def _fmt(val, kind):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/A"
        if kind == "pct":   return fmt_pct(val)
        if kind == "ratio": return fmt_ratio(val)
        if kind == "days":  return fmt_days(val)
        if kind == "num":   return f"{val:.2f}%"
        return str(val)

    for section_title, metric_list in SECTIONS.items():
        with st.expander(section_title, expanded=False):
            rows = [
                {"Metric": label, "Value": _fmt(metrics.get(key), kind)}
                for key, label, kind in metric_list
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── TAB 5: DATA QUALITY ────────────────────────────────────────────────────────
with tab_quality:
    nav    = metrics.get("nav")
    report = build_quality_report(selected_name, nav)

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("History",     f"{report.get('history_years', 0)} yrs")
    q2.metric("Data Points", f"{report.get('data_points', 0):,}")
    q3.metric("Missing %",   f"{report.get('missing_pct', 0):.1f}%")
    q4.metric("Start Date",  fmt_date(report.get("start_date")))

    for w in report.get("warnings", []): st.warning(w)

    st.subheader("Metric Coverage")
    coverage = report.get("coverage", {})
    cov_rows = [
        {
            "Metric":    METRIC_LABELS.get(k, k),
            "Available": "✅ Yes" if v else "❌ No (insufficient history)",
        }
        for k, v in coverage.items()
    ]
    yes_n = sum(1 for r in cov_rows if "Yes" in r["Available"])
    st.caption(f"{yes_n} of {len(cov_rows)} metrics available.")
    st.dataframe(pd.DataFrame(cov_rows), use_container_width=True, hide_index=True)
