"""
pages/2_Category_Explorer.py
=============================
Category Explorer

Lets users browse all funds in a chosen category, then optionally run
full quantitative analytics on the entire category.

Workflow:
  1. User selects a category from sidebar
  2. Fund list loads instantly (name-based, no NAV fetching)
  3. User clicks "Run Full Analytics" → NAV data fetched + metrics computed
  4. Results: heatmaps, scatter, sortable metrics table
"""

import streamlit as st
import pandas as pd
import numpy as np

from data.fund_loader      import get_all_categorized_schemes, get_nav_history, load_navs_for_funds
from analytics.engine      import compute_category_metrics, compute_category_quartiles
from analytics.quartile    import build_metrics_dataframe
from visualizations        import (
    plot_risk_return_scatter, plot_vol_cagr_scatter,
    plot_quartile_heatmap,    plot_metric_heatmap,
)
from utils.constants       import CATEGORIES, APP_TITLE, APP_ICON, METRIC_LABELS
from utils.formatters      import fmt_pct, fmt_ratio, fmt_days, format_metrics_for_display

st.set_page_config(
    page_title = "Category Explorer — MF Analytics",
    page_icon  = "🔍",
    layout     = "wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.divider()

    category = st.selectbox(
        "📂 Select Category",
        options = CATEGORIES,
        index   = CATEGORIES.index(st.session_state.get("selected_category", "Large Cap")),
        help    = "Analytics are computed within the selected category only.",
    )
    st.session_state["selected_category"] = category

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
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.title(f"🔍 Category Explorer — {category}")
st.caption(f"Browsing **{plan_type}** Growth funds in the **{category}** category.")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# LOAD FUND LIST
# ─────────────────────────────────────────────────────────────────────────────

plan_type = st.session_state.get("plan_type", "Direct")

with st.spinner(f"Loading {category} fund list…"):
    all_cat   = get_all_categorized_schemes(plan_type=plan_type)
    fund_list = all_cat.get(category, [])

if not fund_list:
    st.warning(
        f"No funds found in **{category}**. "
        "This may be a connectivity issue — run `python debug_connection.py`."
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# FUND LIST TABLE (instant — no NAV loading)
# ─────────────────────────────────────────────────────────────────────────────

st.subheader(f"📋 {len(fund_list)} Funds in {category}")

# Search/filter
search = st.text_input(
    "🔎 Filter by fund name",
    placeholder = "e.g. HDFC, Axis, Nippon…",
    label_visibility = "collapsed",
)
filtered = [f for f in fund_list if search.lower() in f["name"].lower()] if search else fund_list

fund_df = pd.DataFrame(filtered).rename(columns={"code": "Scheme Code", "name": "Fund Name"})
st.dataframe(
    fund_df[["Fund Name", "Scheme Code"]],
    use_container_width = True,
    hide_index          = True,
    height              = min(400, 38 + 35 * len(filtered)),
)
st.caption(f"Showing {len(filtered)} of {len(fund_list)} funds. Direct + Regular plans included.")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# FULL ANALYTICS (expensive — user-triggered)
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("📊 Full Category Analytics")
st.info(
    f"Computing analytics for **{len(fund_list)} funds** fetches NAV history for each fund "
    f"(~2–5s per fund). **First run: {len(fund_list)*3//60 + 1}–{len(fund_list)*5//60 + 2} minutes.**  "
    "Results are cached for 1 hour.",
    icon = "⏱️",
)

run_analytics = st.button(
    f"⚡ Run Full Analytics for {category}  ({len(fund_list)} funds)",
    type = "primary",
    use_container_width = True,
)

if run_analytics or st.session_state.get(category_analytics_key(category)):

    # ── Load NAVs with progress bar ───────────────────────────────────────────
    if not st.session_state.get(category_analytics_key(category)):
        progress  = st.progress(0, text="Starting…")
        nav_dict  = {}

        for i, fund in enumerate(fund_list):
            progress.progress(
                (i + 1) / len(fund_list),
                text = f"Fetching NAV: {fund['name'][:55]} ({i+1}/{len(fund_list)})",
            )
            nav_dict[fund["name"]] = get_nav_history(fund["code"])

        progress.empty()

        # ── Compute metrics ───────────────────────────────────────────────────
        with st.spinner("Computing quantitative metrics + alpha…"):
            from data.benchmark_loader import get_benchmark_nav, get_benchmark_info
            bm_info   = get_benchmark_info(category)
            bm_nav_df = get_benchmark_nav(category) if bm_info["available"] else None

            fund_metrics = compute_category_metrics(
                nav_dict,
                rf_rate          = rf_rate,
                benchmark_nav_df = bm_nav_df,
                benchmark_name   = bm_info["display_name"],
            )
            full_df    = compute_category_quartiles(fund_metrics)
            metrics_df = build_metrics_dataframe(fund_metrics)

        st.session_state[category_fund_metrics_key(category)] = fund_metrics
        st.session_state[category_full_df_key(category)]      = full_df
        st.session_state[f"metrics_df_{category}_v"]   = metrics_df
        st.session_state[category_analytics_key(category)] = True
        st.success(f"✅ Analytics computed for {len(fund_metrics)} funds!")

    # ── Retrieve cached results ────────────────────────────────────────────────
    fund_metrics = st.session_state.get(category_fund_metrics_key(category), {})
    full_df      = st.session_state.get(category_full_df_key(category), pd.DataFrame())
    metrics_df   = st.session_state.get(f"metrics_df_{category}_v", pd.DataFrame())

    if full_df.empty:
        st.warning("No analytics data available — some funds may lack sufficient history.")
        st.stop()

    valid_funds = sum(1 for m in fund_metrics.values() if m.get("is_valid"))
    st.caption(f"📊 {valid_funds} of {len(fund_metrics)} funds had sufficient data for metric computation.")

    # ── TABS ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Scatter Plots",
        "🗺️ Quartile Heatmap",
        "🔥 Metric Heatmap",
        "📋 Metrics Table",
    ])

    with tab1:
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.plotly_chart(
                plot_risk_return_scatter(full_df),
                use_container_width = True,
            )
        with c2:
            st.plotly_chart(
                plot_vol_cagr_scatter(full_df),
                use_container_width = True,
            )

    with tab2:
        st.caption(
            "Q1 = Best 25% in category (green)  |  Q4 = Worst 25% (red)  |  "
            "N/A = Insufficient history for this metric."
        )
        st.plotly_chart(
            plot_quartile_heatmap(full_df, height=max(400, 100 + 38 * len(full_df))),
            use_container_width = True,
        )

    with tab3:
        st.caption("Values normalised within each metric column. Green = better, Red = worse.")
        st.plotly_chart(
            plot_metric_heatmap(metrics_df, height=max(400, 100 + 38 * len(metrics_df))),
            use_container_width = True,
        )

    with tab4:
        _display_cols = {
            "cagr_1y":               "1Y CAGR",
            "cagr_3y":               "3Y CAGR",
            "cagr_5y":               "5Y CAGR",
            "annualized_volatility": "Ann. Vol",
            "max_drawdown":          "Max DD",
            "sharpe":                "Sharpe",
            "sortino":               "Sortino",
            "win_rate":              "Win Rate",
        }
        rows = []
        for fund_name, m in fund_metrics.items():
            if not m.get("is_valid"):
                continue
            row = {"Fund": fund_name}
            for key, label in _display_cols.items():
                val = m.get(key)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    row[label] = "N/A"
                elif key in {"sharpe", "sortino"}:
                    row[label] = f"{val:.3f}"
                else:
                    row[label] = f"{val*100:.2f}%"
            rows.append(row)

        if rows:
            table_df = pd.DataFrame(rows).set_index("Fund")
            st.dataframe(
                table_df,
                use_container_width = True,
                height              = min(600, 42 + 35 * len(table_df)),
            )
            csv = table_df.reset_index().to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Table (CSV)",
                data      = csv,
                file_name = f"{category.replace(' ','_')}_metrics.csv",
                mime      = "text/csv",
            )
        else:
            st.warning("No valid fund data to display.")
