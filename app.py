"""
app.py
======
MF Quantitative Analytics Platform — Home Page

Entry point. Run with:  streamlit run app.py

This is the HOME page. The 6 pages in pages/ are automatically added
to the sidebar by Streamlit's multi-page app system.

Home page shows:
  - Platform header + description
  - Category overview cards (fund counts)
  - Quick navigation guide
  - Global sidebar controls (risk-free rate, refresh)
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from data.fund_loader     import get_all_schemes
from data.category_mapper import get_category_fund_counts
from utils.constants      import (
    APP_TITLE, APP_ICON, APP_SUBTITLE, APP_VERSION, CATEGORIES,
)

# ── Page config — must be first Streamlit call ────────────────────────────────
st.set_page_config(
    page_title          = f"{APP_TITLE}",
    page_icon           = APP_ICON,
    layout              = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — global controls (shared state via session_state)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.caption(APP_SUBTITLE)
    st.divider()

    # Risk-free rate — persists across all pages via session_state
    rf_pct = st.slider(
        "Risk-Free Rate (%)",
        min_value = 4.0,
        max_value = 9.0,
        value     = st.session_state.get("rf_rate", 6.5),
        step      = 0.1,
        help      = "Indian 91-day T-bill rate. Used for Sharpe & Sortino.",
    )
    st.session_state["rf_rate"] = rf_pct

    plan_type = st.radio(
        "Plan Universe",
        options    = ["Direct", "Regular"],
        index      = 0 if st.session_state.get("plan_type", "Direct") == "Direct" else 1,
        horizontal = True,
        help       = "Direct: no distributor commission. Regular: distributor-advised. Never mix both.",
    )
    st.session_state["plan_type"] = plan_type

    st.divider()

    if st.button("🔄 Refresh NAV Data"
, use_container_width=True,
                 help="Clears the 24-hour NAV cache and re-fetches all data."):
        st.cache_data.clear()
        st.success("Cache cleared — data will reload on next action.")
        st.rerun()

    st.caption(
        f"NAV sourced from AMFI via mfapi.in\n"
        f"Updates once daily after 8 PM IST\n"
        f"v{APP_VERSION}"
    )

# ─────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title(f"{APP_ICON}  {APP_TITLE}")
st.markdown(
    f"<p style='font-size:1.1em; color:#78909C;'>{APP_SUBTITLE}</p>",
    unsafe_allow_html=True,
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# LOAD SCHEME DATA
# ─────────────────────────────────────────────────────────────────────────────
plan_type = st.session_state.get("plan_type", "Direct")

with st.spinner("Loading scheme registry…"):
    all_schemes = get_all_schemes()

if not all_schemes:
    st.error(
        "❌ Unable to load mutual fund data.\n\n"
        "**Steps to fix:**\n"
        "1. Open Anaconda Prompt\n"
        "2. Run `python debug_connection.py`\n"
        "3. Follow the instructions in the output"
    )
    st.stop()

counts       = get_category_fund_counts(all_schemes)
total_growth = sum(counts.values())

# ─────────────────────────────────────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total AMFI Schemes",       f"{len(all_schemes):,}")
k2.metric("Growth Funds Tracked",     f"{total_growth:,}")
k3.metric("Categories Supported",     f"{len(CATEGORIES)}")
k4.metric("Active Risk-Free Rate",    f"{rf_pct:.1f}%")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY CARDS + BAR CHART
# ─────────────────────────────────────────────────────────────────────────────
left, right = st.columns([1.1, 0.9], gap="large")

ICONS = {
    "Large Cap": "🏛️",    "Mid Cap": "🏢",          "Small Cap": "🏗️",
    "Flexi Cap": "🔄",    "Multi Cap": "🗂️",         "ELSS": "💰",
    "Value": "🔍",        "Contra": "↩️",            "Focused": "🎯",
    "Aggressive Hybrid": "⚖️", "Balanced Advantage": "🧮", "Index Funds": "📈",
}

with left:
    st.subheader("📂 Fund Counts by Category")
    st.caption("Growth plans only — ETFs, FoFs, Dividend/IDCW excluded.")
    card_cols = st.columns(3)
    for i, cat in enumerate(CATEGORIES):
        n    = counts.get(cat, 0)
        icon = ICONS.get(cat, "📁")
        with card_cols[i % 3]:
            st.markdown(
                f"""
                <div style="
                    background:rgba(33,150,243,0.07);
                    border:1px solid rgba(33,150,243,0.2);
                    border-radius:8px; padding:12px 14px; margin-bottom:10px;
                ">
                    <div style="font-size:1.4em">{icon}</div>
                    <div style="font-weight:600;font-size:0.88em;margin:4px 0 2px">{cat}</div>
                    <div style="font-size:1.5em;font-weight:700;color:#2196F3">{n}</div>
                    <div style="font-size:0.72em;color:#78909C">funds</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

with right:
    st.subheader("📊 Distribution")
    sorted_cats = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    cats, ns    = zip(*sorted_cats) if sorted_cats else ([], [])
    max_n       = max(ns) if ns else 1

    fig = go.Figure(go.Bar(
        x=ns, y=cats, orientation="h",
        marker=dict(
            color=[f"rgba(33,150,243,{0.35 + 0.65*(v/max_n):.2f})" for v in ns],
            line=dict(color="rgba(33,150,243,0.7)", width=1),
        ),
        text=[str(v) for v in ns], textposition="outside",
        hovertemplate="%{y}: %{x} funds<extra></extra>",
    ))
    fig.update_layout(
        height=420, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(22,27,40,0.5)",
        font=dict(color="#E0E0E0", size=11),
        margin=dict(l=130, r=55, t=20, b=30),
        xaxis=dict(gridcolor="rgba(255,255,255,0.07)", title="Number of Funds"),
        yaxis=dict(autorange="reversed"), showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# NAVIGATION GUIDE
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("🗺️ Navigation Guide")
c1, c2, c3, c4, c5, c6 = st.columns(6)

nav_items = [
    ("📊",  "Dashboard",           "You are here — category overview and fund counts."),
    ("🔍",  "Category Explorer",   "Browse all funds in a category. Run full analytics with one click."),
    ("📋",  "Fund Analytics",      "Deep dive into a single fund — all 31 metrics and 6 charts."),
    ("⚖️",  "Fund Comparison",     "Compare 2–5 funds side by side with normalised charts."),
    ("🏆",  "Rankings",            "Category-wide ranking tables for every metric. CSV export."),
    ("🔬",  "Data Quality",        "Check NAV history coverage and missing data warnings."),
]

for col, (icon, title, desc) in zip([c1, c2, c3, c4, c5, c6], nav_items):
    col.markdown(
        f"""
        <div style="
            background:rgba(255,255,255,0.04);
            border:1px solid rgba(255,255,255,0.08);
            border-radius:8px; padding:14px; height:140px;
        ">
            <div style="font-size:1.6em">{icon}</div>
            <div style="font-weight:700; font-size:0.9em; margin:6px 0 4px">{title}</div>
            <div style="font-size:0.78em; color:#90A4AE">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()
st.info(
    "**This platform provides institutional-style quantitative analytics only.**  "
    "It does not provide investment recommendations, ratings, or advice.  "
    "All rankings and metrics are computed within a single category — "
    "cross-category comparisons are not supported by design.",
    icon="ℹ️",
)
