"""
pages/1_Dashboard.py
====================
Dashboard — Category Overview

The landing page of the platform. Shows:
  - Total funds tracked across all 12 categories
  - Fund count per category as a card grid
  - Bar chart of fund distribution
  - Navigation guide for new users

No analytics computation happens here — only the scheme list is loaded,
which is fast (cached after the first run).
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from data.fund_loader      import get_all_schemes, get_all_categorized_schemes
from data.category_mapper  import get_category_fund_counts
from utils.constants       import CATEGORIES, APP_TITLE, APP_ICON, APP_SUBTITLE
from utils.formatters      import fmt_pct

st.set_page_config(
    page_title = "Dashboard — MF Analytics",
    page_icon  = "📊",
    layout     = "wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.caption(APP_SUBTITLE)
    st.divider()

    rf_pct = st.slider(
        "Risk-Free Rate (%)",
        min_value = 4.0, max_value = 9.0,
        value     = st.session_state.get("rf_rate", 6.5),
        step      = 0.1,
        help      = "Indian 91-day T-bill rate. Used for Sharpe & Sortino calculations.",
    )
    st.session_state["rf_rate"] = rf_pct

    st.divider()
    if st.button("🔄 Refresh NAV Data", use_container_width=True):
        st.cache_data.clear()
        st.success("Cache cleared — data will reload.")
        st.rerun()

    st.caption("NAV data sourced from AMFI via mfapi.in.\nUpdates once daily after 8 PM IST.")

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.title(f"{APP_ICON} Dashboard — Category Overview")
st.caption(APP_SUBTITLE)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

with st.spinner("Loading scheme registry from AMFI…"):
    all_schemes = get_all_schemes()

if not all_schemes:
    st.error(
        "❌ Could not load mutual fund schemes.\n\n"
        "Run `python debug_connection.py` in Anaconda Prompt to diagnose."
    )
    st.stop()

counts = get_category_fund_counts(all_schemes)
total_growth = sum(counts.values())

# ─────────────────────────────────────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────────────────────────────────────

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total AMFI Schemes",        f"{len(all_schemes):,}")
k2.metric("Growth Funds Tracked",      f"{total_growth:,}")
k3.metric("Categories Covered",        f"{len(CATEGORIES)}")
k4.metric("Risk-Free Rate (Active)",   f"{rf_pct:.1f}%")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY CARDS + BAR CHART
# ─────────────────────────────────────────────────────────────────────────────

col_left, col_right = st.columns([1.1, 0.9], gap="large")

with col_left:
    st.subheader("📂 Fund Count by Category")
    st.caption("Growth plans only — ETFs, FoFs, Dividend/IDCW excluded.")

    # 3-column card grid
    card_cols = st.columns(3)
    category_icons = {
        "Large Cap":          "🏛️",
        "Mid Cap":            "🏢",
        "Small Cap":          "🏗️",
        "Flexi Cap":          "🔄",
        "Multi Cap":          "🗂️",
        "ELSS":               "💰",
        "Value":              "🔍",
        "Contra":             "↩️",
        "Focused":            "🎯",
        "Aggressive Hybrid":  "⚖️",
        "Balanced Advantage": "🧮",
        "Index Funds":        "📈",
    }

    for i, cat in enumerate(CATEGORIES):
        n = counts.get(cat, 0)
        icon = category_icons.get(cat, "📁")
        with card_cols[i % 3]:
            st.markdown(
                f"""
                <div style="
                    background: rgba(33,150,243,0.07);
                    border: 1px solid rgba(33,150,243,0.2);
                    border-radius: 8px;
                    padding: 12px 14px;
                    margin-bottom: 10px;
                ">
                    <div style="font-size:1.4em">{icon}</div>
                    <div style="font-weight:600; font-size:0.9em; margin:4px 0 2px 0;">{cat}</div>
                    <div style="font-size:1.5em; font-weight:700; color:#2196F3;">{n}</div>
                    <div style="font-size:0.75em; color:#78909C;">funds</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

with col_right:
    st.subheader("📊 Distribution Chart")

    sorted_cats = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    cats, ns    = zip(*sorted_cats) if sorted_cats else ([], [])

    fig = go.Figure(
        go.Bar(
            x             = ns,
            y             = cats,
            orientation   = "h",
            marker        = dict(
                color     = [f"rgba(33,150,243,{0.4 + 0.6 * (v / max(ns))})" for v in ns],
                line      = dict(color="rgba(33,150,243,0.8)", width=1),
            ),
            text          = [str(v) for v in ns],
            textposition  = "outside",
            hovertemplate = "%{y}: %{x} funds<extra></extra>",
        )
    )
    fig.update_layout(
        height        = 420,
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(22,27,40,0.5)",
        font          = dict(color="#E0E0E0", size=11),
        margin        = dict(l=130, r=50, t=20, b=30),
        xaxis         = dict(gridcolor="rgba(255,255,255,0.07)", title="Number of Funds"),
        yaxis         = dict(gridcolor="rgba(255,255,255,0.0)", autorange="reversed"),
        showlegend    = False,
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# NAVIGATION GUIDE
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("🗺️ How to Use This Platform")
g1, g2, g3 = st.columns(3)

with g1:
    st.markdown("""
    **🔍 Explore a Category**

    Go to **Category Explorer** →
    Select a category → See all funds listed
    with key metrics and quartile rankings.

    Good starting point for screening.
    """)

with g2:
    st.markdown("""
    **📋 Analyse One Fund**

    Go to **Fund Analytics** →
    Select category + fund → See all 31 metrics,
    8 charts, and quartile badges for that fund.

    Deep dive into a single fund.
    """)

with g3:
    st.markdown("""
    **🏆 See Rankings**

    Go to **Rankings** →
    Select category → See funds ranked by
    Sharpe, Sortino, CAGR, Drawdown, and more.

    Export any ranking to CSV.
    """)

st.info(
    "ℹ️ **This platform provides quantitative analytics only.**  "
    "It does not provide investment advice, recommendations, or ratings.  "
    "All metrics are computed within categories — funds are never compared across categories.",
    icon="ℹ️",
)
