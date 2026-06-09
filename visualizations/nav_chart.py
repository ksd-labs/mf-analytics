"""
visualizations/nav_chart.py
============================
NAV History Chart — Chart 1 of 8.

Shows the historical Net Asset Value of one or more mutual funds over time.

Two modes:
    Raw NAV     → Y-axis in ₹ (rupees). Makes sense for a single fund.
    Normalized  → All funds rebased to 100 at the start of the period.
                  Makes sense for multi-fund comparison — removes the
                  "older fund has higher NAV" bias.

Usage:
    from visualizations.nav_chart import plot_nav_history

    # Single fund
    fig = plot_nav_history({"Axis Bluechip": nav_series})
    st.plotly_chart(fig, use_container_width=True)

    # Multi-fund comparison (auto-normalized)
    fig = plot_nav_history({"Fund A": nav_a, "Fund B": nav_b}, normalize=True)
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Optional
from visualizations._theme import base_layout, empty_figure, get_color, UP_COLOR


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CHART FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def plot_nav_history(
    nav_dict:  Dict[str, Optional[pd.Series]],
    normalize: bool = False,
    title:     Optional[str] = None,
    height:    int = 420,
) -> go.Figure:
    """
    Plot NAV history for one or more funds as a line chart.

    Args:
        nav_dict:  {fund_name: nav_series}
                   nav_series must be a clean pd.Series with DatetimeIndex.
                   Pass None as value for funds with no data — they are skipped.
        normalize: If True, rebase all series to 100 at their first data point.
                   Recommended when comparing more than one fund.
                   If False, Y-axis is raw NAV in ₹.
        title:     Optional chart title override.
        height:    Chart height in pixels.

    Returns:
        go.Figure ready for st.plotly_chart()
    """
    # Filter out None series
    valid = {k: v for k, v in nav_dict.items() if v is not None and len(v) > 0}

    if not valid:
        return empty_figure("No NAV data available for chart")

    # Auto-normalize when comparing multiple funds
    if len(valid) > 1:
        normalize = True

    fig = go.Figure()

    for i, (name, nav) in enumerate(valid.items()):
        color = get_color(i)

        if normalize:
            first_valid = nav.iloc[0]
            if first_valid <= 0:
                continue
            y_values = (nav / first_valid - 1) * 100   # % return from start
            y_label = "Return from Start Date (%)"
            hover = (
                "<b>%{fullData.name}</b><br>"
                "Date: %{x|%d %b %Y}<br>"
                "Return: %{y:.2f}%<br>"
                f"(from {nav.index[0].strftime('%d %b %Y')})"
                "<extra></extra>"
            )
        else:
            y_values = nav
            y_label = "NAV (₹)"
            hover = (
                "<b>%{fullData.name}</b><br>"
                "Date: %{x|%d %b %Y}<br>"
                "NAV: ₹%{y:,.4f}"
                "<extra></extra>"
            )

        fig.add_trace(
            go.Scatter(
                x             = nav.index,
                y             = y_values,
                name          = name,
                mode          = "lines",
                line          = dict(color=color, width=2),
                hovertemplate = hover,
            )
        )

    # ── Auto title ────────────────────────────────────────────────────────────
    if title is None:
        if len(valid) == 1:
            fund_name = list(valid.keys())[0]
            title = f"NAV History — {fund_name}"
        elif normalize:
            title = "Return Comparison (% from Start Date)"
        else:
            title = "NAV History"

    fig.update_layout(
        base_layout(
            title     = title,
            x_title   = "Date",
            y_title   = y_label,
            height    = height,
            hovermode = "x unified",
        )
    )

    # Zero reference line for % return charts
    if normalize:
        fig.add_hline(
            y=0, line_dash="dot",
            line_color="rgba(255,255,255,0.25)", line_width=1.2,
            annotation_text="0%", annotation_position="right",
            annotation_font_size=10,
            annotation_font_color="rgba(200,200,200,0.5)",
        )
        fig.update_yaxes(ticksuffix="%")

    # ── Range selector buttons ─────────────────────────────────────────────────
    fig.update_xaxes(
        rangeslider_visible = False,
        rangeselector = dict(
            bgcolor   = "rgba(22,27,40,0.9)",
            activecolor = "#2196F3",
            font      = dict(color="#E0E0E0", size=11),
            buttons   = [
                dict(count=1,  label="1Y", step="year",  stepmode="backward"),
                dict(count=3,  label="3Y", step="year",  stepmode="backward"),
                dict(count=5,  label="5Y", step="year",  stepmode="backward"),
                dict(step="all", label="All"),
            ],
        ),
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE WRAPPERS
# ─────────────────────────────────────────────────────────────────────────────

def plot_single_nav(
    nav:       pd.Series,
    fund_name: str,
    height:    int = 420,
) -> go.Figure:
    """
    Convenience wrapper — plot NAV history for exactly one fund.
    Adds a shaded area under the line for visual polish.

    Args:
        nav:       Clean NAV series
        fund_name: Fund display name
        height:    Chart height

    Returns:
        go.Figure
    """
    if nav is None or len(nav) == 0:
        return empty_figure(f"No NAV data for {fund_name}")

    fig = go.Figure()

    # Filled area trace
    fig.add_trace(
        go.Scatter(
            x             = nav.index,
            y             = nav.values,
            name          = fund_name,
            mode          = "lines",
            line          = dict(color="#2196F3", width=2),
            fill          = "tozeroy",
            fillcolor     = "rgba(33,150,243,0.08)",
            hovertemplate = (
                "<b>" + fund_name + "</b><br>"
                "Date: %{x|%d %b %Y}<br>"
                "NAV: ₹%{y:,.4f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        base_layout(
            title   = f"NAV History — {fund_name}",
            x_title = "Date",
            y_title = "NAV (₹)",
            height  = height,
            legend  = False,
        )
    )

    fig.update_xaxes(
        rangeselector = dict(
            bgcolor     = "rgba(22,27,40,0.9)",
            activecolor = "#2196F3",
            font        = dict(color="#E0E0E0", size=11),
            buttons     = [
                dict(count=1,  label="1Y", step="year",  stepmode="backward"),
                dict(count=3,  label="3Y", step="year",  stepmode="backward"),
                dict(count=5,  label="5Y", step="year",  stepmode="backward"),
                dict(step="all", label="All"),
            ],
        ),
    )

    return fig
