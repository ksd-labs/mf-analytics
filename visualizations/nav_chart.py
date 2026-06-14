"""
visualizations/nav_chart.py
============================
NAV History Chart (Chart 1) + Trailing Returns Chart (Value Research style).

plot_trailing_returns() — Key design:
    - Period selector: 1M / 3M / 6M / 1Y / 3Y / 5Y / All
    - All funds start at 0% at the common start of the selected period
    - Y-axis in percentage with prominent zero reference line
    - Exactly like Value Research's Trailing Returns tab
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Optional
from visualizations._theme import base_layout, empty_figure, get_color, UP_COLOR

# Period label → calendar months (None = full history)
PERIOD_MAP = {
    "1M": 1, "3M": 3, "6M": 6,
    "1Y": 12, "3Y": 36, "5Y": 60, "All": None,
}


# ─────────────────────────────────────────────────────────────────────────────
# TRAILING RETURNS CHART  (Value Research style)
# ─────────────────────────────────────────────────────────────────────────────

def plot_trailing_returns(
    nav_dict:     Dict[str, Optional[pd.Series]],
    period_label: str = "1Y",
    title:        Optional[str] = None,
    height:       int = 460,
) -> go.Figure:
    """
    Trailing returns chart — every fund/benchmark starts at 0% at the
    beginning of the selected period, just like Value Research.

    Args:
        nav_dict:     {display_name: nav_pd.Series (clean, DatetimeIndex)}
        period_label: One of "1M","3M","6M","1Y","3Y","5Y","All"
        title:        Optional chart title override
        height:       Chart height in pixels

    Returns:
        go.Figure with lines starting at 0%, zero reference line,
        y-axis in % with +/- prefix on hover.
    """
    valid = {k: v for k, v in nav_dict.items()
             if v is not None and len(v) >= 5}
    if not valid:
        return empty_figure("No NAV data available")

    period_months = PERIOD_MAP.get(period_label)

    # ── Slice each series to the chosen lookback period ───────────────────────
    sliced: Dict[str, pd.Series] = {}
    for name, nav in valid.items():
        if period_months is not None:
            end   = nav.index[-1]
            start = end - pd.DateOffset(months=period_months)
            chunk = nav[nav.index >= start]
            sliced[name] = chunk if len(chunk) >= 3 else nav
        else:
            sliced[name] = nav

    # ── Common start = latest start date across all funds ─────────────────────
    # Ensures every fund has data from the same calendar date so comparison
    # is fair — no fund has a "head start"
    common_start = max(s.index[0] for s in sliced.values())

    rebased: Dict[str, pd.Series] = {}
    for name, nav in sliced.items():
        chunk = nav[nav.index >= common_start]
        if len(chunk) < 2:
            continue
        rebased[name] = (chunk / chunk.iloc[0] - 1) * 100  # % from 0

    if not rebased:
        return empty_figure(f"No overlapping history for the {period_label} period")

    fig = go.Figure()

    for i, (name, pct) in enumerate(rebased.items()):
        short = (name[:42] + "…") if len(name) > 42 else name
        color = get_color(i)
        fig.add_trace(go.Scatter(
            x    = pct.index,
            y    = pct.values,
            name = short,
            mode = "lines",
            line = dict(color=color, width=2),
            hovertemplate=(
                f"<b>{short}</b><br>"
                "Date: %{x|%d %b %Y}<br>"
                "Return: %{y:+.2f}%"
                "<extra></extra>"
            ),
        ))

    # ── Prominent zero baseline ────────────────────────────────────────────────
    fig.add_hline(
        y=0,
        line_dash   = "dot",
        line_color  = "rgba(255,255,255,0.45)",
        line_width  = 1.8,
        annotation_text       = "0%",
        annotation_position   = "right",
        annotation_font_size  = 11,
        annotation_font_color = "rgba(200,200,200,0.75)",
    )

    if title is None:
        title = f"Trailing Returns — {period_label}"

    fig.update_layout(
        base_layout(
            title     = title,
            x_title   = "Date",
            y_title   = f"Return from {common_start.strftime('%d %b %Y')} (%)",
            height    = height,
            hovermode = "x unified",
        )
    )
    fig.update_yaxes(
        ticksuffix   = "%",
        zeroline     = True,
        zerolinecolor= "rgba(255,255,255,0.45)",
        zerolinewidth= 1.8,
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# NAV HISTORY CHART  (existing — multi-fund)
# ─────────────────────────────────────────────────────────────────────────────

def plot_nav_history(
    nav_dict:  Dict[str, Optional[pd.Series]],
    normalize: bool = False,
    title:     Optional[str] = None,
    height:    int = 420,
) -> go.Figure:
    """
    Plot NAV history for one or more funds.

    normalize=True → each fund rebased to 100 at its own start date.
    For comparing funds use plot_trailing_returns() instead — it gives
    proper % returns from a common start date.
    """
    valid = {k: v for k, v in nav_dict.items() if v is not None and len(v) > 0}
    if not valid:
        return empty_figure("No NAV data available for chart")

    if len(valid) > 1:
        normalize = True

    fig = go.Figure()

    for i, (name, nav) in enumerate(valid.items()):
        color = get_color(i)
        if normalize:
            first_valid = nav.iloc[0]
            if first_valid <= 0:
                continue
            y_values = (nav / first_valid) * 100
            y_label  = "Value (rebased to 100)"
            hover    = (
                "<b>%{fullData.name}</b><br>"
                "Date: %{x|%d %b %Y}<br>"
                "Rebased Value: %{y:.2f}<extra></extra>"
            )
        else:
            y_values = nav
            y_label  = "NAV (₹)"
            hover    = (
                "<b>%{fullData.name}</b><br>"
                "Date: %{x|%d %b %Y}<br>"
                "NAV: ₹%{y:,.4f}<extra></extra>"
            )

        fig.add_trace(go.Scatter(
            x=nav.index, y=y_values,
            name=name, mode="lines",
            line=dict(color=color, width=2),
            hovertemplate=hover,
        ))

    if title is None:
        title = "NAV History" if len(valid) > 1 else f"NAV History — {list(valid.keys())[0]}"

    fig.update_layout(base_layout(
        title=title, x_title="Date", y_title=y_label,
        height=height, hovermode="x unified",
    ))
    fig.update_xaxes(
        rangeselector=dict(
            bgcolor="rgba(22,27,40,0.9)", activecolor="#2196F3",
            font=dict(color="#E0E0E0", size=11),
            buttons=[
                dict(count=1,  label="1Y", step="year",  stepmode="backward"),
                dict(count=3,  label="3Y", step="year",  stepmode="backward"),
                dict(count=5,  label="5Y", step="year",  stepmode="backward"),
                dict(step="all", label="All"),
            ],
        ),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE NAV CHART
# ─────────────────────────────────────────────────────────────────────────────

def plot_single_nav(
    nav:       pd.Series,
    fund_name: str,
    height:    int = 420,
) -> go.Figure:
    """Single fund NAV — filled area, range selector buttons."""
    if nav is None or len(nav) == 0:
        return empty_figure(f"No NAV data for {fund_name}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=nav.index, y=nav.values,
        name=fund_name, mode="lines",
        line=dict(color="#2196F3", width=2),
        fill="tozeroy", fillcolor="rgba(33,150,243,0.08)",
        hovertemplate=(
            f"<b>{fund_name}</b><br>"
            "Date: %{x|%d %b %Y}<br>"
            "NAV: ₹%{y:,.4f}<extra></extra>"
        ),
    ))
    fig.update_layout(base_layout(
        title=f"NAV History — {fund_name}",
        x_title="Date", y_title="NAV (₹)",
        height=height, legend=False,
    ))
    fig.update_xaxes(
        rangeselector=dict(
            bgcolor="rgba(22,27,40,0.9)", activecolor="#2196F3",
            font=dict(color="#E0E0E0", size=11),
            buttons=[
                dict(count=1,  label="1Y", step="year",  stepmode="backward"),
                dict(count=3,  label="3Y", step="year",  stepmode="backward"),
                dict(count=5,  label="5Y", step="year",  stepmode="backward"),
                dict(step="all", label="All"),
            ],
        ),
    )
    return fig
