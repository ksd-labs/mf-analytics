"""
visualizations/alpha_charts.py
================================
Alpha generation charts — benchmark-relative visualizations.

Charts:
    1. Excess Return Chart    — fund vs benchmark NAV (normalised, same canvas)
    2. Rolling Alpha Chart    — Jensen's alpha over time (persistence view)
    3. Capture Ratio Chart    — up-capture vs down-capture scatter per category
    4. Alpha Metrics Bar      — all alpha metrics side-by-side for comparison
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Optional, List
from visualizations._theme import (
    base_layout, empty_figure, get_color,
    UP_COLOR, DOWN_COLOR, NEUTRAL_COLOR,
)


# ─────────────────────────────────────────────────────────────────────────────
# CHART 1 — FUND vs BENCHMARK NAV (normalised)
# ─────────────────────────────────────────────────────────────────────────────

def plot_fund_vs_benchmark(
    fund_nav:       pd.Series,
    benchmark_nav:  pd.Series,
    fund_name:      str,
    benchmark_name: str,
    height:         int = 420,
) -> go.Figure:
    """
    Overlay fund and benchmark NAV on the same chart, both rebased to 100.

    The gap between the two lines IS the fund's alpha visually.
    When the fund line is above the benchmark line, the fund is outperforming.

    Args:
        fund_nav:       Clean NAV series for the fund
        benchmark_nav:  Clean NAV series for the benchmark index fund
        fund_name:      Fund display name
        benchmark_name: Benchmark display name (e.g. "Nifty 100 TRI")
        height:         Chart height in pixels

    Returns:
        go.Figure with two overlaid normalised NAV lines + shaded gap area.
    """
    if fund_nav is None or benchmark_nav is None:
        return empty_figure("Fund or benchmark NAV not available")

    # Align to common dates
    common = fund_nav.index.intersection(benchmark_nav.index)
    if len(common) < 30:
        return empty_figure("Insufficient overlapping history between fund and benchmark")

    f = fund_nav.reindex(common)
    b = benchmark_nav.reindex(common)

    # Rebase to 100 at common start
    f_norm = (f / f.iloc[0]) * 100
    b_norm = (b / b.iloc[0]) * 100

    fig = go.Figure()

    # Shaded area showing outperformance / underperformance
    fig.add_trace(go.Scatter(
        x=f_norm.index, y=f_norm.values,
        name=fund_name, mode="lines",
        line=dict(color="#2196F3", width=2),
        fill=None,
        hovertemplate=(
            f"<b>{fund_name}</b><br>"
            "Date: %{x|%d %b %Y}<br>"
            "Value: %{y:.2f}<extra></extra>"
        ),
    ))

    fig.add_trace(go.Scatter(
        x=b_norm.index, y=b_norm.values,
        name=benchmark_name,
        mode="lines",
        line=dict(color="#FF9800", width=2, dash="dot"),
        fill="tonexty",
        fillcolor="rgba(33,150,243,0.07)",
        hovertemplate=(
            f"<b>{benchmark_name}</b><br>"
            "Date: %{x|%d %b %Y}<br>"
            "Value: %{y:.2f}<extra></extra>"
        ),
    ))

    # Outperformance annotation
    final_diff = float(f_norm.iloc[-1] - b_norm.iloc[-1])
    color = UP_COLOR if final_diff >= 0 else DOWN_COLOR
    sign  = "+" if final_diff >= 0 else ""

    fig.add_annotation(
        text=f"<b>{sign}{final_diff:.1f} pts vs benchmark</b>",
        xref="paper", yref="paper",
        x=0.99, y=0.99,
        showarrow=False,
        xanchor="right",
        font=dict(size=12, color=color),
        bgcolor="rgba(22,27,40,0.85)",
        borderpad=4,
    )

    fig.update_layout(
        base_layout(
            title=f"Fund vs Benchmark — {fund_name}",
            x_title="Date",
            y_title="Value (Rebased to 100)",
            height=height,
            hovermode="x unified",
        )
    )

    fig.update_xaxes(
        rangeselector=dict(
            bgcolor="rgba(22,27,40,0.9)",
            activecolor="#2196F3",
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
# CHART 2 — ROLLING ALPHA (PERSISTENCE)
# ─────────────────────────────────────────────────────────────────────────────

def plot_rolling_alpha(
    rolling_alpha_dict: Dict[str, Optional[pd.Series]],
    window_label:       str = "1-Year",
    height:             int = 400,
) -> go.Figure:
    """
    Rolling Jensen's Alpha over time — shows whether alpha is persistent.

    A fund with a consistently positive line has sustained manager skill.
    A fund that dips frequently below zero generates sporadic or no alpha.

    Args:
        rolling_alpha_dict: {fund_name: rolling_alpha_series}
                            rolling_alpha_series = output of calc_rolling_alpha()
        window_label:       Display label for the window size
        height:             Chart height

    Returns:
        go.Figure with rolling alpha time series + zero reference line.
    """
    valid = {k: v for k, v in rolling_alpha_dict.items() if v is not None and len(v) > 0}
    if not valid:
        return empty_figure("Insufficient data for rolling alpha (requires 2+ years of overlapping history)")

    fig = go.Figure()

    for i, (name, series) in enumerate(valid.items()):
        pct = series * 100    # Convert to percentage
        color = get_color(i)

        fig.add_trace(go.Scatter(
            x=pct.index, y=pct.values,
            name=name, mode="lines",
            line=dict(color=color, width=1.8),
            hovertemplate=(
                f"<b>{name}</b><br>"
                "Date: %{x|%d %b %Y}<br>"
                f"{window_label} Rolling Alpha: %{{y:.2f}}%"
                "<extra></extra>"
            ),
        ))

    # Zero line — alpha above here means manager is adding value
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="rgba(255,152,0,0.6)",
        line_width=1.5,
        annotation_text="0% Alpha",
        annotation_position="right",
        annotation_font_size=10,
        annotation_font_color=NEUTRAL_COLOR,
    )

    fig.update_layout(
        base_layout(
            title=f"{window_label} Rolling Jensen's Alpha",
            x_title="Date (end of rolling window)",
            y_title="Annualized Alpha (%)",
            height=height,
            hovermode="x unified",
        )
    )
    fig.update_yaxes(ticksuffix="%")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CHART 3 — CAPTURE RATIO SCATTER
# ─────────────────────────────────────────────────────────────────────────────

def plot_capture_scatter(
    full_df:        pd.DataFrame,
    category:       str,
    height:         int = 500,
) -> go.Figure:
    """
    Up-Capture vs Down-Capture scatter plot — the alpha quadrant chart.

    Quadrants:
        Top-left  (high up, low down) → IDEAL: captures gains, avoids losses
        Top-right (high up, high down)→ Aggressive: needs bull market to shine
        Bottom-left (low up, low down)→ Defensive: won't lose much but won't gain
        Bottom-right(low up, high down)→ WORST: loses more than it gains

    Args:
        full_df:  Full metrics + quartile DataFrame (from compute_category_quartiles)
                  Must have 'up_capture' and 'down_capture' columns.
        category: Category name for the title.
        height:   Chart height.

    Returns:
        go.Figure with capture ratio scatter.
    """
    required = ["up_capture", "down_capture"]
    for col in required:
        if col not in full_df.columns:
            return empty_figure("Capture ratio data not available — run analytics first")

    plot_df = full_df[required].dropna()
    if plot_df.empty:
        return empty_figure("No capture ratio data available")

    x_vals = plot_df["down_capture"]     # X = down-capture (lower = better)
    y_vals = plot_df["up_capture"]       # Y = up-capture (higher = better)
    names  = [n[:38] + "…" if len(n) > 38 else n for n in plot_df.index]

    # Colour by capture_ratio quartile if available
    cap_q_col = "capture_ratio_quartile"
    from utils.constants import QUARTILE_COLORS
    colors = [
        QUARTILE_COLORS.get(str(full_df.loc[n, cap_q_col]), "#2196F3")
        if cap_q_col in full_df.columns and n in full_df.index else "#2196F3"
        for n in plot_df.index
    ]

    fig = go.Figure()

    # ── Quadrant shading ─────────────────────────────────────────────────────
    x_mid = 100.0   # 100% = neutral capture
    y_mid = 100.0

    quad_style = dict(type="rect", xref="x", yref="y",
                      line=dict(width=0))

    x_min = max(float(x_vals.min()) * 0.92, 40)
    x_max = float(x_vals.max()) * 1.08
    y_min = max(float(y_vals.min()) * 0.92, 40)
    y_max = float(y_vals.max()) * 1.08

    # Top-left (ideal) — green tint
    fig.add_shape(**quad_style, x0=x_min, x1=x_mid, y0=y_mid, y1=y_max,
                  fillcolor="rgba(76,175,80,0.06)")
    # Bottom-right (worst) — red tint
    fig.add_shape(**quad_style, x0=x_mid, x1=x_max, y0=y_min, y1=y_mid,
                  fillcolor="rgba(244,67,54,0.06)")
    # Remaining two quadrants — subtle neutral
    fig.add_shape(**quad_style, x0=x_min, x1=x_mid, y0=y_min, y1=y_mid,
                  fillcolor="rgba(255,255,255,0.02)")
    fig.add_shape(**quad_style, x0=x_mid, x1=x_max, y0=y_mid, y1=y_max,
                  fillcolor="rgba(255,255,255,0.02)")

    # Cross-hair at 100/100
    fig.add_hline(y=100, line_dash="dot",
                  line_color="rgba(255,255,255,0.15)", line_width=1)
    fig.add_vline(x=100, line_dash="dot",
                  line_color="rgba(255,255,255,0.15)", line_width=1)

    # ── Data points ──────────────────────────────────────────────────────────
    cap_ratios = (
        [full_df.loc[n, "capture_ratio"] for n in plot_df.index]
        if "capture_ratio" in full_df.columns
        else [None] * len(plot_df)
    )

    hover = [
        f"<b>{n}</b><br>"
        f"Up-Capture: {float(y_vals.iloc[i]):.1f}%<br>"
        f"Down-Capture: {float(x_vals.iloc[i]):.1f}%<br>"
        f"Capture Ratio: {f'{cap_ratios[i]:.3f}' if cap_ratios[i] else 'N/A'}"
        "<extra></extra>"
        for i, n in enumerate(plot_df.index)
    ]

    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals,
        mode="markers+text",
        marker=dict(size=11, color=colors,
                    line=dict(color="rgba(255,255,255,0.3)", width=1),
                    opacity=0.88),
        text=names,
        textposition="top center",
        textfont=dict(size=8, color="#C0C0C0"),
        hovertemplate=hover,
        showlegend=False,
    ))

    # ── Quadrant labels ───────────────────────────────────────────────────────
    _ql = dict(showarrow=False, font=dict(size=8, color="rgba(200,200,200,0.25)"))
    fig.add_annotation(text="IDEAL ✓", x=x_min*1.02, y=y_max*0.98,
                       xanchor="left", **_ql)
    fig.add_annotation(text="WORST ✗", x=x_max*0.98, y=y_min*1.02,
                       xanchor="right", **_ql)
    fig.add_annotation(text="DEFENSIVE", x=x_min*1.02, y=y_min*1.02,
                       xanchor="left", **_ql)
    fig.add_annotation(text="AGGRESSIVE", x=x_max*0.98, y=y_max*0.98,
                       xanchor="right", **_ql)

    fig.update_layout(
        base_layout(
            title=f"Capture Ratio Map — {category}",
            x_title="Down-Capture (%) — Lower = better downside protection →",
            y_title="Up-Capture (%) — Higher = better upside participation ↑",
            height=height,
            hovermode="closest",
        )
    )
    fig.update_xaxes(ticksuffix="%")
    fig.update_yaxes(ticksuffix="%")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CHART 4 — ALPHA METRICS COMPARISON BAR
# ─────────────────────────────────────────────────────────────────────────────

def plot_alpha_comparison(
    fund_metrics_dict: Dict[str, Dict],
    height:            int = 420,
) -> go.Figure:
    """
    Grouped bar chart comparing key alpha metrics across multiple funds.

    Shows: Jensen's Alpha, Information Ratio, Excess Return, Capture Ratio
    side-by-side for each fund.

    Args:
        fund_metrics_dict: {fund_name: metrics_dict}
                           metrics_dict must contain alpha metric keys.
        height:            Chart height.

    Returns:
        go.Figure with grouped bars.
    """
    if not fund_metrics_dict:
        return empty_figure("No fund metrics provided")

    METRICS_TO_SHOW = [
        ("jensens_alpha",    "Jensen's Alpha",    True,  100),   # multiply by 100 → %
        ("information_ratio","Information Ratio", False, 1),
        ("excess_return",    "Excess Return",     True,  100),
        ("capture_ratio",    "Capture Ratio",     False, 1),
    ]

    fig = go.Figure()

    for i, (name, metrics) in enumerate(fund_metrics_dict.items()):
        if not metrics.get("is_valid"):
            continue

        values = []
        labels = []
        for key, label, is_pct, multiplier in METRICS_TO_SHOW:
            val = metrics.get(key)
            if val is not None and np.isfinite(val):
                values.append(float(val) * multiplier)
                labels.append(label)
            else:
                values.append(None)
                labels.append(label)

        bar_colors = [
            UP_COLOR if (v is not None and v > 0) else DOWN_COLOR
            for v in values
        ]

        fig.add_trace(go.Bar(
            x=labels,
            y=values,
            name=name[:35],
            marker_color=get_color(i),
            opacity=0.82,
            hovertemplate=(
                f"<b>{name[:40]}</b><br>"
                "Metric: %{x}<br>"
                "Value: %{y:.3f}"
                "<extra></extra>"
            ),
        ))

    fig.add_hline(y=0, line_dash="dot",
                  line_color="rgba(255,255,255,0.2)", line_width=1)

    fig.update_layout(
        base_layout(
            title="Alpha Metrics Comparison",
            x_title="Metric",
            y_title="Value",
            height=height,
            hovermode="x unified",
        ),
        barmode="group",
    )

    return fig
