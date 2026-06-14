"""
visualizations/alpha_charts.py
================================
Alpha generation charts — benchmark-relative visualizations.

plot_fund_vs_benchmark: Updated to show % returns from period start (not rebased 100).
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Optional
from visualizations._theme import (
    base_layout, empty_figure, get_color,
    UP_COLOR, DOWN_COLOR, NEUTRAL_COLOR,
)


# ─────────────────────────────────────────────────────────────────────────────
# CHART 1 — FUND vs BENCHMARK  (% return from period start)
# ─────────────────────────────────────────────────────────────────────────────

PERIOD_MAP = {
    "1M": 1, "3M": 3, "6M": 6,
    "1Y": 12, "3Y": 36, "5Y": 60, "All": None,
}


def plot_fund_vs_benchmark(
    fund_nav:       pd.Series,
    benchmark_nav:  pd.Series,
    fund_name:      str,
    benchmark_name: str,
    period_label:   str = "All",
    height:         int = 440,
) -> go.Figure:
    """
    Fund vs Benchmark — both lines start at 0% at the beginning of the
    selected period. Gap between lines = alpha visually.

    Args:
        fund_nav:       Clean NAV series for the fund
        benchmark_nav:  Clean NAV series for the benchmark
        fund_name:      Fund display name
        benchmark_name: Benchmark display name (e.g. "Nifty 100 TRI")
        period_label:   "1M","3M","6M","1Y","3Y","5Y","All"
        height:         Chart height in pixels

    Returns:
        go.Figure — both lines start at 0%, shaded gap area.
    """
    if fund_nav is None or benchmark_nav is None:
        return empty_figure("Fund or benchmark NAV not available")

    # Common date range
    common = fund_nav.index.intersection(benchmark_nav.index)
    if len(common) < 10:
        return empty_figure("Insufficient overlapping history")

    f = fund_nav.reindex(common)
    b = benchmark_nav.reindex(common)

    # Slice to period
    period_months = PERIOD_MAP.get(period_label)
    if period_months is not None:
        end   = f.index[-1]
        start = end - pd.DateOffset(months=period_months)
        f = f[f.index >= start]
        b = b[b.index >= start]
        if len(f) < 5:
            f = fund_nav.reindex(common)
            b = benchmark_nav.reindex(common)

    # Common start after slicing
    cs = max(f.index[0], b.index[0])
    f  = f[f.index >= cs]
    b  = b[b.index >= cs]

    # % return from common start
    f_pct = (f / f.iloc[0] - 1) * 100
    b_pct = (b / b.iloc[0] - 1) * 100

    fig = go.Figure()

    # Shaded area between lines (fund - benchmark)
    diff = f_pct.values - b_pct.reindex(f_pct.index).values

    # Benchmark line
    fig.add_trace(go.Scatter(
        x=b_pct.index, y=b_pct.values,
        name=benchmark_name,
        mode="lines",
        line=dict(color="#FF9800", width=1.8, dash="dot"),
        hovertemplate=(
            f"<b>{benchmark_name}</b><br>"
            "Date: %{x|%d %b %Y}<br>"
            "Return: %{y:+.2f}%<extra></extra>"
        ),
    ))

    # Fund line — filled to benchmark
    fig.add_trace(go.Scatter(
        x=f_pct.index, y=f_pct.values,
        name=fund_name,
        mode="lines",
        line=dict(color="#2196F3", width=2),
        fill="tonexty",
        fillcolor="rgba(33,150,243,0.08)",
        hovertemplate=(
            f"<b>{fund_name}</b><br>"
            "Date: %{x|%d %b %Y}<br>"
            "Return: %{y:+.2f}%<extra></extra>"
        ),
    ))

    # Zero baseline
    fig.add_hline(
        y=0, line_dash="dot",
        line_color="rgba(255,255,255,0.30)", line_width=1.2,
        annotation_text="0%", annotation_position="right",
        annotation_font_size=10,
        annotation_font_color="rgba(200,200,200,0.6)",
    )

    # Outperformance annotation
    final_diff = float(f_pct.iloc[-1] - b_pct.reindex(f_pct.index).iloc[-1])
    col  = UP_COLOR if final_diff >= 0 else DOWN_COLOR
    sign = "+" if final_diff >= 0 else ""
    fig.add_annotation(
        text=f"<b>{sign}{final_diff:.1f}% vs benchmark</b>",
        xref="paper", yref="paper", x=0.99, y=0.99,
        showarrow=False, xanchor="right",
        font=dict(size=12, color=col),
        bgcolor="rgba(22,27,40,0.85)", borderpad=4,
    )

    fig.update_layout(base_layout(
        title=f"Fund vs Benchmark — {period_label}",
        x_title="Date",
        y_title=f"Return from {cs.strftime('%d %b %Y')} (%)",
        height=height, hovermode="x unified",
    ))
    fig.update_yaxes(ticksuffix="%", zeroline=True,
                     zerolinecolor="rgba(255,255,255,0.30)", zerolinewidth=1.2)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CHART 2 — ROLLING ALPHA
# ─────────────────────────────────────────────────────────────────────────────

def plot_rolling_alpha(
    rolling_alpha_dict: Dict[str, Optional[pd.Series]],
    window_label:       str = "1-Year",
    height:             int = 400,
) -> go.Figure:
    valid = {k: v for k, v in rolling_alpha_dict.items()
             if v is not None and len(v) > 0}
    if not valid:
        return empty_figure("Rolling alpha requires 2+ years of overlapping history")

    fig = go.Figure()
    for i, (name, series) in enumerate(valid.items()):
        pct = (series * 100).dropna()
        fig.add_trace(go.Scatter(
            x=pct.index, y=pct.values, name=name,
            mode="lines", line=dict(color=get_color(i), width=1.8),
            hovertemplate=(
                f"<b>{name}</b><br>Date: %{{x|%d %b %Y}}<br>"
                f"{window_label} Rolling Alpha: %{{y:.2f}}%<extra></extra>"
            ),
        ))

    fig.add_hline(y=0, line_dash="dash",
                  line_color="rgba(255,152,0,0.6)", line_width=1.5,
                  annotation_text="0% Alpha", annotation_position="right",
                  annotation_font_color=NEUTRAL_COLOR, annotation_font_size=10)

    fig.update_layout(base_layout(
        title=f"{window_label} Rolling Jensen's Alpha",
        x_title="Date", y_title="Annualized Alpha (%)",
        height=height, hovermode="x unified",
    ))
    fig.update_yaxes(ticksuffix="%")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CHART 3 — CAPTURE RATIO SCATTER
# ─────────────────────────────────────────────────────────────────────────────

def plot_capture_scatter(
    full_df:  pd.DataFrame,
    category: str,
    height:   int = 500,
) -> go.Figure:
    required = ["up_capture", "down_capture"]
    for col in required:
        if col not in full_df.columns:
            return empty_figure("Capture ratio data not available")

    plot_df = full_df[required].dropna()
    if plot_df.empty:
        return empty_figure("No capture ratio data")

    x_vals = plot_df["down_capture"]
    y_vals = plot_df["up_capture"]
    names  = [n[:38] + "…" if len(n) > 38 else n for n in plot_df.index]

    from utils.constants import QUARTILE_COLORS
    cap_q = "capture_ratio_quartile"
    colors = [
        QUARTILE_COLORS.get(str(full_df.loc[n, cap_q]), "#2196F3")
        if cap_q in full_df.columns and n in full_df.index else "#2196F3"
        for n in plot_df.index
    ]

    cap_ratios = [
        full_df.loc[n, "capture_ratio"] if "capture_ratio" in full_df.columns
        and n in full_df.index else None
        for n in plot_df.index
    ]

    hover = [
        f"<b>{n}</b><br>Up-Capture: {float(y_vals.iloc[i]):.1f}%<br>"
        f"Down-Capture: {float(x_vals.iloc[i]):.1f}%<br>"
        f"Capture Ratio: {f'{cap_ratios[i]:.3f}' if cap_ratios[i] else 'N/A'}"
        "<extra></extra>"
        for i, n in enumerate(plot_df.index)
    ]

    fig = go.Figure()

    x_mid, y_mid = 100.0, 100.0
    x_min = max(float(x_vals.min()) * 0.92, 40)
    x_max = float(x_vals.max()) * 1.08
    y_min = max(float(y_vals.min()) * 0.92, 40)
    y_max = float(y_vals.max()) * 1.08

    qs = dict(type="rect", xref="x", yref="y", line=dict(width=0))
    fig.add_shape(**qs, x0=x_min, x1=x_mid, y0=y_mid, y1=y_max, fillcolor="rgba(76,175,80,0.06)")
    fig.add_shape(**qs, x0=x_mid, x1=x_max, y0=y_min, y1=y_mid, fillcolor="rgba(244,67,54,0.06)")
    fig.add_shape(**qs, x0=x_min, x1=x_mid, y0=y_min, y1=y_mid, fillcolor="rgba(255,255,255,0.02)")
    fig.add_shape(**qs, x0=x_mid, x1=x_max, y0=y_mid, y1=y_max, fillcolor="rgba(255,255,255,0.02)")

    fig.add_hline(y=100, line_dash="dot", line_color="rgba(255,255,255,0.15)", line_width=1)
    fig.add_vline(x=100, line_dash="dot", line_color="rgba(255,255,255,0.15)", line_width=1)

    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals, mode="markers+text",
        marker=dict(size=11, color=colors,
                    line=dict(color="rgba(255,255,255,0.3)", width=1), opacity=0.88),
        text=names, textposition="top center",
        textfont=dict(size=8, color="#C0C0C0"),
        hovertemplate=hover, showlegend=False,
    ))

    _ql = dict(showarrow=False, font=dict(size=8, color="rgba(200,200,200,0.25)"))
    fig.add_annotation(text="IDEAL ✓", x=x_min*1.02, y=y_max*0.98, xanchor="left", **_ql)
    fig.add_annotation(text="WORST ✗", x=x_max*0.98, y=y_min*1.02, xanchor="right", **_ql)

    fig.update_layout(base_layout(
        title=f"Capture Ratio Map — {category}",
        x_title="Down-Capture (%) — Lower = better →",
        y_title="Up-Capture (%) — Higher = better ↑",
        height=height, hovermode="closest",
    ))
    fig.update_xaxes(ticksuffix="%")
    fig.update_yaxes(ticksuffix="%")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CHART 4 — ALPHA COMPARISON BAR
# ─────────────────────────────────────────────────────────────────────────────

def plot_alpha_comparison(
    fund_metrics_dict: Dict[str, Dict],
    height:            int = 420,
) -> go.Figure:
    if not fund_metrics_dict:
        return empty_figure("No fund metrics provided")

    SHOW = [
        ("jensens_alpha",    "Jensen's Alpha",    True,  100),
        ("information_ratio","Information Ratio", False, 1),
        ("excess_return",    "Excess Return",     True,  100),
        ("capture_ratio",    "Capture Ratio",     False, 1),
    ]

    fig = go.Figure()
    for i, (name, metrics) in enumerate(fund_metrics_dict.items()):
        if not metrics.get("is_valid"):
            continue
        values, labels = [], []
        for key, label, is_pct, mult in SHOW:
            val = metrics.get(key)
            if val is not None and np.isfinite(val):
                values.append(float(val) * mult)
                labels.append(label)
            else:
                values.append(None)
                labels.append(label)

        fig.add_trace(go.Bar(
            x=labels, y=values, name=name[:35],
            marker_color=get_color(i), opacity=0.82,
            hovertemplate=(
                f"<b>{name[:40]}</b><br>%{{x}}: %{{y:.3f}}<extra></extra>"
            ),
        ))

    fig.add_hline(y=0, line_dash="dot",
                  line_color="rgba(255,255,255,0.2)", line_width=1)
    fig.update_layout(base_layout(
        title="Alpha Metrics Comparison", x_title="Metric",
        y_title="Value", height=height, hovermode="x unified",
    ), barmode="group")
    return fig
