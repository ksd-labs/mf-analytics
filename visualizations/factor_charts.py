"""
visualizations/factor_charts.py
================================
Phase C — Fama-French factor model charts.

Chart 1 — Factor Loading Bar Chart
    Shows β_mkt, β_smb, β_hml, β_wml for one or more funds.
    The zero line separates positive/negative tilts.

Chart 2 — Factor Contribution Chart
    Stacked bar showing how much each factor contributed to the fund's
    total annualized return vs how much came from pure alpha.

Chart 3 — Rolling 4-Factor Alpha
    Same as rolling Jensen's alpha but controlling for all 4 factors.
    Purer measure of skill persistence.

Chart 4 — Factor Exposure Heatmap
    Category-wide: funds × factor loadings, colour-coded.
    Instantly shows which funds have similar factor profiles.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Optional, List
from visualizations._theme import (
    base_layout, empty_figure, get_color,
    UP_COLOR, DOWN_COLOR, NEUTRAL_COLOR,
)


# Factor display config
FACTOR_COLORS = {
    "market":  "#2196F3",   # Blue
    "smb":     "#4CAF50",   # Green
    "hml":     "#FF9800",   # Orange
    "wml":     "#9C27B0",   # Purple
    "alpha":   "#F44336",   # Red — pure skill
    "unexplained": "#607D8B",
}

FACTOR_LABELS = {
    "market":  "Market (β_mkt)",
    "smb":     "Size SMB (β_smb)",
    "hml":     "Value HML (β_hml)",
    "wml":     "Momentum WML (β_wml)",
    "alpha":   "4-Factor Alpha",
    "contrib_market": "Market Contribution",
    "contrib_smb":    "Size Contribution",
    "contrib_hml":    "Value Contribution",
    "contrib_wml":    "Momentum Contribution",
    "contrib_alpha":  "Pure Alpha",
}


# ─────────────────────────────────────────────────────────────────────────────
# CHART 1 — FACTOR LOADING BAR CHART
# ─────────────────────────────────────────────────────────────────────────────

def plot_factor_loadings(
    fund_metrics_dict: Dict[str, Dict],
    height:            int = 420,
) -> go.Figure:
    """
    Grouped bar chart of factor loadings (betas) for one or more funds.

    Each cluster of bars = one fund.
    Each bar = one factor loading (β_mkt, β_smb, β_hml, β_wml).

    Interpretation:
        Bar above zero → positive exposure to that factor
        Bar below zero → negative exposure (contrarian to that factor)
        Bar near zero  → fund is neutral to that factor

    Args:
        fund_metrics_dict: {fund_name: metrics_dict}
        height:            Chart height in pixels.

    Returns:
        go.Figure with grouped factor loading bars.
    """
    FACTOR_KEYS = [
        ("beta_market_4f", "Market β",  FACTOR_COLORS["market"]),
        ("beta_smb",       "Size β",    FACTOR_COLORS["smb"]),
        ("beta_hml",       "Value β",   FACTOR_COLORS["hml"]),
        ("beta_wml",       "Momentum β",FACTOR_COLORS["wml"]),
    ]

    valid = {
        k: v for k, v in fund_metrics_dict.items()
        if v.get("is_valid") and any(
            v.get(fk) is not None for fk, _, _ in FACTOR_KEYS
        )
    }

    if not valid:
        return empty_figure("Factor model data not available — benchmark and factor proxies required")

    names = [n[:35] + "…" if len(n) > 35 else n for n in valid.keys()]
    fig   = go.Figure()

    for fk, label, color in FACTOR_KEYS:
        vals = [v.get(fk) for v in valid.values()]
        if all(v is None for v in vals):
            continue

        fig.add_trace(go.Bar(
            name          = label,
            x             = names,
            y             = [v if v is not None else 0 for v in vals],
            marker_color  = color,
            opacity       = 0.82,
            hovertemplate = f"<b>%{{x}}</b><br>{label}: %{{y:.3f}}<extra></extra>",
        ))

    # Zero reference
    fig.add_hline(y=0, line_dash="dot",
                  line_color="rgba(255,255,255,0.2)", line_width=1)

    # Interpretation guide
    fig.add_annotation(
        text=(
            "β_smb > 0 = small-cap tilt  |  β_hml > 0 = value tilt  |  "
            "β_wml > 0 = momentum tilt"
        ),
        xref="paper", yref="paper",
        x=0.5, y=-0.15, showarrow=False,
        font=dict(size=9, color=NEUTRAL_COLOR),
        xanchor="center",
    )

    fig.update_layout(
        base_layout(
            title     = "Factor Loadings (β Coefficients)",
            x_title   = "Fund",
            y_title   = "Factor Loading (β)",
            height    = height,
            hovermode = "x unified",
        ),
        barmode = "group",
    )
    fig.update_xaxes(tickangle=-25)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CHART 2 — FACTOR CONTRIBUTION (RETURN ATTRIBUTION)
# ─────────────────────────────────────────────────────────────────────────────

def plot_factor_contribution(
    fund_metrics_dict: Dict[str, Dict],
    height:            int = 440,
) -> go.Figure:
    """
    Stacked bar chart showing return attribution across factors.

    Each bar = total annualized return of the fund (approximately).
    Each segment = return explained by one factor or pure alpha.

    Interpretation:
        A tall "Pure Alpha" segment → most return came from stock selection
        A tall "Market Contribution" → fund just rode the market
        Negative segments → factor worked against the fund

    Args:
        fund_metrics_dict: {fund_name: metrics_dict}
        height:            Chart height.

    Returns:
        go.Figure with stacked contribution bars.
    """
    CONTRIB_KEYS = [
        ("contrib_market", "Market",    FACTOR_COLORS["market"]),
        ("contrib_smb",    "Size",      FACTOR_COLORS["smb"]),
        ("contrib_hml",    "Value",     FACTOR_COLORS["hml"]),
        ("contrib_wml",    "Momentum",  FACTOR_COLORS["wml"]),
        ("contrib_alpha",  "Pure Alpha",FACTOR_COLORS["alpha"]),
    ]

    valid = {
        k: v for k, v in fund_metrics_dict.items()
        if v.get("is_valid") and any(
            v.get(ck) is not None for ck, _, _ in CONTRIB_KEYS
        )
    }

    if not valid:
        return empty_figure("Factor contribution data not available")

    names = [n[:35] + "…" if len(n) > 35 else n for n in valid.keys()]
    fig   = go.Figure()

    for ck, label, color in CONTRIB_KEYS:
        vals = [
            (v.get(ck) or 0) * 100   # Convert to percentage
            for v in valid.values()
        ]
        if all(v == 0 for v in vals):
            continue

        fig.add_trace(go.Bar(
            name          = label,
            x             = names,
            y             = vals,
            marker_color  = color,
            opacity       = 0.82,
            hovertemplate = (
                f"<b>%{{x}}</b><br>"
                f"{label} Contribution: %{{y:.2f}}%"
                "<extra></extra>"
            ),
        ))

    fig.add_hline(y=0, line_dash="dot",
                  line_color="rgba(255,255,255,0.2)", line_width=1)

    fig.update_layout(
        base_layout(
            title     = "Return Attribution — Factor Contributions",
            x_title   = "Fund",
            y_title   = "Annualized Contribution (%)",
            height    = height,
            hovermode = "x unified",
        ),
        barmode = "relative",   # Stacked with negative bars going down
    )
    fig.update_yaxes(ticksuffix="%")
    fig.update_xaxes(tickangle=-25)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CHART 3 — ROLLING 4-FACTOR ALPHA
# ─────────────────────────────────────────────────────────────────────────────

def plot_rolling_alpha_4f(
    rolling_dict: Dict[str, Optional[pd.Series]],
    height:       int = 400,
) -> go.Figure:
    """
    Rolling 4-Factor Alpha over time.

    Compares rolling Jensen's alpha (1-factor) vs rolling 4-factor alpha
    to show how much of the apparent alpha is explained by factor tilts.

    Args:
        rolling_dict: {fund_name: rolling_alpha_4f_series}
        height:       Chart height.

    Returns:
        go.Figure with rolling 4-factor alpha line.
    """
    valid = {k: v for k, v in rolling_dict.items()
             if v is not None and len(v) > 0}

    if not valid:
        return empty_figure(
            "Rolling 4-Factor Alpha requires 2+ years of "
            "overlapping fund and factor history"
        )

    fig = go.Figure()

    for i, (name, series) in enumerate(valid.items()):
        pct   = (series * 100).dropna()
        color = get_color(i)

        fig.add_trace(go.Scatter(
            x=pct.index, y=pct.values,
            name=name, mode="lines",
            line=dict(color=color, width=1.8),
            hovertemplate=(
                f"<b>{name}</b><br>"
                "Date: %{x|%d %b %Y}<br>"
                "4F Rolling Alpha: %{y:.2f}%"
                "<extra></extra>"
            ),
        ))

    fig.add_hline(y=0, line_dash="dash",
                  line_color="rgba(244,67,54,0.5)", line_width=1.5,
                  annotation_text="0% True Alpha",
                  annotation_position="right",
                  annotation_font_color=DOWN_COLOR,
                  annotation_font_size=10)

    fig.update_layout(
        base_layout(
            title     = "Rolling 1-Year 4-Factor Alpha (True Alpha after Factor Adjustment)",
            x_title   = "Date",
            y_title   = "Annualized 4-Factor Alpha (%)",
            height    = height,
            hovermode = "x unified",
        )
    )
    fig.update_yaxes(ticksuffix="%")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CHART 4 — FACTOR EXPOSURE HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

def plot_factor_heatmap(
    full_df: pd.DataFrame,
    height:  int = 500,
) -> go.Figure:
    """
    Heatmap of factor loadings across all funds in a category.

    Rows = funds, Columns = Market β, SMB β, HML β, WML β, 4F Alpha.
    Colour = blue (positive loading) → red (negative loading).

    This shows whether funds in a category cluster into similar
    factor profiles (style drift) or are genuinely differentiated.

    Args:
        full_df: Full metrics DataFrame from compute_category_quartiles().
        height:  Chart height.

    Returns:
        go.Figure with annotated factor heatmap.
    """
    FACTOR_COLS = {
        "beta_market_4f": "Market β",
        "beta_smb":       "Size β",
        "beta_hml":       "Value β",
        "beta_wml":       "Momentum β",
        "alpha_4f":       "4F Alpha",
    }

    available = {k: v for k, v in FACTOR_COLS.items() if k in full_df.columns}
    if not available:
        return empty_figure("Factor model data not available in category metrics")

    sub = full_df[list(available.keys())].dropna(how="all")
    if sub.empty:
        return empty_figure("No factor data to display")

    # Build z matrix and text annotations
    z    = sub.values.astype(float)
    text = np.full(z.shape, "", dtype=object)

    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            val = z[i, j]
            col = list(available.keys())[j]
            if np.isnan(val):
                text[i, j] = "N/A"
            elif col == "alpha_4f":
                text[i, j] = f"{val*100:.2f}%"
            else:
                text[i, j] = f"{val:.3f}"

    x_labels = list(available.values())
    y_labels  = [n[:35] + "…" if len(n) > 35 else n for n in sub.index]

    # Diverging colour scale: red (negative) → white (zero) → blue (positive)
    colorscale = [
        [0.00, "#b71c1c"],
        [0.25, "#ef5350"],
        [0.45, "#90A4AE"],
        [0.55, "#90A4AE"],
        [0.75, "#42A5F5"],
        [1.00, "#0d47a1"],
    ]

    fig = go.Figure(go.Heatmap(
        z=z, x=x_labels, y=y_labels,
        text=text, texttemplate="%{text}",
        textfont=dict(size=9, color="white"),
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(
            title=dict(text="Loading", font=dict(color="#E0E0E0", size=10)),
            tickfont=dict(color="#E0E0E0", size=9),
        ),
        xgap=2, ygap=2,
        hovertemplate="<b>%{y}</b><br>%{x}: %{text}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="Factor Exposure Heatmap",
                   x=0.01, font=dict(size=14, color="#E0E0E0")),
        height=max(height, 80 + 35 * len(sub)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(22,27,40,0.6)",
        font=dict(color="#E0E0E0", size=10),
        margin=dict(l=220, r=80, t=55, b=80),
        xaxis=dict(tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=9), autorange="reversed"),
    )

    return fig
