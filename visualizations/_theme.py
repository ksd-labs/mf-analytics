"""
visualizations/_theme.py
========================
Shared Plotly theme and layout helpers used by all chart modules.

All charts in the platform use this theme so they are visually consistent
with each other and with the Streamlit dark config.toml.

Importing:
    from visualizations._theme import apply_theme, COLORS, GRID_COLOR
"""

import plotly.graph_objects as go
from typing import Optional, Dict, Any
from utils.constants import CHART_COLORS, QUARTILE_COLORS

# ─────────────────────────────────────────────────────────────────────────────
# PALETTE
# ─────────────────────────────────────────────────────────────────────────────

COLORS          = CHART_COLORS
QUARTILE_COLS   = QUARTILE_COLORS

BG_PAPER        = "rgba(0,0,0,0)"           # Transparent — inherits Streamlit bg
BG_PLOT         = "rgba(22,27,40,0.6)"      # Subtle dark panel
GRID_COLOR      = "rgba(255,255,255,0.08)"  # Very faint grid lines
ZERO_LINE_COLOR = "rgba(255,255,255,0.20)"  # Slightly visible zero axis
FONT_COLOR      = "#E0E0E0"
FONT_FAMILY     = "sans-serif"

UP_COLOR        = "#4CAF50"   # Green  — positive values
DOWN_COLOR      = "#F44336"   # Red    — negative values
NEUTRAL_COLOR   = "#78909C"   # Grey   — neutral / reference lines


# ─────────────────────────────────────────────────────────────────────────────
# BASE LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

def base_layout(
    title:       Optional[str] = None,
    x_title:     Optional[str] = None,
    y_title:     Optional[str] = None,
    height:      int = 420,
    legend:      bool = True,
    hovermode:   str = "x unified",
    **extra,
) -> go.Layout:
    """
    Return a go.Layout object with the platform's standard dark theme applied.

    Args:
        title:     Chart title string
        x_title:   X-axis label
        y_title:   Y-axis label
        height:    Chart height in pixels
        legend:    Whether to show the legend
        hovermode: Plotly hovermode ('x unified', 'closest', False)
        **extra:   Any additional go.Layout kwargs

    Returns:
        go.Layout with dark theme pre-applied.
    """
    layout_dict: Dict[str, Any] = dict(
        height          = height,
        paper_bgcolor   = BG_PAPER,
        plot_bgcolor    = BG_PLOT,
        hovermode       = hovermode,
        margin          = dict(l=60, r=30, t=55 if title else 30, b=55),
        font            = dict(color=FONT_COLOR, family=FONT_FAMILY, size=12),
        showlegend      = legend,
        legend          = dict(
            bgcolor      = "rgba(22,27,40,0.8)",
            bordercolor  = "rgba(255,255,255,0.1)",
            borderwidth  = 1,
            font         = dict(size=11),
            orientation  = "h",
            yanchor      = "bottom",
            y            = 1.01,
            xanchor      = "left",
            x            = 0,
        ),
        xaxis = dict(
            gridcolor       = GRID_COLOR,
            gridwidth       = 1,
            zerolinecolor   = ZERO_LINE_COLOR,
            showgrid        = True,
            title           = dict(text=x_title or "", font=dict(size=12)),
            tickfont        = dict(size=10),
        ),
        yaxis = dict(
            gridcolor       = GRID_COLOR,
            gridwidth       = 1,
            zerolinecolor   = ZERO_LINE_COLOR,
            zerolinewidth   = 1.5,
            showgrid        = True,
            title           = dict(text=y_title or "", font=dict(size=12)),
            tickfont        = dict(size=10),
        ),
    )

    if title:
        layout_dict["title"] = dict(
            text    = title,
            x       = 0.01,
            xanchor = "left",
            font    = dict(size=15, color=FONT_COLOR),
        )

    layout_dict.update(extra)
    return go.Layout(**layout_dict)


def empty_figure(message: str = "Insufficient data") -> go.Figure:
    """
    Return a blank Plotly figure with a centred message.
    Used when a chart cannot be rendered due to missing data.
    """
    fig = go.Figure()
    fig.add_annotation(
        text      = f"<b>{message}</b>",
        xref      = "paper", yref = "paper",
        x = 0.5,  y = 0.5,
        showarrow = False,
        font      = dict(size=14, color=NEUTRAL_COLOR),
    )
    fig.update_layout(
        paper_bgcolor = BG_PAPER,
        plot_bgcolor  = BG_PLOT,
        xaxis         = dict(visible=False),
        yaxis         = dict(visible=False),
        height        = 350,
    )
    return fig


def get_color(index: int) -> str:
    """Cycle through CHART_COLORS by index."""
    return COLORS[index % len(COLORS)]
