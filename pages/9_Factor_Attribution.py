"""
pages/9_Factor_Attribution.py
==============================
6-Factor Attribution — Cross-Fund Factor Analysis

Compare up to 3 funds using a 6-factor model:
    Market (Mkt-Rf) · Size (SMB) · Value (HML) · Momentum (WML)
    Quality (QMJ)   · Low Vol (BAB)

All 6 factors are required. If any TRI file is missing the page
shows a clear error rather than running with partial data.

Betas are STANDARDISED (zero mean, unit variance per factor) so
comparisons across factors and funds are directly meaningful.

Four tabs:
    📐 Factor Loadings       — betas + significance, model quality
    📈 Rolling Exposures     — 2×3 grid of rolling betas over time
    🧩 Return Attribution    — stacked bar: contribution per factor
    🔄 Regime Analysis       — betas estimated in Bull/Sideways/Bear
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data.fund_loader    import get_all_categorized_schemes, get_nav_history
from data.nav_processor  import process_nav, compute_daily_returns
from data.factor_loader  import (
    get_factor_returns_6f,
    FACTOR_DISPLAY_NAMES,
    FACTOR_COLORS,
)
from analytics.factor_model import (
    calc_factor_model_6f,
    calc_rolling_factor_betas,
    calc_regime_betas,
)
from visualizations._theme import get_color
from utils.constants  import CATEGORIES, APP_TITLE, APP_ICON
from utils.formatters import fmt_pct, fmt_ratio
from utils.session    import render_refresh_button

st.set_page_config(
    page_title = "Factor Attribution — MF Analytics",
    page_icon  = "🔬",
    layout     = "wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

FACTOR_ORDER = ["market", "smb", "hml", "wml", "qmj", "bab"]

_DARK_LAYOUT = dict(
    paper_bgcolor = "rgba(0,0,0,0)",
    plot_bgcolor  = "rgba(22,27,40,0.5)",
    font          = dict(color="#E0E0E0", size=11),
)
_GRID = dict(gridcolor="rgba(255,255,255,0.07)", showgrid=True)
_ZERO = dict(zeroline=True, zerolinecolor="rgba(255,255,255,0.35)", zerolinewidth=1)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _apply_period(obj, period_label):
    """Slice a Series or DataFrame to the trailing period for display."""
    if period_label == "All" or obj is None or len(obj) == 0:
        return obj
    months = {"1Y": 12, "3Y": 36, "5Y": 60}.get(period_label, 0)
    if not months:
        return obj
    cutoff = obj.index[-1] - pd.DateOffset(months=months)
    sliced = obj[obj.index >= cutoff]
    return sliced if len(sliced) > 0 else obj


def _sig(tstat):
    """Return significance stars for a t-statistic."""
    if tstat is None or not np.isfinite(tstat):
        return ""
    at = abs(tstat)
    if at >= 1.96: return "**"
    if at >= 1.65: return "*"
    return ""


def _fmt_beta(beta, tstat):
    if beta is None or not np.isfinite(beta):
        return "—"
    s = _sig(tstat)
    return f"{beta:+.3f}{s}"


def _fund_slot(slot_idx, all_cat, color):
    """Render one fund selection row. Returns dict or None."""
    c_label, c_cat, c_fund = st.columns([0.5, 2, 4])
    c_label.markdown(
        f"<div style='padding-top:8px;font-weight:700;color:{color}'>"
        f"Fund {slot_idx}</div>",
        unsafe_allow_html=True,
    )
    cat = c_cat.selectbox(
        "c", ["—"] + CATEGORIES,
        key=f"fa_cat_{slot_idx}", label_visibility="collapsed",
    )
    if cat == "—":
        c_fund.selectbox(
            "f", ["—"], key=f"fa_fund_{slot_idx}",
            disabled=True, label_visibility="collapsed",
        )
        return None
    fund_list = all_cat.get(cat, [])
    fund_opts = [f["name"] for f in fund_list]
    fund_map  = {f["name"]: f["code"] for f in fund_list}
    fund_sel  = c_fund.selectbox(
        "f", ["—"] + fund_opts,
        key=f"fa_fund_{slot_idx}", label_visibility="collapsed",
    )
    if fund_sel == "—":
        return None
    return {"name": fund_sel, "code": fund_map[fund_sel], "category": cat}


def _plot_rolling_grid(funds_data, period_label, window_label):
    """2×3 subplot grid — one panel per factor, one trace per fund."""
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[FACTOR_DISPLAY_NAMES[f] for f in FACTOR_ORDER],
        horizontal_spacing=0.08,
        vertical_spacing=0.18,
    )
    fund_names = list(funds_data.keys())
    for fidx, fname in enumerate(FACTOR_ORDER):
        row = fidx // 3 + 1
        col = fidx % 3 + 1
        for cidx, fund_name in enumerate(fund_names):
            rb = funds_data[fund_name].get("rolling_betas")
            if rb is None or fname not in rb.columns:
                continue
            rb_p = _apply_period(rb, period_label)
            short = fund_name[:28] + "…" if len(fund_name) > 28 else fund_name
            fig.add_trace(
                go.Scatter(
                    x    = rb_p.index,
                    y    = rb_p[fname].values,
                    name = short,
                    line = dict(color=get_color(cidx), width=1.5),
                    showlegend = (fidx == 0),
                    hovertemplate = (
                        f"%{{x|%d %b %Y}}: %{{y:.3f}}"
                        f"<extra>{short}</extra>"
                    ),
                ),
                row=row, col=col,
            )

    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID, **_ZERO)
    fig.update_layout(
        **_DARK_LAYOUT,
        height  = 600,
        title   = dict(
            text = f"Rolling {window_label} Standardised Factor Betas",
            font = dict(size=14, color="#E0E0E0"),
        ),
        legend  = dict(
            orientation="h", y=-0.06,
            font=dict(color="#E0E0E0", size=10),
        ),
    )
    return fig


def _plot_attribution(funds_data):
    """Horizontal stacked bar of return contributions per factor."""
    fund_names  = list(funds_data.keys())
    short_names = [n[:32] + "…" if len(n) > 32 else n for n in fund_names]
    all_keys    = FACTOR_ORDER + ["alpha"]

    fig = go.Figure()
    for fkey in all_keys:
        x_vals = []
        for fund_name in fund_names:
            model = funds_data[fund_name].get("model", {})
            key   = "contrib_alpha_6f" if fkey == "alpha" else f"contrib_{fkey}"
            val   = model.get(key)
            x_vals.append((val or 0) * 100)

        label = FACTOR_DISPLAY_NAMES.get(fkey, "Alpha (6F)")
        color = FACTOR_COLORS.get(fkey, "#FFEB3B")
        fig.add_trace(go.Bar(
            y             = short_names,
            x             = x_vals,
            name          = label,
            orientation   = "h",
            marker_color  = color,
            hovertemplate = f"%{{y}}: %{{x:.2f}}%<extra>{label}</extra>",
        ))

    fig.add_vline(x=0, line_color="rgba(255,255,255,0.4)", line_width=1)
    fig.update_layout(
        **_DARK_LAYOUT,
        barmode = "relative",
        height  = max(320, 130 * len(fund_names)),
        title   = dict(
            text = "Annualised Return Attribution by Factor",
            font = dict(size=14, color="#E0E0E0"),
        ),
        xaxis  = dict(title="Annualised Contribution (%)", ticksuffix="%",
                      **_GRID),
        yaxis  = dict(**_GRID),
        legend = dict(orientation="h", y=-0.15,
                      font=dict(color="#E0E0E0", size=10)),
    )
    return fig


def _regime_table(regime_data, factor_order):
    """Build a styled regime-beta DataFrame for one fund."""
    rows = []
    for fname in factor_order:
        row = {"Factor": FACTOR_DISPLAY_NAMES[fname]}
        for regime in ["Bull", "Sideways", "Bear"]:
            rd = regime_data.get(regime)
            if rd is None:
                row[regime] = "— (< 60 days)"
            else:
                b = rd.get(fname, {})
                if isinstance(b, dict):
                    row[regime] = _fmt_beta(b.get("beta"), b.get("tstat"))
                else:
                    row[regime] = "—"
        rows.append(row)
    return pd.DataFrame(rows).set_index("Factor")


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.divider()

    plan_type = st.radio(
        "Plan Universe", ["Direct", "Regular"],
        index=0 if st.session_state.get("plan_type", "Direct") == "Direct" else 1,
        horizontal=True,
    )
    st.session_state["plan_type"] = plan_type

    st.divider()

    # RF rate with fine increment buttons
    rf_col, rd_col, ru_col = st.columns([3, 1, 1])
    rf_pct = rf_col.slider(
        "Risk-Free Rate (%)", 4.0, 9.0,
        st.session_state.get("rf_rate", 6.5), 0.1,
    )
    if rd_col.button("−", key="fa_rf_down"):
        rf_pct = max(4.0, round(rf_pct - 0.1, 1))
        st.session_state["rf_rate"] = rf_pct
        st.rerun()
    if ru_col.button("+", key="fa_rf_up"):
        rf_pct = min(9.0, round(rf_pct + 0.1, 1))
        st.session_state["rf_rate"] = rf_pct
        st.rerun()
    rf_rate = rf_pct / 100
    st.session_state["rf_rate"] = rf_pct

    st.divider()

    st.markdown("**Rolling Window** *(Tab 2)*")
    window_label = st.radio(
        "Rolling window", ["63 days (Qtr)", "126 days (6M)", "252 days (1Y)"],
        index=2, label_visibility="collapsed",
    )
    window_days = {"63 days (Qtr)": 63, "126 days (6M)": 126, "252 days (1Y)": 252}[window_label]

    st.divider()

    st.markdown("**Display Period**")
    period_label = st.radio(
        "Period", ["1Y", "3Y", "5Y", "All"],
        index=3, horizontal=True, label_visibility="collapsed",
    )

    st.divider()
    render_refresh_button()

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("🔬 Factor Attribution")
st.caption(
    "6-factor model: Market · Size (SMB) · Value (HML) · Momentum (WML) "
    "· Quality (QMJ) · Low Vol (BAB). "
    "Betas are **standardised** — directly comparable across factors and funds. "
    "All 6 TRI series are required."
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# FUND SELECTION
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("📂 Fund Selection")
st.caption("Select up to 3 funds across any category.")

with st.spinner("Loading fund universe…"):
    all_cat = get_all_categorized_schemes(plan_type=plan_type)

SLOT_COLORS = ["#2196F3", "#FF9800", "#4CAF50"]
selected_funds = []
for i in range(1, 4):
    slot = _fund_slot(i, all_cat, SLOT_COLORS[i - 1])
    if slot:
        selected_funds.append(slot)

n_funds = len(selected_funds)
if n_funds == 0:
    st.info("Select at least 1 fund to run attribution.", icon="ℹ️")
elif n_funds == 1:
    st.caption(f"✅ 1 fund selected — single-fund analysis mode.")
else:
    names = [f["name"][:40] for f in selected_funds]
    st.caption(f"✅ {n_funds} funds selected — comparison mode.")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# RUN BUTTON
# ─────────────────────────────────────────────────────────────────────────────
fa_sig = (
    str([f["code"] for f in selected_funds]) +
    str(rf_pct) + str(window_days)
)
if st.session_state.get("_fa_sig") != fa_sig:
    st.session_state.pop("_fa_result", None)

run_btn = st.button(
    "⚡ Run Factor Attribution",
    type="primary", use_container_width=True,
    disabled=(n_funds == 0),
)

# ─────────────────────────────────────────────────────────────────────────────
# COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────
if run_btn and n_funds > 0:
    st.session_state["_fa_sig"] = fa_sig

    # Load 6F factor data
    prog = st.progress(0, text="Loading 6-factor TRI data…")
    factor_df, source_names, err = get_factor_returns_6f(rf_rate=rf_rate)
    if err or factor_df is None:
        st.error(f"❌ Factor data unavailable: {err}")
        st.stop()

    eff_factor_start = factor_df.index[0].strftime("%d %b %Y")
    eff_factor_end   = factor_df.index[-1].strftime("%d %b %Y")

    funds_data = {}
    for fidx, fund in enumerate(selected_funds):
        pct = int(20 + 75 * (fidx / n_funds))
        prog.progress(pct, text=f"Computing: {fund['name'][:50]}…")

        nav_df  = get_nav_history(fund["code"])
        nav     = process_nav(nav_df)
        if nav is None:
            st.warning(f"Could not load NAV for {fund['name']}")
            continue

        returns = compute_daily_returns(nav)

        model         = calc_factor_model_6f(returns, factor_df, rf_rate)
        rolling_betas = calc_rolling_factor_betas(returns, factor_df, rf_rate, window_days)
        regime_betas  = calc_regime_betas(returns, factor_df, rf_rate)

        funds_data[fund["name"]] = {
            "returns":       returns,
            "model":         model,
            "rolling_betas": rolling_betas,
            "regime_betas":  regime_betas,
            "category":      fund["category"],
        }

    prog.progress(100, text="Done.")
    prog.empty()

    st.session_state["_fa_result"] = {
        "funds_data":       funds_data,
        "factor_df":        factor_df,
        "source_names":     source_names,
        "factor_start":     eff_factor_start,
        "factor_end":       eff_factor_end,
    }
    st.success(f"✅ Attribution complete — factor data: {eff_factor_start} → {eff_factor_end}")

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────────────────────
if "_fa_result" not in st.session_state:
    st.stop()

res         = st.session_state["_fa_result"]
funds_data  = res["funds_data"]
factor_df   = res["factor_df"]
source_names= res["source_names"]
fund_names  = list(funds_data.keys())
n_funds     = len(fund_names)

if n_funds == 0:
    st.warning("No valid fund data computed."); st.stop()

st.caption(
    f"Factor history: **{res['factor_start']} → {res['factor_end']}**  ·  "
    f"{len(factor_df):,} trading days"
)

tab1, tab2, tab3, tab4 = st.tabs([
    "📐 Factor Loadings",
    "📈 Rolling Exposures",
    "🧩 Return Attribution",
    "🔄 Regime Analysis",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — FACTOR LOADINGS
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Standardised Factor Betas")
    st.caption(
        "Beta = 1.0 means one full standard-deviation tilt toward that factor. "
        "** = significant at 95% · * = 90% · No mark = not significant. "
        "Model uses the full common date range for OLS."
    )

    if n_funds == 1:
        # ── Single fund: KPI cards ────────────────────────────────────────
        fund_name = fund_names[0]
        model     = funds_data[fund_name]["model"]
        st.markdown(f"#### {fund_name[:70]}")
        st.caption(
            f"Observations: {model.get('n_obs', 0):,}  |  "
            f"{model['effective_start'].strftime('%d %b %Y') if model.get('effective_start') else '—'} → "
            f"{model['effective_end'].strftime('%d %b %Y')   if model.get('effective_end')   else '—'}"
        )

        # Model quality
        q1, q2, q3 = st.columns(3)
        alpha = model.get("alpha_6f")
        at    = model.get("alpha_6f_tstat")
        r2    = model.get("r_squared_6f")

        q1.metric("6F Alpha (Ann.)",  fmt_pct(alpha) if alpha is not None else "N/A",
                  delta=f"|t| = {abs(at):.2f} {'✅' if at and abs(at)>=1.96 else '⚠️'}"
                  if at is not None else None, delta_color="off")
        q2.metric("6F R²",            f"{r2:.4f}" if r2 is not None else "N/A")
        q3.metric("Observations",     f"{model.get('n_obs', 0):,}")

        st.divider()
        st.markdown("**Factor Loadings**")

        c1, c2, c3 = st.columns(3)
        c4, c5, c6 = st.columns(3)
        cols = [c1, c2, c3, c4, c5, c6]

        for i, fname in enumerate(FACTOR_ORDER):
            beta  = model.get(f"beta_{fname}_std")
            tstat = model.get(f"tstat_{fname}")
            sig   = _sig(tstat)
            cols[i].metric(
                FACTOR_DISPLAY_NAMES[fname],
                f"{beta:+.3f}{sig}" if beta is not None else "N/A",
                delta=f"t = {tstat:.2f}" if tstat is not None else None,
                delta_color="off",
            )

    else:
        # ── Multi fund: comparison table ─────────────────────────────────
        st.markdown("**Beta Comparison Table**  (* = 90% · ** = 95% significance)")

        # Factor betas table
        table_rows = []
        for fname in FACTOR_ORDER:
            row = {"Factor": FACTOR_DISPLAY_NAMES[fname]}
            for fund_name in fund_names:
                model = funds_data[fund_name]["model"]
                row[fund_name[:35]] = _fmt_beta(
                    model.get(f"beta_{fname}_std"),
                    model.get(f"tstat_{fname}"),
                )
            table_rows.append(row)

        beta_df = pd.DataFrame(table_rows).set_index("Factor")
        st.dataframe(beta_df, use_container_width=True)

        st.divider()

        # Model quality table
        st.markdown("**Model Quality**")
        quality_rows = []
        for fund_name in fund_names:
            model = funds_data[fund_name]["model"]
            alpha = model.get("alpha_6f")
            at    = model.get("alpha_6f_tstat")
            r2    = model.get("r_squared_6f")
            quality_rows.append({
                "Fund":            fund_name[:50],
                "6F Alpha (Ann.)": fmt_pct(alpha) if alpha is not None else "N/A",
                "Alpha t-stat":    f"{at:.2f}" if at is not None else "N/A",
                "Alpha sig.":      "✅" if at and abs(at) >= 1.96 else ("⚠️" if at and abs(at) >= 1.65 else "✗"),
                "R²":              f"{r2:.4f}" if r2 is not None else "N/A",
                "N obs":           f"{model.get('n_obs', 0):,}",
            })
        st.dataframe(pd.DataFrame(quality_rows).set_index("Fund"), use_container_width=True)

    # Factor data source note
    with st.expander("Factor data sources", expanded=False):
        for fname in FACTOR_ORDER:
            src = source_names.get(fname, "unknown")
            st.caption(f"**{FACTOR_DISPLAY_NAMES[fname]}:** {src}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — ROLLING FACTOR EXPOSURES
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader(f"Rolling {window_label} Factor Exposures")
    st.caption(
        "Standardised betas estimated over a rolling window. "
        "A rising line indicates an increasing tilt toward that factor. "
        "Zero line = neutral exposure. Period selector controls display range."
    )

    any_rolling = any(
        funds_data[f].get("rolling_betas") is not None for f in fund_names
    )
    if not any_rolling:
        st.info(
            f"Rolling betas require at least {window_days + 30} common trading days. "
            "Select a shorter window or a fund with more history.",
            icon="ℹ️",
        )
    else:
        st.plotly_chart(
            _plot_rolling_grid(funds_data, period_label, window_label),
            use_container_width=True,
        )

        # Tip
        st.caption(
            "**Reading this chart:** A fund whose QMJ (Quality) line rises over time "
            "is increasing its tilt toward quality stocks. Divergence between funds on "
            "the same factor reveals genuine style differences."
        )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — RETURN ATTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Return Attribution by Factor")
    st.caption(
        "Annualised return contribution per factor: "
        "`contribution = raw_beta × mean_factor_return × 252`. "
        "Total bar ≈ fund CAGR explained by the 6-factor model. "
        "Negative contributions point left."
    )

    st.plotly_chart(_plot_attribution(funds_data), use_container_width=True)

    st.divider()

    # Numeric attribution table
    st.subheader("Attribution Table")
    attr_rows = []
    col_headers = ["Factor"] + [n[:40] for n in fund_names]
    for fkey in FACTOR_ORDER + ["alpha"]:
        label = FACTOR_DISPLAY_NAMES.get(fkey, "Alpha (6F)")
        row   = {"Factor": label}
        for fund_name in fund_names:
            model = funds_data[fund_name]["model"]
            k     = "contrib_alpha_6f" if fkey == "alpha" else f"contrib_{fkey}"
            val   = model.get(k)
            row[fund_name[:40]] = fmt_pct(val) if val is not None else "N/A"
        attr_rows.append(row)

    # Total row
    total_row = {"Factor": "**Total Explained**"}
    for fund_name in fund_names:
        model = funds_data[fund_name]["model"]
        total = sum(
            (model.get(f"contrib_{f}") or 0) for f in FACTOR_ORDER
        ) + (model.get("contrib_alpha_6f") or 0)
        total_row[fund_name[:40]] = fmt_pct(total)
    attr_rows.append(total_row)

    attr_df = pd.DataFrame(attr_rows).set_index("Factor")
    st.dataframe(attr_df, use_container_width=True)

    st.info(
        "**Interpretation:** The 'Total Explained' row is the model-implied CAGR. "
        "The 6F alpha row shows returns not explained by any of the six systematic factors. "
        "Large positive alpha with high t-stat (Tab 1) indicates genuine manager skill.",
        icon="ℹ️",
    )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — REGIME ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Regime-Conditional Factor Betas")
    st.caption(
        "Regimes are classified from the rolling 252-day Nifty 500 annualised return "
        "(market factor + rf):  "
        "**Bull** > 10%  ·  **Sideways** 0–10%  ·  **Bear** < 0%.  "
        "Betas estimated separately in each regime. "
        "Requires ≥ 60 days per regime for estimation."
    )

    # Regime legend
    rc1, rc2, rc3 = st.columns(3)
    rc1.markdown(
        "<div style='background:rgba(76,175,80,0.15);border-radius:6px;padding:8px;text-align:center'>"
        "🐂 <b>Bull</b><br><small>Nifty 500 rolling CAGR > 10%</small></div>",
        unsafe_allow_html=True,
    )
    rc2.markdown(
        "<div style='background:rgba(255,152,0,0.15);border-radius:6px;padding:8px;text-align:center'>"
        "↔️ <b>Sideways</b><br><small>0% – 10%</small></div>",
        unsafe_allow_html=True,
    )
    rc3.markdown(
        "<div style='background:rgba(244,67,54,0.15);border-radius:6px;padding:8px;text-align:center'>"
        "🐻 <b>Bear</b><br><small>Nifty 500 rolling CAGR < 0%</small></div>",
        unsafe_allow_html=True,
    )
    st.divider()

    for fund_name in fund_names:
        rd = funds_data[fund_name]["regime_betas"]
        st.markdown(f"#### {fund_name[:70]}")

        # Regime observation counts
        counts = rd.get("regime_counts", {})
        dc1, dc2, dc3, dc4 = st.columns(4)
        dc1.metric("Bull days",     f"{counts.get('Bull', 0):,}")
        dc2.metric("Sideways days", f"{counts.get('Sideways', 0):,}")
        dc3.metric("Bear days",     f"{counts.get('Bear', 0):,}")
        dc4.metric("Total days",    f"{sum(counts.values()):,}")

        # Regime beta table
        rt = _regime_table(rd, FACTOR_ORDER)
        st.dataframe(rt, use_container_width=True)

        # Alpha by regime
        alpha_rows = []
        for regime in ["Bull", "Sideways", "Bear"]:
            rdata = rd.get(regime)
            if rdata and "alpha" in rdata:
                alpha_rows.append({
                    "Regime":       regime,
                    "Alpha (Ann.)": fmt_pct(rdata["alpha"]),
                    "R²":           f"{rdata.get('r2', 0):.3f}",
                    "N obs":        f"{rdata.get('n_obs', 0):,}",
                })
            else:
                alpha_rows.append({
                    "Regime": regime,
                    "Alpha (Ann.)": "— (insufficient data)",
                    "R²": "—", "N obs": "—",
                })
        st.dataframe(
            pd.DataFrame(alpha_rows).set_index("Regime"),
            use_container_width=True,
        )

        st.caption(
            "**How to read:** A fund with QMJ (Quality) beta much higher in Bear than Bull "
            "is defensively positioned — it increases quality exposure during downturns. "
            "A fund with WML (Momentum) beta high in Bull but near zero in Bear is a "
            "momentum rider that reduces trend exposure in falling markets."
        )
        if fund_name != fund_names[-1]:
            st.divider()
