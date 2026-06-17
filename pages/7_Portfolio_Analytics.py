"""
pages/7_Portfolio_Analytics.py
================================
Portfolio Analytics

Build a weighted multi-fund portfolio, choose a rebalancing strategy,
and analyse performance vs Nifty 500 across all standard metrics.

Workflow:
  1. Select up to 8 funds across any category (category → fund per slot)
  2. Assign target weights (must sum to exactly 100%)
  3. Choose rebalancing frequency and analysis period
  4. Click Run → portfolio NAV constructed, all metrics computed
  5. Results in 4 tabs: Overview | Risk | Fund Breakdown | Full Comparison

Rebalancing logic:
  - Static:    weights drift with returns, never reset
  - Monthly:   reset to target weights on first trading day of each month
  - Quarterly: reset on first trading day of each quarter
  - Annual:    reset on first trading day of each year

Portfolio NAV starts at 100 on the first common trading date.
Benchmark: Nifty 500 TRI (via existing benchmark_loader infrastructure).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date

from data.fund_loader        import get_all_categorized_schemes, get_nav_history
from data.nav_processor      import process_nav
from data.benchmark_loader   import get_benchmark_nav, get_benchmark_info
from analytics.performance   import calc_all_cagr
from analytics.volatility    import calc_all_volatility
from analytics.risk          import calc_all_risk
from analytics.risk_adjusted import calc_all_risk_adjusted
from analytics.alpha         import calc_all_alpha
from analytics.consistency   import calc_all_consistency
from visualizations._theme          import base_layout, get_color
from visualizations.drawdown_chart  import plot_drawdown
from visualizations.nav_chart       import plot_trailing_returns
from visualizations.rolling_returns import plot_rolling_combined
from utils.constants  import CATEGORIES, APP_TITLE, APP_ICON, TRADING_DAYS_PER_YEAR, MAR
from utils.formatters import fmt_pct, fmt_ratio, fmt_days
from utils.session    import render_refresh_button

st.set_page_config(
    page_title = "Portfolio Analytics — MF Analytics",
    page_icon  = "💼",
    layout     = "wide",
)

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
    rf_pct  = st.slider("Risk-Free Rate (%)", 4.0, 9.0,
                        st.session_state.get("rf_rate", 6.5), 0.1)
    rf_rate = rf_pct / 100
    st.session_state["rf_rate"] = rf_pct

    st.divider()
    render_refresh_button()

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _construct_portfolio(
    nav_dict:       dict,
    weights_frac:   dict,
    rebalance_freq: str,
):
    """
    Build portfolio returns and NAV from individual fund NAVs.

    Args:
        nav_dict:       {fund_name: pd.Series (clean NAV)}
        weights_frac:   {fund_name: float}  — must sum to 1.0
        rebalance_freq: "static" | "monthly" | "quarterly" | "annual"

    Returns:
        (port_nav, port_returns, fund_returns_df)
        or (None, None, None) if insufficient overlapping history.

    Algorithm:
        1. Compute daily returns for each fund.
        2. Align on common dates (inner join → dropna).
        3. For each trading day:
           a. If it's a rebalance date, reset current_weights to target.
           b. Portfolio return = dot(current_weights, fund_returns).
           c. Drift current_weights using today's returns.
        4. Portfolio NAV = cumulative product of (1 + daily return) × 100.
    """
    funds = list(weights_frac.keys())

    ret_dict = {f: nav_dict[f].pct_change().dropna() for f in funds}
    fund_returns_df = pd.DataFrame(ret_dict).dropna()

    if len(fund_returns_df) < 63:
        return None, None, None

    dates          = fund_returns_df.index
    returns_matrix = fund_returns_df[funds].values          # (T, N)
    w_target       = np.array([weights_frac[f] for f in funds])
    current_w      = w_target.copy()
    portfolio_rets = np.zeros(len(dates))

    for t in range(len(dates)):
        # Rebalance check — reset weights BEFORE computing today's return
        if t > 0 and rebalance_freq != "static":
            prev, curr = dates[t - 1], dates[t]
            do_rebalance = (
                (rebalance_freq == "monthly"   and curr.month != prev.month) or
                (rebalance_freq == "quarterly" and (curr.month - 1) // 3 != (prev.month - 1) // 3) or
                (rebalance_freq == "annual"    and curr.year != prev.year)
            )
            if do_rebalance:
                current_w = w_target.copy()

        # Portfolio return for this day
        portfolio_rets[t] = float(np.dot(current_w, returns_matrix[t]))

        # Drift weights using today's return
        new_values = current_w * (1.0 + returns_matrix[t])
        total      = new_values.sum()
        if total > 0:
            current_w = new_values / total

    port_returns = pd.Series(portfolio_rets, index=dates, name="Portfolio")
    port_nav     = (1.0 + port_returns).cumprod() * 100.0

    return port_nav, port_returns, fund_returns_df


def _apply_period(series, period_label, custom_start=None, custom_end=None):
    """Slice a pd.Series to the requested analysis period."""
    if custom_start is not None and custom_end is not None:
        s, e   = pd.Timestamp(custom_start), pd.Timestamp(custom_end)
        sliced = series[(series.index >= s) & (series.index <= e)]
        return sliced if len(sliced) >= 5 else series
    if period_label == "All":
        return series
    months = {"1Y": 12, "3Y": 36, "5Y": 60}.get(period_label, 0)
    if not months:
        return series
    cutoff = series.index[-1] - pd.DateOffset(months=months)
    sliced = series[series.index >= cutoff]
    return sliced if len(sliced) >= 5 else series


def _plot_correlation_heatmap(fund_returns_df, funds):
    """Plotly heatmap of pairwise daily return correlations."""
    corr   = fund_returns_df[funds].corr().round(3)
    labels = [n[:30] + "…" if len(n) > 30 else n for n in funds]

    fig = go.Figure(go.Heatmap(
        z             = corr.values,
        x             = labels,
        y             = labels,
        colorscale    = "RdYlGn",
        zmin=-1, zmax=1,
        text          = [[f"{v:.2f}" for v in row] for row in corr.values],
        texttemplate  = "%{text}",
        colorbar      = dict(title="ρ", tickformat=".2f"),
        hovertemplate = "%{y} × %{x}: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(base_layout(
        title  = "Fund Pairwise Return Correlations (Daily)",
        height = max(380, 80 * len(funds)),
    ))
    fig.update_layout(xaxis=dict(tickangle=-30))
    return fig


def _plot_rolling_volatility(port_returns, bm_returns, bm_name, window=63):
    """Rolling annualised volatility chart for portfolio and benchmark."""
    port_vol = (
        port_returns
        .rolling(window)
        .std()
        .dropna()
        * np.sqrt(TRADING_DAYS_PER_YEAR)
        * 100
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=port_vol.index, y=port_vol.values,
        name="Portfolio",
        line=dict(color=get_color(0), width=2),
        hovertemplate="%{x|%d %b %Y}: %{y:.2f}%<extra>Portfolio</extra>",
    ))
    if bm_returns is not None and len(bm_returns) > window:
        common   = port_returns.index.intersection(bm_returns.index)
        bm_vol   = (
            bm_returns.reindex(common)
            .rolling(window)
            .std()
            .dropna()
            * np.sqrt(TRADING_DAYS_PER_YEAR)
            * 100
        )
        fig.add_trace(go.Scatter(
            x=bm_vol.index, y=bm_vol.values,
            name=bm_name,
            line=dict(color=get_color(1), width=1.5, dash="dot"),
            hovertemplate=f"%{{x|%d %b %Y}}: %{{y:.2f}}%<extra>{bm_name}</extra>",
        ))
    fig.update_layout(base_layout(
        title  = f"Rolling {window}-Day Annualised Volatility",
        height = 380,
    ))
    fig.update_layout(yaxis=dict(ticksuffix="%"))
    return fig


def _contribution_charts(funds, weights_frac, fund_returns_df, port_ret):
    """
    Build return contribution and risk contribution charts.

    Return contribution  = weight_i × fund_CAGR_over_period  (annualised, %)
    Risk contribution    = component method:
                           (w_i × (Σw)_i) / σ_portfolio       (annualised, %)
    where Σ is the annualised covariance matrix.
    """
    # Return contribution
    ret_contribs = []
    for f in funds:
        fnav  = (1 + fund_returns_df[f]).cumprod() * 100
        fcagr = calc_all_cagr(fnav).get("cagr_inception") or 0.0
        ret_contribs.append(weights_frac[f] * fcagr * 100)

    # Risk contribution (component method)
    cov_ann      = fund_returns_df[funds].cov() * TRADING_DAYS_PER_YEAR
    w_arr        = np.array([weights_frac[f] for f in funds])
    port_var     = float(w_arr @ cov_ann.values @ w_arr)
    port_vol_ann = np.sqrt(port_var) if port_var > 0 else 1e-9
    marginal     = cov_ann.values @ w_arr
    risk_contribs = (w_arr * marginal / port_vol_ann * 100).tolist()

    colors      = [get_color(i) for i in range(len(funds))]
    short_names = [n[:32] + "…" if len(n) > 32 else n for n in funds]

    def _hbar(x_vals, title, x_title):
        f = go.Figure(go.Bar(
            y             = short_names,
            x             = x_vals,
            orientation   = "h",
            marker_color  = colors,
            text          = [f"{v:.2f}%" for v in x_vals],
            textposition  = "outside",
            hovertemplate = "%{y}: %{x:.2f}%<extra></extra>",
        ))
        f.update_layout(base_layout(title=title, height=max(300, 55 * len(funds))))
        f.update_layout(
            xaxis=dict(title=x_title, ticksuffix="%"),
            margin=dict(l=20, r=60, t=50, b=30),
        )
        return f

    return (
        _hbar(ret_contribs,  "Return Contribution  (weight × Fund CAGR)", "% Return"),
        _hbar(risk_contribs, "Risk Contribution  (Component Volatility)",  "% Volatility"),
    )


def _row_metrics(nav_s, ret_s, label, rf):
    """Compute one row of the Full Comparison table."""
    def _f(v):
        return "N/A" if (v is None or (isinstance(v, float) and np.isnan(v))) else fmt_pct(v)
    def _r(v):
        return "N/A" if (v is None or (isinstance(v, float) and np.isnan(v))) else fmt_ratio(v)

    if len(nav_s) < 30:
        return {"Name": label, "1Y CAGR": "N/A", "3Y CAGR": "N/A",
                "Incep. CAGR": "N/A", "Ann. Vol": "N/A", "Max DD": "N/A",
                "Sharpe": "N/A", "Sortino": "N/A", "Calmar": "N/A"}

    perf = calc_all_cagr(nav_s)
    risk = calc_all_risk(nav_s)
    vol  = calc_all_volatility(ret_s, mar=MAR)
    radj = calc_all_risk_adjusted(
        returns         = ret_s,
        cagr_for_calmar = perf.get("cagr_3y") or perf.get("cagr_inception"),
        max_drawdown    = risk.get("max_drawdown"),
        rf_rate         = rf,
    )
    return {
        "Name":        label,
        "1Y CAGR":     _f(perf.get("cagr_1y")),
        "3Y CAGR":     _f(perf.get("cagr_3y")),
        "Incep. CAGR": _f(perf.get("cagr_inception")),
        "Ann. Vol":    _f(vol.get("annualized_volatility")),
        "Max DD":      _f(risk.get("max_drawdown")),
        "Sharpe":      _r(radj.get("sharpe")),
        "Sortino":     _r(radj.get("sortino")),
        "Calmar":      _r(radj.get("calmar")),
    }


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("💼 Portfolio Analytics")
st.caption(
    "Build a multi-fund portfolio with custom weights, choose a rebalancing "
    "strategy, and analyse performance vs Nifty 500."
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — PORTFOLIO BUILDER
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("🏗️ Portfolio Builder")

with st.spinner("Loading fund universe…"):
    all_cat = get_all_categorized_schemes(plan_type=plan_type)

# ── Column headers ─────────────────────────────────────────────────────────
h0, h1, h2, h3 = st.columns([0.25, 1.9, 3.85, 1.0])
for col, txt in zip([h0, h1, h2, h3], ["#", "Category", "Fund", "Weight %"]):
    col.markdown(
        f"<div style='color:#78909C;font-size:0.8em;padding-bottom:2px'>{txt}</div>",
        unsafe_allow_html=True,
    )

# ── Fund slots ─────────────────────────────────────────────────────────────
CAT_OPTIONS   = ["—"] + CATEGORIES
selected_slots = []

for i in range(8):
    c0, c1, c2, c3 = st.columns([0.25, 1.9, 3.85, 1.0])
    c0.markdown(
        f"<div style='padding-top:8px;color:#546E7A;font-size:0.85em'>{i+1}</div>",
        unsafe_allow_html=True,
    )
    cat_sel = c1.selectbox(
        f"cat_{i}", CAT_OPTIONS, key=f"pf_cat_{i}",
        label_visibility="collapsed",
    )

    if cat_sel == "—":
        c2.selectbox(f"fund_{i}", ["—"], key=f"pf_fund_{i}",
                     disabled=True, label_visibility="collapsed")
        c3.number_input(f"w_{i}", 0, 100, 0, key=f"pf_w_{i}",
                        disabled=True, label_visibility="collapsed")
    else:
        fund_list  = all_cat.get(cat_sel, [])
        fund_opts  = [f["name"] for f in fund_list]
        fund_map   = {f["name"]: f["code"] for f in fund_list}
        fund_sel   = c2.selectbox(f"fund_{i}", ["—"] + fund_opts, key=f"pf_fund_{i}",
                                   label_visibility="collapsed")
        weight_val = int(c3.number_input(f"w_{i}", 0, 100, 0, key=f"pf_w_{i}",
                                         label_visibility="collapsed"))
        if fund_sel != "—":
            selected_slots.append({
                "name":     fund_sel,
                "code":     fund_map.get(fund_sel, ""),
                "weight":   weight_val,
                "category": cat_sel,
            })

# ── Weight validation ──────────────────────────────────────────────────────
names_sel    = [s["name"] for s in selected_slots]
has_dupes    = len(names_sel) != len(set(names_sel))
total_weight = sum(s["weight"] for s in selected_slots)
n_sel        = len(selected_slots)
weights_ok   = (abs(total_weight - 100.0) < 0.01) and n_sel >= 2 and not has_dupes

wt_col, status_col = st.columns([1, 3], gap="medium")
wt_col.metric("Total Weight", f"{total_weight:.1f}%")

if n_sel == 0:
    status_col.info("Select at least 2 funds to build a portfolio.", icon="ℹ️")
elif has_dupes:
    status_col.error("⚠️ Duplicate fund detected — each fund must appear only once.")
elif not weights_ok:
    rem = 100.0 - total_weight
    status_col.error(
        f"Weights sum to **{total_weight:.1f}%** — must equal **100%** exactly.  "
        f"({'Add' if rem > 0 else 'Remove'} **{abs(rem):.1f}%**)"
    )
else:
    status_col.success(f"✅ {n_sel} funds selected — weights sum to 100%")

st.divider()

# ── Settings ───────────────────────────────────────────────────────────────
set_l, set_r = st.columns(2, gap="large")

with set_l:
    st.markdown("**⚖️ Rebalancing**")
    rebalance_choice = st.radio(
        "Rebalancing frequency",
        ["Static (no rebalancing)", "Monthly", "Quarterly", "Annual"],
        index=2,
        label_visibility="collapsed",
    )
    rebalance_freq  = rebalance_choice.split(" ")[0].lower()   # "static" / "monthly" / etc.
    rebalance_label = rebalance_choice

with set_r:
    st.markdown("**📅 Analysis Period**")
    period_label = st.radio(
        "Period", ["1Y", "3Y", "5Y", "All"],
        index=3, horizontal=True,
        label_visibility="collapsed",
    )
    use_custom = st.toggle("Use custom date range")
    custom_start, custom_end = None, None
    if use_custom:
        d1, d2 = st.columns(2)
        custom_start = d1.date_input("From", value=date(2018, 1, 1), key="pf_cstart")
        custom_end   = d2.date_input("To",   value=date.today(),     key="pf_cend")
        if custom_start >= custom_end:
            st.error("'From' date must be before 'To' date.")
            custom_start, custom_end = None, None

st.divider()

run_btn = st.button(
    "⚡ Run Portfolio Analysis",
    type="primary", use_container_width=True,
    disabled=not weights_ok,
)

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

# Invalidate cached result when portfolio definition changes
pf_sig = str(sorted([(s["name"], s["weight"]) for s in selected_slots])) + rebalance_freq
if st.session_state.get("_pf_sig") != pf_sig:
    st.session_state.pop("_pf_result", None)

if run_btn and weights_ok:
    st.session_state["_pf_sig"] = pf_sig

    # Load NAV histories
    raw_navs = {}
    prog = st.progress(0, text="Loading NAV histories…")
    for i, slot in enumerate(selected_slots):
        prog.progress(
            (i + 1) / len(selected_slots),
            text=f"Loading: {slot['name'][:55]}…",
        )
        nav_df = get_nav_history(slot["code"])
        nav    = process_nav(nav_df)
        if nav is not None:
            raw_navs[slot["name"]] = nav
    prog.empty()

    missing = [s["name"] for s in selected_slots if s["name"] not in raw_navs]
    if missing:
        st.warning(f"NAV unavailable for: {', '.join(missing)}")

    valid_slots = [s for s in selected_slots if s["name"] in raw_navs]
    if len(valid_slots) < 2:
        st.error("Need at least 2 funds with valid NAV data to build a portfolio.")
        st.stop()

    funds_in    = [s["name"] for s in valid_slots]
    weights_pct = {s["name"]: s["weight"] for s in valid_slots}
    w_sum       = sum(weights_pct.values())
    weights_frac= {k: v / w_sum for k, v in weights_pct.items()}

    with st.spinner("Constructing portfolio NAV…"):
        port_nav, port_returns, fund_returns_df = _construct_portfolio(
            raw_navs, weights_frac, rebalance_freq,
        )

    if port_nav is None:
        st.error(
            "Insufficient overlapping NAV history across selected funds. "
            "Try funds with longer or more aligned histories."
        )
        st.stop()

    # Load Nifty 500 via Flexi Cap mapping (Flexi Cap → Nifty 500 TRI)
    with st.spinner("Loading Nifty 500 benchmark…"):
        bm_nav_df  = get_benchmark_nav("Flexi Cap")
        bm_info    = get_benchmark_info("Flexi Cap")
        bm_nav_raw = process_nav(bm_nav_df) if bm_nav_df is not None else None

    bm_nav_aligned  = None
    bm_returns_full = None
    if bm_nav_raw is not None:
        common_idx     = port_returns.index.intersection(bm_nav_raw.index)
        if len(common_idx) > 30:
            bm_nav_aligned  = bm_nav_raw.reindex(common_idx)
            bm_returns_full = bm_nav_aligned.pct_change().dropna()

    st.session_state["_pf_result"] = {
        "port_nav":        port_nav,
        "port_returns":    port_returns,
        "fund_returns_df": fund_returns_df,
        "bm_nav":          bm_nav_aligned,
        "bm_returns":      bm_returns_full,
        "bm_name":         bm_info.get("display_name", "Nifty 500 TRI"),
        "bm_available":    bm_info.get("available", False),
        "funds":           funds_in,
        "weights_frac":    weights_frac,
        "weights_pct":     weights_pct,
        "raw_navs":        raw_navs,
        "slots":           valid_slots,
        "rebalance_label": rebalance_label,
    }

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — RESULTS
# ─────────────────────────────────────────────────────────────────────────────

if "_pf_result" not in st.session_state:
    st.stop()

res = st.session_state["_pf_result"]

port_nav        = res["port_nav"]
port_returns    = res["port_returns"]
fund_returns_df = res["fund_returns_df"]
bm_nav          = res["bm_nav"]
bm_returns      = res["bm_returns"]
bm_name         = res["bm_name"]
funds           = res["funds"]
weights_frac    = res["weights_frac"]
weights_pct     = res["weights_pct"]
raw_navs        = res["raw_navs"]
rebalance_label = res["rebalance_label"]

# ── Apply analysis period to all series ───────────────────────────────────
port_nav_p  = _apply_period(port_nav,     period_label, custom_start, custom_end)
port_ret_p  = _apply_period(port_returns, period_label, custom_start, custom_end)
bm_nav_p    = _apply_period(bm_nav,    period_label, custom_start, custom_end) if bm_nav    is not None else None
bm_ret_p    = _apply_period(bm_returns,period_label, custom_start, custom_end) if bm_returns is not None else None

eff_start = port_nav_p.index[0].strftime("%d %b %Y")
eff_end   = port_nav_p.index[-1].strftime("%d %b %Y")
n_days    = len(port_nav_p)

st.divider()
st.caption(
    f"Analysis period: **{eff_start} → {eff_end}** "
    f"({n_days} trading days · {rebalance_label})"
)

# ── Metrics on period-sliced series ───────────────────────────────────────
rf    = st.session_state.get("rf_rate", 6.5) / 100
perf  = calc_all_cagr(port_nav_p)
risk  = calc_all_risk(port_nav_p)
vol   = calc_all_volatility(port_ret_p, mar=MAR)
radj  = calc_all_risk_adjusted(
    returns         = port_ret_p,
    cagr_for_calmar = perf.get("cagr_3y") or perf.get("cagr_inception"),
    max_drawdown    = risk.get("max_drawdown"),
    rf_rate         = rf,
)

# Alpha vs benchmark (align to same dates within period)
alpha_metrics = None
if bm_ret_p is not None and len(bm_ret_p) > 60:
    common_p      = port_ret_p.index.intersection(bm_ret_p.index)
    port_ret_aln  = port_ret_p.reindex(common_p).dropna()
    bm_ret_aln    = bm_ret_p.reindex(common_p).dropna()
    common_final  = port_ret_aln.index.intersection(bm_ret_aln.index)
    if len(common_final) > 60:
        alpha_metrics = calc_all_alpha(
            port_ret_aln.reindex(common_final),
            bm_ret_aln.reindex(common_final),
            rf,
        )

# ── Shared KPI helper ─────────────────────────────────────────────────────
def _kpi(col, label, val, pct=True, days=False):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        col.metric(label, "N/A")
        return
    if days: col.metric(label, fmt_days(val))
    elif pct: col.metric(label, fmt_pct(val))
    else:     col.metric(label, fmt_ratio(val))

# ─────────────────────────────────────────────────────────────────────────────
# RESULT TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Overview",
    "⚠️ Risk",
    "🔗 Fund Breakdown",
    "📊 Full Comparison",
])

# ── TAB 1: OVERVIEW ──────────────────────────────────────────────────────────
with tab1:
    st.subheader("Portfolio Performance")

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    _kpi(k1, "Incep. CAGR",  perf.get("cagr_inception"))
    _kpi(k2, "3Y CAGR",      perf.get("cagr_3y"))
    _kpi(k3, "1Y CAGR",      perf.get("cagr_1y"))
    _kpi(k4, "Sharpe",       radj.get("sharpe"),  pct=False)
    _kpi(k5, "Sortino",      radj.get("sortino"), pct=False)
    _kpi(k6, "Ann. Vol",     vol.get("annualized_volatility"))

    st.divider()

    # Trailing returns chart vs benchmark
    st.subheader(f"📈 Portfolio vs {bm_name}")
    chart_period = st.radio(
        "Chart period",
        ["1M", "3M", "6M", "1Y", "3Y", "5Y", "All"],
        index=3, horizontal=True, key="pf_chart_period",
    )
    nav_chart = {"Portfolio": port_nav}
    if bm_nav is not None:
        nav_chart[bm_name] = bm_nav
    st.plotly_chart(
        plot_trailing_returns(nav_chart, period_label=chart_period, height=480),
        use_container_width=True,
    )

    # Alpha vs benchmark
    if alpha_metrics:
        st.divider()
        st.subheader(f"Alpha vs {bm_name}")
        a1, a2, a3, a4, a5, a6 = st.columns(6)
        _kpi(a1, "Jensen's Alpha",  alpha_metrics.get("jensens_alpha"))
        _kpi(a2, "Beta",            alpha_metrics.get("beta"),             pct=False)
        _kpi(a3, "Info Ratio",      alpha_metrics.get("information_ratio"),pct=False)
        _kpi(a4, "Tracking Error",  alpha_metrics.get("tracking_error"))
        _kpi(a5, "Up Capture %",    alpha_metrics.get("up_capture"),       pct=False)
        _kpi(a6, "Down Capture %",  alpha_metrics.get("down_capture"),     pct=False)

        t_stat = alpha_metrics.get("alpha_tstat")
        if t_stat is not None:
            if abs(t_stat) >= 2.0:
                st.success(f"✅ Portfolio alpha statistically significant (|t| = {t_stat:.2f} ≥ 2.0)")
            else:
                st.info(f"ℹ️ Portfolio alpha not statistically significant (|t| = {t_stat:.2f} < 2.0)")
    elif bm_nav is None:
        st.warning("Nifty 500 benchmark not available — alpha metrics cannot be computed.", icon="⚠️")

# ── TAB 2: RISK ───────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Risk Analysis")

    r1, r2, r3, r4 = st.columns(4)
    _kpi(r1, "Max Drawdown",   risk.get("max_drawdown"))
    _kpi(r2, "Avg Drawdown",   risk.get("avg_drawdown"))
    _kpi(r3, "DD Duration",    risk.get("drawdown_duration"), days=True)
    _kpi(r4, "Calmar Ratio",   radj.get("calmar"), pct=False)

    r5, r6, r7, r8 = st.columns(4)
    _kpi(r5, "Ann. Volatility",vol.get("annualized_volatility"))
    _kpi(r6, "Downside Vol",   vol.get("downside_volatility"))
    _kpi(r7, "Sortino",        radj.get("sortino"), pct=False)
    r8.metric("Rebalancing",   rebalance_label.split(" ")[0])

    st.divider()

    # Drawdown chart
    dd_series = risk.get("drawdown_series")
    if dd_series is not None:
        dd_p = _apply_period(dd_series, period_label, custom_start, custom_end)
        st.plotly_chart(
            plot_drawdown({"Portfolio": dd_p}),
            use_container_width=True,
        )

    st.divider()

    # Rolling volatility
    st.subheader("Rolling Annualised Volatility (63-Day Window)")
    st.plotly_chart(
        _plot_rolling_volatility(port_ret_p, bm_ret_p, bm_name),
        use_container_width=True,
    )

    st.divider()

    # Rolling 1Y returns
    st.subheader("1-Year Rolling Returns")
    cons = calc_all_consistency(port_nav_p)
    s1y  = cons.get("_series_1y")
    if s1y is not None:
        st.plotly_chart(
            plot_rolling_combined({"Portfolio": s1y}, window_label="1-Year", height=480),
            use_container_width=True,
        )
    else:
        st.info(
            "Rolling 1Y returns require at least 2 years of data in the selected period.",
            icon="ℹ️",
        )

# ── TAB 3: FUND BREAKDOWN ─────────────────────────────────────────────────────
with tab3:
    st.subheader("Fund Breakdown")

    # Weight table
    w_rows = [
        {"Fund": s["name"], "Category": s["category"], "Weight": f"{s['weight']:.1f}%"}
        for s in res["slots"]
    ]
    st.dataframe(pd.DataFrame(w_rows), use_container_width=True, hide_index=True)

    st.divider()

    # Correlation heatmap
    st.subheader("📐 Fund Return Correlations")
    st.caption("Pairwise Pearson correlations of daily returns over the common analysis period.")

    frd_idx = fund_returns_df.index.intersection(port_ret_p.index)
    frd_p   = fund_returns_df.reindex(frd_idx).dropna()

    if len(frd_p) > 30 and len(funds) >= 2:
        st.plotly_chart(
            _plot_correlation_heatmap(frd_p, funds),
            use_container_width=True,
        )
    else:
        st.info("Insufficient overlapping data for correlation matrix in this period.", icon="ℹ️")

    st.divider()

    # Return and risk contribution
    st.subheader("📊 Return & Risk Contribution")
    st.caption(
        "**Return contribution** = weight × fund CAGR over the analysis period.  "
        "**Risk contribution** = each fund's share of portfolio volatility "
        "(component method using the annualised covariance matrix)."
    )

    if len(frd_p) > 30:
        fig_ret, fig_risk = _contribution_charts(funds, weights_frac, frd_p, port_ret_p)
        cr, ck = st.columns(2, gap="large")
        with cr:
            st.plotly_chart(fig_ret,  use_container_width=True)
        with ck:
            st.plotly_chart(fig_risk, use_container_width=True)
    else:
        st.info("Insufficient data for contribution analysis in this period.", icon="ℹ️")

# ── TAB 4: FULL COMPARISON ───────────────────────────────────────────────────
with tab4:
    st.subheader(f"Full Comparison — Portfolio · {bm_name} · Individual Funds")
    st.caption(
        f"Period: **{eff_start} → {eff_end}**  ·  "
        f"Rebalancing: **{rebalance_label}**  ·  "
        "All metrics computed on the same date range for fair comparison."
    )

    ORDERED_COLS = [
        "Weight", "1Y CAGR", "3Y CAGR", "Incep. CAGR",
        "Ann. Vol", "Max DD", "Sharpe", "Sortino", "Calmar",
    ]

    rows = []

    # Portfolio row
    pf_row = _row_metrics(port_nav_p, port_ret_p, "🏦 Portfolio", rf)
    pf_row["Weight"] = "100%"
    rows.append(pf_row)

    # Benchmark row
    if bm_nav_p is not None and bm_ret_p is not None and len(bm_nav_p) > 30:
        bm_row = _row_metrics(bm_nav_p, bm_ret_p, f"📊 {bm_name}", rf)
        bm_row["Weight"] = "—"
        rows.append(bm_row)

    # Individual fund rows (sliced to the same period as portfolio)
    for fname in funds:
        fret = fund_returns_df[fname].reindex(frd_idx).dropna()
        fnav = (1 + fret).cumprod() * 100
        if len(fnav) > 30:
            short     = fname[:48] + "…" if len(fname) > 48 else fname
            fund_row  = _row_metrics(fnav, fret, f"  {short}", rf)
            fund_row["Weight"] = f"{weights_pct[fname]:.1f}%"
            rows.append(fund_row)

    if rows:
        compare_df = pd.DataFrame(rows).set_index("Name")[ORDERED_COLS]
        st.dataframe(compare_df, use_container_width=True)

        csv = compare_df.reset_index().to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Comparison (CSV)",
            data=csv,
            file_name="portfolio_comparison.csv",
            mime="text/csv",
            key="dl_pf_compare",
        )
    else:
        st.warning("No data available for comparison table.")
