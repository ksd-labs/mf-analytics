"""
pages/5_Rankings.py
====================
Rankings — Category-wise Ranking Tables

Tabs:
  1. 📈 Performance     — CAGR rankings
  2. ⚖️ Risk-Adjusted   — Sharpe, Sortino, Calmar
  3. ⚠️ Risk            — Drawdown, Volatility
  4. 🔁 Consistency     — Rolling returns
  5. 📅 Stability       — Win rate, positive frequency
  6. ⚡ Alpha           — Jensen's Alpha, Capture Ratio, IR
  7. 📊 Absolute Returns — 1M / 3M / 6M point-in-time returns  ← NEW
  8. 📊 Momentum        — 12M momentum, Momentum Sharpe
  9. 🔁 Persistence     — Alpha persistence, bull/bear alpha
 10. 🔬 Factor Model    — 4-Factor alpha, factor loadings
"""

import streamlit as st
import pandas as pd
import numpy as np

from data.fund_loader      import get_all_categorized_schemes, get_nav_history
from analytics.engine      import compute_category_metrics, compute_category_quartiles
from analytics.quartile    import build_metrics_dataframe
from utils.constants       import CATEGORIES, APP_TITLE, APP_ICON, METRIC_LABELS
from utils.formatters      import fmt_pct, fmt_ratio, fmt_days, style_quartile
from utils.session         import (
    rankings_done_key, category_full_df_key, category_fund_metrics_key,
    render_refresh_button,
)
from visualizations.alpha_charts   import plot_capture_scatter
from visualizations.momentum_charts import (
    plot_momentum_bars, plot_bull_bear_alpha, plot_momentum_heatmap,
)
from visualizations.factor_charts   import (
    plot_rolling_alpha_4f,
)

st.set_page_config(page_title="Rankings — MF Analytics", page_icon="🏆", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.divider()

    category = st.selectbox(
        "📂 Category", CATEGORIES,
        index=CATEGORIES.index(st.session_state.get("selected_category", "Large Cap")),
    )
    st.session_state["selected_category"] = category

    top_n = st.slider("Top N funds to show", 5, 20, 10, 1)

    plan_type = st.radio(
        "Plan Universe", ["Direct", "Regular"],
        index=0 if st.session_state.get("plan_type", "Direct") == "Direct" else 1,
        horizontal=True,
    )
    st.session_state["plan_type"] = plan_type

    st.divider()
    rf_pct = st.slider("Risk-Free Rate (%)", 4.0, 9.0,
                       st.session_state.get("rf_rate", 6.5), 0.1)
    rf_rate = rf_pct / 100
    st.session_state["rf_rate"] = rf_pct

    st.divider()
    render_refresh_button()

# ── Header ────────────────────────────────────────────────────────────────────
st.title(f"🏆 Rankings — {category}")
st.caption(
    f"**{plan_type} plans only.** "
    f"Funds ranked within **{category}** only. "
    "Rankings are not comparable across categories."
)
st.divider()

# ── Load Fund List ─────────────────────────────────────────────────────────────
with st.spinner("Loading fund list…"):
    all_cat   = get_all_categorized_schemes(plan_type=plan_type)
    fund_list = all_cat.get(category, [])

if not fund_list:
    st.warning("No funds found for this category."); st.stop()

# ── Analytics Trigger ─────────────────────────────────────────────────────────
st.info(
    f"Rankings require computing metrics for all **{len(fund_list)} funds** in {category}. "
    f"First run takes ~{len(fund_list)*3//60 + 1}–{len(fund_list)*5//60 + 2} minutes. "
    "Results are cached for 1 hour.",
    icon="⏱️",
)

analytics_key = rankings_done_key(category)
run_btn = st.button(
    f"⚡ Compute Rankings for {category}  ({len(fund_list)} funds)",
    type="primary", use_container_width=True,
)

if run_btn or st.session_state.get(analytics_key):

    if not st.session_state.get(analytics_key):
        nav_dict = {}
        progress = st.progress(0, text="Loading NAVs…")
        for i, fund in enumerate(fund_list):
            progress.progress(
                (i + 1) / len(fund_list),
                text=f"Fetching: {fund['name'][:55]} ({i+1}/{len(fund_list)})",
            )
            nav_dict[fund["name"]] = get_nav_history(fund["code"])
        progress.empty()

        with st.spinner("Computing metrics + alpha + factor model for all funds…"):
            from data.benchmark_loader import get_benchmark_nav, get_benchmark_info
            from data.factor_loader    import get_factor_returns
            bm_info   = get_benchmark_info(category)
            bm_nav_df = get_benchmark_nav(category) if bm_info["available"] else None
            factor_df, _ = get_factor_returns(rf_rate=rf_rate)

            fund_metrics = compute_category_metrics(
                nav_dict,
                rf_rate           = rf_rate,
                benchmark_nav_df  = bm_nav_df,
                benchmark_name    = bm_info["display_name"],
                factor_returns_df = factor_df,
            )
            full_df = compute_category_quartiles(fund_metrics)

        st.session_state[category_full_df_key(category)]      = full_df
        st.session_state[category_fund_metrics_key(category)] = fund_metrics
        st.session_state[analytics_key]                        = True
        st.success(f"✅ Rankings ready for {len(fund_metrics)} funds!")

    full_df      = st.session_state.get(category_full_df_key(category),      pd.DataFrame())
    fund_metrics = st.session_state.get(category_fund_metrics_key(category), {})

    if full_df.empty:
        st.warning("No data available for rankings."); st.stop()

    valid_n = sum(1 for m in fund_metrics.values() if m.get("is_valid"))
    st.caption(f"Rankings computed from {valid_n} of {len(fund_metrics)} funds with sufficient data.")

    # ── Helper: render one ranking table ──────────────────────────────────────
    def _fmt(val, kind):
        if val is None:
            return "N/A"
        try:
            v = float(val)
            if np.isnan(v): return "N/A"
            if kind == "pct":   return fmt_pct(v)
            if kind == "ratio": return fmt_ratio(v)
            if kind == "days":  return fmt_days(v)
            if kind == "num":   return f"{v:.2f}%"
        except Exception:
            return "N/A"
        return str(val)

    def _ranking_table(metric_key, label, kind, ascending=False):
        if metric_key not in full_df.columns:
            st.caption(f"_{label} — insufficient data_"); return

        col = pd.to_numeric(full_df[metric_key], errors="coerce").dropna()
        if col.empty:
            st.caption(f"_{label} — no valid values_"); return

        sorted_df  = full_df.sort_values(metric_key, ascending=ascending).head(top_n)
        q_col      = f"{metric_key}_quartile"

        rows = []
        for rank, (fund_name, row) in enumerate(sorted_df.iterrows(), start=1):
            val = row.get(metric_key)
            q   = row.get(q_col, "N/A") if q_col in sorted_df.columns else "N/A"
            rows.append({
                "Rank": rank, "Fund": fund_name,
                label: _fmt(val, kind), "Quartile": str(q),
            })

        df_out = pd.DataFrame(rows)
        st.dataframe(
            df_out.style.map(style_quartile, subset=["Quartile"]),
            use_container_width=True, hide_index=True,
            height=min(450, 42 + 36 * len(df_out)),
        )
        csv = df_out.to_csv(index=False).encode("utf-8")
        st.download_button(
            f"⬇️ Download {label} Ranking (CSV)",
            data=csv,
            file_name=f"{category.replace(' ','_')}_{metric_key}_ranking.csv",
            mime="text/csv", key=f"dl_{metric_key}",
        )

    # ── TABS ──────────────────────────────────────────────────────────────────
    (tab1, tab2, tab3, tab4, tab5,
     tab6, tab_abs, tab7, tab8, tab9) = st.tabs([
        "📈 Performance",
        "⚖️ Risk-Adjusted",
        "⚠️ Risk",
        "🔁 Consistency",
        "📅 Stability",
        "⚡ Alpha",
        "📊 Absolute Returns",
        "📊 Momentum",
        "🔁 Persistence",
        "🔬 Factor Model",
    ])

    # ── Tab 1: Performance ────────────────────────────────────────────────────
    with tab1:
        st.subheader("Performance Rankings")
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("**Top — 1Y CAGR**")
            _ranking_table("cagr_1y", "1Y CAGR", "pct", ascending=False)
            st.markdown("**Top — 5Y CAGR**")
            _ranking_table("cagr_5y", "5Y CAGR", "pct", ascending=False)
        with c2:
            st.markdown("**Top — 3Y CAGR**")
            _ranking_table("cagr_3y", "3Y CAGR", "pct", ascending=False)
            st.markdown("**Top — Since Inception CAGR**")
            _ranking_table("cagr_inception", "Inception CAGR", "pct", ascending=False)

    # ── Tab 2: Risk-Adjusted ──────────────────────────────────────────────────
    with tab2:
        st.subheader("Risk-Adjusted Rankings")
        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            st.markdown("**Top — Sharpe Ratio**")
            _ranking_table("sharpe", "Sharpe", "ratio", ascending=False)
        with c2:
            st.markdown("**Top — Sortino Ratio**")
            _ranking_table("sortino", "Sortino", "ratio", ascending=False)
        with c3:
            st.markdown("**Top — Calmar Ratio**")
            _ranking_table("calmar", "Calmar", "ratio", ascending=False)

    # ── Tab 3: Risk ───────────────────────────────────────────────────────────
    with tab3:
        st.subheader("Risk Rankings")
        st.caption("Lower is better for all metrics in this section.")
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("**Lowest — Annualized Volatility**")
            _ranking_table("annualized_volatility", "Ann. Volatility", "pct", ascending=True)
            st.markdown("**Lowest — Max Drawdown**")
            _ranking_table("max_drawdown", "Max Drawdown", "pct", ascending=False)
        with c2:
            st.markdown("**Lowest — Downside Volatility**")
            _ranking_table("downside_volatility", "Downside Vol", "pct", ascending=True)
            st.markdown("**Lowest — Avg Drawdown**")
            _ranking_table("avg_drawdown", "Avg Drawdown", "pct", ascending=False)

    # ── Tab 4: Consistency ────────────────────────────────────────────────────
    with tab4:
        st.subheader("Consistency Rankings")
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("**Top — Avg 1Y Rolling Return**")
            _ranking_table("avg_rolling_1y", "Avg 1Y Rolling", "pct", ascending=False)
            st.markdown("**Best — Worst 1Y Rolling Return**")
            _ranking_table("worst_rolling_1y", "Worst 1Y Rolling", "pct", ascending=False)
        with c2:
            st.markdown("**Top — Avg 3Y Rolling Return**")
            _ranking_table("avg_rolling_3y", "Avg 3Y Rolling", "pct", ascending=False)
            st.markdown("**% of Positive 1Y Rolling Periods**")
            _ranking_table("pct_positive_rolling_1y", "% Positive 1Y", "pct", ascending=False)

    # ── Tab 5: Stability ──────────────────────────────────────────────────────
    with tab5:
        st.subheader("Stability Rankings")
        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            st.markdown("**Top — Monthly Win Rate**")
            _ranking_table("win_rate", "Win Rate", "pct", ascending=False)
        with c2:
            st.markdown("**Top — Positive Day Frequency**")
            _ranking_table("positive_freq", "Positive Freq", "pct", ascending=False)
        with c3:
            st.markdown("**Top — % Positive 3Y Rolling**")
            _ranking_table("pct_positive_rolling_3y", "% Positive 3Y", "pct", ascending=False)

    # ── Tab 6: Alpha ──────────────────────────────────────────────────────────
    with tab6:
        bm_info_tab = get_benchmark_info(category) if 'get_benchmark_info' in dir() else None
        if bm_info_tab:
            st.info(
                f"**Benchmark:** {bm_info_tab.get('display_name','')}  |  "
                f"**Proxy:** {bm_info_tab.get('scheme_name','')[:60]}",
                icon="📊",
            )

        has_alpha = ("jensens_alpha" in full_df.columns and
                     full_df["jensens_alpha"].notna().any())

        if not has_alpha:
            st.info("Alpha metrics not available — re-run rankings.", icon="ℹ️")
        else:
            st.plotly_chart(plot_capture_scatter(full_df, category),
                            use_container_width=True)
            st.divider()
            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.markdown("**Top — Jensen's Alpha**")
                _ranking_table("jensens_alpha", "Jensen's Alpha", "pct", ascending=False)
                st.markdown("**Top — Information Ratio**")
                _ranking_table("information_ratio", "Info Ratio", "ratio", ascending=False)
            with c2:
                st.markdown("**Top — Capture Ratio**")
                _ranking_table("capture_ratio", "Capture Ratio", "ratio", ascending=False)
                st.markdown("**Lowest — Down-Capture**")
                _ranking_table("down_capture", "Down-Capture %", "num", ascending=True)

    # ── Tab 7 (NEW): Absolute Returns 1M / 3M / 6M ───────────────────────────
    with tab_abs:
        st.subheader("📊 Absolute Returns Rankings")
        st.caption(
            "Point-in-time returns over the last 1, 3, and 6 months. "
            "Sorted highest to lowest. These are trailing returns, not annualised."
        )

        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            st.markdown("**Top — 1 Month Return**")
            _ranking_table("momentum_1m", "1M Return", "pct", ascending=False)
        with c2:
            st.markdown("**Top — 3 Month Return**")
            _ranking_table("momentum_3m", "3M Return", "pct", ascending=False)
        with c3:
            st.markdown("**Top — 6 Month Return**")
            _ranking_table("momentum_6m", "6M Return", "pct", ascending=False)

        st.divider()
        st.caption("Bottom performers (funds with worst recent returns):")
        c4, c5, c6 = st.columns(3, gap="large")
        with c4:
            st.markdown("**Worst — 1 Month Return**")
            _ranking_table("momentum_1m", "1M Return", "pct", ascending=True)
        with c5:
            st.markdown("**Worst — 3 Month Return**")
            _ranking_table("momentum_3m", "3M Return", "pct", ascending=True)
        with c6:
            st.markdown("**Worst — 6 Month Return**")
            _ranking_table("momentum_6m", "6M Return", "pct", ascending=True)

    # ── Tab 8: Momentum ───────────────────────────────────────────────────────
    with tab7:
        st.subheader("📊 Momentum Rankings")
        st.caption("Point-in-time returns over 3, 6, and 12 months. Higher = stronger recent momentum.")

        has_mom = ("momentum_12m" in full_df.columns and
                   full_df["momentum_12m"].notna().any())

        if has_mom:
            st.plotly_chart(plot_momentum_heatmap(full_df), use_container_width=True)
            st.divider()

        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            st.markdown("**Top — 12M Return**")
            _ranking_table("momentum_12m", "12M Return", "pct", ascending=False)
        with c2:
            st.markdown("**Top — 6M Return**")
            _ranking_table("momentum_6m", "6M Return", "pct", ascending=False)
        with c3:
            st.markdown("**Top — Momentum Sharpe**")
            _ranking_table("momentum_sharpe", "Mom. Sharpe", "ratio", ascending=False)

        st.divider()
        c4, c5 = st.columns(2, gap="large")
        with c4:
            st.markdown("**Top — 3M Return**")
            _ranking_table("momentum_3m", "3M Return", "pct", ascending=False)
        with c5:
            st.markdown("**Top — Alpha Momentum**")
            _ranking_table("alpha_momentum", "Alpha Mom.", "pct", ascending=False)

    # ── Tab 9: Persistence ────────────────────────────────────────────────────
    with tab8:
        st.subheader("🔁 Alpha Persistence & Regime Rankings")
        st.caption(
            "Persistence = % of 1Y windows with positive alpha. "
            "Bull/Bear alpha shows manager skill across market regimes."
        )

        has_pers = ("alpha_persistence" in full_df.columns and
                    full_df["alpha_persistence"].notna().any())

        if not has_pers:
            st.info("Persistence metrics require benchmark data. Re-run rankings.", icon="ℹ️")
        else:
            if "bull_alpha" in full_df.columns:
                chart_data = {
                    idx: {"is_valid": True,
                          "bull_alpha": row.get("bull_alpha"),
                          "bear_alpha": row.get("bear_alpha")}
                    for idx, row in full_df.iterrows()
                    if pd.notna(row.get("bull_alpha")) or pd.notna(row.get("bear_alpha"))
                }
                if chart_data:
                    st.plotly_chart(plot_bull_bear_alpha(chart_data),
                                    use_container_width=True)
                st.divider()

            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.markdown("**Top — Alpha Persistence Score**")
                _ranking_table("alpha_persistence", "Persistence", "pct", ascending=False)
                st.markdown("**Top — Bear Market Alpha**")
                _ranking_table("bear_alpha", "Bear Alpha", "pct", ascending=False)
            with c2:
                st.markdown("**Top — Bull Market Alpha**")
                _ranking_table("bull_alpha", "Bull Alpha", "pct", ascending=False)
                st.markdown("**Fastest — Drawdown Recovery**")
                _ranking_table("drawdown_recovery_rate", "Recovery (days)", "days", ascending=True)

    # ── Tab 10: Factor Model ──────────────────────────────────────────────────
    with tab9:
        st.subheader("🔬 Factor Model Rankings")
        st.caption(
            "4-Factor alpha controls for Market, Size (SMB), Value (HML), "
            "and Momentum (WML) tilts. Higher alpha = purer manager skill."
        )

        has_factor = ("alpha_4f" in full_df.columns and
                      full_df["alpha_4f"].notna().any())

        if not has_factor:
            from data.factor_loader import get_factor_availability, FACTOR_DISPLAY_NAMES
            avail  = get_factor_availability()
            n_avail = sum(avail.values())
            if n_avail == 0:
                st.warning("No factor proxy index funds found. Check connectivity.")
            else:
                st.info(
                    f"Factor proxies available ({n_avail}/4 factors). "
                    "Re-run rankings to include factor model.",
                    icon="ℹ️",
                )
        else:
            st.divider()
            c1, c2, c3 = st.columns(3, gap="large")
            with c1:
                st.markdown("**Top — 4-Factor Alpha**")
                _ranking_table("alpha_4f", "4F Alpha", "pct", ascending=False)
            with c2:
                st.markdown("**Top — Pure Alpha Contribution**")
                _ranking_table("contrib_alpha", "Alpha Contrib", "pct", ascending=False)
            with c3:
                st.markdown("**Highest — 4-Factor R-Squared**")
                _ranking_table("r_squared_4f", "4F R²", "ratio", ascending=False)

            st.divider()
            c4, c5, c6 = st.columns(3, gap="large")
            with c4:
                st.markdown("**Size Loading (SMB β)**")
                _ranking_table("beta_smb", "SMB β", "ratio", ascending=False)
            with c5:
                st.markdown("**Value Loading (HML β)**")
                _ranking_table("beta_hml", "HML β", "ratio", ascending=False)
            with c6:
                st.markdown("**Momentum Loading (WML β)**")
                _ranking_table("beta_wml", "WML β", "ratio", ascending=False)
