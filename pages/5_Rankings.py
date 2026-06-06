"""
pages/5_Rankings.py
====================
Rankings — Category-wise Ranking Tables

Shows all funds in a category ranked by each quantitative metric.
Every ranking table is sortable and exportable to CSV.

Organised into tabs:
  1. Performance    — CAGR rankings
  2. Risk-Adjusted  — Sharpe, Sortino, Calmar
  3. Risk           — Drawdown, Volatility
  4. Consistency    — Rolling returns
  5. Stability      — Win rate, positive frequency
"""

import streamlit as st
import pandas as pd
import numpy as np

from data.fund_loader       import get_all_categorized_schemes, get_nav_history
from data.benchmark_loader  import get_benchmark_nav, get_benchmark_info
from utils.session import (
    rankings_done_key, category_full_df_key, category_fund_metrics_key,
)
from visualizations.alpha_charts   import plot_capture_scatter
from visualizations.factor_charts   import (
    plot_factor_loadings, plot_factor_contribution, plot_factor_heatmap,
)
from visualizations.momentum_charts import (
    plot_momentum_bars, plot_bull_bear_alpha, plot_momentum_heatmap,
)
from analytics.engine   import compute_category_metrics, compute_category_quartiles
from analytics.quartile import build_metrics_dataframe, get_rankings_for_metric
from utils.constants    import CATEGORIES, APP_TITLE, APP_ICON, METRIC_LABELS
from utils.formatters   import fmt_pct, fmt_ratio, fmt_days, style_quartile

st.set_page_config(
    page_title = "Rankings — MF Analytics",
    page_icon  = "🏆",
    layout     = "wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.divider()

    category = st.selectbox(
        "📂 Category",
        CATEGORIES,
        index = CATEGORIES.index(st.session_state.get("selected_category", "Large Cap")),
    )
    st.session_state["selected_category"] = category

    top_n = st.slider("Top N funds to show", 5, 20, 10, 1)

    st.divider()
    rf_pct = st.slider(
        "Risk-Free Rate (%)", 4.0, 9.0,
        st.session_state.get("rf_rate", 6.5), 0.1,
    )
    rf_rate = rf_pct / 100
    st.session_state["rf_rate"] = rf_pct

    plan_type = st.radio(
        "Plan Universe",
        options    = ["Direct", "Regular"],
        index      = 0 if st.session_state.get("plan_type", "Direct") == "Direct" else 1,
        horizontal = True,
        help       = "Direct: no distributor commission. Regular: distributor-advised.",
    )
    st.session_state["plan_type"] = plan_type

    st.divider()
    if st.button("🔄 Refresh NAV Data", use_container_width=True):
        from utils.session import clear_analytics_cache
        clear_analytics_cache()
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.title(f"🏆 Rankings — {category}")
st.caption(
    f"**{plan_type} plans only.** Funds ranked within **{category}** only. "
    "Rankings are not comparable across categories."
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# LOAD FUND LIST
# ─────────────────────────────────────────────────────────────────────────────

plan_type = st.session_state.get("plan_type", "Direct")

with st.spinner("Loading fund list…"):
    all_cat   = get_all_categorized_schemes(plan_type=plan_type)
    fund_list = all_cat.get(category, [])

if not fund_list:
    st.warning("No funds found for this category.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# ANALYTICS TRIGGER
# ─────────────────────────────────────────────────────────────────────────────

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

        with st.spinner("Computing metrics + alpha + factor model…"):
            from data.benchmark_loader import get_benchmark_nav, get_benchmark_info
            from data.factor_loader    import get_factor_returns
            bm_info    = get_benchmark_info(category)
            bm_nav_df  = get_benchmark_nav(category) if bm_info["available"] else None
            factor_df, _ = get_factor_returns(rf_rate=rf_rate)

            fund_metrics = compute_category_metrics(
                nav_dict,
                rf_rate           = rf_rate,
                benchmark_nav_df  = bm_nav_df,
                benchmark_name    = bm_info["display_name"],
                factor_returns_df = factor_df,
            )
            full_df = compute_category_quartiles(fund_metrics)

        st.session_state[category_full_df_key(category)]        = full_df
        st.session_state[category_fund_metrics_key(category)]   = fund_metrics
        st.session_state[analytics_key]                 = True
        st.success(f"✅ Rankings ready for {len(fund_metrics)} funds!")

    full_df      = st.session_state.get(category_full_df_key(category), pd.DataFrame())
    fund_metrics = st.session_state.get(category_fund_metrics_key(category), {})

    if full_df.empty:
        st.warning("No data available for rankings.")
        st.stop()

    valid_n = sum(1 for m in fund_metrics.values() if m.get("is_valid"))
    st.caption(f"Rankings computed from {valid_n} of {len(fund_metrics)} funds with sufficient data.")

    # ── Helper ────────────────────────────────────────────────────────────────
    def _fmt(val, kind):
        if val is None or (isinstance(val, float) and np.isnan(float(val) if val is not None else float("nan"))):
            return "N/A"
        try:
            v = float(val)
            if kind == "pct":   return fmt_pct(v)
            if kind == "ratio": return fmt_ratio(v)
            if kind == "days":  return fmt_days(v)
            if kind == "int":   return str(int(v))
        except Exception:
            return "N/A"
        return str(val)

    def _ranking_table(metric_key, label, kind, ascending=False):
        """Render a ranking table for one metric."""
        if metric_key not in full_df.columns:
            st.caption(f"_{label} — insufficient data_")
            return

        col = pd.to_numeric(full_df[metric_key], errors="coerce").dropna()
        if col.empty:
            st.caption(f"_{label} — no valid values_")
            return

        sorted_df = full_df.sort_values(metric_key, ascending=ascending).head(top_n)
        q_col = f"{metric_key}_quartile"

        rows = []
        for rank, (fund_name, row) in enumerate(sorted_df.iterrows(), start=1):
            val = row.get(metric_key)
            q   = row.get(q_col, "N/A") if q_col in sorted_df.columns else "N/A"
            rows.append({
                "Rank":     rank,
                "Fund":     fund_name,
                label:      _fmt(val, kind),
                "Quartile": str(q),
            })

        df_out = pd.DataFrame(rows)
        st.dataframe(
            df_out.style.map(style_quartile, subset=["Quartile"]),
            use_container_width=True,
            hide_index=True,
            height=min(450, 42 + 36 * len(df_out)),
        )

        csv = df_out.to_csv(index=False).encode("utf-8")
        st.download_button(
            f"⬇️ Download {label} Ranking (CSV)",
            data=csv,
            file_name=f"{category.replace(' ','_')}_{metric_key}_ranking.csv",
            mime="text/csv",
            key=f"dl_{metric_key}",
        )

    # ── RANKING TABS ──────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📈 Performance",
        "⚖️ Risk-Adjusted",
        "⚠️ Risk",
        "🔁 Consistency",
        "📅 Stability",
        "⚡ Alpha",
        "📊 Momentum",
        "🔁 Persistence",
        "🔬 Factor Model",
    ])

    # ── Tab 1: Performance ────────────────────────────────────────────────────
    with tab1:
        st.subheader("Performance Rankings")
        cols = st.columns(2, gap="large")
        with cols[0]:
            st.markdown("**Top — 1Y CAGR**")
            _ranking_table("cagr_1y", "1Y CAGR", "pct", ascending=False)

            st.markdown("**Top — 5Y CAGR**")
            _ranking_table("cagr_5y", "5Y CAGR", "pct", ascending=False)

        with cols[1]:
            st.markdown("**Top — 3Y CAGR**")
            _ranking_table("cagr_3y", "3Y CAGR", "pct", ascending=False)

            st.markdown("**Top — Since Inception CAGR**")
            _ranking_table("cagr_inception", "Inception CAGR", "pct", ascending=False)

    # ── Tab 2: Risk-Adjusted ──────────────────────────────────────────────────
    with tab2:
        st.subheader("Risk-Adjusted Rankings")
        cols = st.columns(3, gap="large")
        with cols[0]:
            st.markdown("**Top — Sharpe Ratio**")
            _ranking_table("sharpe", "Sharpe", "ratio", ascending=False)
        with cols[1]:
            st.markdown("**Top — Sortino Ratio**")
            _ranking_table("sortino", "Sortino", "ratio", ascending=False)
        with cols[2]:
            st.markdown("**Top — Calmar Ratio**")
            _ranking_table("calmar", "Calmar", "ratio", ascending=False)

    # ── Tab 3: Risk ───────────────────────────────────────────────────────────
    with tab3:
        st.subheader("Risk Rankings")
        st.caption("Lower is better for all metrics in this section.")
        cols = st.columns(2, gap="large")
        with cols[0]:
            st.markdown("**Lowest — Annualized Volatility**")
            _ranking_table("annualized_volatility", "Ann. Volatility", "pct", ascending=True)

            st.markdown("**Lowest — Max Drawdown**")
            _ranking_table("max_drawdown", "Max Drawdown", "pct", ascending=False)
        with cols[1]:
            st.markdown("**Lowest — Downside Volatility**")
            _ranking_table("downside_volatility", "Downside Vol", "pct", ascending=True)

            st.markdown("**Lowest — Avg Drawdown**")
            _ranking_table("avg_drawdown", "Avg Drawdown", "pct", ascending=False)

    # ── Tab 4: Consistency ────────────────────────────────────────────────────
    with tab4:
        st.subheader("Consistency Rankings")
        cols = st.columns(2, gap="large")
        with cols[0]:
            st.markdown("**Top — Avg 1Y Rolling Return**")
            _ranking_table("avg_rolling_1y", "Avg 1Y Rolling", "pct", ascending=False)

            st.markdown("**Best — Worst 1Y Rolling Return**")
            _ranking_table("worst_rolling_1y", "Worst 1Y Rolling", "pct", ascending=False)
        with cols[1]:
            st.markdown("**Top — Avg 3Y Rolling Return**")
            _ranking_table("avg_rolling_3y", "Avg 3Y Rolling", "pct", ascending=False)

            st.markdown("**% of Positive 1Y Rolling Periods**")
            _ranking_table("pct_positive_rolling_1y", "% Positive 1Y", "pct", ascending=False)

    # ── Tab 5: Stability ──────────────────────────────────────────────────────
    with tab5:
        st.subheader("Stability Rankings")
        cols = st.columns(3, gap="large")
        with cols[0]:
            st.markdown("**Top — Monthly Win Rate**")
            _ranking_table("win_rate", "Win Rate", "pct", ascending=False)
        with cols[1]:
            st.markdown("**Top — Positive Day Frequency**")
            _ranking_table("positive_freq", "Positive Freq", "pct", ascending=False)
        with cols[2]:
            st.markdown("**Top — % Positive 3Y Rolling**")
            _ranking_table("pct_positive_rolling_3y", "% Positive 3Y", "pct", ascending=False)

    # ── Tab 6: Alpha ──────────────────────────────────────────────────────────────
    with tab6:
        bm_info = get_benchmark_info(category)
        st.subheader("⚡ Alpha Rankings")
        st.info(
            f"**Benchmark:** {bm_info['display_name']}  |  "
            f"**Proxy:** {bm_info['scheme_name'][:60]}",
            icon="📊",
        )

        if not bm_info["available"]:
            st.warning("No benchmark found for this category — alpha rankings unavailable.")
        elif not st.session_state.get(analytics_key):
            st.info("Run full analytics first to see alpha rankings.")
        else:
            # Check if alpha metrics are present in full_df
            has_alpha = "jensens_alpha" in full_df.columns and full_df["jensens_alpha"].notna().any()

            if not has_alpha:
                st.info(
                    "Alpha metrics not available — the benchmark index fund for this "
                    "category may not have been found. Check internet connection and "
                    "re-run rankings.",
                    icon="ℹ️",
                )
            else:
                st.plotly_chart(
                    plot_capture_scatter(full_df, category),
                    use_container_width=True,
                )
                st.divider()
                cols_alpha = st.columns(2, gap="large")
                with cols_alpha[0]:
                    st.markdown("**Top — Jensen's Alpha**")
                    _ranking_table("jensens_alpha", "Jensen's Alpha", "pct", ascending=False)
                    st.markdown("**Top — Information Ratio**")
                    _ranking_table("information_ratio", "Info Ratio", "ratio", ascending=False)
                with cols_alpha[1]:
                    st.markdown("**Top — Capture Ratio**")
                    _ranking_table("capture_ratio", "Capture Ratio", "ratio", ascending=False)
                    st.markdown("**Lowest — Down-Capture**")
                    _ranking_table("down_capture", "Down-Capture %", "num", ascending=True)

    # ── Tab 7: Momentum ───────────────────────────────────────────────────────
    with tab7:
        st.subheader("📊 Momentum Rankings")
        st.caption("Point-in-time returns over 3, 6, and 12 months. Higher = stronger recent momentum.")

        if not st.session_state.get(analytics_key):
            st.info("Run full analytics first to see momentum rankings.")
        else:
            # Momentum heatmap (category-wide)
            if "momentum_12m" in full_df.columns:
                st.plotly_chart(
                    plot_momentum_heatmap(full_df),
                    use_container_width=True,
                )
                st.divider()

            cols_mom = st.columns(3, gap="large")
            with cols_mom[0]:
                st.markdown("**Top — 12M Momentum**")
                _ranking_table("momentum_12m", "12M Return", "pct", ascending=False)
            with cols_mom[1]:
                st.markdown("**Top — 6M Momentum**")
                _ranking_table("momentum_6m", "6M Return", "pct", ascending=False)
            with cols_mom[2]:
                st.markdown("**Top — Momentum Sharpe**")
                _ranking_table("momentum_sharpe", "Mom. Sharpe", "ratio", ascending=False)

            st.divider()
            cols_mom2 = st.columns(2, gap="large")
            with cols_mom2[0]:
                st.markdown("**Top — 3M Momentum**")
                _ranking_table("momentum_3m", "3M Return", "pct", ascending=False)
            with cols_mom2[1]:
                st.markdown("**Top — Alpha Momentum**")
                _ranking_table("alpha_momentum", "Alpha Mom.", "pct", ascending=False)

    # ── Tab 8: Alpha Persistence ──────────────────────────────────────────────
    with tab8:
        st.subheader("🔁 Alpha Persistence & Regime Rankings")
        st.caption(
            "Persistence = % of 1Y windows with positive alpha. "
            "Bull/Bear alpha shows manager skill across market regimes."
        )

        if not st.session_state.get(analytics_key):
            st.info("Run full analytics first to see persistence rankings.")
        else:
            has_persistence = (
                "alpha_persistence" in full_df.columns and
                full_df["alpha_persistence"].notna().any()
            )

            if not has_persistence:
                st.info(
                    "Persistence metrics require benchmark data. "
                    "They will appear after re-running rankings.",
                    icon="ℹ️",
                )
            else:
                # Bull vs Bear chart
                if "bull_alpha" in full_df.columns:
                    # Build metrics dict for chart from full_df rows
                    chart_data = {
                        idx: {
                            "is_valid": True,
                            "bull_alpha": row.get("bull_alpha"),
                            "bear_alpha": row.get("bear_alpha"),
                        }
                        for idx, row in full_df.iterrows()
                        if pd.notna(row.get("bull_alpha")) or pd.notna(row.get("bear_alpha"))
                    }
                    if chart_data:
                        st.plotly_chart(
                            plot_bull_bear_alpha(chart_data),
                            use_container_width=True,
                        )
                    st.divider()

                cols_p = st.columns(2, gap="large")
                with cols_p[0]:
                    st.markdown("**Top — Alpha Persistence Score**")
                    _ranking_table("alpha_persistence", "Persistence", "pct", ascending=False)
                    st.markdown("**Top — Bear Market Alpha**")
                    _ranking_table("bear_alpha", "Bear Alpha", "pct", ascending=False)
                with cols_p[1]:
                    st.markdown("**Top — Bull Market Alpha**")
                    _ranking_table("bull_alpha", "Bull Alpha", "pct", ascending=False)
                    st.markdown("**Fastest — Drawdown Recovery**")
                    _ranking_table("drawdown_recovery_rate", "Recovery (days)", "days", ascending=True)

    # ── Tab 9: Factor Model ───────────────────────────────────────────────────
    with tab9:
        st.subheader("🔬 Factor Model Rankings")
        st.caption(
            "4-Factor alpha controls for Market, Size (SMB), Value (HML), "
            "and Momentum (WML) tilts. Higher alpha = purer manager skill."
        )

        if not st.session_state.get(analytics_key):
            st.info("Run full analytics first to see factor model rankings.")
        else:
            has_factor = (
                "alpha_4f" in full_df.columns and
                full_df["alpha_4f"].notna().any()
            )

            if not has_factor:
                from data.factor_loader import get_factor_availability, FACTOR_DISPLAY_NAMES
                avail = get_factor_availability()
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
                # Factor heatmap
                st.plotly_chart(
                    plot_factor_heatmap(full_df),
                    use_container_width=True,
                )
                st.divider()

                # Build chart data from full_df
                chart_data = {
                    idx: {"is_valid": True, **{
                        col: row.get(col)
                        for col in ["alpha_4f","beta_market_4f","beta_smb",
                                    "beta_hml","beta_wml","contrib_market",
                                    "contrib_smb","contrib_hml","contrib_wml",
                                    "contrib_alpha"]
                    }}
                    for idx, row in full_df.iterrows()
                    if pd.notna(row.get("alpha_4f"))
                }

                ch1, ch2 = st.columns(2, gap="medium")
                with ch1:
                    st.plotly_chart(
                        plot_factor_loadings(chart_data),
                        use_container_width=True,
                    )
                with ch2:
                    st.plotly_chart(
                        plot_factor_contribution(chart_data),
                        use_container_width=True,
                    )

                st.divider()
                cols_f = st.columns(3, gap="large")
                with cols_f[0]:
                    st.markdown("**Top — 4-Factor Alpha**")
                    _ranking_table("alpha_4f", "4F Alpha", "pct", ascending=False)
                with cols_f[1]:
                    st.markdown("**Top — Pure Alpha Contribution**")
                    _ranking_table("contrib_alpha", "Alpha Contrib", "pct", ascending=False)
                with cols_f[2]:
                    st.markdown("**Highest — 4-Factor R-Squared**")
                    _ranking_table("r_squared_4f", "4F R²", "ratio", ascending=False)

                st.divider()
                cols_f2 = st.columns(3, gap="large")
                with cols_f2[0]:
                    st.markdown("**Size Loading (SMB β)**")
                    _ranking_table("beta_smb", "SMB β", "ratio", ascending=False)
                with cols_f2[1]:
                    st.markdown("**Value Loading (HML β)**")
                    _ranking_table("beta_hml", "HML β", "ratio", ascending=False)
                with cols_f2[2]:
                    st.markdown("**Momentum Loading (WML β)**")
                    _ranking_table("beta_wml", "WML β", "ratio", ascending=False)
