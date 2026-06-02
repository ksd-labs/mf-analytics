# 📊 MF Quantitative Analytics Platform

An institutional-grade quantitative analytics dashboard for Indian mutual funds, built with Python and Streamlit.

> **Disclaimer:** This platform provides quantitative analytics only. It does not provide investment advice, recommendations, or ratings. All metrics are computed within a single category — cross-category comparisons are not supported by design.

---

## 🖥️ Screenshots

| Dashboard | Fund Analytics | Rankings |
|---|---|---|
| Category overview with fund counts | All 31 metrics + 6 charts per fund | Sortable rankings with quartile badges |

---

## ✨ Features

- **31 quantitative metrics** per fund across 8 categories:
  - Performance (1Y / 3Y / 5Y / Inception CAGR)
  - Volatility (Annualized, Downside)
  - Risk (Max Drawdown, Avg Drawdown, Duration)
  - Risk-Adjusted (Sharpe, Sortino, Calmar)
  - Consistency (1Y & 3Y Rolling Returns — avg, median, std, best, worst)
  - Distribution (Skewness, Excess Kurtosis)
  - Stability (Win Rate, Positive/Negative Frequency)
  - Persistence (% Positive Rolling Periods, Consecutive Streaks)

- **Quartile system** — every metric ranked Q1–Q4 within its category
- **8 Plotly charts** per fund (NAV history, drawdown, rolling returns, heatmaps, scatter plots)
- **6 Streamlit pages** with a consistent dark theme
- **Data quality reporting** — NAV coverage and missing data warnings
- **CSV export** on every ranking and comparison table
- **Live data** from AMFI via mfapi.in — refreshes daily

---

## 📂 Project Structure

```
mf_analytics/
│
├── app.py                      # Home page + global sidebar
├── requirements.txt
├── debug_connection.py         # Run this first to check connectivity
│
├── .streamlit/
│   └── config.toml             # Dark theme configuration
│
├── pages/                      # Streamlit multi-page app
│   ├── 1_Dashboard.py
│   ├── 2_Category_Explorer.py
│   ├── 3_Fund_Analytics.py
│   ├── 4_Fund_Comparison.py
│   ├── 5_Rankings.py
│   └── 6_Data_Quality.py
│
├── data/                       # Data layer (mftool wrappers + processors)
│   ├── fund_loader.py
│   ├── category_mapper.py
│   └── nav_processor.py
│
├── analytics/                  # Quantitative metrics engine
│   ├── engine.py               # Master orchestrator
│   ├── performance.py
│   ├── volatility.py
│   ├── risk.py
│   ├── risk_adjusted.py
│   ├── consistency.py
│   ├── distribution.py
│   ├── stability.py
│   ├── persistence.py
│   └── quartile.py
│
├── visualizations/             # Plotly chart functions
│   ├── nav_chart.py
│   ├── drawdown_chart.py
│   ├── rolling_returns.py
│   ├── heatmaps.py
│   └── scatter_plots.py
│
└── utils/                      # Shared utilities
    ├── constants.py
    ├── formatters.py
    └── validators.py
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/mf_analytics.git
cd mf_analytics
```

### 2. Create and activate a conda environment

```bash
conda create -n mf_env python=3.11
conda activate mf_env
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Check connectivity (run this before the app)

```bash
python debug_connection.py
```

This verifies that your machine can reach AMFI and mfapi.in. If it fails, see the Troubleshooting section below.

### 5. Launch the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` in your browser.

---

## 📦 Dependencies

| Library | Version | Purpose |
|---|---|---|
| `streamlit` | ≥ 1.35.0 | Frontend UI |
| `mftool` | ≥ 3.3.0 | Mutual fund data |
| `pandas` | ≥ 3.0.0 | Data processing |
| `numpy` | ≥ 1.26.0 | Numerical computations |
| `scipy` | ≥ 1.11.0 | Skewness, kurtosis |
| `plotly` | ≥ 5.18.0 | Interactive charts |

---

## 📊 Supported Fund Categories

| Category | Description |
|---|---|
| Large Cap | Top 100 companies by market cap |
| Mid Cap | 101st–250th companies |
| Small Cap | 251st company onwards |
| Flexi Cap | Flexible allocation across caps |
| Multi Cap | Minimum 25% each in large/mid/small |
| ELSS | Tax-saving equity funds (80C) |
| Value | Value-investing style funds |
| Contra | Contrarian investment strategy |
| Focused | Maximum 30 stocks portfolio |
| Aggressive Hybrid | 65–80% equity + 20–35% debt |
| Balanced Advantage | Dynamic equity/debt allocation |
| Index Funds | Passive index-tracking funds |

---

## 🔧 Troubleshooting

### "No schemes returned" error
Run `python debug_connection.py` — it will identify which URL is blocked.

**Common fixes:**
- Disable VPN
- Switch to personal WiFi / mobile hotspot
- Run `pip install --upgrade certifi requests`
- Add Python to Windows Defender / antivirus exceptions

### Pandas version errors (`applymap`, `infer_datetime_format`)
The project is tested on **pandas 3.x**. If you see these errors, upgrade:
```bash
pip install --upgrade pandas
```

### Slow first load
The first run fetches NAV history for every fund in a category (~2–5 seconds per fund). Results are cached for 24 hours — subsequent loads are instant.

---

## 🏗️ Architecture

```
mftool / mfapi.in
      ↓
data/fund_loader.py     ← all API calls, cached with @st.cache_data
      ↓
data/nav_processor.py   ← NAV cleaning, returns computation
      ↓
analytics/engine.py     ← orchestrates all 31 metric calculations
      ↓
visualizations/*.py     ← Plotly chart builders
      ↓
pages/*.py              ← Streamlit UI pages
```

---

## 📈 Quantitative Methodology

### CAGR
```
CAGR = (End NAV / Start NAV) ^ (1 / actual_years) - 1
```

### Sharpe Ratio
```
Sharpe = mean(daily_return - rf_daily) / std(daily_return - rf_daily) × √252
```

### Sortino Ratio
```
Sortino = annualised_excess_return / (std(returns below MAR) × √252)
```

### Maximum Drawdown
```
MDD = min((NAV_t - max(NAV_0..NAV_t)) / max(NAV_0..NAV_t))
```

### Rolling Returns
Annualised CAGR computed over a rolling window of 252 (1Y) or 756 (3Y) trading days.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Data Sources

- **AMFI India** — [amfiindia.com](https://www.amfiindia.com) — Official NAV data
- **mfapi.in** — Free open API for Indian mutual fund data
- **mftool** — Python library for AMFI data access
