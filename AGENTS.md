# AGENTS.md — AlgoTrade-X Platform

> This is the single source of truth for the entire project.
> Read this file completely before taking any action.
> You are operating autonomously. Do not ask for confirmation on decisions
> that are answerable from this document. State assumptions inline and proceed.
> Test your own work. If a test fails, fix it before moving on.
> Build in the exact order specified in the Build Execution Plan.

---

## Project Identity

**Name:** AlgoTrade-X
**Type:** Production-grade Algorithmic Trading Platform
**Stack:** Python backend + Streamlit dashboard
**Data:** 100% free APIs, zero paid services, self-downloading
**Deployment:** Docker Compose, single command launch
**Target:** Runs fully autonomously from cold start to live paper trading

---

## Design Language

The UI is inspired by professional financial terminal aesthetics —
specifically the visual grammar of institutional trading interfaces.
This is an original implementation, not a copy of any existing product.

### Color System

```css
/* Core palette — use these exact hex values everywhere */
--bg-primary:      #0A0A0A;   /* near-black base */
--bg-secondary:    #111318;   /* card/panel background */
--bg-tertiary:     #1A1D26;   /* elevated surface */
--bg-border:       #252830;   /* subtle borders */

--amber-primary:   #FFB300;   /* primary accent — data labels, highlights */
--amber-dim:       #CC8F00;   /* secondary amber — subtext */
--amber-glow:      rgba(255,179,0,0.08); /* ambient glow on hover */

--green-up:        #00E676;   /* price increase, positive P&L */
--green-dim:       #00A152;   /* muted positive */
--red-down:        #FF1744;   /* price decrease, loss */
--red-dim:         #C62828;   /* muted negative */

--blue-signal:     #2979FF;   /* model signals, info */
--cyan-data:       #00E5FF;   /* live data stream indicators */
--purple-ml:       #E040FB;   /* ML/AI specific elements */

--text-primary:    #E8E9EB;   /* main text */
--text-secondary:  #8B9099;   /* labels, captions */
--text-muted:      #4A4F5A;   /* disabled, placeholder */

--font-mono:       'JetBrains Mono', 'Fira Code', monospace; /* all numbers */
--font-ui:         'Inter', 'DM Sans', sans-serif;            /* all labels */
```

### Design Rules (never break these)

- ALL numbers, prices, percentages → monospace font, right-aligned
- Positive values → `--green-up`, negative → `--red-down`, neutral → `--amber-primary`
- No rounded corners > 4px on data tables
- No gradients on data — gradients only on decorative elements
- No animations on numbers — they update in place, no bounce/flash
- Borders: 1px solid `--bg-border` only. No drop shadows on data panels.
- Amber is for labels and structure. Green/red are for semantics only.
- Charts: dark background `--bg-secondary`, grid lines `--bg-tertiary`
- No icons in data cells. Icons only in navigation and status indicators.
- Loading states: amber pulsing dots, never spinners

---

## Free Data Sources (all zero cost, all self-configuring)

The system auto-downloads all data on first run. No manual API key setup required for the free tiers. The system handles rate limits, retries, and fallbacks automatically.

### Primary Sources

```python
DATA_SOURCES = {
    "yfinance": {
        "cost": "free",
        "auth": "none",
        "assets": "equities, ETFs, indices, crypto, forex",
        "history": "max available (decades for major stocks)",
        "rate_limit": "none enforced",
        "use_for": "primary OHLCV, fundamentals, dividends"
    },
    "alpha_vantage": {
        "cost": "free tier: 25 requests/day",
        "auth": "free API key — auto-register via DEMO key for testing",
        "key_env": "ALPHA_VANTAGE_KEY",
        "use_for": "technical indicators, earnings, economic data",
        "fallback": "compute indicators locally if quota exhausted"
    },
    "finnhub": {
        "cost": "free tier: 60 calls/min",
        "auth": "free API key required",
        "key_env": "FINNHUB_KEY",
        "use_for": "news sentiment, insider trading, earnings calendar",
        "note": "register at finnhub.io/register — takes 30 seconds"
    },
    "fred": {
        "cost": "completely free, no key needed for basic access",
        "auth": "optional FRED_API_KEY for higher limits",
        "key_env": "FRED_API_KEY",
        "library": "fredapi or pandas_datareader",
        "use_for": "VIX, GDP, CPI, Fed Funds Rate, 10Y yield, DXY",
        "series": ["VIXCLS","DFF","GS10","DEXUSEU","CPIAUCSL","GDP","UNRATE"]
    },
    "binance_public": {
        "cost": "free, no auth needed for public endpoints",
        "auth": "none for market data",
        "use_for": "crypto OHLCV (BTC, ETH, SOL, BNB)",
        "library": "ccxt (binance exchange, public endpoints)"
    },
    "coingecko": {
        "cost": "free tier: 30 calls/min",
        "auth": "none",
        "use_for": "crypto market cap, dominance, fear & greed index"
    },
    "sec_edgar": {
        "cost": "completely free, government data",
        "auth": "User-Agent header required (email address)",
        "use_for": "10-K, 10-Q filings, insider trades, institutional holdings",
        "library": "requests + custom parser"
    },
    "newsapi_org": {
        "cost": "free tier: 100 requests/day",
        "auth": "free API key",
        "key_env": "NEWS_API_KEY",
        "use_for": "financial news headlines for FinBERT sentiment"
    },
    "alpaca_paper": {
        "cost": "completely free paper trading",
        "auth": "free account at alpaca.markets",
        "key_env": "ALPACA_API_KEY, ALPACA_SECRET_KEY",
        "base_url": "https://paper-api.alpaca.markets",
        "use_for": "live paper trade execution, portfolio state"
    }
}
```

### Auto-Setup Protocol

On first run, `bootstrap.py` does ALL of this automatically:

1. Check which API keys exist in `.env`
2. For sources requiring no key (yfinance, FRED basic, Binance public, CoinGecko, SEC EDGAR) → proceed immediately
3. For sources requiring free keys (Finnhub, NewsAPI, Alpha Vantage, Alpaca) → print exact signup URL, pause, wait for user to paste key, save to `.env`
4. Download all historical data for all configured assets
5. Run data quality checks
6. Print readiness report

### Asset Universe

```python
EQUITY_UNIVERSE = {
    # US Large Cap
    "SPY":  "S&P 500 ETF — benchmark",
    "QQQ":  "Nasdaq 100 ETF",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
    "GOOGL":"Alphabet",
    "AMZN": "Amazon",
    "META": "Meta",
    "JPM":  "JPMorgan Chase",
    "GS":   "Goldman Sachs",
    "XOM":  "ExxonMobil",
    "BRK-B":"Berkshire Hathaway",
    # Sector ETFs for cross-asset features
    "XLF":  "Financials",
    "XLK":  "Technology",
    "XLE":  "Energy",
    "XLV":  "Healthcare",
    "XLI":  "Industrials",
    # Volatility / Macro
    "^VIX": "CBOE Volatility Index",
    "^TNX": "10-Year Treasury Yield",
    "GLD":  "Gold ETF",
    "TLT":  "20-Year Treasury ETF",
}

CRYPTO_UNIVERSE = {
    "BTC/USDT": "Bitcoin",
    "ETH/USDT": "Ethereum",
    "SOL/USDT": "Solana",
    "BNB/USDT": "Binance Coin",
}

MACRO_FRED_SERIES = {
    "VIXCLS":   "VIX Close",
    "DFF":      "Fed Funds Rate",
    "GS10":     "10Y Treasury",
    "CPIAUCSL": "CPI",
    "UNRATE":   "Unemployment Rate",
    "GDP":      "GDP",
    "T10YIE":   "10Y Breakeven Inflation",
    "DEXUSEU":  "USD/EUR Exchange Rate",
}
```

---

## Complete File Structure

```
algotrade-x/
│
├── AGENTS.md                          ← this file, master reference
├── README.md                          ← auto-generated after build
├── .env.example                       ← template with all required keys
├── .env                               ← actual keys (gitignored)
├── .claudeignore                      ← fable ignore rules
├── .gitignore
├── docker-compose.yml                 ← single command launch
├── Dockerfile                         ← multi-stage build
├── requirements.txt                   ← pinned versions
├── requirements-dev.txt               ← test/lint dependencies
├── pyproject.toml                     ← project metadata, tool config
│
├── bootstrap.py                       ← ENTRY POINT: run this first
│                                         auto-downloads all data,
│                                         checks keys, validates setup
│
├── src/
│   ├── __init__.py
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py                ← pydantic BaseSettings, all config
│   │   ├── assets.py                  ← asset universe definitions
│   │   ├── constants.py               ← magic numbers, thresholds
│   │   └── logging_config.py          ← structlog setup, JSON logs
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── bootstrap/
│   │   │   ├── downloader.py          ← orchestrates all downloads
│   │   │   ├── equity_loader.py       ← yfinance async batch download
│   │   │   ├── crypto_loader.py       ← ccxt Binance OHLCV
│   │   │   ├── macro_loader.py        ← FRED series download
│   │   │   ├── news_loader.py         ← NewsAPI headline download
│   │   │   └── sec_loader.py          ← SEC EDGAR filings
│   │   ├── pipeline/
│   │   │   ├── cleaner.py             ← outlier removal, gap fill,
│   │   │   │                             corporate action adj,
│   │   │   │                             timezone normalization
│   │   │   ├── validator.py           ← data quality checks,
│   │   │   │                             OHLC consistency,
│   │   │   │                             volume sanity, gap detection
│   │   │   ├── aligner.py             ← align all assets to common
│   │   │   │                             trading calendar, handle halts
│   │   │   └── storage.py             ← parquet (historical) +
│   │   │                                 SQLite (metadata + live state)
│   │   ├── features/
│   │   │   ├── __init__.py
│   │   │   ├── momentum.py            ← RSI, MACD, ROC, Stoch, Williams
│   │   │   ├── volatility.py          ← ATR, Bollinger, HV, Parkinson
│   │   │   ├── volume.py              ← OBV, VWAP, CMF, ADL, vol ratio
│   │   │   ├── statistical.py         ← z-score, Hurst, autocorr,
│   │   │   │                             skew, kurt, fractal dim
│   │   │   ├── cross_asset.py         ← correlations, beta, rel strength
│   │   │   ├── microstructure.py      ← bid-ask proxy, Amihud illiquidity,
│   │   │   │                             Kyle's lambda, order flow imbalance
│   │   │   ├── sentiment.py           ← FinBERT on headlines,
│   │   │   │                             rolling sentiment score,
│   │   │   │                             sentiment momentum/divergence
│   │   │   ├── fundamental.py         ← P/E, P/B, EV/EBITDA from yfinance,
│   │   │   │                             earnings surprise, revision trend
│   │   │   ├── regime.py              ← HMM regime labels as features,
│   │   │   │                             VIX regime (low/mid/high/spike)
│   │   │   └── engineer.py            ← master orchestrator,
│   │   │                                 calls all modules, returns
│   │   │                                 final feature dataframe
│   │   └── live/
│   │       ├── stream.py              ← Alpaca WebSocket price stream
│   │       ├── bar_builder.py         ← build OHLCV bars from ticks
│   │       └── feature_updater.py     ← rolling feature update on new bar
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                    ← Abstract BaseModel interface
│   │   │                                 predict(), predict_proba(),
│   │   │                                 fit(), save(), load(),
│   │   │                                 get_feature_importance()
│   │   ├── classical/
│   │   │   ├── xgb_model.py           ← XGBoost + Optuna tuning
│   │   │   ├── lgbm_model.py          ← LightGBM + monotonic constraints
│   │   │   └── random_forest.py       ← RF as diversity member
│   │   ├── deep/
│   │   │   ├── lstm_model.py          ← 3-layer LSTM + attention
│   │   │   ├── tcn_model.py           ← Temporal Conv Net, dilated causal
│   │   │   ├── transformer_model.py   ← PatchTST architecture
│   │   │   └── nbeats_model.py        ← N-BEATS for pure time-series
│   │   ├── rl/
│   │   │   ├── environment.py         ← gymnasium TradingEnv
│   │   │   │                             state, action, reward,
│   │   │   │                             curriculum schedule
│   │   │   ├── ppo_agent.py           ← SB3 PPO with LSTM policy
│   │   │   ├── sac_agent.py           ← SAC for continuous action space
│   │   │   └── callbacks.py           ← custom SB3 callbacks,
│   │   │                                 early stopping, checkpointing
│   │   ├── ensemble/
│   │   │   ├── stacker.py             ← Ridge meta-learner, OOF stacking
│   │   │   ├── regime_router.py       ← HMM regime → model weights
│   │   │   └── ensemble.py            ← master ensemble, final signal
│   │   └── registry.py                ← ModelRegistry, versioning,
│   │                                     load any model by name+version
│   │
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── engine.py                  ← core event-driven backtester
│   │   ├── portfolio.py               ← position tracking, cash, P&L
│   │   ├── order_book.py              ← simulated order book,
│   │   │                                 limit order queue, fills
│   │   ├── costs.py                   ← commission + slippage +
│   │   │                                 market impact model
│   │   ├── metrics.py                 ← all 25 performance metrics
│   │   ├── walk_forward.py            ← WFO orchestrator
│   │   ├── monte_carlo.py             ← MC simulation on equity curve,
│   │   │                                 bootstrap confidence intervals
│   │   ├── stress_test.py             ← replay 2008, 2020, 2022 crashes,
│   │   │                                 custom scenario injection
│   │   └── report.py                  ← generate HTML + JSON report
│   │
│   ├── risk/
│   │   ├── __init__.py
│   │   ├── position_sizer.py          ← Kelly, fixed fractional,
│   │   │                                 vol targeting, max concentration
│   │   ├── circuit_breaker.py         ← all halt conditions,
│   │   │                                 stateful, thread-safe
│   │   ├── correlation_filter.py      ← dynamic correlation matrix,
│   │   │                                 position pruning on crisis
│   │   ├── regime_detector.py         ← HMM 3-state (bull/bear/sideways)
│   │   │                                 + VIX regime overlay
│   │   ├── var_calculator.py          ← historical VaR, CVaR,
│   │   │                                 parametric VaR, Monte Carlo VaR
│   │   └── risk_manager.py            ← master risk gate:
│   │                                     input: proposed Trade
│   │                                     output: approved/rejected/resized
│   │
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── broker/
│   │   │   ├── base_broker.py         ← abstract broker interface
│   │   │   ├── alpaca_broker.py       ← Alpaca paper trading
│   │   │   └── sim_broker.py          ← simulation broker for backtest
│   │   ├── order_manager.py           ← order lifecycle, fill tracking
│   │   ├── paper_trader.py            ← main live trading loop
│   │   └── rebalancer.py              ← periodic portfolio rebalancing
│   │
│   ├── explainability/
│   │   ├── __init__.py
│   │   ├── shap_analyzer.py           ← SHAP waterfall, beeswarm, force
│   │   ├── lime_explainer.py          ← LIME local explanations
│   │   ├── drift_monitor.py           ← PSI feature drift detection
│   │   ├── attribution.py             ← P&L attribution by signal/asset
│   │   └── audit_log.py               ← every trade decision logged
│   │                                     with full feature snapshot
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                    ← FastAPI app, all routes
│   │   ├── routes/
│   │   │   ├── portfolio.py           ← GET /portfolio, /positions
│   │   │   ├── signals.py             ← GET /signals, /predictions
│   │   │   ├── backtest.py            ← POST /backtest/run, GET /results
│   │   │   ├── models.py              ← GET /models, /metrics
│   │   │   └── health.py              ← GET /health, /status
│   │   ├── schemas.py                 ← Pydantic models for all responses
│   │   └── websocket.py               ← WS /live — push portfolio updates
│   │
│   └── dashboard/
│       ├── __init__.py
│       ├── app.py                     ← Streamlit entry point
│       ├── components/
│       │   ├── theme.py               ← inject CSS, color vars, fonts
│       │   ├── charts.py              ← all Plotly chart builders
│       │   ├── tables.py              ← styled dataframe renderers
│       │   ├── metrics_bar.py         ← KPI strip component
│       │   └── signal_card.py         ← per-asset signal display card
│       └── pages/
│           ├── 01_live_overview.py
│           ├── 02_backtest_results.py
│           ├── 03_model_analysis.py
│           ├── 04_risk_monitor.py
│           ├── 05_trade_log.py
│           ├── 06_explainability.py
│           └── 07_documentation.py
│
├── tests/
│   ├── conftest.py                    ← shared fixtures, test data
│   ├── unit/
│   │   ├── test_no_lookahead.py       ← CRITICAL: data leakage test
│   │   ├── test_cost_model.py         ← costs always > 0
│   │   ├── test_risk_rules.py         ← circuit breaker tests
│   │   ├── test_backtest_math.py      ← metric formula verification
│   │   ├── test_feature_engineer.py   ← no NaN, no inf, correct shape
│   │   ├── test_walk_forward.py       ← no train/test overlap
│   │   └── test_position_sizer.py     ← Kelly, vol target correctness
│   ├── integration/
│   │   ├── test_full_backtest.py      ← end-to-end backtest run
│   │   ├── test_model_interface.py    ← all models implement BaseModel
│   │   └── test_api_endpoints.py      ← FastAPI route tests
│   └── performance/
│       └── test_backtest_speed.py     ← 10yr backtest < 60 seconds
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   ├── 03_model_comparison.ipynb
│   ├── 04_backtest_deep_dive.ipynb
│   └── 05_risk_analysis.ipynb
│
├── docs/
│   ├── architecture.md                ← system architecture + diagrams
│   ├── data_sources.md                ← all APIs, rate limits, fallbacks
│   ├── models.md                      ← each model documented
│   ├── backtest_methodology.md        ← WFO protocol, cost model
│   ├── risk_management.md             ← all rules documented
│   ├── api_reference.md               ← FastAPI endpoint docs
│   ├── dashboard_guide.md             ← each page documented
│   ├── deployment.md                  ← Docker, env setup
│   └── CHANGELOG.md                   ← version history
│
├── data/
│   ├── raw/                           ← downloaded parquet files
│   ├── processed/                     ← feature-engineered parquet
│   ├── models/                        ← saved model artifacts
│   ├── backtest_results/              ← HTML reports, JSON metrics
│   └── logs/                          ← structured JSON logs
│
└── scripts/
    ├── download_data.py               ← standalone data downloader
    ├── train_all_models.py            ← train all models sequentially
    ├── run_backtest.py                ← run full walk-forward backtest
    ├── start_paper_trading.py         ← launch live paper trader
    └── generate_report.py             ← generate full HTML report
```

---

## Data Pipeline — Detailed Spec

### `src/data/bootstrap/downloader.py`

```python
"""
Master download orchestrator. Called by bootstrap.py.
Downloads everything needed for the full system to run.

Protocol:
1. Create data/ directory structure
2. Download equity OHLCV (yfinance) — all EQUITY_UNIVERSE assets
   - Period: max available history
   - Interval: 1d (daily bars)
   - Save to data/raw/equities/{ticker}.parquet
3. Download crypto OHLCV (ccxt Binance) — all CRYPTO_UNIVERSE
   - Period: 2017-01-01 to today
   - Interval: 1d
   - Save to data/raw/crypto/{pair}.parquet
4. Download FRED macro series — all MACRO_FRED_SERIES
   - Period: 2000-01-01 to today
   - Save to data/raw/macro/{series}.parquet
5. Download news headlines (NewsAPI) — past 30 days for each ticker
   - Save to data/raw/news/{ticker}_{date}.json
6. Download SEC filings index — top 20 assets
   - Save to data/raw/sec/{ticker}_filings.json
7. Run data validation on everything
8. Run feature engineering on all assets
9. Save processed data to data/processed/
10. Print summary: assets loaded, date ranges, row counts, any gaps
"""
```

### `src/data/features/engineer.py` — Feature Catalog

```python
"""
Produces a feature matrix with NO future data contamination.
All features use only data available at time T to produce features at T.
Shift rule: any feature using close[t] is valid at end of bar t.
Features using tomorrow's data are FORBIDDEN.

Feature groups and expected column names:

MOMENTUM (prefix: mom_):
  mom_rsi_7, mom_rsi_14, mom_rsi_21
  mom_macd_line, mom_macd_signal, mom_macd_hist, mom_macd_diverge (bool)
  mom_roc_5, mom_roc_10, mom_roc_20
  mom_willr_14
  mom_stoch_k, mom_stoch_d
  mom_cci_20
  mom_dpo_20                          ← Detrended Price Oscillator
  mom_trix_15                         ← TRIX momentum oscillator

VOLATILITY (prefix: vol_):
  vol_atr_7, vol_atr_14, vol_atr_21
  vol_atr_norm_14                     ← ATR / close price
  vol_bb_upper, vol_bb_lower, vol_bb_mid
  vol_bb_pct, vol_bb_width
  vol_bb_squeeze                      ← bool: bandwidth < 20th percentile
  vol_hv_5, vol_hv_10, vol_hv_21     ← historical vol (rolling std log ret)
  vol_parkinson                       ← Parkinson high-low estimator
  vol_garman_klass                    ← Garman-Klass estimator
  vol_vol_ratio                       ← vol_hv_5 / vol_hv_21 (vol regime)

VOLUME (prefix: vl_):
  vl_obv, vl_obv_slope_5
  vl_vwap_dev                         ← (close - VWAP) / VWAP
  vl_vol_ratio                        ← volume / vol_sma_20
  vl_adl                              ← Accumulation/Distribution Line
  vl_cmf_20                           ← Chaikin Money Flow
  vl_mfi_14                           ← Money Flow Index
  vl_ease_of_movement                 ← EOM
  vl_volume_price_trend               ← VPT

STATISTICAL (prefix: stat_):
  stat_zscore_20, stat_zscore_50, stat_zscore_200
  stat_hurst_60                       ← Hurst Exponent (0.5=random, >0.5=trend)
  stat_autocorr_1, stat_autocorr_2, stat_autocorr_5, stat_autocorr_10
  stat_skew_21, stat_kurt_21
  stat_jb_stat                        ← Jarque-Bera stat (normality)
  stat_half_life                      ← mean reversion half-life (OU process)
  stat_adf_stat                       ← ADF test statistic (stationarity)

CROSS-ASSET (prefix: ca_):
  ca_corr_spy_21, ca_corr_spy_63
  ca_corr_vix_21
  ca_corr_gold_21
  ca_corr_tlt_21
  ca_beta_spy_60
  ca_rel_strength_sector              ← asset return / sector ETF return
  ca_vix_level                        ← current VIX value
  ca_vix_regime                       ← categorical: low/mid/high/spike
  ca_yield_curve                      ← 10Y - 2Y spread (recession indicator)
  ca_dollar_index                     ← DXY level

MICROSTRUCTURE (prefix: ms_):
  ms_amihud                           ← Amihud illiquidity (|ret| / volume)
  ms_roll_spread                      ← Roll bid-ask spread estimate
  ms_kyle_lambda                      ← price impact coefficient

SENTIMENT (prefix: sent_):
  sent_score_1d                       ← FinBERT mean score on today's headlines
  sent_score_3d, sent_score_7d        ← rolling averages
  sent_momentum_5d                    ← sent_score_1d - sent_score_7d
  sent_diverge                        ← bool: price trend ≠ sentiment trend
  sent_volume_headlines               ← number of headlines today

FUNDAMENTAL (prefix: fund_):
  fund_pe_ratio                       ← trailing P/E
  fund_pb_ratio
  fund_ev_ebitda
  fund_earnings_surprise              ← latest EPS surprise %
  fund_revision_trend                 ← analyst estimate revision direction

REGIME (prefix: reg_):
  reg_hmm_state                       ← 0=bear, 1=sideways, 2=bull
  reg_hmm_bull_prob, reg_hmm_bear_prob, reg_hmm_side_prob
  reg_trend_strength                  ← ADX(14)
  reg_trend_direction                 ← +1 / -1 based on 50d vs 200d MA

TARGET VARIABLES (computed but not used as features):
  target_1d                           ← sign(next_close - close): -1, 0, 1
  target_5d                           ← sign(close.shift(-5) - close)
  target_ret_1d                       ← log(next_close / close)
  target_ret_5d
  target_vol_adj_1d                   ← target_ret_1d / vol_hv_5
"""
```

---

## Model Specs — Detailed

### `src/models/base.py`

```python
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class ModelPrediction:
    signal: int           # -1 (short), 0 (flat), 1 (long)
    confidence: float     # [0.0, 1.0]
    raw_proba: np.ndarray # [P(short), P(flat), P(long)]
    model_name: str
    timestamp: pd.Timestamp
    feature_snapshot: dict # top 10 features + values at prediction time

class BaseModel(ABC):
    @abstractmethod
    def fit(self, X_train, y_train, X_val, y_val,
            sample_weights=None) -> dict: ...  # returns train metrics

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...

    @abstractmethod
    def get_feature_importance(self) -> pd.Series: ...

    @abstractmethod
    def save(self, path: str) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'BaseModel': ...

    def predict_with_metadata(self, X: pd.DataFrame,
                               timestamp: pd.Timestamp) -> ModelPrediction:
        proba = self.predict_proba(X)
        signal = np.argmax(proba) - 1  # [0,1,2] → [-1,0,1]
        confidence = float(proba.max())
        top_features = (self.get_feature_importance()
                           .nlargest(10).to_dict())
        return ModelPrediction(
            signal=signal, confidence=confidence,
            raw_proba=proba, model_name=self.__class__.__name__,
            timestamp=timestamp, feature_snapshot=top_features
        )
```

### XGBoost (`src/models/classical/xgb_model.py`)

- 3-class classification: {-1: short, 0: flat, 1: long}
- Features: all engineered features except targets
- Hyperparameter search via Optuna (100 trials)
- Optuna parameters to tune: n_estimators[100-2000], max_depth[3-10], learning_rate[1e-4, 0.3 log], subsample[0.5-1.0], colsample_bytree[0.3-1.0], reg_alpha[1e-8, 1.0 log], reg_lambda[1e-8, 1.0 log]
- Use TimeSeriesSplit(n_splits=5) inside Optuna for CV
- Class weights: compute_class_weight('balanced')
- SHAP: TreeExplainer, compute on every predict call, cache for 1 bar
- Save: joblib + metadata JSON (params, feature_names, train_date)

### LightGBM (`src/models/classical/lgbm_model.py`)

- Same structure as XGBoost
- Additional: monotonic constraints
  - RSI: constraint=-1 (higher RSI → weaker buy signal, stronger sell)
  - Sentiment: constraint=+1 (better sentiment → stronger buy)
  - VIX: constraint=-1 (higher VIX → weaker long signal)
- DART boosting for regularization
- Feature interaction constraints:
  - Group 1: all momentum features (interact freely)
  - Group 2: all volatility features
  - Group 3: [vix, beta, correlation] cross-asset
  - Groups interact only through ensemble

### LSTM (`src/models/deep/lstm_model.py`)

```
Architecture:
  Input: (batch, seq_len=60, n_features)
  → BatchNorm1d
  → LSTM(hidden=256, num_layers=3, dropout=0.3, bidirectional=False)
     Note: unidirectional only — no future leakage from bidirectional
  → TemporalAttention(d_model=256)
     Attention over time dimension, returns weighted context vector
  → Dropout(0.3)
  → Linear(256, 128) → GELU → Dropout(0.2)
  → Linear(128, 3) → Softmax

Training:
  Optimizer: AdamW(lr=1e-3, weight_decay=1e-4)
  Schedule: CosineAnnealingWarmRestarts(T_0=10, T_mult=2)
  Gradient clipping: max_norm=1.0
  Early stopping: patience=15, monitor val_loss
  Batch: 64, Epochs: 100 max
  Loss: CrossEntropyLoss with class weights
```

### TCN (`src/models/deep/tcn_model.py`)

```
Architecture: Temporal Convolutional Network with dilated causal convolutions
Dilation schedule: [1, 2, 4, 8, 16, 32, 64, 128]
Kernel size: 3
Each block: Conv1d(causal) → WeightNorm → ReLU → Dropout(0.2) → ResidualAdd
Receptive field: kernel_size * sum(dilations) = 3 * 255 = 765 bars
Output: Global average pool → Linear(3) → Softmax
Note: causal padding ensures no future information leaks
Advantage over LSTM: fully parallelizable, no gradient vanishing
```

### PatchTST Transformer (`src/models/deep/transformer_model.py`)

```
Architecture: Patch-based Time Series Transformer (2023)
Patch length: 16 bars
Stride: 8 bars (50% overlap)
Number of patches: (60 - 16) // 8 + 1 = 6 patches
Each patch: Linear projection to d_model=128
Transformer encoder: 4 layers, 8 heads, d_ff=256, dropout=0.1
Channel independence: separate transformer per feature channel,
  outputs concatenated before final head
Prediction head: MLP(n_channels * d_model → 128 → 3) → Softmax
Positional encoding: learnable
Pre-training: optional masked patch prediction on full history
  (self-supervised, then fine-tune on labeled targets)
```

### N-BEATS (`src/models/deep/nbeats_model.py`)

```
Architecture: Neural Basis Expansion Analysis for Time Series
Stack 1: Trend stack (polynomial basis, degree=3)
Stack 2: Seasonality stack (Fourier basis, harmonics=20)
Stack 3: Generic stack (fully learnable basis)
Each block: FC(256) → ReLU × 4 → backcast + forecast heads
Backcast subtraction creates residual signal for next stack
Output: 5-day forward return prediction (regression)
Convert to signal: ret > threshold → long, ret < -threshold → short
```

### PPO RL Agent (`src/models/rl/`)

#### Environment (`environment.py`)

```python
class TradingEnv(gymnasium.Env):
    """
    State space:
      - Last 60 bars of top 50 features (selected by XGBoost importance)
      - Current position: scalar in {-1, 0, 1}
      - Unrealized P&L: normalized by portfolio value
      - Portfolio heat: sum of absolute position weights
      - Days since last trade
      - Current regime (HMM state as one-hot)
      Shape: (60 * 50 + 6,) = 3006 floats

    Action space:
      Discrete(5): [strong_short, short, flat, long, strong_long]
      Maps to position sizes: [-1.0, -0.5, 0.0, 0.5, 1.0] × max_position

    Reward function:
      daily_return = portfolio_value_change / portfolio_value_prev
      volatility_penalty = -0.5 * rolling_vol_21d
      transaction_cost = -|position_change| * cost_rate
      drawdown_penalty = -2.0 * max(0, drawdown - 0.05)

      reward = daily_return + volatility_penalty + transaction_cost + drawdown_penalty

      Shaped reward encourages: profitability, low volatility, infrequent trading,
      controlled drawdown

    Episode: 252 trading days (1 year)
    Reset: random start date from training period

    Curriculum stages:
      Stage 1 (0-500k steps): Low volatility periods only (VIX < 15)
      Stage 2 (500k-2M steps): Normal market conditions (VIX < 25)
      Stage 3 (2M+ steps): All conditions including crises
    """
```

#### PPO Config (`ppo_agent.py`)

```python
PPO_CONFIG = {
    "policy": "MlpLstmPolicy",     # recurrent policy
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "clip_range_vf": None,
    "normalize_advantage": True,
    "ent_coef": 0.01,              # entropy for exploration
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "learning_rate": 3e-4,         # linear decay schedule
    "total_timesteps": 5_000_000,
}
```

### Ensemble (`src/models/ensemble/ensemble.py`)

```
Level 0 models: XGBoost, LightGBM, RandomForest, LSTM, TCN, PatchTST, PPO
  Each produces: signal (-1/0/1) + confidence (0-1) + raw probas (3,)

Level 1 meta-learner: Ridge classifier
  Input: concatenation of all level-0 raw_proba outputs (7 * 3 = 21 features)
  + regime one-hot (3 features)
  Training: out-of-fold predictions from walk-forward

Regime-aware weighting:
  HMM bull regime:  XGB×0.15, LGBM×0.15, RF×0.10, LSTM×0.20, TCN×0.15, TST×0.15, PPO×0.10
  HMM bear regime:  XGB×0.20, LGBM×0.20, RF×0.15, LSTM×0.15, TCN×0.10, TST×0.10, PPO×0.10
  HMM sideways:     XGB×0.10, LGBM×0.10, RF×0.10, LSTM×0.10, TCN×0.10, TST×0.10, PPO×0.40

Dynamic adjustment: every 21 days, recompute weights based on
  recent Sharpe of each model's signals. Blend 50/50 with static weights.

Trade gate: only emit signal when ensemble_confidence > 0.62
  AND at least 5 of 7 models agree on direction
  AND regime_detector is not in "transition" state
```

---

## Backtesting Engine — Detailed Spec

### Walk-Forward Protocol

```
ASSETS: all EQUITY_UNIVERSE + CRYPTO_UNIVERSE simultaneously

FOLDS:
  Training window:   730 calendar days (2 years)
  Test window:       91 calendar days (3 months)
  Step size:         91 days
  Minimum history:   5 years required before first fold
  Typical folds:     ~20 folds over 7-year dataset

STRICT DATA ISOLATION:
  - Feature normalization (z-score) computed on train window only
  - Test window scaled using train-window mean/std (no test data leakage)
  - Model hyperparameters tuned inside train window only
  - Optuna trials use inner TimeSeriesSplit(n_splits=5) within train window

PER-FOLD EXECUTION:
  For each bar t in test window:
    1. Use only features computed from data[0:t] (no data[t+1:] used)
    2. ensemble.predict(features[t]) → signal, confidence
    3. risk_manager.check(signal, portfolio_state) → approved_trade
    4. portfolio.execute(approved_trade, prices[t]) → fill at open[t+1]
       (signal at close t → fill at open t+1: realistic execution)
    5. deduct costs from cash
    6. record portfolio state, signal, costs
```

### Cost Model

```python
def transaction_cost(trade_value: float, asset_type: str,
                     realized_vol: float, avg_daily_volume: float) -> float:
    """
    Models all real costs a retail algorithmic trader faces.
    All costs denominated in dollars.
    """
    if asset_type == "equity":
        commission = max(1.00, trade_value * 0.0005)
        slippage = trade_value * 0.0003 * (1 + realized_vol / 0.015)
        market_impact = trade_value * 0.0001 * sqrt(trade_value / avg_daily_volume)
        sec_fee = trade_value * 0.0000229  # SEC transaction fee
        finra_taf = min(5.95, trade_size_shares * 0.000119)

    elif asset_type == "crypto":
        commission = trade_value * 0.001    # 10bps taker fee
        slippage = trade_value * 0.0005 * (1 + realized_vol / 0.025)
        market_impact = trade_value * 0.0002 * sqrt(trade_value / avg_daily_volume)
        sec_fee = 0
        finra_taf = 0

    funding_cost = 0  # added in future if margin is modeled
    total = commission + slippage + market_impact + sec_fee + finra_taf
    return total
```

### Performance Metrics

```python
def compute_all_metrics(equity_curve: pd.Series,
                         trades: pd.DataFrame,
                         risk_free_rate: float = 0.045) -> dict:
    """
    Compute all 25 metrics. equity_curve is daily portfolio values.
    """
    returns = equity_curve.pct_change().dropna()
    log_returns = np.log(equity_curve / equity_curve.shift(1)).dropna()

    TRADING_DAYS = 252
    excess = returns - risk_free_rate / TRADING_DAYS

    dd = (equity_curve / equity_curve.cummax() - 1)
    max_dd = dd.min()
    dd_duration = compute_drawdown_duration(dd)

    winning_trades = trades[trades.pnl > 0]
    losing_trades  = trades[trades.pnl < 0]

    return {
        # Returns
        "total_return":         (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1,
        "annualized_return":    (1 + returns.mean()) ** TRADING_DAYS - 1,
        "cagr":                 (equity_curve.iloc[-1] / equity_curve.iloc[0])
                                ** (TRADING_DAYS / len(returns)) - 1,

        # Risk-adjusted
        "sharpe_ratio":         excess.mean() / excess.std() * sqrt(TRADING_DAYS),
        "sortino_ratio":        excess.mean() / excess[excess<0].std() * sqrt(TRADING_DAYS),
        "calmar_ratio":         annualized_return / abs(max_dd),
        "omega_ratio":          compute_omega(returns, threshold=risk_free_rate/TRADING_DAYS),
        "information_ratio":    compute_information_ratio(returns, benchmark_returns),

        # Drawdown
        "max_drawdown":         max_dd,
        "max_drawdown_duration": dd_duration,
        "avg_drawdown":         dd[dd < 0].mean(),
        "recovery_factor":      total_return / abs(max_dd),

        # Trade statistics
        "total_trades":         len(trades),
        "trades_per_year":      len(trades) / (len(returns) / TRADING_DAYS),
        "win_rate":             len(winning_trades) / len(trades),
        "profit_factor":        winning_trades.pnl.sum() / abs(losing_trades.pnl.sum()),
        "avg_trade_return":     trades.pnl.mean() / equity_curve.iloc[0],
        "avg_win":              winning_trades.pnl.mean(),
        "avg_loss":             losing_trades.pnl.mean(),
        "largest_win":          winning_trades.pnl.max(),
        "largest_loss":         losing_trades.pnl.min(),
        "avg_holding_days":     trades.holding_days.mean(),
        "expectancy":           (win_rate * avg_win + (1-win_rate) * avg_loss),

        # Distribution
        "return_skewness":      returns.skew(),
        "return_kurtosis":      returns.kurt(),
        "var_95":               returns.quantile(0.05),
        "cvar_95":              returns[returns < returns.quantile(0.05)].mean(),

        # Regime breakdown
        "sharpe_bull":          compute_sharpe(returns[regime=="bull"]),
        "sharpe_bear":          compute_sharpe(returns[regime=="bear"]),
        "sharpe_sideways":      compute_sharpe(returns[regime=="sideways"]),
    }
```

### Monte Carlo Simulation

```python
"""
After backtest completes, run Monte Carlo on the trade return series:
  - Bootstrap resample trades (with replacement) 10,000 times
  - For each simulation: reconstruct equity curve
  - Compute: 5th percentile Sharpe, 95th percentile Max Drawdown,
    probability of 20% loss, probability of doubling
  - Plot: fan chart of equity curve confidence bands
  - This shows robustness — a good strategy should have tight bands
"""
```

---

## Risk Management — Complete Rules

### Position Sizing (`position_sizer.py`)

```python
def compute_position_size(signal: int, confidence: float,
                           portfolio: Portfolio, asset: str) -> float:
    """
    Three-layer sizing system:

    Layer 1: Base Kelly
      win_rate, avg_win, avg_loss from trailing 63-day trade history
      kelly = (win_rate * avg_win - (1-win_rate) * |avg_loss|) / avg_win
      kelly_half = kelly / 2  (half-Kelly for safety)

    Layer 2: Volatility targeting
      vol_target = 0.15  # 15% annualized target portfolio vol
      asset_vol = realized_vol_21d * sqrt(252)
      vol_scalar = vol_target / asset_vol
      sized = kelly_half * vol_scalar

    Layer 3: Confidence scaling
      sized = sized * confidence  # high confidence → larger position

    Hard limits (never exceeded regardless of formula output):
      max_single_position = 0.20    # 20% of portfolio in one name
      max_sector_concentration = 0.40  # 40% in any one sector
      max_total_gross_exposure = 1.50  # 150% (some leverage allowed)
      min_trade_size = 0.01         # 1% minimum to avoid noise trades

    return min(sized, max_single_position) * portfolio.value
    """
```

### Circuit Breakers (all stateful, all thread-safe)

```python
CIRCUIT_BREAKERS = {
    "portfolio_drawdown": {
        "threshold": -0.15,         # -15% from peak
        "action":    "HALT_ALL",    # liquidate everything
        "cooldown":  5,             # trading days before resuming
        "log":       True,
    },
    "daily_loss": {
        "threshold": -0.02,         # -2% portfolio in one day
        "action":    "FLAT_ALL",    # close all, resume next day
        "cooldown":  1,
    },
    "single_position_loss": {
        "threshold": -0.05,         # -5% portfolio from single name
        "action":    "CLOSE_POSITION",
        "cooldown":  0,
    },
    "volatility_spike": {
        "threshold": 2.5,           # ATR > 2.5x 21d average
        "action":    "REDUCE_HALF", # cut all positions by 50%
        "cooldown":  0,
        "resume_condition": "atr < 1.5 * atr_21d_avg",
    },
    "correlation_crisis": {
        "threshold": 0.85,          # avg pairwise corr > 0.85
        "action":    "MAX_3_POSITIONS",
        "cooldown":  0,
    },
    "model_confidence_low": {
        "threshold": 0.55,          # ensemble confidence < 0.55
        "action":    "NO_NEW_TRADES",
        "cooldown":  0,
    },
    "vix_spike": {
        "threshold": 40,            # VIX > 40
        "action":    "REDUCE_75_PCT",
        "cooldown":  3,
    },
}
```

---

## Dashboard — Page-by-Page Spec

### CSS Injection (`src/dashboard/components/theme.py`)

```python
def inject_terminal_theme():
    """
    Inject custom CSS into Streamlit.
    Override all default Streamlit styling.

    Requirements:
    - Background: --bg-primary (#0A0A0A) on entire page
    - Sidebar: --bg-secondary (#111318)
    - All metric values: --font-mono, right-aligned
    - Positive deltas: --green-up (#00E676)
    - Negative deltas: --red-down (#FF1744)
    - Chart backgrounds: --bg-secondary
    - Remove all Streamlit default padding/margins that break grid
    - Custom scrollbars: thin, amber track
    - Page title in amber, uppercase, letter-spacing: 0.15em
    - All st.metric() values formatted with monospace font
    - Tables: no cell borders except bottom 1px --bg-border
    - Hover on table rows: background --amber-glow
    """
```

### Page 1: Live Overview (`pages/01_live_overview.py`)

```
Layout: 4-column KPI strip at top, then 2-column grid below

KPI Strip (updates every 30 seconds):
  [Portfolio Value] [Today P&L] [Total Return] [Sharpe (live)] [Open Positions] [Active Signals]
  Numbers in amber monospace. Positive P&L green, negative red.

Left column (60% width):
  Live equity curve (Plotly, updating)
  Below: Open positions table
    Columns: Asset | Entry Price | Current | Unrealized P&L% | Size | Signal | Confidence

Right column (40% width):
  Regime indicator: large colored badge (BULL/BEAR/SIDEWAYS)
    with probability bars for each state
  Below: Today's signal cards — one per asset
    Each card: ticker | signal arrow (↑↓—) | confidence bar | model agreement
  Below: Circuit breaker status — traffic light grid
```

### Page 2: Backtest Results (`pages/02_backtest_results.py`)

```
Top: Fold selector (dropdown) or "All Folds Aggregated"

Section 1: Summary Table
  Rows: each walk-forward fold
  Columns: Period | Sharpe | Sortino | Max DD | Win Rate | Trades | Return
  Color-code Sharpe: <0.5 red, 0.5-1.0 amber, >1.0 green
  Final row: Mean ± Std across all folds (bold)

Section 2: Equity Curves (overlaid)
  All fold equity curves on same chart, normalized to 100
  Benchmark (SPY buy-and-hold) as thick amber dashed line
  Monte Carlo confidence bands as shaded regions

Section 3: Monthly Returns Heatmap
  Calendar grid: years as rows, months as columns
  Color: red-to-green diverging, white at zero
  Values in monospace

Section 4: Trade Analysis
  Left: P&L distribution histogram (log scale x-axis)
  Right: Win/loss streak chart, holding period boxplot
```

### Page 3: Model Analysis (`pages/03_model_analysis.py`)

```
Section 1: Model Leaderboard
  Table: Model | Sharpe | Win Rate | Avg Confidence | Prediction Count
  Sorted by Sharpe descending

Section 2: SHAP Beeswarm
  Dropdown to select model
  Plotly beeswarm plot of SHAP values for top 20 features
  Color: feature value (low=blue, high=red)

Section 3: Model Agreement
  For each trading day: how many models agreed with final signal?
  Line chart: agreement score over time
  Scatter: agreement score vs trade P&L (does high agreement = better?)

Section 4: Confidence Distribution
  Histogram of confidence scores over time, per model
  Vertical line at 0.62 threshold
```

### Page 4: Risk Monitor (`pages/04_risk_monitor.py`)

```
Top row:
  [Drawdown Gauge] [VaR 95% Gauge] [Portfolio Heat Gauge] [VIX Level]
  Gauges use Plotly indicator charts, colored by risk level

Middle: Live Correlation Matrix
  Heatmap of current 21d rolling correlations
  Cells > 0.85 highlighted in amber (warning)

Bottom row:
  Left: Circuit breaker status grid (7 breakers, each with status + last triggered)
  Right: Position risk table
    Asset | Weight | Vol Contribution | Beta | Corr to Portfolio
```

### Page 5: Trade Log (`pages/05_trade_log.py`)

```
Full paginated trade history table:
  Date | Asset | Direction | Entry | Exit | Return% | P&L | Holding Days | Signal Source

Filters: date range, asset, direction, profitable only
Export to CSV button

Below table:
  Best 5 trades: entry context, signal that triggered, SHAP explanation
  Worst 5 trades: same structure, post-mortem analysis
```

### Page 6: Explainability (`pages/06_explainability.py`)

```
Select any historical trade → see full explanation:
  - Which model generated the primary signal
  - SHAP waterfall plot for that specific prediction
  - Feature values at prediction time vs their historical distribution
  - What would have changed the prediction (counterfactual)

Feature Drift Monitor:
  Table: top 20 features, PSI score, trend (stable/warning/critical)
  Time series of PSI for selected feature

P&L Attribution:
  Stacked bar: how much P&L came from each signal source
  Pie: P&L attribution by asset, by regime, by model
```

### Page 7: Documentation (`pages/07_documentation.py`)

```
Full inline documentation page:
  - Architecture diagram (rendered from mermaid or embedded SVG)
  - All formulas used (LaTeX rendered via st.latex)
  - Data sources table
  - Model descriptions
  - Risk rules reference
  - How to add a new model (step-by-step)
  - API endpoints reference
  - Glossary of all terms used in the UI
```

---

## FastAPI Backend — Endpoints

```
GET  /health                    → system status, uptime, last data update
GET  /portfolio                 → current portfolio state
GET  /portfolio/history         → equity curve as JSON
GET  /positions                 → all open positions
GET  /signals                   → latest signals for all assets
GET  /signals/{ticker}          → signal + confidence + explanation for one asset
GET  /models                    → all models, their status, last metrics
GET  /models/{name}/metrics     → detailed metrics for one model
GET  /backtest/results          → latest backtest summary
GET  /backtest/results/{fold}   → specific fold results
POST /backtest/run              → trigger new backtest run (async)
GET  /risk/status               → circuit breaker states
GET  /risk/var                  → current VaR/CVaR
GET  /regime                    → current regime state + probabilities
GET  /trades                    → trade history with pagination
WS   /live                      → WebSocket: portfolio updates every 5s
```

---

## Documentation Requirements

### `docs/architecture.md`

Must include:
- Mermaid diagram of full system (data → features → models → ensemble → risk → execution → dashboard)
- Data flow diagram (how a bar of price data becomes a trade)
- Component dependency graph
- Deployment architecture (Docker services)

### `docs/backtest_methodology.md`

Must include:
- Mathematical proof of no lookahead bias in WFO setup
- Complete cost model derivation
- All metric formulas with LaTeX
- Comparison to naive buy-and-hold benchmark
- Known limitations and assumptions

### `docs/models.md`

For each of the 7 models:
- Architecture description
- Input/output specification
- Hyperparameters and tuning methodology
- Known strengths and weaknesses
- Typical prediction latency

### `README.md` (auto-generated by `scripts/generate_report.py`)

```markdown
# AlgoTrade-X

## Quick Start
git clone ...
cd algotrade-x
cp .env.example .env
# Fill in free API keys (instructions inside .env.example)
docker-compose up

## Results
[auto-populated backtest results table]
[equity curve image]
[model comparison table]

## Architecture
[diagram]

## API Keys Required
[table: service, cost, signup URL, what it's used for]
```

---

## Docker Setup

### `docker-compose.yml`

```yaml
version: "3.9"
services:
  bootstrap:
    build: .
    command: python bootstrap.py
    volumes:
      - ./data:/app/data
      - ./.env:/app/.env
    profiles: [setup]

  api:
    build: .
    command: uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
    ports: ["8000:8000"]
    volumes:
      - ./data:/app/data
      - ./.env:/app/.env
    depends_on: [redis]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]

  dashboard:
    build: .
    command: streamlit run src/dashboard/app.py --server.port 8501 --server.address 0.0.0.0
    ports: ["8501:8501"]
    volumes:
      - ./data:/app/data
      - ./.env:/app/.env
    depends_on: [api]

  paper_trader:
    build: .
    command: python scripts/start_paper_trading.py
    volumes:
      - ./data:/app/data
      - ./.env:/app/.env
    depends_on: [api]
    profiles: [live]

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]

  jupyter:
    build: .
    command: jupyter lab --ip=0.0.0.0 --no-browser --allow-root
    ports: ["8888:8888"]
    volumes:
      - ./notebooks:/app/notebooks
      - ./data:/app/data
    profiles: [dev]
```

---

## Build Execution Plan

Execute in this exact order. Do not skip steps. Do not start step N+1 until all tests for step N pass.

```
PHASE 1: FOUNDATION
  Step 01: src/config/ — settings, assets, constants, logging
  Step 02: src/data/bootstrap/equity_loader.py — yfinance download
  Step 03: src/data/bootstrap/crypto_loader.py — ccxt Binance
  Step 04: src/data/bootstrap/macro_loader.py — FRED
  Step 05: src/data/pipeline/cleaner.py + validator.py
  Step 06: src/data/pipeline/storage.py — parquet read/write
  Step 07: bootstrap.py — end-to-end: download → clean → store

  CHECKPOINT: run bootstrap.py, verify data/ directory populated
              with clean parquet files, print data summary

PHASE 2: FEATURES
  Step 08: src/data/features/momentum.py
  Step 09: src/data/features/volatility.py
  Step 10: src/data/features/volume.py
  Step 11: src/data/features/statistical.py
  Step 12: src/data/features/cross_asset.py
  Step 13: src/data/features/microstructure.py
  Step 14: src/data/features/sentiment.py (FinBERT via HuggingFace)
  Step 15: src/data/features/fundamental.py
  Step 16: src/data/features/regime.py (HMM)
  Step 17: src/data/features/engineer.py (master orchestrator)
  Step 18: tests/unit/test_feature_engineer.py — run, must pass
  Step 19: tests/unit/test_no_lookahead.py — run, MUST PASS BEFORE MODELS

PHASE 3: BACKTEST ENGINE
  Step 20: src/backtest/costs.py
  Step 21: src/backtest/portfolio.py
  Step 22: src/backtest/order_book.py
  Step 23: src/backtest/metrics.py
  Step 24: src/backtest/engine.py
  Step 25: tests/unit/test_cost_model.py + test_backtest_math.py — must pass
  Step 26: src/backtest/walk_forward.py
  Step 27: tests/unit/test_walk_forward.py — must pass

PHASE 4: MODELS (classical first, deep second, RL last)
  Step 28: src/models/base.py
  Step 29: src/models/classical/xgb_model.py
  Step 30: Run XGBoost through walk_forward — get baseline metrics
           Sharpe must be computed correctly (verify against manual calc)
  Step 31: src/models/classical/lgbm_model.py
  Step 32: src/models/classical/random_forest.py
  Step 33: src/models/deep/lstm_model.py
  Step 34: src/models/deep/tcn_model.py
  Step 35: src/models/deep/transformer_model.py
  Step 36: src/models/deep/nbeats_model.py
  Step 37: src/models/rl/environment.py
  Step 38: src/models/rl/ppo_agent.py
  Step 39: src/models/rl/sac_agent.py
  Step 40: src/models/ensemble/stacker.py + regime_router.py + ensemble.py
  Step 41: src/models/registry.py
  Step 42: tests/integration/test_model_interface.py — all models implement BaseModel

PHASE 5: RISK
  Step 43: src/risk/position_sizer.py
  Step 44: src/risk/circuit_breaker.py
  Step 45: src/risk/correlation_filter.py
  Step 46: src/risk/regime_detector.py
  Step 47: src/risk/var_calculator.py
  Step 48: src/risk/risk_manager.py
  Step 49: tests/unit/test_risk_rules.py + test_position_sizer.py — must pass

PHASE 6: EXPLAINABILITY
  Step 50: src/explainability/shap_analyzer.py
  Step 51: src/explainability/lime_explainer.py
  Step 52: src/explainability/drift_monitor.py
  Step 53: src/explainability/attribution.py
  Step 54: src/explainability/audit_log.py

PHASE 7: EXECUTION
  Step 55: src/execution/broker/base_broker.py
  Step 56: src/execution/broker/sim_broker.py
  Step 57: src/execution/broker/alpaca_broker.py
  Step 58: src/execution/order_manager.py
  Step 59: src/execution/paper_trader.py

PHASE 8: API
  Step 60: src/api/schemas.py
  Step 61: src/api/routes/ (all route files)
  Step 62: src/api/websocket.py
  Step 63: src/api/main.py
  Step 64: tests/integration/test_api_endpoints.py — must pass

PHASE 9: DASHBOARD
  Step 65: src/dashboard/components/theme.py (CSS injection)
  Step 66: src/dashboard/components/charts.py
  Step 67: src/dashboard/components/tables.py
  Step 68: src/dashboard/components/metrics_bar.py
  Step 69: src/dashboard/components/signal_card.py
  Step 70: src/dashboard/pages/ (all 7 pages)
  Step 71: src/dashboard/app.py

PHASE 10: DOCUMENTATION + POLISH
  Step 72: scripts/ (all 5 scripts)
  Step 73: docs/ (all 8 markdown files)
  Step 74: notebooks/ (all 5 notebooks with outputs cleared)
  Step 75: README.md via scripts/generate_report.py
  Step 76: .env.example with instructions
  Step 77: docker-compose.yml + Dockerfile
  Step 78: pyproject.toml, requirements.txt (pinned), requirements-dev.txt

PHASE 11: FULL INTEGRATION TEST
  Step 79: docker-compose up → verify all services start
  Step 80: run full walk-forward backtest via docker
  Step 81: verify dashboard loads, all 7 pages render
  Step 82: verify API returns valid JSON on all GET endpoints
  Step 83: run paper_trader for 1 simulated day → verify trade logs
  Step 84: tests/performance/test_backtest_speed.py → must complete < 60s
```

---

## Absolute Hard Constraints

These are never negotiable. If any test catches a violation, stop and fix before proceeding.

1. **No lookahead bias.** `test_no_lookahead.py` must pass at step 19 and remain passing forever.
2. **Transaction costs on every fill.** `costs.py` must return > 0 for any trade. `test_cost_model.py` verifies this.
3. **Fill at next open.** Signal at close[t] → fill at open[t+1]. Never fill at the bar that generated the signal.
4. **No financial metric is accuracy.** Never print accuracy, F1, or precision as a primary metric. Sharpe is the headline number.
5. **Walk-forward only.** No single train/test split for final evaluation. All reported metrics are out-of-sample from WFO.
6. **Type hints everywhere.** Every function signature has type hints. `pyproject.toml` enables mypy strict mode.
7. **Logging everywhere.** Every significant action logged via structlog. Log level configurable via `LOG_LEVEL` env var.
8. **No hardcoded paths.** All paths relative to project root via `settings.py`.
9. **No secrets in code.** All API keys via `.env` only, never in source.
10. **Tests green before merge.** `pytest tests/` must pass completely before any phase is considered done.
