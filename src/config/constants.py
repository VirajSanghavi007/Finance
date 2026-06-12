from pathlib import Path

# Project root — all paths derived from here
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR          = PROJECT_ROOT / "data"
RAW_DIR           = DATA_DIR / "raw"
PROCESSED_DIR     = DATA_DIR / "processed"
MODELS_DIR        = DATA_DIR / "models"
RESULTS_DIR       = DATA_DIR / "backtest_results"
LOGS_DIR          = DATA_DIR / "logs"

RAW_EQUITIES_DIR  = RAW_DIR / "equities"
RAW_CRYPTO_DIR    = RAW_DIR / "crypto"
RAW_MACRO_DIR     = RAW_DIR / "macro"
RAW_NEWS_DIR      = RAW_DIR / "news"
RAW_SEC_DIR       = RAW_DIR / "sec"

# Trading calendar
TRADING_DAYS_PER_YEAR = 252
TRADING_DAYS = 252

# Risk thresholds
MAX_SINGLE_POSITION   = 0.20
MAX_SECTOR_CONC       = 0.40
MAX_GROSS_EXPOSURE    = 1.50
MIN_TRADE_SIZE        = 0.01
VOL_TARGET            = 0.15
MAX_PORTFOLIO_DRAWDOWN  = 0.20   # 20% max drawdown before circuit breaker
MAX_DAILY_LOSS          = 0.05   # 5% daily loss limit
MAX_CONSECUTIVE_LOSSES  = 7      # days
MAX_CORRELATION         = 0.75   # reject new position if corr > this with held

# Ensemble gate
MIN_ENSEMBLE_CONFIDENCE = 0.62
MIN_MODEL_AGREEMENT     = 5  # out of 7

# Backtest WFO
WFO_TRAIN_DAYS  = 730
WFO_TEST_DAYS   = 91
WFO_STEP_DAYS   = 91
WFO_MIN_HISTORY = 5 * 365

# Feature windows
RSI_WINDOWS         = [7, 14, 21]
ATR_WINDOWS         = [7, 14, 21]
HV_WINDOWS          = [5, 10, 21]
ZSCORE_WINDOWS      = [20, 50, 200]
AUTOCORR_LAGS       = [1, 2, 5, 10]
CORR_WINDOWS        = [21, 63]
SENTIMENT_WINDOWS   = [1, 3, 7]

# Regime
HMM_N_STATES = 3
HMM_STATE_BEAR    = 0
HMM_STATE_SIDEWAYS = 1
HMM_STATE_BULL    = 2

VIX_LOW    = 15.0
VIX_MID    = 25.0
VIX_HIGH   = 35.0
VIX_SPIKE  = 40.0
