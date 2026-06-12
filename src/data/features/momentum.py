from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=window - 1, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(com=window - 1, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
           ) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig  = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return line, sig, hist


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                k_window: int = 14, d_window: int = 3
                ) -> tuple[pd.Series, pd.Series]:
    low_min  = low.rolling(k_window).min()
    high_max = high.rolling(k_window).max()
    k = 100 * (close - low_min) / (high_max - low_min).replace(0, np.nan)
    d = k.rolling(d_window).mean()
    return k, d


def _cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    tp = (high + low + close) / 3
    ma = tp.rolling(window).mean()
    md = tp.rolling(window).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def _roc(close: pd.Series, window: int) -> pd.Series:
    return close.pct_change(window) * 100


def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    high_max = high.rolling(window).max()
    low_min  = low.rolling(window).min()
    return -100 * (high_max - close) / (high_max - low_min).replace(0, np.nan)


def _dpo(close: pd.Series, window: int = 20) -> pd.Series:
    shift = window // 2 + 1
    ma = close.rolling(window).mean()
    return close - ma.shift(shift)


def _trix(close: pd.Series, window: int = 15) -> pd.Series:
    ema1 = close.ewm(span=window, adjust=False).mean()
    ema2 = ema1.ewm(span=window, adjust=False).mean()
    ema3 = ema2.ewm(span=window, adjust=False).mean()
    return ema3.pct_change() * 100


def compute_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all momentum features. Input: OHLCV df. Returns feature columns only."""
    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    out   = pd.DataFrame(index=df.index)

    # RSI
    for w in [7, 14, 21]:
        out[f"mom_rsi_{w}"] = _rsi(close, w)

    # MACD
    macd_line, macd_sig, macd_hist = _macd(close)
    out["mom_macd_line"]    = macd_line
    out["mom_macd_signal"]  = macd_sig
    out["mom_macd_hist"]    = macd_hist
    out["mom_macd_diverge"] = (macd_hist > 0) & (macd_hist.shift(1) <= 0)  # bullish cross

    # ROC
    for w in [5, 10, 20]:
        out[f"mom_roc_{w}"] = _roc(close, w)

    # Williams %R
    out["mom_willr_14"] = _williams_r(high, low, close, 14)

    # Stochastic
    k, d = _stochastic(high, low, close)
    out["mom_stoch_k"] = k
    out["mom_stoch_d"] = d

    # CCI
    out["mom_cci_20"] = _cci(high, low, close, 20)

    # DPO
    out["mom_dpo_20"] = _dpo(close, 20)

    # TRIX
    out["mom_trix_15"] = _trix(close, 15)

    return out
