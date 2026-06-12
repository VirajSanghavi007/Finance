from __future__ import annotations

import numpy as np
import pandas as pd


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def _historical_vol(close: pd.Series, window: int) -> pd.Series:
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252)


def _bollinger(close: pd.Series, window: int = 20, n_std: float = 2.0
               ) -> tuple[pd.Series, pd.Series, pd.Series]:
    ma  = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    return upper, lower, ma


def _parkinson(high: pd.Series, low: pd.Series, window: int = 21) -> pd.Series:
    log_hl = np.log(high / low) ** 2
    return (log_hl.rolling(window).mean() / (4 * np.log(2))) ** 0.5 * np.sqrt(252)


def _garman_klass(open_: pd.Series, high: pd.Series, low: pd.Series,
                  close: pd.Series, window: int = 21) -> pd.Series:
    log_hl = np.log(high / low) ** 2
    log_co = np.log(close / open_) ** 2
    daily  = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    return (daily.rolling(window).mean()) ** 0.5 * np.sqrt(252)


def compute_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    open_ = df["open"]
    out   = pd.DataFrame(index=df.index)

    # ATR (raw and normalised)
    for w in [7, 14, 21]:
        out[f"vol_atr_{w}"] = _atr(high, low, close, w)
    out["vol_atr_norm_14"] = out["vol_atr_14"] / close

    # Bollinger Bands
    bb_upper, bb_lower, bb_mid = _bollinger(close)
    out["vol_bb_upper"] = bb_upper
    out["vol_bb_lower"] = bb_lower
    out["vol_bb_mid"]   = bb_mid
    bb_range = (bb_upper - bb_lower).replace(0, np.nan)
    out["vol_bb_pct"]   = (close - bb_lower) / bb_range
    out["vol_bb_width"] = bb_range / bb_mid

    # BB squeeze: width < 20th percentile of rolling 252d
    width_pct = out["vol_bb_width"].rolling(252, min_periods=50).quantile(0.20)
    out["vol_bb_squeeze"] = (out["vol_bb_width"] < width_pct).astype(float)

    # Historical volatility
    for w in [5, 10, 21]:
        out[f"vol_hv_{w}"] = _historical_vol(close, w)

    # Parkinson & Garman-Klass
    out["vol_parkinson"]    = _parkinson(high, low)
    out["vol_garman_klass"] = _garman_klass(open_, high, low, close)

    # Vol ratio: short-term vs long-term (regime indicator)
    out["vol_vol_ratio"] = out["vol_hv_5"] / out["vol_hv_21"].replace(0, np.nan)

    return out
