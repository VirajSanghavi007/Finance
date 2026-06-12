from __future__ import annotations

import numpy as np
import pandas as pd


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def _vwap_deviation(high: pd.Series, low: pd.Series, close: pd.Series,
                    volume: pd.Series) -> pd.Series:
    tp    = (high + low + close) / 3
    vwap  = (tp * volume).rolling(20).sum() / volume.rolling(20).sum().replace(0, np.nan)
    return (close - vwap) / vwap.replace(0, np.nan)


def _adl(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    clv = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    return (clv * volume).cumsum()


def _cmf(high: pd.Series, low: pd.Series, close: pd.Series,
         volume: pd.Series, window: int = 20) -> pd.Series:
    mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mfv = mfm * volume
    return mfv.rolling(window).sum() / volume.rolling(window).sum().replace(0, np.nan)


def _mfi(high: pd.Series, low: pd.Series, close: pd.Series,
         volume: pd.Series, window: int = 14) -> pd.Series:
    tp = (high + low + close) / 3
    mf = tp * volume
    delta = tp.diff()
    pos_mf = mf.where(delta > 0, 0).rolling(window).sum()
    neg_mf = mf.where(delta < 0, 0).rolling(window).sum().abs()
    mfr    = pos_mf / neg_mf.replace(0, np.nan)
    return 100 - (100 / (1 + mfr))


def _eom(high: pd.Series, low: pd.Series, volume: pd.Series, window: int = 14) -> pd.Series:
    mid_move = (high + low) / 2 - (high.shift(1) + low.shift(1)) / 2
    box_ratio = (volume / 1e6) / (high - low).replace(0, np.nan)
    eom = mid_move / box_ratio.replace(0, np.nan)
    return eom.rolling(window).mean()


def _vpt(close: pd.Series, volume: pd.Series) -> pd.Series:
    return (close.pct_change() * volume).cumsum()


def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]
    out    = pd.DataFrame(index=df.index)

    # OBV + slope
    out["vl_obv"] = _obv(close, volume)
    out["vl_obv_slope_5"] = out["vl_obv"].diff(5)

    # VWAP deviation
    out["vl_vwap_dev"] = _vwap_deviation(high, low, close, volume)

    # Volume ratio vs 20-day SMA
    vol_sma20 = volume.rolling(20).mean().replace(0, np.nan)
    out["vl_vol_ratio"] = volume / vol_sma20

    # Accumulation/Distribution Line
    out["vl_adl"] = _adl(high, low, close, volume)

    # Chaikin Money Flow
    out["vl_cmf_20"] = _cmf(high, low, close, volume, 20)

    # Money Flow Index
    out["vl_mfi_14"] = _mfi(high, low, close, volume, 14)

    # Ease of Movement
    out["vl_ease_of_movement"] = _eom(high, low, volume, 14)

    # Volume Price Trend
    out["vl_volume_price_trend"] = _vpt(close, volume)

    return out
