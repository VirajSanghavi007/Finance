from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats  # type: ignore


def _zscore(close: pd.Series, window: int) -> pd.Series:
    ma  = close.rolling(window).mean()
    std = close.rolling(window).std()
    return (close - ma) / std.replace(0, np.nan)


def _hurst_exponent(ts: np.ndarray) -> float:
    """RS analysis for Hurst exponent. ~0.5=random, >0.5=trending, <0.5=mean-reverting."""
    n = len(ts)
    if n < 20:
        return 0.5
    lags = range(2, min(n // 2, 40))
    rs_vals = []
    for lag in lags:
        chunks = [ts[i:i + lag] for i in range(0, n - lag, lag)]
        if not chunks:
            continue
        rs = []
        for chunk in chunks:
            mean = np.mean(chunk)
            devs = np.cumsum(chunk - mean)
            r = devs.max() - devs.min()
            s = np.std(chunk, ddof=1)
            if s > 0:
                rs.append(r / s)
        if rs:
            rs_vals.append((lag, np.mean(rs)))
    if len(rs_vals) < 2:
        return 0.5
    lags_arr = np.log([x[0] for x in rs_vals])
    rs_arr   = np.log([x[1] for x in rs_vals])
    slope, _ = np.polyfit(lags_arr, rs_arr, 1)
    return float(np.clip(slope, 0.0, 1.0))


def _rolling_hurst(close: pd.Series, window: int = 60, stride: int = 5) -> pd.Series:
    """Compute every `stride` rows then forward-fill -- Hurst changes slowly."""
    log_ret = np.log(close / close.shift(1)).dropna()
    result = pd.Series(np.nan, index=close.index)
    indices = range(window, len(close), stride)
    for i in indices:
        subset = log_ret.iloc[max(0, i - window):i].values
        result.iloc[i] = _hurst_exponent(subset)
    return result.ffill()


def _autocorr(series: pd.Series, lag: int, window: int = 60) -> pd.Series:
    """Numpy-based autocorr -- avoids creating a pd.Series per window."""
    def _np_autocorr(x: np.ndarray) -> float:
        if len(x) <= lag + 2:
            return np.nan
        x = x - x.mean()
        denom = np.dot(x, x)
        if denom == 0:
            return np.nan
        return float(np.dot(x[:-lag], x[lag:]) / denom)
    return series.rolling(window).apply(_np_autocorr, raw=True)


def _jarque_bera(series: pd.Series, window: int = 60) -> pd.Series:
    def jb(x: np.ndarray) -> float:
        if len(x) < 8:
            return np.nan
        try:
            stat, _ = scipy_stats.jarque_bera(x)
            return float(stat)
        except Exception:
            return np.nan
    return series.rolling(window).apply(jb, raw=True)


def _ou_half_life(close: pd.Series, window: int = 60) -> pd.Series:
    log_p = np.log(close)
    lag_p = log_p.shift(1)
    spread = log_p - lag_p

    def hl(x: np.ndarray) -> float:
        if len(x) < 10:
            return np.nan
        y   = x[1:]
        lag = x[:-1]
        try:
            res = np.polyfit(lag, y, 1)
            lam = res[0]
            if lam >= 0 or np.isnan(lam):
                return np.nan
            return float(-np.log(2) / lam)
        except Exception:
            return np.nan

    return log_p.rolling(window).apply(hl, raw=True)


def _adf_stat(close: pd.Series, window: int = 60, stride: int = 5) -> pd.Series:
    """ADF with fixed maxlag=1 and stride -- autolag='AIC' is O(N²) per window."""
    from statsmodels.tsa.stattools import adfuller  # type: ignore

    result = pd.Series(np.nan, index=close.index)
    arr = close.values
    for i in range(window, len(arr), stride):
        x = arr[i - window:i]
        if len(x) < 15:
            continue
        try:
            res = adfuller(x, maxlag=1, autolag=None)
            result.iloc[i] = float(res[0])
        except Exception:
            pass
    return result.ffill()


def compute_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    log_ret = np.log(close / close.shift(1))
    out = pd.DataFrame(index=df.index)

    # Z-scores
    for w in [20, 50, 200]:
        out[f"stat_zscore_{w}"] = _zscore(close, w)

    # Hurst exponent (slow -- computed with rolling window)
    out["stat_hurst_60"] = _rolling_hurst(close, 60)

    # Autocorrelation of returns
    for lag in [1, 2, 5, 10]:
        out[f"stat_autocorr_{lag}"] = _autocorr(log_ret, lag=lag, window=63)

    # Skew and kurtosis of returns
    out["stat_skew_21"] = log_ret.rolling(21).skew()
    out["stat_kurt_21"] = log_ret.rolling(21).kurt()

    # Jarque-Bera stat (normality test)
    out["stat_jb_stat"] = _jarque_bera(log_ret, window=60)

    # Mean-reversion half-life (OU process)
    out["stat_half_life"] = _ou_half_life(close, window=60)

    # ADF test statistic (stationarity)
    try:
        out["stat_adf_stat"] = _adf_stat(close, window=60)
    except ImportError:
        out["stat_adf_stat"] = np.nan

    return out
