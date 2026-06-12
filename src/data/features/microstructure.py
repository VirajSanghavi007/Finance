from __future__ import annotations

import numpy as np
import pandas as pd


def _amihud_illiquidity(close: pd.Series, volume: pd.Series, window: int = 21) -> pd.Series:
    """Amihud (2002): |ret| / dollar_volume — higher = less liquid."""
    log_ret = np.log(close / close.shift(1)).abs()
    dollar_vol = close * volume
    ratio = log_ret / dollar_vol.replace(0, np.nan)
    return ratio.rolling(window).mean() * 1e6  # scale for readability


def _roll_spread(close: pd.Series, window: int = 21) -> pd.Series:
    """Roll (1984) bid-ask spread estimator: 2 * sqrt(-cov(ret_t, ret_{t-1}))."""
    ret = close.diff()

    def roll(x: np.ndarray) -> float:
        if len(x) < 4:
            return np.nan
        cov = np.cov(x[1:], x[:-1])[0, 1]
        val = -cov
        return float(2 * np.sqrt(val) if val > 0 else 0.0)

    return ret.rolling(window).apply(roll, raw=True)


def _kyle_lambda(close: pd.Series, volume: pd.Series, window: int = 21) -> pd.Series:
    """Kyle's lambda: price impact per unit of signed order flow (proxy)."""
    ret = np.log(close / close.shift(1))
    signed_vol = ret.apply(np.sign) * volume
    ratio = ret.abs() / (signed_vol.abs() + 1e-10)
    return ratio.rolling(window).mean() * 1e6


def compute_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    close  = df["close"]
    volume = df["volume"]
    out    = pd.DataFrame(index=df.index)

    out["ms_amihud"]     = _amihud_illiquidity(close, volume)
    out["ms_roll_spread"] = _roll_spread(close)
    out["ms_kyle_lambda"] = _kyle_lambda(close, volume)

    return out
