from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.logging_config import get_logger

logger = get_logger(__name__)

OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def clean_ohlcv(df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
    df = df.copy()

    # Ensure datetime index, tz-naive
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    df = df.sort_index()

    # Drop exact duplicates on index
    df = df[~df.index.duplicated(keep="last")]

    # Coerce to float
    for col in OHLCV_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where close is NaN or zero
    if "close" in df.columns:
        df = df[df["close"].notna() & (df["close"] > 0)]

    # Fill small gaps in OHLC via forward-fill (up to 3 trading days)
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = df[col].ffill(limit=3)

    # Volume: fill NaN with 0 (some assets have no volume data)
    if "volume" in df.columns:
        df["volume"] = df["volume"].fillna(0).clip(lower=0)

    # OHLC consistency: high >= close >= low, high >= open >= low
    if all(c in df.columns for c in ["open", "high", "low", "close"]):
        bad_mask = (
            (df["high"] < df["low"]) |
            (df["close"] > df["high"] * 1.001) |
            (df["close"] < df["low"] * 0.999) |
            (df["open"] > df["high"] * 1.001) |
            (df["open"] < df["low"] * 0.999)
        )
        n_bad = bad_mask.sum()
        if n_bad > 0:
            logger.warning("ohlc_inconsistency_fixed", ticker=ticker, count=int(n_bad))
            # Clamp to fix minor floating-point issues
            df["high"] = df[["open", "high", "close"]].max(axis=1)
            df["low"]  = df[["open", "low",  "close"]].min(axis=1)

    # Outlier removal: drop rows where close changes > 50% in a single day
    # (likely split/bad data that wasn't adjusted)
    if "close" in df.columns and len(df) > 1:
        ret = df["close"].pct_change().abs()
        extreme = ret > 0.50
        if extreme.any():
            n = extreme.sum()
            logger.warning("extreme_return_dropped", ticker=ticker, count=int(n))
            df = df[~extreme]

    return df


def clean_macro(df: pd.DataFrame, series_id: str = "") -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    # Forward fill up to 5 days (macro data is often weekly/monthly)
    df = df.ffill(limit=5)
    df = df.dropna()
    return df
