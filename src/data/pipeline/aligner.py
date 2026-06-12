from __future__ import annotations

import pandas as pd
import numpy as np

from src.config.logging_config import get_logger

logger = get_logger(__name__)


def build_trading_calendar(
    start: str,
    end: str,
    freq: str = "B",  # business days
) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, end=end, freq=freq)


def align_to_calendar(
    df: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    ffill_limit: int = 3,
) -> pd.DataFrame:
    df = df.reindex(calendar)
    df = df.ffill(limit=ffill_limit)
    return df


def align_all(
    data_map: dict[str, pd.DataFrame],
    start: str | None = None,
    end: str | None = None,
    ffill_limit: int = 3,
    min_coverage: float = 0.7,
) -> dict[str, pd.DataFrame]:
    if not data_map:
        return {}

    # Determine common date range
    if start is None:
        start = max(df.index[0] for df in data_map.values() if not df.empty).strftime("%Y-%m-%d")
    if end is None:
        end = min(df.index[-1] for df in data_map.values() if not df.empty).strftime("%Y-%m-%d")

    calendar = build_trading_calendar(start, end)
    logger.info("aligning_data", start=start, end=end, calendar_days=len(calendar))

    aligned: dict[str, pd.DataFrame] = {}
    for ticker, df in data_map.items():
        if df.empty:
            continue
        aligned_df = align_to_calendar(df, calendar, ffill_limit=ffill_limit)
        coverage = aligned_df["close"].notna().mean() if "close" in aligned_df.columns else 0
        if coverage < min_coverage:
            logger.warning("low_coverage_dropped", ticker=ticker, coverage=f"{coverage:.1%}")
            continue
        aligned[ticker] = aligned_df

    logger.info("alignment_complete", tickers=len(aligned))
    return aligned
