from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from src.config.assets import EQUITY_UNIVERSE
from src.config.constants import RAW_EQUITIES_DIR
from src.config.logging_config import get_logger

logger = get_logger(__name__)

# Columns we keep (standardised lowercase)
OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def _download_single(ticker: str, period: str = "max") -> Optional[pd.DataFrame]:
    try:
        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if df.empty:
            logger.warning("empty_download", ticker=ticker)
            return None

        df.columns = [c.lower() for c in df.columns]
        df = df[OHLCV_COLS].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "date"
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df

    except Exception as exc:
        logger.error("download_failed", ticker=ticker, error=str(exc))
        return None


def download_all_equities(
    output_dir: Path = RAW_EQUITIES_DIR,
    tickers: Optional[list[str]] = None,
    period: str = "max",
    delay_seconds: float = 0.5,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tickers = tickers or list(EQUITY_UNIVERSE.keys())
    results: dict[str, int] = {}

    for ticker in tickers:
        safe_name = ticker.replace("/", "_").replace("^", "IDX_")
        out_path = output_dir / f"{safe_name}.parquet"

        if out_path.exists():
            # Only re-download last 30 days to update
            existing = pd.read_parquet(out_path)
            df = _download_single(ticker, period="30d")
            if df is not None and not df.empty:
                df = pd.concat([existing, df])
                df = df[~df.index.duplicated(keep="last")].sort_index()
        else:
            df = _download_single(ticker, period=period)

        if df is not None and not df.empty:
            df.to_parquet(out_path, compression="snappy")
            results[ticker] = len(df)
            logger.info("equity_saved", ticker=ticker, rows=len(df),
                        start=str(df.index[0].date()), end=str(df.index[-1].date()))
        else:
            results[ticker] = 0

        time.sleep(delay_seconds)

    total_ok = sum(1 for v in results.values() if v > 0)
    logger.info("equity_download_complete", total=len(tickers), success=total_ok)
    return results


def load_equity(ticker: str, output_dir: Path = RAW_EQUITIES_DIR) -> Optional[pd.DataFrame]:
    safe_name = ticker.replace("/", "_").replace("^", "IDX_")
    path = output_dir / f"{safe_name}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)
