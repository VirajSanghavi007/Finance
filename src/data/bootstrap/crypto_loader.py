from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config.assets import CRYPTO_UNIVERSE
from src.config.constants import RAW_CRYPTO_DIR
from src.config.logging_config import get_logger

logger = get_logger(__name__)


def _download_single_ccxt(symbol: str, since_ms: int) -> Optional[pd.DataFrame]:
    try:
        import ccxt  # type: ignore

        exchange = ccxt.binance({"enableRateLimit": True})
        all_ohlcv: list = []
        since = since_ms

        while True:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe="1d", since=since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            if len(ohlcv) < 1000:
                break
            since = ohlcv[-1][0] + 1
            time.sleep(exchange.rateLimit / 1000)

        if not all_ohlcv:
            return None

        df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.normalize()
        df = df.drop(columns=["timestamp"]).set_index("date")
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df

    except ImportError:
        logger.error("ccxt_not_installed")
        return None
    except Exception as exc:
        logger.error("crypto_download_failed", symbol=symbol, error=str(exc))
        return None


def download_all_crypto(
    output_dir: Path = RAW_CRYPTO_DIR,
    symbols: Optional[list[str]] = None,
    start_date: str = "2017-01-01",
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    symbols = symbols or list(CRYPTO_UNIVERSE.keys())

    import datetime
    since_ms = int(pd.Timestamp(start_date).timestamp() * 1000)
    results: dict[str, int] = {}

    for symbol in symbols:
        safe_name = symbol.replace("/", "_")
        out_path = output_dir / f"{safe_name}.parquet"

        if out_path.exists():
            existing = pd.read_parquet(out_path)
            last_ts = int(existing.index[-1].timestamp() * 1000)
            df_new = _download_single_ccxt(symbol, last_ts)
            if df_new is not None and not df_new.empty:
                df = pd.concat([existing, df_new])
                df = df[~df.index.duplicated(keep="last")].sort_index()
            else:
                df = existing
        else:
            df = _download_single_ccxt(symbol, since_ms)

        if df is not None and not df.empty:
            df.to_parquet(out_path, compression="snappy")
            results[symbol] = len(df)
            logger.info("crypto_saved", symbol=symbol, rows=len(df),
                        start=str(df.index[0].date()), end=str(df.index[-1].date()))
        else:
            results[symbol] = 0

    logger.info("crypto_download_complete", total=len(symbols),
                success=sum(1 for v in results.values() if v > 0))
    return results


def load_crypto(symbol: str, output_dir: Path = RAW_CRYPTO_DIR) -> Optional[pd.DataFrame]:
    safe_name = symbol.replace("/", "_")
    path = output_dir / f"{safe_name}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)
