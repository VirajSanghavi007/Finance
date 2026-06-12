from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.config.assets import MACRO_FRED_SERIES
from src.config.constants import RAW_MACRO_DIR
from src.config.logging_config import get_logger
from src.config.settings import get_settings

logger = get_logger(__name__)


def _download_fred_series(series_id: str, start: str = "2000-01-01") -> Optional[pd.DataFrame]:
    settings = get_settings()
    try:
        from fredapi import Fred  # type: ignore

        fred = Fred(api_key=settings.fred_api_key) if settings.fred_api_key else Fred()
        s = fred.get_series(series_id, observation_start=start)
        df = s.to_frame(name=series_id.lower())
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "date"
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df = df.dropna()
        return df

    except ImportError:
        logger.warning("fredapi_not_installed_trying_datareader", series=series_id)
        return _download_fred_datareader(series_id, start)
    except Exception as exc:
        logger.error("fred_download_failed", series=series_id, error=str(exc))
        return None


def _download_fred_datareader(series_id: str, start: str) -> Optional[pd.DataFrame]:
    try:
        import pandas_datareader.data as web  # type: ignore

        df = web.DataReader(series_id, "fred", start=start)
        df.columns = [series_id.lower()]
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "date"
        df = df.sort_index().dropna()
        return df
    except Exception as exc:
        logger.error("datareader_failed", series=series_id, error=str(exc))
        return None


def download_all_macro(
    output_dir: Path = RAW_MACRO_DIR,
    series: Optional[dict[str, str]] = None,
    start: str = "2000-01-01",
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    series = series or MACRO_FRED_SERIES
    results: dict[str, int] = {}

    for series_id in series:
        out_path = output_dir / f"{series_id}.parquet"
        df = _download_fred_series(series_id, start)

        if df is not None and not df.empty:
            df.to_parquet(out_path, compression="snappy")
            results[series_id] = len(df)
            logger.info("macro_saved", series=series_id, rows=len(df))
        else:
            results[series_id] = 0

    logger.info("macro_download_complete", total=len(series),
                success=sum(1 for v in results.values() if v > 0))
    return results


def load_macro(series_id: str, output_dir: Path = RAW_MACRO_DIR) -> Optional[pd.DataFrame]:
    path = output_dir / f"{series_id}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def load_all_macro(output_dir: Path = RAW_MACRO_DIR) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for series_id in MACRO_FRED_SERIES:
        df = load_macro(series_id, output_dir)
        if df is not None:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, axis=1)
    combined = combined.sort_index().ffill()
    return combined
