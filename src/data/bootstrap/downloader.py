from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config.assets import EQUITY_UNIVERSE, CRYPTO_UNIVERSE, MACRO_FRED_SERIES
from src.config.constants import (
    RAW_EQUITIES_DIR, RAW_CRYPTO_DIR, RAW_MACRO_DIR,
    RAW_NEWS_DIR, RAW_SEC_DIR,
)
from src.config.logging_config import get_logger
from src.data.bootstrap.equity_loader import download_all_equities, load_equity
from src.data.bootstrap.crypto_loader import download_all_crypto, load_crypto
from src.data.bootstrap.macro_loader import download_all_macro, load_macro
from src.data.bootstrap.news_loader import download_news
from src.data.bootstrap.sec_loader import download_sec_filings
from src.data.pipeline.cleaner import clean_ohlcv, clean_macro
from src.data.pipeline.validator import validate_ohlcv, validate_all
from src.data.pipeline.storage import init_db, upsert_download_status

logger = get_logger(__name__)


def _ensure_dirs() -> None:
    for d in [RAW_EQUITIES_DIR, RAW_CRYPTO_DIR, RAW_MACRO_DIR, RAW_NEWS_DIR, RAW_SEC_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def run_full_download(
    skip_news: bool = False,
    skip_sec: bool = False,
    force_refresh: bool = False,
) -> dict:
    _ensure_dirs()
    init_db()

    summary: dict = {
        "equities": {},
        "crypto": {},
        "macro": {},
        "news": {},
        "sec": {},
        "validation": {},
    }

    # ── Equities ──────────────────────────────────────────────────────────────
    logger.info("downloading_equities", count=len(EQUITY_UNIVERSE))
    eq_results = download_all_equities()
    summary["equities"] = eq_results

    # Clean and validate equities
    loaded: dict[str, pd.DataFrame] = {}
    for ticker in EQUITY_UNIVERSE:
        df = load_equity(ticker)
        if df is not None and not df.empty:
            df = clean_ohlcv(df, ticker=ticker)
            report = validate_ohlcv(df, ticker=ticker)
            summary["validation"][ticker] = report.passed
            if df is not None:
                upsert_download_status(
                    ticker=ticker, asset_type="equity",
                    rows=len(df),
                    date_start=str(df.index[0].date()),
                    date_end=str(df.index[-1].date()),
                )
            loaded[ticker] = df

    # ── Crypto ────────────────────────────────────────────────────────────────
    logger.info("downloading_crypto", count=len(CRYPTO_UNIVERSE))
    try:
        cr_results = download_all_crypto()
        summary["crypto"] = cr_results
        for symbol in CRYPTO_UNIVERSE:
            df = load_crypto(symbol)
            if df is not None and not df.empty:
                df = clean_ohlcv(df, ticker=symbol)
                upsert_download_status(
                    ticker=symbol, asset_type="crypto",
                    rows=len(df),
                    date_start=str(df.index[0].date()),
                    date_end=str(df.index[-1].date()),
                )
    except Exception as exc:
        logger.warning("crypto_download_skipped", error=str(exc))

    # ── Macro ─────────────────────────────────────────────────────────────────
    logger.info("downloading_macro", count=len(MACRO_FRED_SERIES))
    try:
        macro_results = download_all_macro()
        summary["macro"] = macro_results
        for series_id in MACRO_FRED_SERIES:
            df = load_macro(series_id)
            if df is not None and not df.empty:
                df = clean_macro(df, series_id=series_id)
                upsert_download_status(
                    ticker=series_id, asset_type="macro",
                    rows=len(df),
                    date_start=str(df.index[0].date()),
                    date_end=str(df.index[-1].date()),
                )
    except Exception as exc:
        logger.warning("macro_download_skipped", error=str(exc))

    # ── News ──────────────────────────────────────────────────────────────────
    if not skip_news:
        logger.info("downloading_news")
        try:
            news_results = download_news()
            summary["news"] = news_results
        except Exception as exc:
            logger.warning("news_download_skipped", error=str(exc))

    # ── SEC ───────────────────────────────────────────────────────────────────
    if not skip_sec:
        logger.info("downloading_sec")
        try:
            sec_results = download_sec_filings()
            summary["sec"] = sec_results
        except Exception as exc:
            logger.warning("sec_download_skipped", error=str(exc))

    return summary


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 60)
    print("  AlgoTrade  --  Data Download Summary")
    print("=" * 60)

    eq_ok = sum(1 for v in summary["equities"].values() if v > 0)
    eq_total = len(summary["equities"])
    print(f"  Equities : {eq_ok}/{eq_total} downloaded")

    cr_ok = sum(1 for v in summary["crypto"].values() if v > 0)
    cr_total = len(summary["crypto"])
    print(f"  Crypto   : {cr_ok}/{cr_total} downloaded")

    macro_ok = sum(1 for v in summary["macro"].values() if v > 0)
    macro_total = len(summary["macro"])
    print(f"  Macro    : {macro_ok}/{macro_total} FRED series downloaded")

    val_pass = sum(1 for v in summary["validation"].values() if v)
    val_total = len(summary["validation"])
    print(f"  Validated: {val_pass}/{val_total} passed quality checks")

    print("=" * 60 + "\n")
