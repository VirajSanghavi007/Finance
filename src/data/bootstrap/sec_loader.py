from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from src.config.assets import TRADEABLE_EQUITIES
from src.config.constants import RAW_SEC_DIR
from src.config.logging_config import get_logger
from src.config.settings import get_settings

logger = get_logger(__name__)

EDGAR_COMPANY_SEARCH = "https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt=2020-01-01&forms=10-K,10-Q"
EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
EDGAR_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


def _get_headers(settings=None) -> dict[str, str]:
    if settings is None:
        settings = get_settings()
    return {"User-Agent": settings.sec_user_agent}


def _fetch_cik_map() -> dict[str, str]:
    try:
        import requests

        r = requests.get(EDGAR_TICKERS_URL, headers=_get_headers(), timeout=10)
        r.raise_for_status()
        data = r.json()
        return {v["ticker"].upper(): str(v["cik_str"]) for v in data.values()}
    except Exception as exc:
        logger.error("cik_map_failed", error=str(exc))
        return {}


def _fetch_filings(cik: str) -> list[dict]:
    try:
        import requests

        url = EDGAR_SUBMISSIONS.format(cik=int(cik))
        r = requests.get(url, headers=_get_headers(), timeout=15)
        r.raise_for_status()
        data = r.json()
        filings = data.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        dates = filings.get("filingDate", [])
        accession = filings.get("accessionNumber", [])
        result = []
        for form, date, acc in zip(forms, dates, accession):
            if form in ("10-K", "10-Q", "4"):
                result.append({"form": form, "date": date, "accession": acc})
        return result[:50]  # cap at 50 most recent
    except Exception as exc:
        logger.error("filings_fetch_failed", cik=cik, error=str(exc))
        return []


def download_sec_filings(
    output_dir: Path = RAW_SEC_DIR,
    tickers: Optional[list[str]] = None,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tickers = tickers or TRADEABLE_EQUITIES[:20]
    cik_map = _fetch_cik_map()
    results: dict[str, int] = {}

    for ticker in tickers:
        out_path = output_dir / f"{ticker}_filings.json"
        if out_path.exists():
            results[ticker] = 1
            continue

        cik = cik_map.get(ticker.upper())
        if not cik:
            logger.warning("cik_not_found", ticker=ticker)
            results[ticker] = 0
            continue

        filings = _fetch_filings(cik)
        if filings:
            out_path.write_text(json.dumps({"ticker": ticker, "cik": cik, "filings": filings}, indent=2))
            results[ticker] = len(filings)
            logger.info("sec_saved", ticker=ticker, count=len(filings))
        else:
            results[ticker] = 0

        time.sleep(0.15)  # SEC rate limit: 10 req/sec

    return results
