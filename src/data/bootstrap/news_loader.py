from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from src.config.assets import TRADEABLE_EQUITIES
from src.config.constants import RAW_NEWS_DIR
from src.config.logging_config import get_logger
from src.config.settings import get_settings

logger = get_logger(__name__)


def _fetch_newsapi(ticker: str, from_date: str, to_date: str, api_key: str) -> list[dict]:
    try:
        import requests

        url = "https://newsapi.org/v2/everything"
        params = {
            "q": ticker,
            "from": from_date,
            "to": to_date,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 100,
            "apiKey": api_key,
        }
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        articles = data.get("articles", [])
        return [
            {
                "title": a.get("title", ""),
                "description": a.get("description", ""),
                "published_at": a.get("publishedAt", ""),
                "source": a.get("source", {}).get("name", ""),
            }
            for a in articles
        ]
    except Exception as exc:
        logger.warning("newsapi_fetch_failed", ticker=ticker, error=str(exc))
        return []


def download_news(
    output_dir: Path = RAW_NEWS_DIR,
    tickers: Optional[list[str]] = None,
    days_back: int = 30,
) -> dict[str, int]:
    settings = get_settings()
    if not settings.news_api_key:
        logger.warning("news_api_key_missing", msg="Skipping news download")
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)
    tickers = tickers or TRADEABLE_EQUITIES[:10]  # limit to avoid quota burn on first run

    today = datetime.utcnow().date()
    from_date = str(today - timedelta(days=days_back))
    to_date = str(today)
    results: dict[str, int] = {}

    for ticker in tickers:
        out_path = output_dir / f"{ticker}_{to_date}.json"
        if out_path.exists():
            results[ticker] = 1
            continue

        articles = _fetch_newsapi(ticker, from_date, to_date, settings.news_api_key)
        if articles:
            out_path.write_text(json.dumps(articles, indent=2))
            results[ticker] = len(articles)
            logger.info("news_saved", ticker=ticker, count=len(articles))
        else:
            results[ticker] = 0

        time.sleep(1.0)  # respect 100 req/day limit

    return results


def load_news(ticker: str, date: Optional[str] = None,
              output_dir: Path = RAW_NEWS_DIR) -> list[dict]:
    if date is None:
        date = str(datetime.utcnow().date())
    path = output_dir / f"{ticker}_{date}.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []
