from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config.constants import RAW_NEWS_DIR
from src.config.logging_config import get_logger

logger = get_logger(__name__)

_finbert_pipeline = None


def _get_finbert():
    global _finbert_pipeline
    if _finbert_pipeline is not None:
        return _finbert_pipeline
    try:
        from transformers import pipeline  # type: ignore
        _finbert_pipeline = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            device=-1,  # CPU
            top_k=None,
        )
        logger.info("finbert_loaded")
    except ImportError:
        logger.warning("transformers_not_installed_using_dummy_sentiment")
        _finbert_pipeline = None
    return _finbert_pipeline


def _score_texts(texts: list[str]) -> float:
    """Return mean sentiment score in [-1, 1]."""
    if not texts:
        return 0.0

    pipe = _get_finbert()
    if pipe is None:
        return 0.0  # neutral fallback

    scores = []
    for text in texts[:20]:  # cap at 20 to avoid OOM on CPU
        try:
            result = pipe(text[:512], truncation=True)
            label_scores = {r["label"]: r["score"] for r in result[0]}
            score = label_scores.get("positive", 0) - label_scores.get("negative", 0)
            scores.append(score)
        except Exception:
            pass

    return float(np.mean(scores)) if scores else 0.0


def _load_daily_headlines(ticker: str, date: str) -> list[str]:
    path = RAW_NEWS_DIR / f"{ticker}_{date}.json"
    if not path.exists():
        return []
    try:
        articles = json.loads(path.read_text())
        return [a.get("title", "") + " " + a.get("description", "")
                for a in articles if a.get("title")]
    except Exception:
        return []


def build_sentiment_series(ticker: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Build daily sentiment scores for a ticker over the given date index."""
    out = pd.DataFrame(index=index)
    daily_scores = pd.Series(np.nan, index=index)

    for date in index:
        date_str = str(date.date())
        headlines = _load_daily_headlines(ticker, date_str)
        if headlines:
            daily_scores[date] = _score_texts(headlines)
            out.at[date, "sent_volume_headlines"] = len(headlines)
        else:
            out.at[date, "sent_volume_headlines"] = 0

    # Fill with 0 where no data (neutral)
    daily_scores = daily_scores.fillna(0.0)

    out["sent_score_1d"] = daily_scores
    out["sent_score_3d"] = daily_scores.rolling(3, min_periods=1).mean()
    out["sent_score_7d"] = daily_scores.rolling(7, min_periods=1).mean()
    out["sent_momentum_5d"] = out["sent_score_1d"] - out["sent_score_7d"]

    # Sentiment divergence: price trend vs sentiment trend
    # Will be filled in by engineer.py after close is available
    out["sent_diverge"] = 0.0

    return out


def compute_sentiment_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    return build_sentiment_series(ticker, df.index)
