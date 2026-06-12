from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.config.logging_config import get_logger

logger = get_logger(__name__)


def _fetch_yfinance_fundamentals(ticker: str) -> dict:
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = t.info
        return {
            "pe_ratio":          info.get("trailingPE"),
            "pb_ratio":          info.get("priceToBook"),
            "ev_ebitda":         info.get("enterpriseToEbitda"),
            "earnings_surprise": None,  # Requires earnings history (see below)
            "revision_trend":    None,
        }
    except Exception as exc:
        logger.warning("fundamentals_fetch_failed", ticker=ticker, error=str(exc))
        return {}


def _get_earnings_surprise(ticker: str) -> Optional[float]:
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        cal = t.earnings_history
        if cal is None or cal.empty:
            return None
        if "Surprise(%)" in cal.columns:
            return float(cal["Surprise(%)"].iloc[-1]) / 100
        return None
    except Exception:
        return None


def compute_fundamental_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Fundamental features are point-in-time (change quarterly).
    We broadcast the latest available value across the full date index.
    """
    out = pd.DataFrame(index=df.index, dtype=float)

    fundamentals = _fetch_yfinance_fundamentals(ticker)
    pe    = fundamentals.get("pe_ratio")
    pb    = fundamentals.get("pb_ratio")
    evebi = fundamentals.get("ev_ebitda")
    surp  = _get_earnings_surprise(ticker)

    # Broadcast scalar values to entire time series
    # In a real WFO system these would be point-in-time from a database.
    # Here we use the latest available value from yfinance.
    out["fund_pe_ratio"]         = float(pe)    if pe    is not None else np.nan
    out["fund_pb_ratio"]         = float(pb)    if pb    is not None else np.nan
    out["fund_ev_ebitda"]        = float(evebi) if evebi is not None else np.nan
    out["fund_earnings_surprise"] = float(surp) if surp  is not None else np.nan
    out["fund_revision_trend"]   = np.nan  # placeholder — needs analyst revision data

    return out
