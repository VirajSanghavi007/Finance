from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.constants import MAX_CORRELATION
from src.config.logging_config import get_logger

logger = get_logger(__name__)


class CorrelationFilter:
    """
    Prevents adding a position that would be highly correlated with
    existing holdings, controlling for concentration risk.

    Rule: reject new signal if |corr(new_asset, any_held_asset)| > MAX_CORRELATION.
    """

    def __init__(self, max_correlation: float = MAX_CORRELATION) -> None:
        self.max_correlation = max_correlation

    def is_allowed(
        self,
        new_ticker: str,
        held_tickers: list[str],
        returns_df: pd.DataFrame,
        lookback: int = 60,
    ) -> bool:
        """
        Returns True if new_ticker can be added without breaching correlation limit.
        returns_df should contain daily returns for all tickers.
        """
        if not held_tickers:
            return True
        if new_ticker not in returns_df.columns:
            return True

        avail = [t for t in held_tickers if t in returns_df.columns]
        if not avail:
            return True

        recent = returns_df.tail(lookback)
        new_ret = recent[new_ticker].dropna()
        for held in avail:
            held_ret = recent[held].dropna()
            common   = new_ret.index.intersection(held_ret.index)
            if len(common) < 10:
                continue
            corr = float(np.corrcoef(new_ret[common], held_ret[common])[0, 1])
            if abs(corr) > self.max_correlation:
                logger.info(
                    "correlation_filter_rejected",
                    new=new_ticker, held=held, corr=f"{corr:.3f}",
                )
                return False
        return True

    def filter_signals(
        self,
        signals: dict[str, int],
        returns_df: pd.DataFrame,
        lookback: int = 60,
    ) -> dict[str, int]:
        """
        Given a dict of {ticker: signal}, remove signals that breach
        correlation limits, in order of confidence (descending by abs signal).
        Returns filtered signals.
        """
        approved: dict[str, int] = {}
        held: list[str] = []
        for ticker, sig in signals.items():
            if sig == 0:
                approved[ticker] = 0
                continue
            if self.is_allowed(ticker, held, returns_df, lookback):
                approved[ticker] = sig
                held.append(ticker)
            else:
                approved[ticker] = 0  # zeroed out
        return approved
