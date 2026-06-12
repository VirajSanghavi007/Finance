from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.logging_config import get_logger

logger = get_logger(__name__)


class Rebalancer:
    """
    Computes target portfolio weights from ensemble signals + sizing.
    Handles rebalance frequency (daily, weekly), transaction cost threshold.
    """

    def __init__(
        self,
        rebalance_freq: str = "daily",  # "daily" | "weekly" | "monthly"
        min_weight_change: float = 0.01,
    ) -> None:
        self.rebalance_freq    = rebalance_freq
        self.min_weight_change = min_weight_change
        self._last_weights: dict[str, float] = {}

    def should_rebalance(self, today: pd.Timestamp) -> bool:
        if self.rebalance_freq == "daily":
            return True
        if self.rebalance_freq == "weekly":
            return today.dayofweek == 0  # Monday
        if self.rebalance_freq == "monthly":
            return today.day == 1
        return True

    def compute_target_weights(
        self,
        signals: dict[str, int],
        sizes: dict[str, float],
        max_gross: float = 1.5,
    ) -> dict[str, float]:
        """
        Returns final portfolio weight dict, scaled to fit max_gross constraint.
        """
        weights = {}
        for ticker, sig in signals.items():
            size = sizes.get(ticker, 0.0)
            weights[ticker] = float(sig) * abs(size) if sig != 0 else 0.0

        gross = sum(abs(w) for w in weights.values())
        if gross > max_gross:
            scale = max_gross / gross
            weights = {t: w * scale for t, w in weights.items()}

        return weights

    def filter_small_changes(
        self,
        target_weights: dict[str, float],
    ) -> dict[str, float]:
        """
        Suppress rebalance orders for tickers where weight change is below threshold.
        Returns adjusted target weights (keeping last weights for small changes).
        """
        result = {}
        for ticker, w in target_weights.items():
            last = self._last_weights.get(ticker, 0.0)
            if abs(w - last) < self.min_weight_change:
                result[ticker] = last   # no change
            else:
                result[ticker] = w
        self._last_weights = dict(result)
        return result
