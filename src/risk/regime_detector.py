from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.constants import TRADING_DAYS
from src.config.logging_config import get_logger

logger = get_logger(__name__)


class RegimeDetector:
    """
    Online regime detector: low / mid / high volatility (0 / 1 / 2).
    Uses expanding-window quantile on realized vol -- no lookahead.

    Also computes:
      - Trend regime: bullish (1), bearish (-1), sideways (0) via SMA crossover
      - Combined 5-state regime for finer routing
    """

    def __init__(
        self,
        vol_window: int = 21,
        min_periods: int = 21,
        low_q: float = 0.33,
        high_q: float = 0.67,
    ) -> None:
        self.vol_window  = vol_window
        self.min_periods = min_periods
        self.low_q       = low_q
        self.high_q      = high_q

    def vol_regime(self, returns: pd.Series) -> pd.Series:
        """
        Returns 0 (low-vol), 1 (mid-vol), 2 (high-vol) per bar.
        Strictly expanding window -- no lookahead.
        """
        ann_vol = returns.rolling(self.vol_window, min_periods=self.min_periods).std() * np.sqrt(TRADING_DAYS)
        ann_vol = ann_vol.dropna()
        lo = ann_vol.expanding(min_periods=self.min_periods).quantile(self.low_q)
        hi = ann_vol.expanding(min_periods=self.min_periods).quantile(self.high_q)
        regime = pd.Series(1, index=ann_vol.index)  # default mid
        regime[ann_vol < lo] = 0
        regime[ann_vol > hi] = 2
        return regime.reindex(returns.index).fillna(1).astype(int)

    def trend_regime(self, close: pd.Series, fast: int = 20, slow: int = 50) -> pd.Series:
        """
        Simple moving-average crossover: +1 (bullish), -1 (bearish), 0 (sideways).
        """
        sma_fast = close.rolling(fast, min_periods=fast).mean()
        sma_slow = close.rolling(slow, min_periods=slow).mean()
        diff = sma_fast - sma_slow
        regime = pd.Series(0, index=close.index)
        regime[diff > 0] = 1
        regime[diff < 0] = -1
        return regime

    def detect(self, close: pd.Series) -> pd.DataFrame:
        """
        Returns DataFrame with columns: vol_regime, trend_regime, combined.
        combined = vol_regime * 3 + (trend_regime + 1)  → 9 states → grouped to 5
        """
        returns = close.pct_change()
        vol = self.vol_regime(returns)
        trend = self.trend_regime(close)
        df = pd.DataFrame({"vol_regime": vol, "trend_regime": trend}, index=close.index)
        df = df.fillna({"vol_regime": 1, "trend_regime": 0})
        df["combined"] = df["vol_regime"].astype(int) * 3 + (df["trend_regime"].astype(int) + 1)
        return df
