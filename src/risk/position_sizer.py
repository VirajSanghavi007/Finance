from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.constants import (
    MAX_SINGLE_POSITION,
    TRADING_DAYS,
)
from src.config.logging_config import get_logger

logger = get_logger(__name__)

_RISK_PER_TRADE = 0.01   # 1% portfolio risk per trade (Kelly fraction)
_VOL_TARGET     = 0.15   # 15% annualized vol target


class PositionSizer:
    """
    Converts a confidence score + regime + volatility estimate into a
    fractional position size in [0, MAX_SINGLE_POSITION].

    Methods:
      - volatility_targeting: size = (vol_target / asset_vol) * confidence
      - kelly_fraction: size = win_rate − loss_rate / avg_win * avg_loss
    """

    def __init__(
        self,
        vol_target: float = _VOL_TARGET,
        risk_per_trade: float = _RISK_PER_TRADE,
        max_position: float = MAX_SINGLE_POSITION,
    ) -> None:
        self.vol_target    = vol_target
        self.risk_per_trade = risk_per_trade
        self.max_position  = max_position

    def volatility_targeting(
        self,
        confidence: float,
        realized_vol_annual: float,
        regime: int = 1,
    ) -> float:
        """
        Target a fixed annualized volatility contribution.
        Returns position fraction in [0, max_position].
        """
        if realized_vol_annual <= 0:
            return 0.0

        # Regime multiplier: reduce size in high-vol (regime 2)
        regime_scale = {0: 1.1, 1: 1.0, 2: 0.7}.get(regime, 1.0)
        raw_size = (self.vol_target / realized_vol_annual) * confidence * regime_scale
        return float(np.clip(raw_size, 0.0, self.max_position))

    def kelly_fraction(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        half_kelly: bool = True,
    ) -> float:
        """
        Full Kelly: f = W/L − (1−W)/W  where W=win_rate, L=avg_loss, W=avg_win.
        Half-Kelly by default for safety.
        """
        if avg_loss <= 0 or avg_win <= 0:
            return 0.0
        b = avg_win / avg_loss
        kelly = win_rate - (1 - win_rate) / b
        if kelly <= 0:
            return 0.0
        if half_kelly:
            kelly *= 0.5
        return float(np.clip(kelly, 0.0, self.max_position))

    def size(
        self,
        signal: int,
        confidence: float,
        realized_vol_daily: float,
        regime: int = 1,
    ) -> float:
        """
        Main entry: return signed position size given a signal.
        realized_vol_daily is σ in daily terms — converted to annual internally.
        """
        if signal == 0:
            return 0.0
        ann_vol = realized_vol_daily * np.sqrt(TRADING_DAYS)
        frac = self.volatility_targeting(confidence, ann_vol, regime)
        return float(signal) * frac
