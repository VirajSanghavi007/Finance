from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from src.config.constants import (
    MAX_PORTFOLIO_DRAWDOWN,
    MAX_DAILY_LOSS,
    MAX_CONSECUTIVE_LOSSES,
)
from src.config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class CircuitBreakerState:
    triggered: bool = False
    reason: str = ""
    triggered_at: date | None = None
    consecutive_losses: int = 0
    peak_value: float = 0.0
    current_drawdown: float = 0.0


class CircuitBreaker:
    """
    Halts trading when risk limits are breached.

    Conditions:
      1. Portfolio drawdown > MAX_PORTFOLIO_DRAWDOWN (20%)
      2. Daily loss > MAX_DAILY_LOSS (5%)
      3. Consecutive losing days > MAX_CONSECUTIVE_LOSSES (7)
    """

    def __init__(
        self,
        max_drawdown: float = MAX_PORTFOLIO_DRAWDOWN,
        max_daily_loss: float = MAX_DAILY_LOSS,
        max_consecutive_losses: int = MAX_CONSECUTIVE_LOSSES,
    ) -> None:
        self.max_drawdown           = max_drawdown
        self.max_daily_loss         = max_daily_loss
        self.max_consecutive_losses = max_consecutive_losses
        self._state = CircuitBreakerState()

    @property
    def is_open(self) -> bool:
        return self._state.triggered

    def update(
        self,
        portfolio_value: float,
        daily_pnl: float,
        today: date | None = None,
    ) -> bool:
        """
        Update state with latest portfolio metrics.
        Returns True if circuit is now open (trading halted).
        """
        state = self._state

        # Track peak
        if portfolio_value > state.peak_value:
            state.peak_value = portfolio_value

        # Compute drawdown
        if state.peak_value > 0:
            state.current_drawdown = (state.peak_value - portfolio_value) / state.peak_value

        # Consecutive losses
        if daily_pnl < 0:
            state.consecutive_losses += 1
        else:
            state.consecutive_losses = 0

        # Check conditions
        if state.current_drawdown >= self.max_drawdown:
            self._trip(f"drawdown={state.current_drawdown:.1%}", today)
        elif daily_pnl / (portfolio_value + 1e-8) <= -self.max_daily_loss:
            self._trip(f"daily_loss={daily_pnl:.2f}", today)
        elif state.consecutive_losses >= self.max_consecutive_losses:
            self._trip(f"consecutive_losses={state.consecutive_losses}", today)

        return state.triggered

    def _trip(self, reason: str, today: date | None) -> None:
        if not self._state.triggered:
            self._state.triggered   = True
            self._state.reason      = reason
            self._state.triggered_at = today
            logger.warning("circuit_breaker_open", reason=reason)

    def reset(self) -> None:
        """Manually reset after review."""
        self._state = CircuitBreakerState(peak_value=self._state.peak_value)
        logger.info("circuit_breaker_reset")

    @property
    def state(self) -> CircuitBreakerState:
        return self._state
