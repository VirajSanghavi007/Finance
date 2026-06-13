from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from src.risk.position_sizer import PositionSizer
from src.risk.circuit_breaker import CircuitBreaker
from src.risk.correlation_filter import CorrelationFilter
from src.risk.var_calculator import VaRCalculator
from src.risk.regime_detector import RegimeDetector
from src.config.constants import MAX_GROSS_EXPOSURE, MAX_SECTOR_CONC
from src.config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class RiskDecision:
    approved: bool
    reason: str
    final_sizes: dict[str, float]  # ticker → signed position fraction
    var_1d: float
    circuit_open: bool


class RiskManager:
    """
    Central risk orchestrator that combines all risk checks into a single
    approval decision for a batch of proposed signals.

    Flow per bar:
      1. Circuit breaker check → halt if open
      2. Position sizing via vol-targeting
      3. Correlation filter → zero out correlated additions
      4. Gross exposure cap
      5. Sector concentration cap
      6. Final VaR check
    """

    def __init__(
        self,
        position_sizer:     PositionSizer | None     = None,
        circuit_breaker:    CircuitBreaker | None    = None,
        correlation_filter: CorrelationFilter | None = None,
        var_calculator:     VaRCalculator | None     = None,
        regime_detector:    RegimeDetector | None    = None,
        sector_map:         dict[str, str] | None    = None,
    ) -> None:
        self._sizer  = position_sizer     or PositionSizer()
        self._cb     = circuit_breaker    or CircuitBreaker()
        self._corr   = correlation_filter or CorrelationFilter()
        self._var    = var_calculator     or VaRCalculator()
        self._regime = regime_detector    or RegimeDetector()
        self._sector_map = sector_map or {}

    # ------------------------------------------------------------------
    def evaluate(
        self,
        signals: dict[str, int],            # ticker → {-1,0,1}
        confidences: dict[str, float],       # ticker → [0,1]
        daily_vols: dict[str, float],        # ticker → daily σ
        portfolio_value: float,
        daily_pnl: float,
        returns_df: pd.DataFrame,            # for correlation + VaR
        regime: int = 1,
        today: date | None = None,
    ) -> RiskDecision:

        # 1. Circuit breaker
        circuit_open = self._cb.update(portfolio_value, daily_pnl, today)
        if circuit_open:
            return RiskDecision(
                approved=False,
                reason=f"circuit_open:{self._cb.state.reason}",
                final_sizes={t: 0.0 for t in signals},
                var_1d=0.0,
                circuit_open=True,
            )

        # 2. Position sizing
        sized: dict[str, float] = {}
        for ticker, sig in signals.items():
            conf    = confidences.get(ticker, 0.5)
            vol_d   = daily_vols.get(ticker, 0.02)
            sized[ticker] = self._sizer.size(sig, conf, vol_d, regime)

        # 3. Correlation filter -- only for new long/short entries
        held = [t for t, s in sized.items() if abs(s) > 0.01]
        filtered_signals = self._corr.filter_signals(
            {t: int(np.sign(s)) for t, s in sized.items()},
            returns_df,
        )
        for ticker in list(sized):
            if filtered_signals.get(ticker, 0) == 0 and signals.get(ticker, 0) != 0:
                sized[ticker] = 0.0

        # 4. Gross exposure cap
        gross = sum(abs(s) for s in sized.values())
        if gross > MAX_GROSS_EXPOSURE:
            scale = MAX_GROSS_EXPOSURE / gross
            sized = {t: s * scale for t, s in sized.items()}

        # 5. Sector concentration cap
        sized = self._apply_sector_cap(sized)

        # 6. Portfolio VaR
        port_returns = self._compute_portfolio_returns(sized, returns_df)
        var_1d = abs(self._var.historical_var(port_returns)) if len(port_returns) > 30 else 0.0

        logger.info("risk_evaluation_done", n_approved=sum(1 for s in sized.values() if abs(s) > 0.01))
        return RiskDecision(
            approved=True,
            reason="ok",
            final_sizes=sized,
            var_1d=var_1d,
            circuit_open=False,
        )

    def _apply_sector_cap(self, sized: dict[str, float]) -> dict[str, float]:
        sector_exposure: dict[str, float] = {}
        for ticker, size in sized.items():
            sector = self._sector_map.get(ticker, "other")
            sector_exposure[sector] = sector_exposure.get(sector, 0.0) + abs(size)

        result = dict(sized)
        for sector, exp in sector_exposure.items():
            if exp > MAX_SECTOR_CONC:
                scale = MAX_SECTOR_CONC / exp
                for ticker, size in result.items():
                    if self._sector_map.get(ticker, "other") == sector:
                        result[ticker] = size * scale
        return result

    def _compute_portfolio_returns(
        self, sized: dict[str, float], returns_df: pd.DataFrame
    ) -> np.ndarray:
        tickers = [t for t in sized if t in returns_df.columns and abs(sized[t]) > 0.01]
        if not tickers:
            return np.array([])
        weights = np.array([sized[t] for t in tickers])
        rets    = returns_df[tickers].fillna(0).values
        port_rets = rets @ weights
        return port_rets[-252:] if len(port_rets) > 252 else port_rets
