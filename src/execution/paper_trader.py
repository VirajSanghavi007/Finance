from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd
import numpy as np

from src.execution.broker.paper_broker import PaperBroker
from src.execution.order_manager import OrderManager
from src.risk.risk_manager import RiskManager
from src.config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class BarResult:
    date: date
    portfolio_value: float
    cash: float
    positions: dict
    orders_submitted: int
    daily_pnl: float


class PaperTrader:
    """
    End-to-end paper trading loop.

    On each bar:
      1. Update market prices on broker
      2. Pass ensemble signals + confidences through RiskManager
      3. Submit rebalance orders via OrderManager
      4. Fill orders at next bar's open (simulated)
      5. Record BarResult
    """

    def __init__(
        self,
        risk_manager: RiskManager,
        initial_cash: float = 100_000.0,
    ) -> None:
        self._broker   = PaperBroker(initial_cash)
        self._om       = OrderManager(self._broker)
        self._risk     = risk_manager
        self._prev_value = initial_cash
        self._history: list[BarResult] = []
        self._pending_fills: list[tuple] = []  # (order_id, fill_price)

    def step(
        self,
        today: date,
        signals: dict[str, int],
        confidences: dict[str, float],
        current_prices: dict[str, float],
        returns_df: pd.DataFrame,
        daily_vols: dict[str, float],
        regime: int = 1,
    ) -> BarResult:
        # Fill any pending orders from previous bar
        for order_id, fill_price in self._pending_fills:
            self._broker.fill_order(order_id, fill_price)
        self._pending_fills.clear()

        # Update market prices
        self._broker.update_market_prices(current_prices)
        portfolio_value = self._broker.get_portfolio_value()
        daily_pnl       = portfolio_value - self._prev_value

        # Risk evaluation
        risk_decision = self._risk.evaluate(
            signals=signals,
            confidences=confidences,
            daily_vols=daily_vols,
            portfolio_value=portfolio_value,
            daily_pnl=daily_pnl,
            returns_df=returns_df,
            regime=regime,
            today=today,
        )

        if risk_decision.approved:
            orders = self._om.rebalance(risk_decision.final_sizes, current_prices)
            # Schedule fills at tomorrow's open (caller must provide next open prices)
            self._pending_fills = [(o.order_id, current_prices.get(o.ticker, 0))
                                   for o in orders if o.status == "pending"]
        else:
            orders = []
            logger.info("paper_trader_halted", reason=risk_decision.reason, date=str(today))

        self._prev_value = portfolio_value
        result = BarResult(
            date=today,
            portfolio_value=portfolio_value,
            cash=self._broker.get_cash(),
            positions={t: p.qty for t, p in self._broker.get_positions().items()},
            orders_submitted=len(orders),
            daily_pnl=daily_pnl,
        )
        self._history.append(result)
        return result

    def history_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"date": r.date, "portfolio_value": r.portfolio_value,
             "cash": r.cash, "daily_pnl": r.daily_pnl}
            for r in self._history
        ])
