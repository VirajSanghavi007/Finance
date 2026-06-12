from __future__ import annotations

import uuid
from typing import Any

import pandas as pd

from src.execution.broker.base_broker import BaseBroker, Order
from src.config.logging_config import get_logger

logger = get_logger(__name__)


class OrderManager:
    """
    Translates risk-approved position sizes into broker orders.

    Flow:
      1. Receive target_weights {ticker: float} (signed fractions of portfolio)
      2. Diff against current positions
      3. Generate buy/sell orders for the delta
      4. Submit via broker
    """

    def __init__(self, broker: BaseBroker, min_order_value: float = 100.0) -> None:
        self._broker = broker
        self.min_order_value = min_order_value

    def rebalance(
        self,
        target_weights: dict[str, float],
        current_prices: dict[str, float],
    ) -> list[Order]:
        """
        Generate and submit orders to move from current to target weights.
        Returns list of submitted Order objects.
        """
        portfolio_value = self._broker.get_portfolio_value()
        current_positions = self._broker.get_positions()

        orders: list[Order] = []

        # Build set of all tickers to consider
        all_tickers = set(target_weights) | set(current_positions)

        for ticker in all_tickers:
            target_weight   = target_weights.get(ticker, 0.0)
            price           = current_prices.get(ticker, 0.0)
            if price <= 0:
                continue

            # Current qty
            current_qty = current_positions.get(ticker)
            current_qty_val = current_qty.qty if current_qty else 0.0

            # Target qty
            target_value = portfolio_value * abs(target_weight)
            target_qty   = target_value / price
            if target_weight < 0:
                target_qty = -target_qty

            delta_qty = target_qty - current_qty_val
            delta_value = abs(delta_qty) * price

            if delta_value < self.min_order_value:
                continue

            side = "buy" if delta_qty > 0 else "sell"
            order = Order(
                order_id=str(uuid.uuid4())[:8],
                ticker=ticker,
                side=side,
                qty=abs(delta_qty),
                order_type="market",
            )
            submitted = self._broker.submit_order(order)
            orders.append(submitted)

        return orders
