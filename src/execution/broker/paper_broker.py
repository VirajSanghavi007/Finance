from __future__ import annotations

import uuid
from datetime import datetime, timezone

import numpy as np

from src.execution.broker.base_broker import BaseBroker, Order, Position
from src.backtest.costs import transaction_cost
from src.config.logging_config import get_logger

logger = get_logger(__name__)


class PaperBroker(BaseBroker):
    """
    Simulated broker for paper trading.
    Fills market orders at next-bar open (caller provides fill price).
    Applies realistic transaction costs from backtest.costs.
    """

    def __init__(self, initial_cash: float = 100_000.0) -> None:
        self._cash       = initial_cash
        self._positions: dict[str, Position] = {}
        self._orders:    dict[str, Order]    = {}

    def submit_order(self, order: Order) -> Order:
        order.order_id = order.order_id or str(uuid.uuid4())[:8]
        self._orders[order.order_id] = order
        logger.info("paper_order_submitted", **{
            "id": order.order_id, "ticker": order.ticker,
            "side": order.side, "qty": order.qty,
        })
        return order

    def fill_order(self, order_id: str, fill_price: float) -> Order:
        """
        Fill a pending order at the given price (called by paper trader at next open).
        Deducts cash + transaction costs.
        """
        order = self._orders.get(order_id)
        if order is None or order.status != "pending":
            return order or Order(order_id=order_id, ticker="?", side="?", qty=0)

        trade_value = fill_price * order.qty
        cost = transaction_cost(trade_value, asset_type="equity",
                                realized_vol=0.20, avg_daily_volume=1_000_000)

        if order.side == "buy":
            total_debit = trade_value + cost
            if total_debit > self._cash:
                # Partial fill — buy only what we can afford
                affordable_qty = (self._cash * 0.999) / (fill_price + cost / order.qty)
                order.qty = max(0.0, affordable_qty)
                trade_value = fill_price * order.qty
                cost = transaction_cost(trade_value, asset_type="equity",
                                realized_vol=0.20, avg_daily_volume=1_000_000)
                total_debit = trade_value + cost
            self._cash -= total_debit

        elif order.side == "sell":
            self._cash += trade_value - cost

        # Update position
        self._update_position(order.ticker, order.side, order.qty, fill_price)

        order.filled_price = fill_price
        order.filled_qty   = order.qty
        order.status       = "filled"
        order.filled_at    = datetime.now(timezone.utc)
        logger.info("paper_order_filled", id=order.order_id,
                    price=fill_price, qty=order.qty)
        return order

    def _update_position(self, ticker: str, side: str, qty: float, price: float) -> None:
        pos = self._positions.get(ticker, Position(ticker=ticker, qty=0.0, avg_price=0.0))
        if side == "buy":
            total_qty   = pos.qty + qty
            pos.avg_price = (pos.avg_price * pos.qty + price * qty) / (total_qty or 1)
            pos.qty = total_qty
        elif side == "sell":
            pos.qty -= qty
        if abs(pos.qty) < 1e-6:
            self._positions.pop(ticker, None)
        else:
            self._positions[ticker] = pos

    def cancel_order(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if order and order.status == "pending":
            order.status = "cancelled"
            return True
        return False

    def get_positions(self) -> dict[str, Position]:
        return dict(self._positions)

    def update_market_prices(self, prices: dict[str, float]) -> None:
        """Call each bar to refresh unrealized PnL."""
        for ticker, pos in self._positions.items():
            price = prices.get(ticker, pos.avg_price)
            pos.market_value    = price * pos.qty
            pos.unrealized_pnl  = (price - pos.avg_price) * pos.qty

    def get_portfolio_value(self) -> float:
        market_value = sum(p.market_value for p in self._positions.values())
        return self._cash + market_value

    def get_cash(self) -> float:
        return self._cash
