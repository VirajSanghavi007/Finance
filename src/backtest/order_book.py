from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import uuid

import pandas as pd

from src.config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class Order:
    order_id:   str
    ticker:     str
    side:       str          # "buy" or "sell"
    quantity:   float
    order_type: str          # "market" or "limit"
    limit_price: Optional[float]
    created_at: pd.Timestamp
    asset_type: str = "equity"
    signal:     int = 0
    confidence: float = 0.0
    model_name: str = ""

    filled:     bool  = False
    fill_price: Optional[float] = None
    fill_date:  Optional[pd.Timestamp] = None
    cost:       float = 0.0


@dataclass
class Fill:
    order_id:   str
    ticker:     str
    side:       str
    quantity:   float
    fill_price: float
    fill_date:  pd.Timestamp
    cost:       float


class SimOrderBook:
    """
    Simulated order book for backtesting.
    Market orders fill at next-bar open (no bar-of-signal fills).
    Limit orders fill when price crosses the limit.
    """

    def __init__(self) -> None:
        self._pending: list[Order] = []
        self._filled:  list[Fill]  = []

    def submit_market_order(
        self,
        ticker: str,
        side: str,
        quantity: float,
        created_at: pd.Timestamp,
        asset_type: str = "equity",
        signal: int = 0,
        confidence: float = 0.0,
        model_name: str = "",
    ) -> Order:
        order = Order(
            order_id=str(uuid.uuid4())[:8],
            ticker=ticker,
            side=side,
            quantity=quantity,
            order_type="market",
            limit_price=None,
            created_at=created_at,
            asset_type=asset_type,
            signal=signal,
            confidence=confidence,
            model_name=model_name,
        )
        self._pending.append(order)
        logger.debug("order_submitted", ticker=ticker, side=side, qty=quantity)
        return order

    def process_bar(
        self,
        date: pd.Timestamp,
        open_prices: dict[str, float],
        cost_fn,       # callable(trade_value, asset_type, vol, adv) -> float
        vol_map:       dict[str, float] | None = None,
        adv_map:       dict[str, float] | None = None,
    ) -> list[Fill]:
        """
        Process all pending orders against the given open prices.
        Returns list of fills generated this bar.
        Rule: fill at open[t+1] — this is called at the START of the next bar.
        """
        fills: list[Fill] = []
        remaining: list[Order] = []

        for order in self._pending:
            if order.ticker not in open_prices:
                remaining.append(order)
                continue

            fill_price = open_prices[order.ticker]
            trade_value = fill_price * order.quantity
            vol  = (vol_map or {}).get(order.ticker, 0.015)
            adv  = (adv_map or {}).get(order.ticker, trade_value * 10)
            cost = cost_fn(trade_value, order.asset_type, vol, adv)

            order.filled     = True
            order.fill_price = fill_price
            order.fill_date  = date
            order.cost       = cost

            fill = Fill(
                order_id=order.order_id,
                ticker=order.ticker,
                side=order.side,
                quantity=order.quantity,
                fill_price=fill_price,
                fill_date=date,
                cost=cost,
            )
            self._filled.append(fill)
            fills.append(fill)

            logger.debug(
                "order_filled", ticker=order.ticker,
                price=fill_price, cost=cost, date=str(date.date()),
            )

        self._pending = remaining
        return fills

    def cancel_all(self, ticker: Optional[str] = None) -> int:
        if ticker:
            n = sum(1 for o in self._pending if o.ticker == ticker)
            self._pending = [o for o in self._pending if o.ticker != ticker]
        else:
            n = len(self._pending)
            self._pending = []
        return n

    @property
    def pending_count(self) -> int:
        return len(self._pending)
