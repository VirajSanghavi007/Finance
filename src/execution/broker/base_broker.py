from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Order:
    order_id: str
    ticker: str
    side: str       # "buy" | "sell"
    qty: float
    order_type: str = "market"   # "market" | "limit"
    limit_price: float | None = None
    status: str = "pending"      # "pending" | "filled" | "rejected" | "cancelled"
    filled_price: float | None = None
    filled_qty: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    filled_at: datetime | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Position:
    ticker: str
    qty: float          # positive = long, negative = short
    avg_price: float
    market_value: float = 0.0
    unrealized_pnl: float = 0.0


class BaseBroker(ABC):
    """Abstract broker interface for paper and live brokers."""

    @abstractmethod
    def submit_order(self, order: Order) -> Order:
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        ...

    @abstractmethod
    def get_positions(self) -> dict[str, Position]:
        ...

    @abstractmethod
    def get_portfolio_value(self) -> float:
        ...

    @abstractmethod
    def get_cash(self) -> float:
        ...
