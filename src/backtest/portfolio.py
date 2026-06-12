from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class Position:
    ticker:        str
    quantity:      float        # positive = long, negative = short
    entry_price:   float
    entry_date:    pd.Timestamp
    asset_type:    str = "equity"

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    def market_value(self, current_price: float) -> float:
        return self.quantity * current_price

    def unrealized_pnl(self, current_price: float) -> float:
        return self.quantity * (current_price - self.entry_price)

    def unrealized_pnl_pct(self, current_price: float) -> float:
        if self.entry_price == 0:
            return 0.0
        return (current_price - self.entry_price) / self.entry_price


@dataclass
class Trade:
    ticker:      str
    direction:   str        # "long" or "short"
    entry_price: float
    exit_price:  float
    quantity:    float
    entry_date:  pd.Timestamp
    exit_date:   pd.Timestamp
    cost:        float
    asset_type:  str = "equity"
    signal:      int = 0
    confidence:  float = 0.0
    model_name:  str = ""

    @property
    def pnl(self) -> float:
        sign = 1 if self.direction == "long" else -1
        return sign * self.quantity * (self.exit_price - self.entry_price) - self.cost

    @property
    def holding_days(self) -> int:
        return max(1, (self.exit_date - self.entry_date).days)


class Portfolio:
    def __init__(self, initial_capital: float = 100_000.0) -> None:
        self.initial_capital = initial_capital
        self.cash            = initial_capital
        self.positions: dict[str, Position] = {}
        self.trades:    list[Trade]         = []
        self.equity_curve: list[tuple[pd.Timestamp, float]] = []
        self._peak_value = initial_capital

    @property
    def total_value(self) -> float:
        return self.cash  # positions added via mark_to_market

    def mark_to_market(self, prices: dict[str, float], date: pd.Timestamp) -> float:
        pos_value = sum(
            pos.market_value(prices[t])
            for t, pos in self.positions.items()
            if t in prices
        )
        total = self.cash + pos_value
        self.equity_curve.append((date, total))
        if total > self._peak_value:
            self._peak_value = total
        return total

    @property
    def drawdown(self) -> float:
        if self._peak_value == 0:
            return 0.0
        current = self.equity_curve[-1][1] if self.equity_curve else self.initial_capital
        return (current - self._peak_value) / self._peak_value

    def open_position(
        self,
        ticker: str,
        quantity: float,
        price: float,
        date: pd.Timestamp,
        cost: float,
        asset_type: str = "equity",
        signal: int = 0,
        confidence: float = 0.0,
        model_name: str = "",
    ) -> None:
        trade_value = abs(quantity) * price
        self.cash -= (trade_value + cost) if quantity > 0 else -(trade_value - cost)
        self.positions[ticker] = Position(
            ticker=ticker, quantity=quantity,
            entry_price=price, entry_date=date,
            asset_type=asset_type,
        )
        logger.debug(
            "position_opened", ticker=ticker, qty=quantity,
            price=price, cost=cost, signal=signal,
        )

    def close_position(
        self,
        ticker: str,
        price: float,
        date: pd.Timestamp,
        cost: float,
    ) -> Optional[Trade]:
        pos = self.positions.pop(ticker, None)
        if pos is None:
            return None

        # Cash adjustment on close
        trade_value = abs(pos.quantity) * price
        if pos.quantity > 0:   # closing long: receive cash
            self.cash += trade_value - cost
        else:                  # closing short: pay cash
            self.cash -= trade_value + cost

        trade = Trade(
            ticker=ticker,
            direction="long" if pos.quantity > 0 else "short",
            entry_price=pos.entry_price,
            exit_price=price,
            quantity=abs(pos.quantity),
            entry_date=pos.entry_date,
            exit_date=date,
            cost=cost,
            asset_type=pos.asset_type,
        )
        self.trades.append(trade)
        logger.debug(
            "position_closed", ticker=ticker, pnl=trade.pnl, days=trade.holding_days
        )
        return trade

    def get_equity_series(self) -> pd.Series:
        if not self.equity_curve:
            return pd.Series(dtype=float)
        idx, vals = zip(*self.equity_curve)
        return pd.Series(vals, index=idx, name="portfolio_value")

    def get_trades_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "ticker":      t.ticker,
                "direction":   t.direction,
                "entry_price": t.entry_price,
                "exit_price":  t.exit_price,
                "quantity":    t.quantity,
                "entry_date":  t.entry_date,
                "exit_date":   t.exit_date,
                "cost":        t.cost,
                "pnl":         t.pnl,
                "holding_days": t.holding_days,
                "signal":      t.signal,
                "confidence":  t.confidence,
                "model":       t.model_name,
            }
            for t in self.trades
        ])

    def position_weights(self, prices: dict[str, float]) -> dict[str, float]:
        total = self.cash + sum(
            pos.market_value(prices.get(t, pos.entry_price))
            for t, pos in self.positions.items()
        )
        if total == 0:
            return {}
        return {
            t: pos.market_value(prices.get(t, pos.entry_price)) / total
            for t, pos in self.positions.items()
        }
