"""Unit tests for Phase 7: Execution layer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# PaperBroker
# ---------------------------------------------------------------------------

class TestPaperBroker:
    def _broker(self, cash: float = 100_000):
        from src.execution.broker.paper_broker import PaperBroker
        return PaperBroker(initial_cash=cash)

    def test_initial_state(self):
        b = self._broker()
        assert b.get_cash() == 100_000
        assert b.get_portfolio_value() == 100_000
        assert b.get_positions() == {}

    def test_buy_order_reduces_cash(self):
        from src.execution.broker.base_broker import Order
        b = self._broker()
        order = Order(order_id="o1", ticker="SPY", side="buy", qty=10)
        b.submit_order(order)
        b.fill_order("o1", fill_price=400.0)
        assert b.get_cash() < 100_000  # cash reduced

    def test_buy_creates_position(self):
        from src.execution.broker.base_broker import Order
        b = self._broker()
        o = Order(order_id="o2", ticker="QQQ", side="buy", qty=5)
        b.submit_order(o)
        b.fill_order("o2", 300.0)
        pos = b.get_positions()
        assert "QQQ" in pos
        assert abs(pos["QQQ"].qty - 5) < 0.1

    def test_sell_after_buy(self):
        from src.execution.broker.base_broker import Order
        b = self._broker()
        b.submit_order(Order(order_id="b", ticker="AAPL", side="buy", qty=10))
        b.fill_order("b", 150.0)
        cash_after_buy = b.get_cash()

        b.submit_order(Order(order_id="s", ticker="AAPL", side="sell", qty=10))
        b.fill_order("s", 155.0)
        # Position should be gone (or near zero)
        pos = b.get_positions()
        assert "AAPL" not in pos or abs(pos["AAPL"].qty) < 0.01
        assert b.get_cash() > cash_after_buy  # made profit

    def test_cancel_pending_order(self):
        from src.execution.broker.base_broker import Order
        b = self._broker()
        o = Order(order_id="cx", ticker="GLD", side="buy", qty=1)
        b.submit_order(o)
        assert b.cancel_order("cx")

    def test_portfolio_value_updates_with_prices(self):
        from src.execution.broker.base_broker import Order
        b = self._broker()
        b.submit_order(Order(order_id="u", ticker="SPY", side="buy", qty=100))
        b.fill_order("u", 400.0)
        b.update_market_prices({"SPY": 450.0})
        # Portfolio value = cash + (100 * 450)
        assert b.get_portfolio_value() > 100_000


# ---------------------------------------------------------------------------
# OrderManager
# ---------------------------------------------------------------------------

class TestOrderManager:
    def test_rebalance_generates_orders(self):
        from src.execution.broker.paper_broker import PaperBroker
        from src.execution.order_manager import OrderManager
        broker = PaperBroker(100_000)
        om     = OrderManager(broker, min_order_value=10)
        prices = {"SPY": 400.0, "QQQ": 300.0}
        orders = om.rebalance({"SPY": 0.10, "QQQ": -0.05}, prices)
        assert len(orders) > 0
        sides = {o.ticker: o.side for o in orders}
        assert sides.get("SPY") == "buy"

    def test_zero_weights_no_orders(self):
        from src.execution.broker.paper_broker import PaperBroker
        from src.execution.order_manager import OrderManager
        broker = PaperBroker(100_000)
        om     = OrderManager(broker)
        orders = om.rebalance({}, {"SPY": 400.0})
        assert orders == []


# ---------------------------------------------------------------------------
# Rebalancer
# ---------------------------------------------------------------------------

class TestRebalancer:
    def test_daily_always_rebalances(self):
        from src.execution.rebalancer import Rebalancer
        r = Rebalancer(rebalance_freq="daily")
        assert r.should_rebalance(pd.Timestamp("2024-01-01"))

    def test_weekly_only_monday(self):
        from src.execution.rebalancer import Rebalancer
        r = Rebalancer(rebalance_freq="weekly")
        assert r.should_rebalance(pd.Timestamp("2024-01-01"))   # Monday
        assert not r.should_rebalance(pd.Timestamp("2024-01-02"))  # Tuesday

    def test_compute_weights_respects_gross_limit(self):
        from src.execution.rebalancer import Rebalancer
        r = Rebalancer()
        signals = {"A": 1, "B": 1, "C": 1, "D": 1, "E": 1}
        sizes   = {t: 0.4 for t in signals}  # would be 2.0 gross
        weights = r.compute_target_weights(signals, sizes, max_gross=1.5)
        assert sum(abs(w) for w in weights.values()) <= 1.5 + 1e-6

    def test_small_change_filtered(self):
        from src.execution.rebalancer import Rebalancer
        r = Rebalancer(min_weight_change=0.02)
        r._last_weights = {"SPY": 0.10}
        result = r.filter_small_changes({"SPY": 0.105})  # change = 0.005 < 0.02
        assert result["SPY"] == pytest.approx(0.10)  # unchanged
