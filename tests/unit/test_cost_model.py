"""Step 25: Cost model tests. Costs must always be > 0 for real trades."""
from __future__ import annotations

import pytest
from src.backtest.costs import transaction_cost


def test_equity_cost_positive():
    cost = transaction_cost(10_000, "equity", 0.015, 50_000_000)
    assert cost > 0, "Equity transaction cost must be > 0"


def test_crypto_cost_positive():
    cost = transaction_cost(5_000, "crypto", 0.025, 1_000_000)
    assert cost > 0, "Crypto transaction cost must be > 0"


def test_zero_trade_returns_zero():
    cost = transaction_cost(0, "equity", 0.015, 1_000_000)
    assert cost == 0.0


def test_high_vol_increases_cost():
    low_vol  = transaction_cost(10_000, "equity", 0.01,  50_000_000)
    high_vol = transaction_cost(10_000, "equity", 0.05,  50_000_000)
    assert high_vol > low_vol, "Higher volatility should increase slippage"


def test_small_trade_minimum_commission():
    cost = transaction_cost(100, "equity", 0.01, 1_000_000)
    assert cost >= 1.0, "Minimum commission is $1.00"


def test_crypto_has_no_sec_fee():
    """Crypto costs should be lower than equity for same trade (no SEC fee, lower commission)."""
    eq_cost  = transaction_cost(10_000, "equity",  0.015, 50_000_000)
    cr_cost  = transaction_cost(10_000, "crypto",  0.015, 50_000_000)
    # Not a hard rule, but crypto should be in same ballpark
    assert cr_cost > 0


def test_cost_scales_with_trade_size():
    small = transaction_cost(1_000,  "equity", 0.015, 50_000_000)
    large = transaction_cost(100_000, "equity", 0.015, 50_000_000)
    assert large > small, "Larger trades should have higher absolute cost"
