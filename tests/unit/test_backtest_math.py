"""Step 25: Backtest metric formula verification."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

from src.backtest.metrics import compute_all_metrics, _sharpe, _max_drawdown


def make_equity(returns: list[float], start: float = 100_000) -> pd.Series:
    vals = [start]
    for r in returns:
        vals.append(vals[-1] * (1 + r))
    return pd.Series(vals, index=pd.date_range("2020-01-01", periods=len(vals), freq="B"))


def test_sharpe_zero_for_constant_returns():
    """Constant returns = 0 excess over rf → Sharpe approaches rf-normalised value."""
    returns = pd.Series([0.001] * 252)
    rf_daily = 0.045 / 252
    sharpe = _sharpe(returns, rf_daily)
    # Should be positive since daily return 0.001 > rf_daily
    assert sharpe > 0


def test_sharpe_negative_for_losing_strategy():
    returns = pd.Series([-0.002] * 252)
    rf_daily = 0.045 / 252
    sharpe = _sharpe(returns, rf_daily)
    assert sharpe < 0


def test_max_drawdown_negative():
    equity = make_equity([0.01, -0.05, -0.03, 0.02])
    dd = _max_drawdown(equity)
    assert dd < 0


def test_max_drawdown_zero_for_monotone():
    equity = make_equity([0.01] * 100)
    dd = _max_drawdown(equity)
    assert dd == pytest.approx(0.0, abs=1e-9)


def test_total_return_formula():
    equity = make_equity([0.01] * 252)  # ~252% over 252 days (compounded)
    metrics = compute_all_metrics(equity, pd.DataFrame(), risk_free_rate=0.0)
    expected = (equity.iloc[-1] / equity.iloc[0]) - 1
    assert metrics["total_return"] == pytest.approx(expected, rel=1e-6)


def test_win_rate_correct():
    trades = pd.DataFrame({
        "pnl":          [100, -50, 200, -30, 80],
        "holding_days": [5, 3, 10, 2, 7],
    })
    equity = make_equity([0.001] * 252)
    metrics = compute_all_metrics(equity, trades)
    assert metrics["win_rate"] == pytest.approx(3 / 5, rel=1e-6)


def test_profit_factor():
    trades = pd.DataFrame({
        "pnl":          [100, -50],
        "holding_days": [5, 3],
    })
    equity = make_equity([0.001] * 252)
    metrics = compute_all_metrics(equity, trades)
    assert metrics["profit_factor"] == pytest.approx(2.0, rel=1e-6)


def test_sharpe_ratio_is_headline_not_accuracy():
    """Confirm sharpe_ratio key exists and is a float, not accuracy/F1."""
    equity  = make_equity([0.001] * 252)
    trades  = pd.DataFrame({"pnl": [100], "holding_days": [5]})
    metrics = compute_all_metrics(equity, trades)
    assert "sharpe_ratio" in metrics
    assert isinstance(metrics["sharpe_ratio"], float)
    assert "accuracy" not in metrics
    assert "f1" not in metrics
