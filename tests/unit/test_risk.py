"""Unit tests for Phase 5: Risk Management."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_returns(n: int = 252, seed: int = 42) -> pd.Series:
    np.random.seed(seed)
    return pd.Series(np.random.randn(n) * 0.01)


def _make_returns_df(tickers=("SPY", "QQQ", "AAPL"), n=252) -> pd.DataFrame:
    np.random.seed(0)
    return pd.DataFrame(
        np.random.randn(n, len(tickers)) * 0.01,
        columns=list(tickers),
    )


# ---------------------------------------------------------------------------
# PositionSizer
# ---------------------------------------------------------------------------

class TestPositionSizer:
    def test_zero_signal_returns_zero(self):
        from src.risk.position_sizer import PositionSizer
        ps = PositionSizer()
        assert ps.size(signal=0, confidence=0.9, realized_vol_daily=0.01) == 0.0

    def test_long_signal_positive(self):
        from src.risk.position_sizer import PositionSizer
        ps = PositionSizer()
        s = ps.size(signal=1, confidence=0.8, realized_vol_daily=0.01)
        assert s > 0

    def test_short_signal_negative(self):
        from src.risk.position_sizer import PositionSizer
        ps = PositionSizer()
        s = ps.size(signal=-1, confidence=0.8, realized_vol_daily=0.01)
        assert s < 0

    def test_capped_at_max_position(self):
        from src.risk.position_sizer import PositionSizer
        from src.config.constants import MAX_SINGLE_POSITION
        ps = PositionSizer()
        s = ps.size(signal=1, confidence=1.0, realized_vol_daily=0.001)  # very low vol → big size
        assert abs(s) <= MAX_SINGLE_POSITION + 1e-9

    def test_high_vol_regime_smaller_size(self):
        from src.risk.position_sizer import PositionSizer
        ps = PositionSizer()
        # Use high daily vol so sizes don't saturate at max_position
        low_vol_regime  = ps.size(signal=1, confidence=0.8, realized_vol_daily=0.05, regime=0)
        high_vol_regime = ps.size(signal=1, confidence=0.8, realized_vol_daily=0.05, regime=2)
        assert high_vol_regime < low_vol_regime

    def test_kelly_fraction_valid(self):
        from src.risk.position_sizer import PositionSizer
        ps = PositionSizer()
        k = ps.kelly_fraction(win_rate=0.55, avg_win=0.02, avg_loss=0.01)
        assert 0 < k <= ps.max_position

    def test_kelly_negative_expectancy(self):
        from src.risk.position_sizer import PositionSizer
        ps = PositionSizer()
        k = ps.kelly_fraction(win_rate=0.3, avg_win=0.01, avg_loss=0.05)
        assert k == 0.0


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    def test_not_triggered_initially(self):
        from src.risk.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker()
        assert not cb.is_open

    def test_triggers_on_drawdown(self):
        from src.risk.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker(max_drawdown=0.20)
        cb.update(100_000, 0)     # set peak
        cb.update(79_000, -1000)  # 21% drawdown
        assert cb.is_open
        assert "drawdown" in cb.state.reason

    def test_triggers_on_daily_loss(self):
        from src.risk.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker(max_daily_loss=0.05)
        cb.update(100_000, 0)
        cb.update(100_000, -6000)  # 6% daily loss
        assert cb.is_open

    def test_triggers_on_consecutive_losses(self):
        from src.risk.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker(max_consecutive_losses=3)
        for i in range(4):
            cb.update(100_000 - i * 100, -100)
        assert cb.is_open

    def test_reset_clears_state(self):
        from src.risk.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker(max_drawdown=0.10)
        cb.update(100_000, 0)
        cb.update(85_000, -5000)
        assert cb.is_open
        cb.reset()
        assert not cb.is_open


# ---------------------------------------------------------------------------
# VaRCalculator
# ---------------------------------------------------------------------------

class TestVaRCalculator:
    def test_var_negative(self):
        from src.risk.var_calculator import VaRCalculator
        vc = VaRCalculator(confidence=0.99)
        r  = _make_returns()
        assert vc.historical_var(r) < 0

    def test_cvar_leq_var(self):
        from src.risk.var_calculator import VaRCalculator
        vc = VaRCalculator(confidence=0.99)
        r  = _make_returns()
        assert vc.cvar(r) <= vc.historical_var(r)

    def test_compute_all_keys(self):
        from src.risk.var_calculator import VaRCalculator
        vc = VaRCalculator()
        d  = vc.compute_all(_make_returns())
        expected = {"var_historical", "var_parametric", "var_cornish_fisher", "cvar"}
        assert expected.issubset(d.keys())

    def test_insufficient_data_returns_zero(self):
        from src.risk.var_calculator import VaRCalculator
        vc = VaRCalculator()
        assert vc.historical_var(pd.Series([0.01, -0.01])) == 0.0


# ---------------------------------------------------------------------------
# CorrelationFilter
# ---------------------------------------------------------------------------

class TestCorrelationFilter:
    def test_empty_held_always_allowed(self):
        from src.risk.correlation_filter import CorrelationFilter
        cf = CorrelationFilter()
        df = _make_returns_df()
        assert cf.is_allowed("SPY", [], df)

    def test_high_corr_rejected(self):
        from src.risk.correlation_filter import CorrelationFilter
        cf = CorrelationFilter(max_correlation=0.3)
        df = _make_returns_df()
        # Make QQQ perfectly correlated with SPY
        df["QQQ"] = df["SPY"]
        assert not cf.is_allowed("QQQ", ["SPY"], df)

    def test_low_corr_allowed(self):
        from src.risk.correlation_filter import CorrelationFilter
        cf = CorrelationFilter(max_correlation=0.9)
        df = _make_returns_df()
        assert cf.is_allowed("QQQ", ["SPY"], df)


# ---------------------------------------------------------------------------
# RegimeDetector
# ---------------------------------------------------------------------------

class TestRegimeDetector:
    def _make_price(self, n: int = 300) -> pd.Series:
        np.random.seed(0)
        return pd.Series(100 + np.cumsum(np.random.randn(n) * 0.5))

    def test_vol_regime_values(self):
        from src.risk.regime_detector import RegimeDetector
        rd = RegimeDetector()
        close = self._make_price()
        returns = close.pct_change()
        regime = rd.vol_regime(returns)
        assert set(regime.dropna().unique()).issubset({0, 1, 2})

    def test_trend_regime_values(self):
        from src.risk.regime_detector import RegimeDetector
        rd = RegimeDetector()
        close = self._make_price()
        regime = rd.trend_regime(close)
        assert set(regime.dropna().unique()).issubset({-1, 0, 1})

    def test_detect_returns_df(self):
        from src.risk.regime_detector import RegimeDetector
        rd = RegimeDetector()
        close = self._make_price()
        df = rd.detect(close)
        assert {"vol_regime", "trend_regime", "combined"}.issubset(df.columns)
        assert len(df) == len(close)


# ---------------------------------------------------------------------------
# RiskManager integration
# ---------------------------------------------------------------------------

class TestRiskManager:
    def test_circuit_open_zeroes_all_positions(self):
        from src.risk.risk_manager import RiskManager
        from src.risk.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker(max_drawdown=0.01)  # very tight
        rm = RiskManager(circuit_breaker=cb)

        # Trip the breaker first
        cb.update(100_000, 0)
        cb.update(95_000, -5000)
        assert cb.is_open

        signals = {"SPY": 1, "QQQ": -1}
        confidences = {"SPY": 0.8, "QQQ": 0.8}
        vols = {"SPY": 0.01, "QQQ": 0.01}
        df = _make_returns_df()

        decision = rm.evaluate(signals, confidences, vols, 95_000, -5000, df)
        assert not decision.approved
        assert all(v == 0.0 for v in decision.final_sizes.values())

    def test_normal_operation_returns_nonzero(self):
        from src.risk.risk_manager import RiskManager
        rm = RiskManager()
        signals = {"SPY": 1}
        confidences = {"SPY": 0.8}
        vols = {"SPY": 0.01}
        df = _make_returns_df()

        decision = rm.evaluate(signals, confidences, vols, 100_000, 100, df)
        assert decision.approved
        assert abs(decision.final_sizes.get("SPY", 0)) > 0
