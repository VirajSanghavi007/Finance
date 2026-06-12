"""
Application state singleton — shared across API routes.
Loads data from disk on startup and caches in memory.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config.constants import PROCESSED_DIR
from src.config.logging_config import get_logger
from src.data.pipeline.storage import (
    load_features, load_trade_log, get_live_state, set_live_state
)
from src.risk.var_calculator import VaRCalculator
from src.risk.circuit_breaker import CircuitBreaker

logger = get_logger(__name__)

_var_calc = VaRCalculator(confidence=0.99)
_cb       = CircuitBreaker()


class AppState:
    """Holds all live runtime state accessible by API routes."""

    def __init__(self) -> None:
        self._signals:    dict[str, dict]  = {}
        self._features:   dict[str, pd.DataFrame] = {}
        self._equity_curve: pd.Series | None = None
        self._portfolio_value: float = 100_000.0
        self._cash:             float = 100_000.0
        self._positions:        dict[str, dict] = {}
        self._daily_pnl:        float = 0.0
        self._regime:           int   = 1
        self._var_1d:           float = 0.0
        self._cvar_1d:          float = 0.0
        self._drawdown:         float = 0.0
        self._circuit_open:     bool  = False
        self._last_updated:     datetime = datetime.now(timezone.utc)

    # ── Signals ────────────────────────────────────────────────────────────────

    def update_signals(self, ticker: str, signal: int, confidence: float,
                       regime: int, features: dict[str, float]) -> None:
        self._signals[ticker] = {
            "ticker":       ticker,
            "signal":       signal,
            "confidence":   confidence,
            "regime":       regime,
            "top_features": features,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._last_updated = datetime.now(timezone.utc)

    def get_signal(self, ticker: str) -> dict | None:
        return self._signals.get(ticker)

    def get_all_signals(self) -> list[dict]:
        return list(self._signals.values())

    # ── Portfolio ──────────────────────────────────────────────────────────────

    def update_portfolio(self, value: float, cash: float, positions: dict,
                          daily_pnl: float) -> None:
        self._portfolio_value = value
        self._cash            = cash
        self._positions       = positions
        self._daily_pnl       = daily_pnl
        self._last_updated    = datetime.now(timezone.utc)

    def get_portfolio(self) -> dict:
        return {
            "portfolio_value": self._portfolio_value,
            "cash":            self._cash,
            "positions":       self._positions,
            "daily_pnl":       self._daily_pnl,
            "as_of":           self._last_updated.isoformat(),
        }

    # ── Risk ───────────────────────────────────────────────────────────────────

    def update_risk(self, var_1d: float, cvar_1d: float,
                    drawdown: float, circuit_open: bool) -> None:
        self._var_1d        = var_1d
        self._cvar_1d       = cvar_1d
        self._drawdown      = drawdown
        self._circuit_open  = circuit_open

    def get_risk(self) -> dict:
        return {
            "var_1d_99":       self._var_1d,
            "cvar_1d_99":      self._cvar_1d,
            "current_drawdown": self._drawdown,
            "gross_exposure":  sum(abs(p.get("weight", 0)) for p in self._positions.values()),
            "circuit_open":    self._circuit_open,
            "as_of":           self._last_updated.isoformat(),
        }

    # ── Features (for SHAP / explainability) ──────────────────────────────────

    def load_features_for(self, ticker: str) -> pd.DataFrame | None:
        if ticker not in self._features:
            df = load_features(ticker)
            if df is not None:
                self._features[ticker] = df
        return self._features.get(ticker)

    # ── Compute signals from features + a simple RF if available ───────────────

    def compute_live_signals(self) -> None:
        """
        Try to compute signals for all tickers that have feature data on disk.
        Uses RandomForest if available, else returns 0/neutral signal.
        """
        from src.config.assets import EQUITY_UNIVERSE
        from src.data.features.engineer import get_feature_columns

        for ticker in list(EQUITY_UNIVERSE.keys())[:10]:  # limit to top 10 for speed
            feat_df = self.load_features_for(ticker)
            if feat_df is None or len(feat_df) < 60:
                self.update_signals(ticker, 0, 0.0, 1, {})
                continue

            feat_cols = get_feature_columns(feat_df)
            X_last = feat_df[feat_cols].fillna(0).iloc[-1:]

            signal, confidence = 0, 0.0
            try:
                from src.models.registry import ModelRegistry
                reg = ModelRegistry()
                if ticker in reg.list_models():
                    model = reg.load_model(ticker)
                    proba  = model.predict_proba(feat_df[feat_cols].fillna(0).iloc[-60:])
                    last_p = proba[-1]
                    signal     = int(np.argmax(last_p)) - 1
                    confidence = float(last_p.max())
            except Exception:
                pass

            # Top feature importances from last row (simple version)
            row_abs = X_last.iloc[0].abs().sort_values(ascending=False).head(5)
            top_feats = row_abs.to_dict()

            self.update_signals(ticker, signal, confidence, 1, top_feats)


# Global singleton
_state: AppState | None = None


def get_state() -> AppState:
    global _state
    if _state is None:
        _state = AppState()
        _state.compute_live_signals()
    return _state
