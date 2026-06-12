"""Unit tests for Phase 6: Explainability."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path


def _make_df(n: int = 100, n_feat: int = 10) -> pd.DataFrame:
    np.random.seed(0)
    return pd.DataFrame(np.random.randn(n, n_feat),
                        columns=[f"f{i}" for i in range(n_feat)])


# ---------------------------------------------------------------------------
# DriftMonitor
# ---------------------------------------------------------------------------

class TestDriftMonitor:
    def test_no_drift_same_data(self):
        from src.explainability.drift_monitor import DriftMonitor
        dm = DriftMonitor()
        df = _make_df(200)
        dm.fit(df.iloc[:100])
        score = dm.drift_score(df.iloc[100:])
        assert score < 0.5   # same distribution → low drift

    def test_full_drift_different_distribution(self):
        from src.explainability.drift_monitor import DriftMonitor
        dm = DriftMonitor(significance=0.05)
        ref  = pd.DataFrame(np.random.randn(200, 5), columns=[f"f{i}" for i in range(5)])
        live = pd.DataFrame(np.random.randn(200, 5) * 10 + 100, columns=[f"f{i}" for i in range(5)])
        dm.fit(ref)
        score = dm.drift_score(live)
        assert score == 1.0

    def test_no_fit_returns_empty(self):
        from src.explainability.drift_monitor import DriftMonitor
        dm = DriftMonitor()
        assert dm.check(_make_df()) == {}

    def test_drifted_features_list(self):
        from src.explainability.drift_monitor import DriftMonitor
        dm = DriftMonitor()
        ref  = pd.DataFrame({"x": np.random.randn(200), "y": np.random.randn(200)})
        live = pd.DataFrame({"x": np.random.randn(200) * 20 + 50, "y": np.random.randn(200)})
        dm.fit(ref)
        drifted = dm.drifted_features(live)
        assert "x" in drifted


# ---------------------------------------------------------------------------
# AttributionEngine
# ---------------------------------------------------------------------------

class TestAttributionEngine:
    def test_attribution_no_shap(self):
        from src.explainability.attribution import AttributionEngine
        engine = AttributionEngine()
        row    = pd.Series({"f0": 0.1, "f1": -0.3})
        attr   = engine.attribute(
            ticker="SPY",
            timestamp=pd.Timestamp("2024-01-01"),
            signal=1,
            confidence=0.75,
            regime=1,
            X_row=row,
            model_predictions={"rf": 1, "xgb": 1, "lstm": 0},
        )
        assert attr.ticker == "SPY"
        assert attr.signal == 1
        assert attr.confidence == 0.75
        assert attr.model_votes == {"rf": 1, "xgb": 1, "lstm": 0}

    def test_attribution_uses_feature_importance(self):
        from src.explainability.attribution import AttributionEngine
        from src.models.classical.random_forest import RandomForestModel

        X, y = _make_df(200), pd.Series(np.random.choice([-1, 0, 1], 200))
        model = RandomForestModel(n_estimators=10, max_depth=3)
        model.fit(X.iloc[:150], y.iloc[:150], X.iloc[150:], y.iloc[150:])

        engine = AttributionEngine()
        row    = X.iloc[0]
        attr   = engine.attribute(
            ticker="TEST",
            timestamp=pd.Timestamp("2024-01-02"),
            signal=1,
            confidence=0.7,
            regime=0,
            X_row=row,
            model_predictions={"rf": 1},
            model=model,
        )
        assert len(attr.top_features) > 0


# ---------------------------------------------------------------------------
# AuditLog
# ---------------------------------------------------------------------------

class TestAuditLog:
    def test_record_and_read(self, tmp_path):
        from src.explainability.audit_log import AuditLog
        from src.explainability.attribution import SignalAttribution
        log = AuditLog(log_dir=tmp_path)
        attr = SignalAttribution(
            timestamp=pd.Timestamp("2024-01-01"),
            ticker="SPY",
            signal=1,
            confidence=0.8,
            regime=1,
            top_features={"rsi_14": 0.3, "macd": 0.2},
            model_votes={"rf": 1, "xgb": 1},
        )
        log.record(attr)
        df = log.read()
        assert len(df) == 1
        assert df.iloc[0]["ticker"] == "SPY"

    def test_summary_counts(self, tmp_path):
        from src.explainability.audit_log import AuditLog
        from src.explainability.attribution import SignalAttribution
        log = AuditLog(log_dir=tmp_path)
        for sig in [1, -1, 0, 1]:
            attr = SignalAttribution(
                timestamp=pd.Timestamp("2024-01-01"),
                ticker="TEST",
                signal=sig,
                confidence=0.6,
                regime=1,
                top_features={},
                model_votes={},
            )
            log.record(attr)
        s = log.summary()
        assert s["total"] == 4
        assert s["long"] == 2
        assert s["short"] == 1
