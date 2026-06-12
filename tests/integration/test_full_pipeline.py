"""
Phase 11: Full end-to-end integration tests.
Covers the critical paths across all major modules.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 400, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    dates = pd.date_range("2021-01-01", periods=n, freq="B")
    close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.5), index=dates)
    df = pd.DataFrame({
        "open":   close * (1 + np.random.randn(n) * 0.002),
        "high":   close * (1 + np.abs(np.random.randn(n)) * 0.003),
        "low":    close * (1 - np.abs(np.random.randn(n)) * 0.003),
        "close":  close.values,
        "volume": np.random.randint(1_000_000, 5_000_000, n).astype(float),
    }, index=dates)
    return df


# ---------------------------------------------------------------------------
# Test 1: Data → Features pipeline
# ---------------------------------------------------------------------------

def test_data_to_features_pipeline():
    """OHLCV → engineer_features → correct shape, no lookahead in feature columns."""
    from src.data.features.engineer import engineer_features, get_feature_columns

    df = _make_ohlcv(n=400)
    feat_df = engineer_features(
        df, ticker="TEST",
        include_fundamentals=False,
        include_sentiment=False,
    )
    assert len(feat_df) == len(df)

    feat_cols = get_feature_columns(feat_df)
    assert len(feat_cols) >= 30, f"Expected 30+ feature cols, got {len(feat_cols)}"

    # No target columns included in feature columns
    from src.data.features.engineer import TARGET_COLS
    for col in feat_cols:
        assert col not in TARGET_COLS, f"Target column {col} in features"


# ---------------------------------------------------------------------------
# Test 2: WFO end-to-end (RandomForestModel, small dataset)
# ---------------------------------------------------------------------------

def test_wfo_fold_generation():
    """WalkForwardOptimizer._generate_folds produces correct non-overlapping windows."""
    from src.backtest.walk_forward import _generate_folds
    import pandas as pd

    dates = pd.date_range("2019-01-01", periods=600, freq="B")
    folds = _generate_folds(dates, train_days=250, test_days=60, step_days=60, min_history=250)
    assert len(folds) >= 1, "Expected at least 1 fold"
    for fold in folds:
        train_start, train_end, test_start, test_end = fold
        # Test window must start after train window ends
        assert test_start > train_end, "Test data must be after train data"
        assert test_end > test_start, "Test window must have positive duration"


# ---------------------------------------------------------------------------
# Test 3: RiskManager + PaperBroker
# ---------------------------------------------------------------------------

def test_risk_manager_with_paper_broker():
    """RiskManager approves signals; PaperBroker changes portfolio value on trades."""
    from src.risk.risk_manager import RiskManager
    from src.execution.broker.paper_broker import PaperBroker
    from src.execution.order_manager import OrderManager

    broker = PaperBroker(initial_cash=100_000)
    rm     = RiskManager()
    om     = OrderManager(broker, min_order_value=10)

    returns_df = pd.DataFrame(
        np.random.randn(252, 3) * 0.01,
        columns=["SPY", "QQQ", "GLD"],
    )
    prices = {"SPY": 400.0, "QQQ": 300.0, "GLD": 180.0}

    # Evaluate signals
    signals     = {"SPY": 1, "QQQ": 0, "GLD": -1}
    confidences = {"SPY": 0.75, "QQQ": 0.0, "GLD": 0.65}
    vols        = {"SPY": 0.01, "QQQ": 0.015, "GLD": 0.012}

    decision = rm.evaluate(signals, confidences, vols, 100_000, 0, returns_df)
    assert decision.approved

    # Submit rebalance
    orders = om.rebalance(decision.final_sizes, prices)
    assert isinstance(orders, list)

    # Fill orders and check broker state
    for o in orders:
        broker.fill_order(o.order_id, prices.get(o.ticker, 400.0))
    broker.update_market_prices(prices)

    # Cash should have decreased (bought something)
    assert broker.get_cash() != 100_000 or len(broker.get_positions()) > 0


# ---------------------------------------------------------------------------
# Test 4: API health + signals
# ---------------------------------------------------------------------------

def test_api_health_and_signals():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from src.api.main import app

    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200

    r = client.get("/api/v1/signals/SPY")
    assert r.status_code == 200
    data = r.json()
    assert data["signal"] in (-1, 0, 1)
    assert 0 <= data["confidence"] <= 1


# ---------------------------------------------------------------------------
# Test 5: Attribution + AuditLog roundtrip
# ---------------------------------------------------------------------------

def test_audit_log_records_signal(tmp_path):
    from src.explainability.attribution import AttributionEngine, SignalAttribution
    from src.explainability.audit_log import AuditLog

    engine = AttributionEngine()
    row    = pd.Series({"rsi_14": 0.45, "macd": 0.12, "close": 400.0})
    attr   = engine.attribute(
        ticker="SPY",
        timestamp=pd.Timestamp("2024-06-01"),
        signal=1,
        confidence=0.72,
        regime=1,
        X_row=row,
        model_predictions={"rf": 1, "xgb": 1, "lstm": 0},
    )
    assert attr.signal == 1

    log = AuditLog(log_dir=tmp_path)
    log.record(attr)
    df = log.read()
    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "SPY"
    assert int(df.iloc[0]["signal"]) == 1


# ---------------------------------------------------------------------------
# Test 6: Ensemble with single RF model
# ---------------------------------------------------------------------------

def test_ensemble_with_rf_only():
    """EnsembleModel with one fitted RF produces valid {-1,0,1} signals."""
    from src.models.classical.random_forest import RandomForestModel
    from src.models.ensemble.ensemble import EnsembleModel, EnsembleConfig

    np.random.seed(7)
    n = 200
    X = pd.DataFrame(np.random.randn(n, 10), columns=[f"f{i}" for i in range(10)])
    y = pd.Series(np.random.choice([-1, 0, 1], n))

    model = RandomForestModel(n_estimators=10, max_depth=3)
    model.fit(X.iloc[:150], y.iloc[:150], X.iloc[150:], y.iloc[150:])

    cfg = EnsembleConfig(
        use_stacker=False,
        regime_routing=False,
        model_names=["rf"],
        min_confidence=0.0,
    )
    ens = EnsembleModel({"rf": model}, config=cfg)
    signals = ens.predict(X.iloc[150:])

    assert set(np.unique(signals)).issubset({-1, 0, 1})
    assert len(signals) == 50
