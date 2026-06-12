"""
Integration test: all 7 models must implement BaseModel and produce valid signals.
Does NOT require fitted models — tests structural conformance + unfitted fallback paths.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.base import BaseModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(n: int = 200, n_features: int = 10):
    np.random.seed(0)
    dates  = pd.date_range("2020-01-01", periods=n, freq="B")
    close  = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.5), index=dates, name="close")
    X = pd.DataFrame(
        np.random.randn(n, n_features),
        index=dates,
        columns=[f"feat_{i}" for i in range(n_features - 1)] + ["close"],
    )
    X["close"] = close.values
    y = pd.Series(np.random.choice([-1, 0, 1], size=n), index=dates)
    return X, y, close


# ---------------------------------------------------------------------------
# Fixtures: one per model class
# ---------------------------------------------------------------------------

@pytest.fixture
def xgb_model():
    from src.models.classical.xgb_model import XGBoostModel
    return XGBoostModel(n_trials=1)


@pytest.fixture
def lgbm_model():
    from src.models.classical.lgbm_model import LightGBMModel
    return LightGBMModel(n_trials=1)


@pytest.fixture
def rf_model():
    from src.models.classical.random_forest import RandomForestModel
    return RandomForestModel(n_estimators=10, max_depth=3)


@pytest.fixture
def lstm_model():
    from src.models.deep.lstm_model import LSTMModel
    return LSTMModel(seq_len=30, epochs=1, batch_size=32)


@pytest.fixture
def tcn_model():
    from src.models.deep.tcn_model import TCNModel
    return TCNModel(seq_len=30, epochs=1, batch_size=32)


@pytest.fixture
def patchtst_model():
    from src.models.deep.transformer_model import PatchTSTModel
    return PatchTSTModel(seq_len=30, epochs=1, batch_size=32)


@pytest.fixture
def nbeats_model():
    from src.models.deep.nbeats_model import NBeatsModel
    return NBeatsModel(seq_len=30, epochs=1, batch_size=32)


ALL_MODEL_FIXTURES = [
    "xgb_model", "lgbm_model", "rf_model",
    "lstm_model", "tcn_model", "patchtst_model", "nbeats_model",
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_fixture", ALL_MODEL_FIXTURES)
def test_is_base_model(model_fixture, request):
    model = request.getfixturevalue(model_fixture)
    assert isinstance(model, BaseModel), f"{type(model).__name__} must inherit BaseModel"


@pytest.mark.parametrize("model_fixture", ALL_MODEL_FIXTURES)
def test_has_required_methods(model_fixture, request):
    model = request.getfixturevalue(model_fixture)
    for method in ("fit", "predict", "predict_proba", "get_feature_importance", "save", "load"):
        assert hasattr(model, method), f"{type(model).__name__} missing {method}"


@pytest.mark.parametrize("model_fixture", ["rf_model", "xgb_model", "lgbm_model"])
def test_classical_fit_predict(model_fixture, request):
    """Classical models fit without PyTorch — fast test."""
    model = request.getfixturevalue(model_fixture)
    pytest.importorskip("xgboost") if "xgb" in model_fixture else None
    pytest.importorskip("lightgbm") if "lgbm" in model_fixture else None
    X, y, _ = _make_data(n=200, n_features=10)
    split = 150
    X_tr, y_tr = X.iloc[:split], y.iloc[:split]
    X_va, y_va = X.iloc[split:], y.iloc[split:]

    result = model.fit(X_tr, y_tr, X_va, y_va)
    assert isinstance(result, dict)

    preds = model.predict(X_va)
    assert set(np.unique(preds)).issubset({-1, 0, 1}), "Signals must be in {-1,0,1}"
    assert len(preds) == len(X_va)

    proba = model.predict_proba(X_va)
    assert proba.shape == (len(X_va), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.parametrize("model_fixture", ALL_MODEL_FIXTURES)
def test_feature_importance_returns_series(model_fixture, request):
    model = request.getfixturevalue(model_fixture)
    fi = model.get_feature_importance()
    assert isinstance(fi, pd.Series)


@pytest.mark.parametrize("model_fixture", ["rf_model"])
def test_save_load_roundtrip(tmp_path, model_fixture, request):
    model = request.getfixturevalue(model_fixture)
    X, y, _ = _make_data(n=200, n_features=10)
    split = 150
    model.fit(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])

    save_dir = str(tmp_path / "model")
    model.save(save_dir)

    from src.models.classical.random_forest import RandomForestModel
    loaded = RandomForestModel.load(save_dir)
    orig_preds   = model.predict(X.iloc[split:])
    loaded_preds = loaded.predict(X.iloc[split:])
    np.testing.assert_array_equal(orig_preds, loaded_preds)


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("xgboost") or
    not __import__("importlib").util.find_spec("lightgbm"),
    reason="xgboost and lightgbm required"
)
def test_ensemble_predict():
    """Ensemble with 3 fitted classical models produces valid signals."""
    from src.models.classical.xgb_model import XGBoostModel
    from src.models.classical.lgbm_model import LightGBMModel
    from src.models.classical.random_forest import RandomForestModel
    from src.models.ensemble.ensemble import EnsembleModel, EnsembleConfig

    X, y, _ = _make_data(n=200, n_features=10)
    split = 150
    X_tr, y_tr = X.iloc[:split], y.iloc[:split]
    X_va, y_va = X.iloc[split:], y.iloc[split:]

    models = {
        "xgb":  XGBoostModel(n_trials=1),
        "lgbm": LightGBMModel(n_trials=1),
        "rf":   RandomForestModel(n_estimators=10, max_depth=3),
    }
    for m in models.values():
        m.fit(X_tr, y_tr, X_va, y_va)

    cfg = EnsembleConfig(
        use_stacker=False,
        regime_routing=True,
        model_names=list(models.keys()),
        min_confidence=0.0,  # emit all signals in test
    )
    ens = EnsembleModel(models, config=cfg)
    signals = ens.predict(X_va)
    assert set(np.unique(signals)).issubset({-1, 0, 1})
    assert len(signals) == len(X_va)


def test_registry_register_and_load(tmp_path):
    """ModelRegistry can register and reload a fitted RandomForest."""
    from src.models.classical.random_forest import RandomForestModel
    from src.models.registry import ModelRegistry

    X, y, _ = _make_data(n=200, n_features=10)
    split = 150
    model = RandomForestModel(n_estimators=10, max_depth=3)
    model.fit(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])

    registry = ModelRegistry(registry_dir=tmp_path / "registry")
    version  = registry.register(model, "rf_test", metrics={"sharpe_ratio": 1.2})
    assert version == "v1"

    loaded = registry.load_model("rf_test", version)
    assert isinstance(loaded, RandomForestModel)
    preds = loaded.predict(X.iloc[split:])
    assert len(preds) == len(X.iloc[split:])
