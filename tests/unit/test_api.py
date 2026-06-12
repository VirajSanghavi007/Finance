"""Unit tests for Phase 8: FastAPI backend."""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")


from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_signal_endpoint():
    r = client.get("/api/v1/signals/SPY")
    assert r.status_code == 200
    data = r.json()
    assert data["ticker"] == "SPY"
    assert data["signal"] in (-1, 0, 1)


def test_signal_all():
    r = client.get("/api/v1/signals/")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_portfolio_endpoint():
    r = client.get("/api/v1/portfolio/")
    assert r.status_code == 200
    data = r.json()
    assert "portfolio_value" in data
    assert "positions" in data


def test_risk_metrics_endpoint():
    r = client.get("/api/v1/risk/metrics")
    assert r.status_code == 200
    data = r.json()
    assert "circuit_open" in data


def test_models_endpoint():
    r = client.get("/api/v1/models/")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_unknown_model_404():
    r = client.get("/api/v1/models/nonexistent_model_xyz")
    assert r.status_code == 404
