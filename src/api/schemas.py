from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime


class SignalResponse(BaseModel):
    ticker: str
    signal: int          # -1, 0, 1
    confidence: float
    regime: int
    top_features: dict[str, float]
    generated_at: datetime


class PortfolioPosition(BaseModel):
    ticker: str
    qty: float
    avg_price: float
    market_value: float
    unrealized_pnl: float


class PortfolioResponse(BaseModel):
    portfolio_value: float
    cash: float
    positions: list[PortfolioPosition]
    daily_pnl: float
    as_of: datetime


class BacktestRequest(BaseModel):
    ticker: str
    start_date: date
    end_date: date
    initial_capital: float = 100_000.0


class BacktestResponse(BaseModel):
    ticker: str
    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    n_trades: int
    metrics: dict[str, Any]


class RiskMetricsResponse(BaseModel):
    var_1d_99: float
    cvar_1d_99: float
    current_drawdown: float
    gross_exposure: float
    circuit_open: bool
    as_of: datetime


class ModelStatusResponse(BaseModel):
    model_name: str
    version: str
    last_trained: datetime | None
    metrics: dict[str, Any]


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
