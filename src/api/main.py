from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import signals, portfolio, risk, models
from src.api.websocket import signal_feed
from src.api.schemas import HealthResponse
from src.config.logging_config import get_logger

logger = get_logger(__name__)

_VERSION = "0.1.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("api_startup", version=_VERSION)
    yield
    logger.info("api_shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="AlgoTrade API",
        description="Production-grade algorithmic trading platform",
        version=_VERSION,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(signals.router, prefix="/api/v1")
    app.include_router(portfolio.router, prefix="/api/v1")
    app.include_router(risk.router, prefix="/api/v1")
    app.include_router(models.router, prefix="/api/v1")

    @app.get("/health", response_model=HealthResponse, tags=["health"])
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=_VERSION,
            timestamp=datetime.now(timezone.utc),
        )

    @app.websocket("/ws/signals")
    async def ws_signals(ws: WebSocket):
        await signal_feed(ws)

    return app


app = create_app()
