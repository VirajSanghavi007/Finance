from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

from src.api.schemas import PortfolioResponse, PortfolioPosition
from src.config.logging_config import get_logger

router = APIRouter(prefix="/portfolio", tags=["portfolio"])
logger = get_logger(__name__)


@router.get("/", response_model=PortfolioResponse)
async def get_portfolio() -> PortfolioResponse:
    from src.api.state import get_state
    state = get_state()
    data  = state.get_portfolio()

    positions = [
        PortfolioPosition(
            ticker=t,
            qty=float(p.get("qty", 0)),
            avg_price=float(p.get("avg_price", 0)),
            market_value=float(p.get("market_value", 0)),
            unrealized_pnl=float(p.get("unrealized_pnl", 0)),
        )
        for t, p in data["positions"].items()
    ]

    return PortfolioResponse(
        portfolio_value=data["portfolio_value"],
        cash=data["cash"],
        positions=positions,
        daily_pnl=data["daily_pnl"],
        as_of=datetime.fromisoformat(data["as_of"]),
    )
