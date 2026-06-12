from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

from src.api.schemas import RiskMetricsResponse
from src.config.logging_config import get_logger

router = APIRouter(prefix="/risk", tags=["risk"])
logger = get_logger(__name__)


@router.get("/metrics", response_model=RiskMetricsResponse)
async def get_risk_metrics() -> RiskMetricsResponse:
    from src.api.state import get_state
    state = get_state()
    data  = state.get_risk()
    return RiskMetricsResponse(
        var_1d_99=data["var_1d_99"],
        cvar_1d_99=data["cvar_1d_99"],
        current_drawdown=data["current_drawdown"],
        gross_exposure=data["gross_exposure"],
        circuit_open=data["circuit_open"],
        as_of=datetime.fromisoformat(data["as_of"]),
    )
