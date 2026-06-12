from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from src.api.schemas import SignalResponse
from src.config.logging_config import get_logger

router = APIRouter(prefix="/signals", tags=["signals"])
logger = get_logger(__name__)


@router.get("/{ticker}", response_model=SignalResponse)
async def get_signal(ticker: str) -> SignalResponse:
    ticker = ticker.upper()
    from src.api.state import get_state
    state = get_state()
    sig   = state.get_signal(ticker)
    if sig is None:
        return SignalResponse(
            ticker=ticker, signal=0, confidence=0.0, regime=1,
            top_features={}, generated_at=datetime.now(timezone.utc),
        )
    return SignalResponse(
        ticker=sig["ticker"],
        signal=sig["signal"],
        confidence=sig["confidence"],
        regime=sig["regime"],
        top_features=sig.get("top_features", {}),
        generated_at=datetime.fromisoformat(sig["generated_at"]),
    )


@router.get("/", response_model=list[SignalResponse])
async def get_all_signals() -> list[SignalResponse]:
    from src.api.state import get_state
    state   = get_state()
    records = state.get_all_signals()
    result  = []
    for s in records:
        result.append(SignalResponse(
            ticker=s["ticker"],
            signal=s["signal"],
            confidence=s["confidence"],
            regime=s["regime"],
            top_features=s.get("top_features", {}),
            generated_at=datetime.fromisoformat(s["generated_at"]),
        ))
    return result
