from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.schemas import ModelStatusResponse
from src.config.logging_config import get_logger

router = APIRouter(prefix="/models", tags=["models"])
logger = get_logger(__name__)


@router.get("/", response_model=list[ModelStatusResponse])
async def list_models() -> list[ModelStatusResponse]:
    """List registered model versions."""
    try:
        from src.models.registry import ModelRegistry
        reg = ModelRegistry()
        result = []
        for name in reg.list_models():
            entry = reg.get_latest_version(name)
            if entry:
                result.append(ModelStatusResponse(
                    model_name=name,
                    version=entry["version"],
                    last_trained=entry.get("registered_at"),
                    metrics=entry.get("metrics", {}),
                ))
        return result
    except Exception as e:
        logger.error("model_list_failed", error=str(e))
        return []


@router.get("/{model_name}", response_model=ModelStatusResponse)
async def get_model(model_name: str) -> ModelStatusResponse:
    from src.models.registry import ModelRegistry
    reg   = ModelRegistry()
    entry = reg.get_latest_version(model_name)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Model {model_name!r} not found")
    return ModelStatusResponse(
        model_name=model_name,
        version=entry["version"],
        last_trained=entry.get("registered_at"),
        metrics=entry.get("metrics", {}),
    )
