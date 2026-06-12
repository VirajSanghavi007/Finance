from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config.constants import PROJECT_ROOT
from src.config.logging_config import get_logger
from src.models.base import BaseModel

logger = get_logger(__name__)

_REGISTRY_DIR = PROJECT_ROOT / "artifacts" / "model_registry"
_REGISTRY_INDEX = _REGISTRY_DIR / "index.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ModelRegistry:
    """
    Version-aware model registry.  Stores model artifacts under:
        artifacts/model_registry/{model_name}/{version}/
    and maintains an index.json with metadata for each registered version.
    """

    def __init__(self, registry_dir: Path | None = None) -> None:
        self._dir = registry_dir or _REGISTRY_DIR
        self._index_path = self._dir / "index.json"
        self._dir.mkdir(parents=True, exist_ok=True)
        if not self._index_path.exists():
            self._index_path.write_text(json.dumps({}))

    # ------------------------------------------------------------------
    def _load_index(self) -> dict:
        return json.loads(self._index_path.read_text())

    def _save_index(self, index: dict) -> None:
        self._index_path.write_text(json.dumps(index, indent=2))

    # ------------------------------------------------------------------
    def register(
        self,
        model: BaseModel,
        model_name: str,
        metrics: dict[str, Any],
        tags: dict[str, str] | None = None,
    ) -> str:
        """Save model + metadata, return version string."""
        index = self._load_index()
        versions = index.get(model_name, [])
        version = f"v{len(versions) + 1}"

        save_path = self._dir / model_name / version
        model.save(str(save_path))

        entry = {
            "version": version,
            "model_name": model_name,
            "model_class": type(model).__name__,
            "metrics": metrics,
            "tags": tags or {},
            "registered_at": _now_iso(),
            "path": str(save_path),
        }
        versions.append(entry)
        index[model_name] = versions
        self._save_index(index)
        logger.info("model_registered", model=model_name, version=version,
                    sharpe=metrics.get("sharpe_ratio"))
        return version

    def get_latest_version(self, model_name: str) -> dict | None:
        index = self._load_index()
        versions = index.get(model_name, [])
        return versions[-1] if versions else None

    def get_latest_info(self, model_name: str) -> dict | None:
        """Return metadata dict for the latest version (no model loaded)."""
        entry = self.get_latest_version(model_name)
        if entry is None:
            return None
        info = dict(entry)
        info["is_champion"] = "champion" in info.get("tags", {})
        return info

    def get_version(self, model_name: str, version: str) -> dict | None:
        index = self._load_index()
        for entry in index.get(model_name, []):
            if entry["version"] == version:
                return entry
        return None

    def list_models(self) -> list[str]:
        return list(self._load_index().keys())

    def list_versions(self, model_name: str) -> list[dict]:
        return self._load_index().get(model_name, [])

    def load_model(
        self,
        model_name: str,
        version: str | None = None,
    ) -> BaseModel:
        entry = (
            self.get_version(model_name, version) if version
            else self.get_latest_version(model_name)
        )
        if entry is None:
            raise KeyError(f"No registered version for {model_name!r} ({version})")

        class_name = entry["model_class"]
        path       = entry["path"]

        # Dynamic import based on class name
        _CLASS_MAP = {
            "XGBoostModel":     "src.models.classical.xgb_model.XGBoostModel",
            "LightGBMModel":    "src.models.classical.lgbm_model.LightGBMModel",
            "XGBModel":         "src.models.classical.xgb_model.XGBoostModel",
            "LGBMModel":        "src.models.classical.lgbm_model.LightGBMModel",
            "RandomForestModel":"src.models.classical.random_forest.RandomForestModel",
            "LSTMModel":        "src.models.deep.lstm_model.LSTMModel",
            "TCNModel":         "src.models.deep.tcn_model.TCNModel",
            "PatchTSTModel":    "src.models.deep.transformer_model.PatchTSTModel",
            "NBeatsModel":      "src.models.deep.nbeats_model.NBeatsModel",
            "PPOAgent":         "src.models.rl.ppo_agent.PPOAgent",
            "SACAgent":         "src.models.rl.sac_agent.SACAgent",
        }
        full_path = _CLASS_MAP.get(class_name)
        if full_path is None:
            raise ValueError(f"Unknown model class: {class_name}")

        module_path, cls_name = full_path.rsplit(".", 1)
        import importlib
        module = importlib.import_module(module_path)
        cls    = getattr(module, cls_name)
        return cls.load(path)

    def promote_to_champion(self, model_name: str, version: str) -> None:
        """Tag a version as 'champion' — used for live trading."""
        index = self._load_index()
        for entry in index.get(model_name, []):
            entry["tags"].pop("champion", None)  # remove old champion tag
            if entry["version"] == version:
                entry["tags"]["champion"] = "true"
        self._save_index(index)
        logger.info("champion_promoted", model=model_name, version=version)
