from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.models.base import BaseModel, ModelPrediction
from src.models.ensemble.stacker import StackingEnsemble
from src.models.ensemble.regime_router import RegimeRouter
from src.config.logging_config import get_logger

logger = get_logger(__name__)

LABEL_UNMAP = {0: -1, 1: 0, 2: 1}
_MIN_CONFIDENCE = 0.40   # don't trade below this confidence


@dataclass
class EnsembleConfig:
    use_stacker: bool = True
    min_confidence: float = _MIN_CONFIDENCE
    regime_routing: bool = True
    model_names: list[str] = field(default_factory=list)


class EnsembleModel:
    """
    7-model ensemble orchestrator.

    Flow:
      1. Each model produces predict_proba(X) → (N, 3)
      2. Regime router applies regime-aware weights
      3. Optional stacker produces final meta-learner probability
      4. Signal emitted only when confidence ≥ min_confidence
    """

    def __init__(
        self,
        models: dict[str, BaseModel],
        config: EnsembleConfig | None = None,
    ) -> None:
        self._models  = models
        self._config  = config or EnsembleConfig(model_names=list(models.keys()))
        self._stacker = StackingEnsemble()
        self._router  = RegimeRouter()
        self._stacker_fitted = False

    # ------------------------------------------------------------------
    def fit_stacker(
        self,
        oof_probas: dict[str, np.ndarray],
        y_true: np.ndarray,
    ) -> None:
        self._stacker.fit(oof_probas, y_true)
        self._stacker_fitted = True

    # ------------------------------------------------------------------
    def predict_proba(
        self,
        X: pd.DataFrame,
        regime_series: np.ndarray | None = None,
    ) -> np.ndarray:
        model_probas: dict[str, np.ndarray] = {}
        min_n = None

        for name, model in self._models.items():
            try:
                p = model.predict_proba(X)
                model_probas[name] = p
                min_n = len(p) if min_n is None else min(min_n, len(p))
            except Exception as e:
                logger.warning("model_predict_failed", model=name, error=str(e))

        if not model_probas:
            n = max(1, len(X))
            return np.ones((n, 3)) / 3

        # align all to min length
        model_probas = {k: v[-min_n:] for k, v in model_probas.items()}
        N = min_n

        if regime_series is None:
            regime_series = np.ones(N, dtype=int)  # default mid-vol

        if len(regime_series) > N:
            regime_series = regime_series[-N:]

        if self._config.regime_routing:
            blended = self._router.blend(model_probas, regime_series)
        else:
            stacked = np.stack(list(model_probas.values()), axis=0)
            blended = stacked.mean(axis=0)

        if self._config.use_stacker and self._stacker_fitted:
            blended = self._stacker.predict_proba(model_probas)

        # normalize
        row_sums = blended.sum(axis=1, keepdims=True)
        return blended / (row_sums + 1e-8)

    def predict(
        self,
        X: pd.DataFrame,
        regime_series: np.ndarray | None = None,
    ) -> np.ndarray:
        proba   = self.predict_proba(X, regime_series)
        signals = []
        for row in proba:
            best_class = int(np.argmax(row))
            confidence = float(row[best_class])
            if confidence < self._config.min_confidence:
                signals.append(0)  # flat when uncertain
            else:
                signals.append(LABEL_UNMAP[best_class])
        return np.array(signals)

    def predict_with_confidence(
        self,
        X: pd.DataFrame,
        regime_series: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (signals, confidences)."""
        proba = self.predict_proba(X, regime_series)
        best_class  = np.argmax(proba, axis=1)
        confidence  = proba[np.arange(len(proba)), best_class]
        signals = np.where(
            confidence >= self._config.min_confidence,
            np.array([LABEL_UNMAP[int(c)] for c in best_class]),
            0,
        )
        return signals, confidence
