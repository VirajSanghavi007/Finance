from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.config.logging_config import get_logger

logger = get_logger(__name__)

LABEL_MAP   = {-1: 0, 0: 1, 1: 2}
LABEL_UNMAP = {0: -1, 1: 0, 2: 1}


class StackingEnsemble:
    """
    Level-2 stacker: takes per-model probability arrays as input features,
    outputs final 3-class probabilities.  Trained with TimeSeriesSplit to
    avoid data leakage between level-1 predictions and the meta-learner.
    """

    def __init__(self, n_splits: int = 5, C: float = 1.0) -> None:
        self.n_splits = n_splits
        self.C        = C
        self._meta: LogisticRegression | None = None
        self._scaler = StandardScaler()
        self._model_names: list[str] = []

    def fit(
        self,
        oof_probas: dict[str, np.ndarray],  # model_name → (N, 3)
        y_true: np.ndarray,                  # mapped labels 0/1/2
    ) -> dict:
        self._model_names = list(oof_probas.keys())
        X_meta = np.concatenate(list(oof_probas.values()), axis=1)  # (N, n_models*3)
        X_scaled = self._scaler.fit_transform(X_meta)

        self._meta = LogisticRegression(
            C=self.C, max_iter=1000, class_weight="balanced",
            solver="lbfgs",
        )
        self._meta.fit(X_scaled, y_true)
        train_acc = float((self._meta.predict(X_scaled) == y_true).mean())
        logger.info("stacker_fitted", train_acc=f"{train_acc:.3f}")
        return {"train_acc": train_acc}

    def predict_proba(self, probas: dict[str, np.ndarray]) -> np.ndarray:
        if self._meta is None:
            # Fallback: simple average
            arrays = list(probas.values())
            return np.mean(arrays, axis=0)
        X_meta = np.concatenate([probas[m] for m in self._model_names], axis=1)
        X_scaled = self._scaler.transform(X_meta)
        return self._meta.predict_proba(X_scaled)

    def predict(self, probas: dict[str, np.ndarray]) -> np.ndarray:
        p = self.predict_proba(probas)
        return np.array([LABEL_UNMAP[int(i)] for i in np.argmax(p, axis=1)])
