from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ModelPrediction:
    signal:           int            # -1 (short), 0 (flat), 1 (long)
    confidence:       float          # [0.0, 1.0]
    raw_proba:        np.ndarray     # shape (3,): [P(short), P(flat), P(long)]
    model_name:       str
    timestamp:        pd.Timestamp
    feature_snapshot: dict           # top 10 features + values at prediction time


class BaseModel(ABC):
    """Abstract interface all AlgoTrade models must implement."""

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val:   pd.DataFrame,
        y_val:   pd.Series,
        sample_weights: np.ndarray | None = None,
    ) -> dict:
        """Train the model. Returns dict of training metrics."""
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted class labels: array of -1, 0, 1."""
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability array, shape (n_samples, 3): [P(short), P(flat), P(long)]."""
        ...

    @abstractmethod
    def get_feature_importance(self) -> pd.Series:
        """Return feature importances sorted descending."""
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist model artifacts to disk."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseModel":
        """Load a saved model from disk."""
        ...

    def predict_with_metadata(
        self,
        X: pd.DataFrame,
        timestamp: pd.Timestamp,
    ) -> ModelPrediction:
        proba = self.predict_proba(X)
        if proba.ndim == 2:
            proba = proba[0]  # single-row input
        signal     = int(np.argmax(proba) - 1)   # [0,1,2] → [-1,0,1]
        confidence = float(proba.max())
        top_features = self.get_feature_importance().nlargest(10).to_dict()
        return ModelPrediction(
            signal=signal,
            confidence=confidence,
            raw_proba=proba,
            model_name=self.__class__.__name__,
            timestamp=timestamp,
            feature_snapshot=top_features,
        )

    @property
    def name(self) -> str:
        return self.__class__.__name__
