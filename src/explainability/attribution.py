from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SignalAttribution:
    """Records what drove a trading signal at a given timestamp."""
    timestamp: pd.Timestamp
    ticker: str
    signal: int           # -1, 0, 1
    confidence: float
    regime: int
    top_features: dict[str, float]  # feature → contribution score
    model_votes: dict[str, int]     # model_name → predicted signal


class AttributionEngine:
    """
    Builds SignalAttribution records by combining:
      - SHAP feature importances
      - Per-model votes from ensemble
      - Current market regime
    """

    def __init__(
        self,
        shap_analyzer: Any | None = None,
        n_top_features: int = 10,
    ) -> None:
        self._shap          = shap_analyzer
        self.n_top_features = n_top_features

    def attribute(
        self,
        ticker: str,
        timestamp: pd.Timestamp,
        signal: int,
        confidence: float,
        regime: int,
        X_row: pd.Series,
        model_predictions: dict[str, int],
        model: Any | None = None,
    ) -> SignalAttribution:
        top_features: dict[str, float] = {}

        if self._shap is not None and model is not None:
            try:
                df = pd.DataFrame([X_row])
                self._shap.fit(model, df)
                top_series = self._shap.top_features(df, n=self.n_top_features)
                top_features = top_series.to_dict()
            except Exception as e:
                logger.warning("attribution_shap_failed", error=str(e))

        if not top_features and hasattr(model, "get_feature_importance"):
            fi = model.get_feature_importance()
            if not fi.empty:
                top_features = fi.head(self.n_top_features).to_dict()

        return SignalAttribution(
            timestamp=timestamp,
            ticker=ticker,
            signal=signal,
            confidence=confidence,
            regime=regime,
            top_features=top_features,
            model_votes=model_predictions,
        )
