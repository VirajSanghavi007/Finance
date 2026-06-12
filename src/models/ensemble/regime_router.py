from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from src.config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class RegimeWeights:
    """Per-model weights for each market regime (0=low-vol, 1=mid-vol, 2=high-vol)."""
    weights: dict[int, dict[str, float]] = field(default_factory=dict)

    def get(self, regime: int, model_names: list[str]) -> np.ndarray:
        """Return normalized weight vector for a regime."""
        if regime not in self.weights:
            return np.ones(len(model_names)) / len(model_names)
        w_map = self.weights[regime]
        w = np.array([w_map.get(m, 1.0) for m in model_names], dtype=float)
        return w / (w.sum() + 1e-8)


class RegimeRouter:
    """
    Routes ensemble weighting based on current market regime.

    In low-volatility (regime 0): upweight trend-following models (LSTM, TCN)
    In mid-volatility (regime 1): equal-weight
    In high-volatility (regime 2): upweight conservative models (RF, XGB) + RL
    """

    _DEFAULT_WEIGHTS = {
        0: {  # low vol — trend-following
            "xgb": 1.2, "lgbm": 1.2, "rf": 0.8,
            "lstm": 1.5, "tcn": 1.5, "patchtst": 1.3,
            "ppo": 1.0, "nbeats": 1.1,
        },
        1: {  # mid vol — balanced
            "xgb": 1.0, "lgbm": 1.0, "rf": 1.0,
            "lstm": 1.0, "tcn": 1.0, "patchtst": 1.0,
            "ppo": 1.0, "nbeats": 1.0,
        },
        2: {  # high vol — conservative + RL
            "xgb": 1.3, "lgbm": 1.0, "rf": 1.3,
            "lstm": 0.7, "tcn": 0.7, "patchtst": 0.8,
            "ppo": 1.4, "nbeats": 0.9,
        },
    }

    def __init__(self, custom_weights: dict | None = None) -> None:
        self._regime_weights = RegimeWeights(
            weights=custom_weights or self._DEFAULT_WEIGHTS
        )

    def get_weights(self, regime: int, model_names: list[str]) -> np.ndarray:
        return self._regime_weights.get(regime, model_names)

    def blend(
        self,
        probas: dict[str, np.ndarray],  # model_name → (N, 3)
        regime_series: np.ndarray,       # length N, values 0/1/2
    ) -> np.ndarray:
        """Produce regime-weighted blended probability array of shape (N, 3)."""
        model_names = list(probas.keys())
        stacked = np.stack(list(probas.values()), axis=0)  # (M, N, 3)
        N = stacked.shape[1]
        blended = np.zeros((N, 3))

        for i in range(N):
            regime = int(regime_series[i]) if i < len(regime_series) else 1
            w = self.get_weights(regime, model_names)  # (M,)
            # weighted sum over models
            blended[i] = np.einsum("m,mc->c", w, stacked[:, i, :])

        # normalize rows
        row_sums = blended.sum(axis=1, keepdims=True)
        return blended / (row_sums + 1e-8)
