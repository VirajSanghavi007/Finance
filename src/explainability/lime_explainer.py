from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from src.config.logging_config import get_logger

logger = get_logger(__name__)


class LIMEExplainer:
    """
    LIME-based local explanation for a single prediction.
    Gracefully degrades if lime is not installed.
    """

    def __init__(self, n_samples: int = 1000, n_features: int = 10) -> None:
        self.n_samples  = n_samples
        self.n_features = n_features
        self._explainer: Any | None = None

    def fit(self, feature_names: list[str]) -> None:
        try:
            from lime.lime_tabular import LimeTabularExplainer
            import numpy as np
            self._feature_names = feature_names
        except ImportError:
            logger.warning("lime_not_installed")

    def explain_instance(
        self,
        row: pd.Series | np.ndarray,
        predict_fn: Callable,
        training_data: np.ndarray | None = None,
    ) -> dict[str, float] | None:
        """
        Explain a single prediction.  Returns dict of {feature: weight}.
        """
        try:
            from lime.lime_tabular import LimeTabularExplainer
        except ImportError:
            return None

        if training_data is None:
            training_data = np.random.randn(100, len(self._feature_names))

        explainer = LimeTabularExplainer(
            training_data,
            feature_names=self._feature_names,
            mode="classification",
        )
        x = np.asarray(row).reshape(1, -1).flatten()
        try:
            exp = explainer.explain_instance(
                x, predict_fn,
                num_features=self.n_features,
                num_samples=self.n_samples,
            )
            return dict(exp.as_list())
        except Exception as e:
            logger.warning("lime_explain_failed", error=str(e))
            return None
