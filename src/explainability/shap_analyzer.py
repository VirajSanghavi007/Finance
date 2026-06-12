from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config.logging_config import get_logger

logger = get_logger(__name__)


class SHAPAnalyzer:
    """
    SHAP-based feature importance and explanation for any scikit-learn-compatible model.
    Gracefully degrades if shap is not installed.
    """

    def __init__(self, max_samples: int = 500) -> None:
        self.max_samples = max_samples
        self._explainer: Any | None = None

    def fit(self, model: Any, X_background: pd.DataFrame) -> None:
        """Build SHAP explainer from background data."""
        try:
            import shap
        except ImportError:
            logger.warning("shap_not_installed")
            return

        bg = X_background.fillna(0)
        if len(bg) > self.max_samples:
            bg = bg.sample(self.max_samples, random_state=42)

        try:
            # TreeExplainer for tree-based models; fallback to KernelExplainer
            if hasattr(model, "get_booster") or hasattr(model, "feature_importances_"):
                self._explainer = shap.TreeExplainer(model)
            else:
                self._explainer = shap.KernelExplainer(model.predict_proba, bg)
        except Exception as e:
            logger.warning("shap_explainer_build_failed", error=str(e))

    def explain(self, X: pd.DataFrame) -> pd.DataFrame | None:
        """
        Returns DataFrame of SHAP values, shape (n_samples, n_features).
        For multi-class, returns values for the predicted class.
        """
        if self._explainer is None:
            return None
        try:
            import shap
        except ImportError:
            return None

        X_clean = X.fillna(0)
        if len(X_clean) > self.max_samples:
            X_clean = X_clean.iloc[:self.max_samples]

        try:
            vals = self._explainer.shap_values(X_clean)
            # Multi-class: vals is a list of (n,f) arrays; take argmax class
            if isinstance(vals, list):
                vals = np.array(vals).mean(axis=0)  # average across classes
            return pd.DataFrame(vals, columns=X_clean.columns, index=X_clean.index)
        except Exception as e:
            logger.warning("shap_explain_failed", error=str(e))
            return None

    def top_features(self, X: pd.DataFrame, n: int = 10) -> pd.Series:
        """Mean absolute SHAP values across samples — top N features."""
        df = self.explain(X)
        if df is None:
            return pd.Series(dtype=float)
        return df.abs().mean().sort_values(ascending=False).head(n)
