"""
Meta-Labeling (López de Prado, AFML Ch.3)

The idea:
  - A primary model predicts direction: {-1, 0, +1}
  - A secondary (meta) model predicts whether to ACT on that signal: {0=skip, 1=trade}
  - Position size = primary_signal × meta_confidence (bet sizing)

Why it helps:
  - The primary model focuses on direction, which it's trained for.
  - The meta-model learns to recognise when the primary model is likely right.
  - This decomposition typically cuts false positives by 30-50%.

Meta-label construction:
  For each bar where primary_signal != 0:
    meta_label = 1 if trade was profitable, else 0
  Bars where primary_signal == 0 get meta_label = 0 (no trade).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_predict

from src.models.base import BaseModel
from src.config.logging_config import get_logger

logger = get_logger(__name__)


def make_meta_labels(
    primary_signals: pd.Series,
    returns_1d: pd.Series,
) -> pd.Series:
    """
    Build binary meta-labels from primary signals and realised returns.

    meta_label[t] = 1  if primary_signal[t] != 0 AND the trade was profitable
                  = 0  otherwise (flat signal, or trade lost money)

    Args:
        primary_signals : Series of {-1, 0, +1} from any primary model.
        returns_1d      : Series of next-day log returns (already shifted).

    Returns:
        Series of {0, 1} aligned to the same index.
    """
    aligned_ret = returns_1d.reindex(primary_signals.index).fillna(0)
    # Profit = signal agrees with realised direction
    profitable  = (primary_signals * aligned_ret.apply(np.sign)) > 0
    meta        = pd.Series(0, index=primary_signals.index, dtype=int)
    meta[profitable & (primary_signals != 0)] = 1
    return meta


class MetaLabeler:
    """
    Secondary binary classifier that filters primary model signals.

    Usage:
        # 1. Get primary signals
        primary_signals = xgb_model.predict(X_train)

        # 2. Build meta-labels
        meta_y = make_meta_labels(primary_signals, target_ret_1d)

        # 3. Fit meta-labeler on SAME feature set
        ml = MetaLabeler()
        ml.fit(X_train, meta_y, primary_signals)

        # 4. At prediction time:
        final_signal, bet_size = ml.predict_with_size(
            primary_signal=xgb_model.predict(X_test),
            X=X_test,
        )
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int    = 6,
        calibrate: bool   = True,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth    = max_depth
        self.calibrate    = calibrate
        self._model: RandomForestClassifier | CalibratedClassifierCV | None = None
        self._feature_names: list[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        meta_y: pd.Series,
        primary_signals: pd.Series | None = None,
    ) -> dict:
        """
        Train meta-labeler.

        The feature set optionally includes the primary signal as a feature
        (the meta-model can learn "when is XGBoost confident AND right?").
        """
        X_aug = X.copy()
        if primary_signals is not None:
            X_aug = X_aug.copy()
            X_aug["_primary_signal"] = primary_signals.reindex(X.index).fillna(0)

        self._feature_names = list(X_aug.columns)
        y = meta_y.reindex(X_aug.index).fillna(0).astype(int)

        # Only train on bars where primary had a view
        if primary_signals is not None:
            active = primary_signals.reindex(X_aug.index).fillna(0) != 0
            if active.sum() < 50:
                logger.warning("meta_labeler_few_active_bars", n=int(active.sum()))
            X_fit = X_aug[active].fillna(0)
            y_fit = y[active]
        else:
            X_fit = X_aug.fillna(0)
            y_fit = y

        if len(X_fit) < 20:
            logger.warning("meta_labeler_insufficient_data", n=len(X_fit))
            return {"status": "skipped", "n_samples": len(X_fit)}

        base = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        if self.calibrate:
            self._model = CalibratedClassifierCV(base, method="isotonic", cv=3)
        else:
            self._model = base

        self._model.fit(X_fit, y_fit)

        # Accuracy on training set
        acc = float((self._model.predict(X_fit) == y_fit).mean())
        logger.info("meta_labeler_trained", n_samples=len(X_fit), train_acc=f"{acc:.3f}")
        return {"train_acc": acc, "n_samples": len(X_fit)}

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(meta=1) for each bar. Shape: (N,)"""
        if self._model is None:
            return np.ones(len(X)) * 0.5
        X_aug = X.copy()
        if "_primary_signal" not in X_aug.columns and "_primary_signal" in self._feature_names:
            X_aug["_primary_signal"] = 0.0
        # Only use columns the model was trained on
        cols = [c for c in self._feature_names if c in X_aug.columns]
        p = self._model.predict_proba(X_aug[cols].fillna(0))
        # p shape: (N, 2) — return P(class=1) column
        return p[:, 1] if p.shape[1] == 2 else p[:, 0]

    def predict_with_size(
        self,
        primary_signals: np.ndarray,
        X: pd.DataFrame,
        threshold: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Filter primary signals and compute bet sizes.

        Returns:
            final_signals : primary_signal if meta_prob >= threshold, else 0
            bet_sizes     : meta_prob (used to scale Kelly fraction)
        """
        meta_prob = self.predict_proba(X)
        # Apply threshold: only act when meta says "this signal is trustworthy"
        act       = (meta_prob >= threshold).astype(int)
        final     = primary_signals * act
        return final, meta_prob

    def get_feature_importance(self) -> pd.Series:
        if self._model is None:
            return pd.Series(dtype=float)
        base = getattr(self._model, "estimator", self._model)
        if hasattr(base, "feature_importances_"):
            fi = base.feature_importances_
            return pd.Series(fi, index=self._feature_names[:len(fi)]).sort_values(ascending=False)
        return pd.Series(dtype=float)
