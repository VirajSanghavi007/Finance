from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

from src.models.base import BaseModel
from src.config.logging_config import get_logger

logger = get_logger(__name__)

LABEL_MAP   = {-1: 0, 0: 1, 1: 2}
LABEL_UNMAP = {0: -1, 1: 0, 2: 1}


class RandomForestModel(BaseModel):
    """Random Forest 3-class classifier — diversity member in ensemble."""

    def __init__(self, n_estimators: int = 300, max_depth: int = 8) -> None:
        self.n_estimators = n_estimators
        self.max_depth    = max_depth
        self._model: RandomForestClassifier | None = None
        self._feature_names: list[str] = []

    def _remap(self, y: pd.Series) -> pd.Series:
        return y.map(LABEL_MAP).fillna(1).astype(int)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val:   pd.DataFrame,
        y_val:   pd.Series,
        sample_weights: np.ndarray | None = None,
    ) -> dict:
        self._feature_names = list(X_train.columns)
        yt = self._remap(y_train)
        classes = np.array([0, 1, 2])
        weights = compute_class_weight("balanced", classes=classes, y=yt.values)
        weight_map = dict(zip(classes, weights))
        sw = yt.map(weight_map).values

        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
        self._model.fit(X_train.fillna(0), yt, sample_weight=sw)
        val_acc = float((self._model.predict(X_val.fillna(0)) == self._remap(y_val)).mean())
        logger.info("rf_trained", val_acc=f"{val_acc:.3f}")
        return {"val_acc": val_acc}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        mapped = np.argmax(proba, axis=1)
        return np.array([LABEL_UNMAP[int(m)] for m in mapped])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted")
        return self._model.predict_proba(X.fillna(0))

    def get_feature_importance(self) -> pd.Series:
        if self._model is None:
            return pd.Series(dtype=float)
        fi = self._model.feature_importances_
        return pd.Series(fi, index=self._feature_names or range(len(fi))).sort_values(ascending=False)

    def save(self, path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, p / "model.joblib")
        (p / "meta.json").write_text(json.dumps({"feature_names": self._feature_names}))

    @classmethod
    def load(cls, path: str) -> "RandomForestModel":
        p = Path(path)
        obj = cls.__new__(cls)
        obj._model = joblib.load(p / "model.joblib")
        meta = json.loads((p / "meta.json").read_text())
        obj._feature_names = meta["feature_names"]
        obj.n_estimators = 300
        obj.max_depth = 8
        return obj
