from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from src.models.base import BaseModel
from src.config.logging_config import get_logger

logger = get_logger(__name__)

LABEL_MAP   = {-1: 0, 0: 1, 1: 2}
LABEL_UNMAP = {0: -1, 1: 0, 2: 1}


class LightGBMModel(BaseModel):
    """
    3-class LightGBM with DART boosting, monotonic constraints, and Optuna tuning.
    """

    def __init__(self, n_trials: int = 100, n_cv_splits: int = 5) -> None:
        self.n_trials    = n_trials
        self.n_cv_splits = n_cv_splits
        self._model: Any | None        = None
        self._feature_names: list[str] = []
        self._best_params: dict        = {}

    def _remap_labels(self, y: pd.Series) -> pd.Series:
        return y.map(LABEL_MAP).fillna(1).astype(int)

    def _build_monotone_constraints(self, feature_names: list[str]) -> list[int]:
        """
        RSI features → constraint=-1 (higher RSI = stronger sell signal)
        Sentiment     → constraint=+1 (better sentiment = stronger buy)
        VIX level     → constraint=-1 (higher VIX = weaker long)
        All others    → 0 (unconstrained)
        """
        constraints = []
        for f in feature_names:
            if "rsi" in f:
                constraints.append(-1)
            elif "sent_score" in f:
                constraints.append(1)
            elif "vix_level" in f:
                constraints.append(-1)
            else:
                constraints.append(0)
        return constraints

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val:   pd.DataFrame,
        y_val:   pd.Series,
        sample_weights: np.ndarray | None = None,
    ) -> dict:
        try:
            import lightgbm as lgb
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError as e:
            raise ImportError(f"lightgbm and optuna required: {e}")

        from src.utils.purged_kfold import PurgedKFold

        self._feature_names = list(X_train.columns)
        y_train_mapped = self._remap_labels(y_train)
        y_val_mapped   = self._remap_labels(y_val)

        classes = np.array([0, 1, 2])
        weights = compute_class_weight("balanced", classes=classes, y=y_train_mapped.values)
        weight_map = dict(zip(classes, weights))
        sw = y_train_mapped.map(weight_map).values

        mono = self._build_monotone_constraints(self._feature_names)

        X_all = pd.concat([X_train, X_val]).fillna(0)
        y_all = pd.concat([y_train_mapped, y_val_mapped])
        tscv  = PurgedKFold(n_splits=self.n_cv_splits, embargo_pct=0.01)

        def objective(trial) -> float:
            params = {
                "n_estimators":          trial.suggest_int("n_estimators", 100, 500),
                "max_depth":             trial.suggest_int("max_depth", 3, 8),
                "learning_rate":         trial.suggest_float("lr", 1e-4, 0.3, log=True),
                "num_leaves":            trial.suggest_int("num_leaves", 20, 200),
                "subsample":             trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree":      trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "reg_alpha":             trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                "reg_lambda":            trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
                "boosting_type":         "dart",
                "objective":             "multiclass",
                "num_class":             3,
                "metric":                "multi_logloss",
                "monotone_constraints":  mono,
                "random_state":          42,
                "verbose":              -1,
            }
            scores = []
            for train_idx, val_idx in tscv.split(X_all):
                clf = lgb.LGBMClassifier(**params)
                clf.fit(
                    X_all.iloc[train_idx].fillna(0), y_all.iloc[train_idx],
                    eval_set=[(X_all.iloc[val_idx].fillna(0), y_all.iloc[val_idx])],
                    callbacks=[lgb.log_evaluation(-1)],
                )
                preds  = clf.predict(X_all.iloc[val_idx].fillna(0))
                aln    = (y_all.iloc[val_idx].values == preds).astype(float) * 2 - 1
                sharpe = float(aln.mean() / aln.std()) if aln.std() > 0 else 0.0
                scores.append(sharpe)
            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        self._best_params = study.best_params

        final_params = {
            **self._best_params,
            "boosting_type":        "dart",
            "objective":            "multiclass",
            "num_class":            3,
            "metric":               "multi_logloss",
            "monotone_constraints": mono,
            "random_state":         42,
            "verbose":             -1,
        }
        self._model = lgb.LGBMClassifier(**final_params)
        self._model.fit(
            X_train.fillna(0), y_train_mapped,
            sample_weight=sw,
            eval_set=[(X_val.fillna(0), y_val_mapped)],
            callbacks=[lgb.log_evaluation(-1)],
        )
        logger.info("lgbm_trained", best_params=self._best_params)
        return {"best_params": self._best_params}

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
        (p / "meta.json").write_text(json.dumps({
            "feature_names": self._feature_names,
            "best_params": self._best_params,
        }))

    @classmethod
    def load(cls, path: str) -> "LightGBMModel":
        p = Path(path)
        obj = cls.__new__(cls)
        obj._model = joblib.load(p / "model.joblib")
        meta = json.loads((p / "meta.json").read_text())
        obj._feature_names = meta["feature_names"]
        obj._best_params   = meta["best_params"]
        obj.n_trials = 100
        obj.n_cv_splits = 5
        return obj
