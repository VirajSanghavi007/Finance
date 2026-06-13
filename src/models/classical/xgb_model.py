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

# Label mapping: XGBoost expects 0,1,2 -- we map from -1,0,1
LABEL_MAP   = {-1: 0, 0: 1, 1: 2}
LABEL_UNMAP = {0: -1, 1: 0, 2: 1}


class XGBoostModel(BaseModel):
    """
    3-class XGBoost classifier with Optuna hyperparameter tuning.
    Classes: 0=short(-1), 1=flat(0), 2=long(+1)
    """

    def __init__(self, n_trials: int = 100, n_cv_splits: int = 5) -> None:
        self.n_trials    = n_trials
        self.n_cv_splits = n_cv_splits
        self._model: Any | None        = None
        self._feature_names: list[str] = []
        self._best_params: dict        = {}
        self._shap_cache: Any | None   = None

    def _remap_labels(self, y: pd.Series) -> pd.Series:
        return y.map(LABEL_MAP).fillna(1).astype(int)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val:   pd.DataFrame,
        y_val:   pd.Series,
        sample_weights: np.ndarray | None = None,
    ) -> dict:
        try:
            import xgboost as xgb
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError as e:
            raise ImportError(f"xgboost and optuna required: {e}")

        from src.utils.purged_kfold import PurgedKFold

        self._feature_names = list(X_train.columns)
        y_train_mapped = self._remap_labels(y_train)
        y_val_mapped   = self._remap_labels(y_val)

        # Class weights
        classes = np.array([0, 1, 2])
        weights = compute_class_weight("balanced", classes=classes, y=y_train_mapped.values)
        weight_map = dict(zip(classes, weights))
        sw = y_train_mapped.map(weight_map).values

        X_all = pd.concat([X_train, X_val]).fillna(0)
        y_all = pd.concat([y_train_mapped, y_val_mapped])

        tscv = PurgedKFold(n_splits=self.n_cv_splits, embargo_pct=0.01)

        def objective(trial) -> float:
            params = {
                "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
                "max_depth":        trial.suggest_int("max_depth", 3, 8),
                "learning_rate":    trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
                "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
                "objective":        "multi:softprob",
                "num_class":        3,
                "eval_metric":      "mlogloss",
                "tree_method":      "hist",
                "random_state":     42,
                "verbosity":        0,
            }
            scores = []
            for train_idx, val_idx in tscv.split(X_all):
                X_tr = X_all.iloc[train_idx].fillna(0)
                y_tr = y_all.iloc[train_idx]
                X_v  = X_all.iloc[val_idx].fillna(0)
                y_v  = y_all.iloc[val_idx]
                clf  = xgb.XGBClassifier(**params)
                clf.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
                preds  = clf.predict(X_v)
                sharpe = self._approx_sharpe(y_v.values, preds)
                scores.append(sharpe)
            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        self._best_params = study.best_params

        # Final fit on full train data
        final_params = {
            **self._best_params,
            "objective":   "multi:softprob",
            "num_class":   3,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "random_state": 42,
            "verbosity":   0,
        }
        self._model = xgb.XGBClassifier(**final_params)
        self._model.fit(
            X_train.fillna(0), y_train_mapped,
            sample_weight=sw,
            eval_set=[(X_val.fillna(0), y_val_mapped)],
            verbose=False,
        )
        self._shap_cache = None

        train_acc = float((self._model.predict(X_train.fillna(0)) == y_train_mapped).mean())
        logger.info("xgb_trained", best_params=self._best_params, train_acc=f"{train_acc:.3f}")
        return {"train_acc": train_acc, "best_params": self._best_params}

    @staticmethod
    def _approx_sharpe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Proxy Sharpe: mean signal alignment, penalised by std."""
        alignment = (y_true == y_pred).astype(float) * 2 - 1
        if alignment.std() == 0:
            return 0.0
        return float(alignment.mean() / alignment.std())

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
        meta = {"feature_names": self._feature_names, "best_params": self._best_params}
        (p / "meta.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path: str) -> "XGBoostModel":
        p = Path(path)
        obj = cls.__new__(cls)
        obj._model = joblib.load(p / "model.joblib")
        meta = json.loads((p / "meta.json").read_text())
        obj._feature_names = meta["feature_names"]
        obj._best_params   = meta["best_params"]
        obj._shap_cache    = None
        obj.n_trials       = 100
        obj.n_cv_splits    = 5
        return obj
