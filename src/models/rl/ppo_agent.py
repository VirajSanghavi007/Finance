from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.base import BaseModel, ModelPrediction
from src.config.logging_config import get_logger

logger = get_logger(__name__)

LABEL_UNMAP = {0: -1, 1: 0, 2: 1}

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


class PPOAgent(BaseModel):
    """
    PPO agent using MlpLstmPolicy for temporal dependencies.
    Wraps SB3 PPO as a BaseModel -- produces {-1,0,1} signals.
    """

    def __init__(
        self,
        total_timesteps: int = 100_000,
        n_steps: int = 2048,
        learning_rate: float = 3e-4,
        curriculum_levels: int = 3,
    ) -> None:
        self.total_timesteps   = total_timesteps
        self.n_steps           = n_steps
        self.learning_rate     = learning_rate
        self.curriculum_levels = curriculum_levels
        self._model: Any | None = None
        self._feature_names: list[str] = []

    def _make_env(self, X: pd.DataFrame, price: pd.Series, level: int = 1):
        from src.models.rl.environment import TradingEnv

        def _init():
            return TradingEnv(X, price, curriculum_level=level)
        return _init

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val:   pd.DataFrame,
        y_val:   pd.Series,
        sample_weights: np.ndarray | None = None,
        price_train: pd.Series | None = None,
    ) -> dict:
        if not SB3_AVAILABLE:
            logger.warning("sb3_not_available_skipping_ppo")
            return {"error": "sb3_not_available"}

        self._feature_names = list(X_train.columns)

        if price_train is None:
            close_col = "close" if "close" in X_train.columns else X_train.columns[0]
            price_train = X_train[close_col]

        from stable_baselines3.common.vec_env import DummyVecEnv
        from src.models.rl.callbacks import EarlyStoppingCallback, CheckpointCallback

        # Curriculum training: gradually increase difficulty
        steps_per_level = self.total_timesteps // self.curriculum_levels
        for level in range(1, self.curriculum_levels + 1):
            env = DummyVecEnv([self._make_env(X_train, price_train, level)])
            if self._model is None:
                self._model = PPO(
                    "MlpLstmPolicy",
                    env,
                    n_steps=self.n_steps,
                    learning_rate=self.learning_rate,
                    verbose=0,
                )
            else:
                self._model.set_env(env)
            self._model.learn(
                total_timesteps=steps_per_level,
                callback=EarlyStoppingCallback(patience=5),
                reset_num_timesteps=(level == 1),
            )
            logger.info("ppo_curriculum_level_done", level=level)

        return {"curriculum_levels": self.curriculum_levels}

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        For compatibility: run deterministic policy, convert action to one-hot proba.
        """
        if not SB3_AVAILABLE or self._model is None:
            return np.ones((len(X), 3)) / 3
        from src.models.rl.environment import TradingEnv

        close_col = "close" if "close" in X.columns else X.columns[0]
        price = X[close_col]
        env = TradingEnv(X, price)
        obs, _ = env.reset()

        probas = []
        for _ in range(len(X) - 1):
            action, _ = self._model.predict(obs, deterministic=True)
            proba = np.zeros(3)
            proba[int(action)] = 1.0
            probas.append(proba)
            obs, _, done, _, _ = env.step(int(action))
            if done:
                break

        if not probas:
            return np.ones((1, 3)) / 3
        return np.array(probas, dtype=np.float32)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba  = self.predict_proba(X)
        mapped = np.argmax(proba, axis=1)
        return np.array([LABEL_UNMAP[int(m)] for m in mapped])

    def get_feature_importance(self) -> pd.Series:
        if not self._feature_names:
            return pd.Series(dtype=float)
        n = len(self._feature_names)
        return pd.Series(np.ones(n) / n, index=self._feature_names)

    def save(self, path: str) -> None:
        if self._model is None:
            return
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self._model.save(str(p / "ppo_model"))
        (p / "meta.json").write_text(json.dumps({
            "feature_names": self._feature_names,
        }))

    @classmethod
    def load(cls, path: str) -> "PPOAgent":
        p   = Path(path)
        obj = cls.__new__(cls)
        obj._feature_names = []
        meta_path = p / "meta.json"
        if meta_path.exists():
            obj._feature_names = json.loads(meta_path.read_text())["feature_names"]
        obj._model = None
        if SB3_AVAILABLE:
            try:
                obj._model = PPO.load(str(p / "ppo_model"))
            except Exception:
                pass
        return obj
