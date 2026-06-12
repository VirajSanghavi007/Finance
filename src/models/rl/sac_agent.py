from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.base import BaseModel
from src.config.logging_config import get_logger

logger = get_logger(__name__)

LABEL_UNMAP = {0: -1, 1: 0, 2: 1}

try:
    from stable_baselines3 import SAC
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


class SACContinuousEnv:
    """Thin wrapper exposing continuous action space for SAC."""

    def __init__(self, feature_df: pd.DataFrame, price_series: pd.Series) -> None:
        try:
            import gymnasium as gym
            from gymnasium import spaces
        except ImportError:
            raise ImportError("gymnasium required for SACContinuousEnv")

        from src.models.rl.environment import TradingEnv
        self._base = TradingEnv(feature_df, price_series)
        n_obs = feature_df.shape[1] + 4
        self.observation_space = self._base.observation_space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, **kwargs):
        return self._base.reset(**kwargs)

    def step(self, action: np.ndarray):
        # Continuous action → discrete: threshold at ±0.33
        a = float(action[0])
        if a < -0.33:
            discrete = 0
        elif a > 0.33:
            discrete = 2
        else:
            discrete = 1
        return self._base.step(discrete)

    def render(self):
        pass


class SACAgent(BaseModel):
    """SAC agent with continuous action for soft exploration."""

    def __init__(
        self,
        total_timesteps: int = 50_000,
        learning_rate: float = 3e-4,
    ) -> None:
        self.total_timesteps = total_timesteps
        self.learning_rate   = learning_rate
        self._model: Any | None = None
        self._feature_names: list[str] = []

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
            logger.warning("sb3_not_available_skipping_sac")
            return {"error": "sb3_not_available"}

        self._feature_names = list(X_train.columns)

        if price_train is None:
            close_col = "close" if "close" in X_train.columns else X_train.columns[0]
            price_train = X_train[close_col]

        from stable_baselines3.common.vec_env import DummyVecEnv

        def _make():
            return SACContinuousEnv(X_train, price_train)

        env = DummyVecEnv([_make])
        self._model = SAC(
            "MlpPolicy",
            env,
            learning_rate=self.learning_rate,
            verbose=0,
        )
        self._model.learn(total_timesteps=self.total_timesteps)
        logger.info("sac_trained", timesteps=self.total_timesteps)
        return {"total_timesteps": self.total_timesteps}

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not SB3_AVAILABLE or self._model is None:
            return np.ones((len(X), 3)) / 3

        close_col = "close" if "close" in X.columns else X.columns[0]
        price = X[close_col]
        env = SACContinuousEnv(X, price)
        obs, _ = env.reset()

        probas = []
        for _ in range(len(X) - 1):
            action, _ = self._model.predict(obs, deterministic=True)
            a = float(action[0])
            proba = np.zeros(3)
            if a < -0.33:
                proba[0] = 1.0
            elif a > 0.33:
                proba[2] = 1.0
            else:
                proba[1] = 1.0
            probas.append(proba)
            obs, _, done, _, _ = env.step(action)
            if done:
                break

        return np.array(probas, dtype=np.float32) if probas else np.ones((1, 3)) / 3

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
        self._model.save(str(p / "sac_model"))
        (p / "meta.json").write_text(json.dumps({"feature_names": self._feature_names}))

    @classmethod
    def load(cls, path: str) -> "SACAgent":
        p   = Path(path)
        obj = cls.__new__(cls)
        obj._feature_names = []
        meta_path = p / "meta.json"
        if meta_path.exists():
            obj._feature_names = json.loads(meta_path.read_text())["feature_names"]
        obj._model = None
        if SB3_AVAILABLE:
            try:
                obj._model = SAC.load(str(p / "sac_model"))
            except Exception:
                pass
        return obj
