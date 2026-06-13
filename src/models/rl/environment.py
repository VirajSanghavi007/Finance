from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False

from src.config.logging_config import get_logger

logger = get_logger(__name__)

_INITIAL_CASH = 100_000.0
_MAX_POSITION = 0.20      # max 20% of portfolio per position
_TRANSACTION_COST = 0.001 # 10 bps round-trip


class TradingEnv(gym.Env if GYM_AVAILABLE else object):
    """
    Single-asset trading environment for PPO/SAC agents.

    Actions: 0=short, 1=flat, 2=long  (discrete, maps to -1/0/+1 signal)
    Observations: feature_vector (n_features,) + portfolio_state (4,)
      portfolio_state = [position, unrealized_pnl_pct, cash_pct, drawdown_pct]
    Reward: risk-adjusted daily PnL -- transaction-cost penalized
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        feature_df: pd.DataFrame,
        price_series: pd.Series,
        curriculum_level: int = 1,
    ) -> None:
        if not GYM_AVAILABLE:
            raise ImportError("gymnasium is required for TradingEnv")

        super().__init__()
        self._features    = feature_df.fillna(0).values.astype(np.float32)
        self._prices      = price_series.values.astype(np.float32)
        self._n_features  = self._features.shape[1]
        self._n_steps     = len(self._prices)
        self.curriculum_level = curriculum_level

        n_obs = self._n_features + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self._t: int = 0
        self._position: int = 0      # -1, 0, 1
        self._cash: float = _INITIAL_CASH
        self._portfolio_value: float = _INITIAL_CASH
        self._peak_value: float = _INITIAL_CASH
        self._entry_price: float = 0.0

    def _obs(self) -> np.ndarray:
        feat = self._features[self._t]
        price = self._prices[self._t]
        unreal = 0.0
        if self._position != 0 and self._entry_price > 0:
            unreal = self._position * (price - self._entry_price) / self._entry_price
        pv = self._portfolio_value
        draw = (self._peak_value - pv) / self._peak_value if self._peak_value > 0 else 0.0
        state = np.array([
            float(self._position),
            unreal,
            self._cash / (_INITIAL_CASH + 1e-8),
            draw,
        ], dtype=np.float32)
        return np.concatenate([feat, state])

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._t              = 0
        self._position       = 0
        self._cash           = _INITIAL_CASH
        self._portfolio_value = _INITIAL_CASH
        self._peak_value      = _INITIAL_CASH
        self._entry_price     = 0.0
        return self._obs(), {}

    def step(self, action: int):
        assert 0 <= action <= 2
        signal = action - 1  # 0→-1, 1→0, 2→+1

        price_now  = self._prices[self._t]
        price_next = self._prices[min(self._t + 1, self._n_steps - 1)]

        trade = signal != self._position
        cost  = _TRANSACTION_COST * abs(signal - self._position) * price_now if trade else 0.0

        # Position PnL
        ret = self._position * (price_next - price_now) / (price_now + 1e-8)
        pnl = ret * self._portfolio_value - cost * self._portfolio_value

        self._portfolio_value = max(1.0, self._portfolio_value + pnl)
        self._peak_value      = max(self._peak_value, self._portfolio_value)

        if trade:
            self._position    = signal
            self._entry_price = price_next

        # Reward: Sharpe-like (return - cost) scaled by running vol estimate
        reward = float(pnl / (self._portfolio_value * 0.01 + 1e-8))
        # Curriculum: clip reward range based on level (wider as level increases)
        reward = np.clip(reward, -self.curriculum_level * 5, self.curriculum_level * 5)

        self._t += 1
        done     = self._t >= self._n_steps - 1
        truncated = False

        return self._obs(), reward, done, truncated, {}

    def render(self):
        pass
