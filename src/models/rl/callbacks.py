from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


class EarlyStoppingCallback(BaseCallback if SB3_AVAILABLE else object):
    """Stop training when mean reward stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4,
                 verbose: int = 0) -> None:
        if not SB3_AVAILABLE:
            return
        super().__init__(verbose)
        self.patience   = patience
        self.min_delta  = min_delta
        self._best      = -np.inf
        self._no_improve = 0

    def _on_step(self) -> bool:
        if not self.locals.get("dones", [False])[-1]:
            return True
        rews = self.locals.get("rewards", [])
        if not len(rews):
            return True
        mean_rew = float(np.mean(rews))
        if mean_rew > self._best + self.min_delta:
            self._best       = mean_rew
            self._no_improve = 0
        else:
            self._no_improve += 1
        if self._no_improve >= self.patience:
            if self.verbose:
                print(f"EarlyStopping: no improvement for {self.patience} rollouts")
            return False
        return True


class CheckpointCallback(BaseCallback if SB3_AVAILABLE else object):
    """Save model every N steps to a checkpoint directory."""

    def __init__(self, save_freq: int, save_dir: str,
                 name_prefix: str = "rl_model", verbose: int = 0) -> None:
        if not SB3_AVAILABLE:
            return
        super().__init__(verbose)
        self.save_freq   = save_freq
        self.save_dir    = Path(save_dir)
        self.name_prefix = name_prefix
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            p = self.save_dir / f"{self.name_prefix}_{self.num_timesteps}"
            self.model.save(str(p))
            if self.verbose:
                print(f"Saved checkpoint: {p}")
        return True
