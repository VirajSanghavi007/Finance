"""
Conservative Q-Learning (CQL) — Offline RL for trading.
(Kumar et al., NeurIPS 2020)

Standard RL requires live interaction: the agent tries actions, observes rewards,
updates. That's fine for games but terrible for trading — you can't "try" bad trades.

Offline RL (CQL) solves this: train from a STATIC dataset of historical trajectories.
The CQL penalty prevents overestimating Q-values at (state, action) pairs that don't
appear in the dataset, which is the core failure mode of offline Q-learning.

Conservative loss:
  L_CQL(Q) = α * E_s[log Σ_a exp(Q(s,a))] - E_{s,a~D}[Q(s,a)] + L_Bellman(Q)

  - First term: penalises high Q-values at actions NOT in the dataset
  - Second term: rewards high Q-values at actions that WERE taken
  - Together: conservative estimate — only trust what you've seen

Architecture:
  State:  (last 30 bars × top 20 features) + position + drawdown  →  flat vector
  Action: Discrete(3) — {0=short, 1=flat, 2=long}
  Q-net:  MLP(state_dim → 256 → 128 → 3), double Q-networks for stability
  Reward: same as TradingEnv but computed offline from labels

Same BaseModel interface so it can be dropped into the ensemble.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.base import BaseModel
from src.config.logging_config import get_logger

logger = get_logger(__name__)

LABEL_MAP   = {-1: 0, 0: 1, 1: 2}   # signal → action index
LABEL_UNMAP = {0: -1, 1: 0, 2: 1}   # action index → signal


def _build_q_net(state_dim: int, hidden: int = 256):
    """Build double Q-network. Returns (q1, q2) or (None, None)."""
    try:
        import torch.nn as nn

        class QNetwork(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(state_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                    nn.Linear(hidden, hidden // 2), nn.SiLU(),
                    nn.Linear(hidden // 2, 3),
                )

            def forward(self, x):
                return self.net(x)

        return QNetwork(), QNetwork()
    except ImportError:
        return None, None


class OfflineRLAgent(BaseModel):
    """
    CQL-inspired offline RL agent.

    Training:
      1. Build state-action-reward-next_state tuples from historical features + returns
      2. Minimise Bellman loss + CQL conservative penalty
      3. Predict Q(s, ·) at inference time; take argmax action

    Graceful fallback: returns uniform 1/3 probabilities if PyTorch absent.
    """

    def __init__(
        self,
        seq_len:    int   = 30,
        n_features: int   = 20,
        hidden:     int   = 256,
        alpha:      float = 1.0,   # CQL penalty weight
        gamma:      float = 0.99,  # discount factor
        lr:         float = 3e-4,
        epochs:     int   = 50,
        batch_size: int   = 256,
    ) -> None:
        self.seq_len    = seq_len
        self.n_features = n_features
        self.hidden     = hidden
        self.alpha      = alpha
        self.gamma      = gamma
        self.lr         = lr
        self.epochs     = epochs
        self.batch_size = batch_size

        self._q1: Any | None = None
        self._q2: Any | None = None
        self._feature_names: list[str] = []
        self._top_features:  list[str] = []

    # ── State construction ────────────────────────────────────────────────────

    def _select_top_features(self, X: pd.DataFrame) -> list[str]:
        """Select top N features by variance (proxy for informativeness)."""
        variances = X.var().sort_values(ascending=False)
        return list(variances.head(self.n_features).index)

    def _build_states(self, X: pd.DataFrame) -> np.ndarray:
        """Flatten (seq_len × n_features) windows into state vectors."""
        cols = [c for c in self._top_features if c in X.columns]
        arr  = X[cols].fillna(0).values.astype(np.float32)
        n    = len(arr)

        states = []
        for i in range(self.seq_len - 1, n):
            window = arr[i - self.seq_len + 1: i + 1].flatten()
            states.append(window)
        return np.array(states, dtype=np.float32)

    # ── Reward computation ────────────────────────────────────────────────────

    @staticmethod
    def _compute_rewards(
        actions: np.ndarray,   # {0, 1, 2}
        returns: np.ndarray,   # log returns
        cost_rate: float = 0.001,
    ) -> np.ndarray:
        """
        Reward = signed log return − transaction cost on direction change.
        Flat (action=1) gets 0 reward regardless of market.
        """
        directions = actions - 1   # {0,1,2} → {-1,0,1}
        rewards    = directions * returns
        # Cost on position changes
        direction_change = np.abs(np.diff(directions, prepend=0))
        rewards -= direction_change * cost_rate
        return rewards.astype(np.float32)

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val:   pd.DataFrame,
        y_val:   pd.Series,
        sample_weights: np.ndarray | None = None,
    ) -> dict:
        try:
            import torch
            import torch.nn.functional as F
            from torch.utils.data import TensorDataset, DataLoader
        except ImportError:
            logger.warning("offline_rl_torch_not_available")
            return {"status": "skipped_no_torch"}

        self._feature_names = list(X_train.columns)
        self._top_features  = self._select_top_features(X_train)

        state_dim = self.seq_len * len(self._top_features)
        self._q1, self._q2 = _build_q_net(state_dim, self.hidden)
        if self._q1 is None:
            return {"status": "skipped_no_torch"}

        # Build dataset
        states   = self._build_states(X_train)
        n_states = len(states)
        if n_states < self.batch_size:
            return {"status": "insufficient_data", "n_states": n_states}

        # Actions from y_train labels (shifted to align with states window).
        # y_train may be shorter than X_train — use the minimum to guarantee alignment.
        y_tail   = y_train.iloc[self.seq_len - 1:].map(LABEL_MAP).fillna(1)
        n_states = min(n_states, len(y_tail))   # enforce consistent length
        states   = states[:n_states]
        y_aligned = y_tail.values[:n_states]
        actions   = y_aligned.astype(np.int64)

        # Rewards from aligned returns
        if "target_ret_1d" in X_train.columns:
            rets = X_train["target_ret_1d"].fillna(0).iloc[self.seq_len - 1:].values[:n_states]
        else:
            rets = np.sign(y_aligned - 1) * 0.001   # tiny proxy reward
        rewards = self._compute_rewards(actions, rets.astype(np.float32))

        # Next states — terminal state gets zero vector (no bootstrapping at end)
        next_states = np.vstack([states[1:], np.zeros_like(states[-1:])])

        S  = torch.tensor(states,      dtype=torch.float32)
        A  = torch.tensor(actions,     dtype=torch.long)
        R  = torch.tensor(rewards,     dtype=torch.float32)
        NS = torch.tensor(next_states, dtype=torch.float32)

        loader = DataLoader(
            TensorDataset(S, A, R, NS),
            batch_size=self.batch_size, shuffle=True,
        )

        params = list(self._q1.parameters()) + list(self._q2.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr)

        for epoch in range(self.epochs):
            for s_b, a_b, r_b, ns_b in loader:
                # ── Bellman target ────────────────────────────────────────────
                with torch.no_grad():
                    q1_ns = self._q1(ns_b)
                    q2_ns = self._q2(ns_b)
                    q_min_ns   = torch.min(q1_ns, q2_ns)
                    target_q   = r_b + self.gamma * q_min_ns.max(dim=1).values

                q1_sa = self._q1(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)
                q2_sa = self._q2(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)

                bellman_loss = F.mse_loss(q1_sa, target_q) + F.mse_loss(q2_sa, target_q)

                # ── CQL conservative penalty ──────────────────────────────────
                # Penalise log-sum-exp of all actions (prevents overestimation)
                q1_all = self._q1(s_b)   # (batch, 3)
                q2_all = self._q2(s_b)
                cql_loss = (
                    torch.logsumexp(q1_all, dim=1).mean() - q1_sa.mean()
                    + torch.logsumexp(q2_all, dim=1).mean() - q2_sa.mean()
                )

                loss = bellman_loss + self.alpha * cql_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()

        logger.info("offline_rl_trained", epochs=self.epochs, state_dim=state_dim)
        return {"epochs": self.epochs, "state_dim": state_dim, "n_transitions": n_states}

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._q1 is None:
            return np.ones((len(X), 3)) / 3.0
        try:
            import torch
            import torch.nn.functional as F

            states = self._build_states(X)
            if len(states) == 0:
                return np.ones((len(X), 3)) / 3.0

            S = torch.tensor(states, dtype=torch.float32)
            self._q1.eval()
            self._q2.eval()
            with torch.no_grad():
                q_vals = (self._q1(S) + self._q2(S)) / 2.0  # average Q-nets
                proba  = F.softmax(q_vals, dim=-1).numpy()

            # Pad front rows with uniform (for early bars lacking seq_len history)
            pad = len(X) - len(proba)
            if pad > 0:
                proba = np.vstack([np.ones((pad, 3)) / 3.0, proba])
            return proba
        except Exception as e:
            logger.warning("offline_rl_predict_failed", error=str(e))
            return np.ones((len(X), 3)) / 3.0

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.array([LABEL_UNMAP[int(np.argmax(p))] for p in proba])

    def get_feature_importance(self) -> pd.Series:
        # Q-network doesn't expose feature importances; return uniform
        if not self._top_features:
            return pd.Series(dtype=float)
        n = len(self._top_features)
        return pd.Series(np.ones(n) / n, index=self._top_features)

    def save(self, path: str) -> None:
        try:
            import torch
            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            if self._q1:
                torch.save(self._q1.state_dict(), p / "q1.pt")
                torch.save(self._q2.state_dict(), p / "q2.pt")
            meta = {
                "seq_len": self.seq_len, "n_features": self.n_features,
                "hidden": self.hidden, "alpha": self.alpha, "gamma": self.gamma,
                "feature_names": self._feature_names,
                "top_features":  self._top_features,
            }
            (p / "meta.json").write_text(json.dumps(meta))
        except Exception as e:
            logger.warning("offline_rl_save_failed", error=str(e))

    @classmethod
    def load(cls, path: str) -> "OfflineRLAgent":
        try:
            import torch
            p    = Path(path)
            meta = json.loads((p / "meta.json").read_text())
            obj  = cls(
                seq_len=meta["seq_len"], n_features=meta["n_features"],
                hidden=meta["hidden"],  alpha=meta["alpha"], gamma=meta["gamma"],
            )
            obj._feature_names = meta["feature_names"]
            obj._top_features  = meta["top_features"]
            state_dim = meta["seq_len"] * len(meta["top_features"])
            q1, q2 = _build_q_net(state_dim, meta["hidden"])
            if q1 and (p / "q1.pt").exists():
                q1.load_state_dict(torch.load(p / "q1.pt", map_location="cpu"))
                q2.load_state_dict(torch.load(p / "q2.pt", map_location="cpu"))
                obj._q1 = q1
                obj._q2 = q2
            return obj
        except Exception as e:
            logger.warning("offline_rl_load_failed", error=str(e))
            return cls()
