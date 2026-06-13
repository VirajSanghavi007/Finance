from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.base import BaseModel
from src.config.logging_config import get_logger

logger = get_logger(__name__)

LABEL_MAP   = {-1: 0, 0: 1, 1: 2}
LABEL_UNMAP = {0: -1, 1: 0, 2: 1}
SEQ_LEN = 60


def _try_torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


class _TemporalAttention:
    pass  # defined inside LSTMNet to avoid top-level torch import


def _build_model(n_features: int, hidden: int = 256, n_layers: int = 3,
                 dropout: float = 0.3):
    torch = _try_torch()
    if torch is None:
        return None

    import torch.nn as nn

    class TemporalAttention(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.attn = nn.Linear(d, 1)

        def forward(self, x):  # x: (B, T, D)
            w = torch.softmax(self.attn(x), dim=1)  # (B, T, 1)
            return (w * x).sum(dim=1)               # (B, D)

    class LSTMNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bn    = nn.BatchNorm1d(n_features)
            self.lstm  = nn.LSTM(n_features, hidden, n_layers,
                                 batch_first=True, dropout=dropout,
                                 bidirectional=False)
            self.attn  = TemporalAttention(hidden)
            self.drop  = nn.Dropout(dropout)
            self.fc1   = nn.Linear(hidden, 128)
            self.act   = nn.GELU()
            self.drop2 = nn.Dropout(0.2)
            self.fc2   = nn.Linear(128, 3)

        def forward(self, x):  # x: (B, T, F)
            B, T, F = x.shape
            xn = self.bn(x.reshape(B * T, F)).reshape(B, T, F)
            out, _ = self.lstm(xn)
            ctx = self.attn(out)
            ctx = self.drop(ctx)
            h = self.act(self.fc1(ctx))
            h = self.drop2(h)
            return self.fc2(h)

    return LSTMNet()


class LSTMModel(BaseModel):
    """3-layer LSTM with temporal attention. Seq_len=60 bars."""

    def __init__(self, seq_len: int = SEQ_LEN, epochs: int = 50,
                 batch_size: int = 64, lr: float = 1e-3) -> None:
        self.seq_len    = seq_len
        self.epochs     = epochs
        self.batch_size = batch_size
        self.lr         = lr
        self._model     = None
        self._feature_names: list[str] = []
        self._n_features: int = 0

    def _make_sequences(self, X: pd.DataFrame, y: pd.Series | None = None):
        torch = _try_torch()
        if torch is None:
            raise ImportError("PyTorch required for LSTMModel")
        vals = X.fillna(0).values.astype(np.float32)
        seqs, labels = [], []
        for i in range(self.seq_len, len(vals)):
            seqs.append(vals[i - self.seq_len:i])
            if y is not None:
                labels.append(LABEL_MAP.get(int(y.iloc[i]), 1))
        Xt = torch.from_numpy(np.stack(seqs))
        yt = torch.tensor(labels, dtype=torch.long) if labels else None
        return Xt, yt

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val:   pd.DataFrame,
        y_val:   pd.Series,
        sample_weights: np.ndarray | None = None,
    ) -> dict:
        torch = _try_torch()
        if torch is None:
            logger.warning("pytorch_not_available_skipping_lstm")
            return {"error": "pytorch_not_available"}

        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        self._feature_names = list(X_train.columns)
        self._n_features    = len(self._feature_names)

        Xt, yt   = self._make_sequences(X_train, y_train)
        Xv, yv   = self._make_sequences(X_val,   y_val)

        from sklearn.utils.class_weight import compute_class_weight
        classes = np.array([0, 1, 2])
        cw = compute_class_weight("balanced", classes=classes, y=yt.numpy())
        class_weight_tensor = torch.FloatTensor(cw)

        model = _build_model(self._n_features)
        if model is None:
            return {"error": "model_build_failed"}

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)

        ds     = TensorDataset(Xt, yt)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                out  = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_out  = model(Xv)
                val_loss = criterion(val_out, yv).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= 15:
                    logger.info("lstm_early_stop", epoch=epoch)
                    break

        if best_state:
            model.load_state_dict(best_state)
        self._model = model
        logger.info("lstm_trained", best_val_loss=f"{best_val_loss:.4f}")
        return {"best_val_loss": best_val_loss}

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        torch = _try_torch()
        if torch is None or self._model is None:
            n = max(1, len(X) - self.seq_len)
            return np.ones((n, 3)) / 3

        import torch.nn.functional as F
        Xt, _ = self._make_sequences(X)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(Xt)
            proba  = F.softmax(logits, dim=-1).numpy()
        return proba

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        mapped = np.argmax(proba, axis=1)
        return np.array([LABEL_UNMAP[int(m)] for m in mapped])

    def get_feature_importance(self) -> pd.Series:
        # LSTM doesn't have built-in feature importance -- return uniform
        if not self._feature_names:
            return pd.Series(dtype=float)
        n = len(self._feature_names)
        return pd.Series(np.ones(n) / n, index=self._feature_names)

    def save(self, path: str) -> None:
        torch = _try_torch()
        if torch is None or self._model is None:
            return
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), p / "model.pt")
        (p / "meta.json").write_text(json.dumps({
            "feature_names": self._feature_names,
            "n_features": self._n_features,
            "seq_len": self.seq_len,
        }))

    @classmethod
    def load(cls, path: str) -> "LSTMModel":
        torch = _try_torch()
        p = Path(path)
        meta = json.loads((p / "meta.json").read_text())
        obj  = cls(seq_len=meta["seq_len"])
        obj._feature_names = meta["feature_names"]
        obj._n_features    = meta["n_features"]
        if torch is not None:
            model = _build_model(obj._n_features)
            if model is not None:
                model.load_state_dict(torch.load(p / "model.pt", weights_only=True))
                obj._model = model
        return obj
