from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.base import BaseModel
from src.config.logging_config import get_logger

logger = get_logger(__name__)

LABEL_MAP   = {-1: 0, 0: 1, 1: 2}
LABEL_UNMAP = {0: -1, 1: 0, 2: 1}
SEQ_LEN = 60
DILATIONS = [1, 2, 4, 8, 16, 32, 64, 128]
KERNEL_SIZE = 3


def _build_tcn(n_features: int, n_channels: int = 64):
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError:
        return None

    class CausalConvBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, dilation: int) -> None:
            super().__init__()
            pad = (KERNEL_SIZE - 1) * dilation
            self.conv  = nn.utils.weight_norm(
                nn.Conv1d(in_ch, out_ch, KERNEL_SIZE,
                          padding=pad, dilation=dilation)
            )
            self.drop  = nn.Dropout(0.2)
            self.resid = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

        def forward(self, x):
            out = F.relu(self.conv(x)[:, :, :-( (KERNEL_SIZE - 1) * self.conv.dilation[0])])
            out = self.drop(out)
            res = x if self.resid is None else self.resid(x)
            return F.relu(out + res)

    class TCN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers = []
            in_ch = n_features
            for d in DILATIONS:
                layers.append(CausalConvBlock(in_ch, n_channels, d))
                in_ch = n_channels
            self.network = nn.Sequential(*layers)
            self.fc = nn.Linear(n_channels, 3)

        def forward(self, x):  # x: (B, T, F) → permute to (B, F, T) for Conv1d
            out = self.network(x.permute(0, 2, 1))  # (B, C, T)
            pooled = out.mean(dim=2)                # global avg pool
            return self.fc(pooled)

    return TCN()


class TCNModel(BaseModel):
    """Temporal Convolutional Network with dilated causal convolutions."""

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
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required")
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
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import TensorDataset, DataLoader
        except ImportError:
            logger.warning("pytorch_not_available_skipping_tcn")
            return {"error": "pytorch_not_available"}

        self._feature_names = list(X_train.columns)
        self._n_features    = len(self._feature_names)

        Xt, yt = self._make_sequences(X_train, y_train)
        Xv, yv = self._make_sequences(X_val,   y_val)

        from sklearn.utils.class_weight import compute_class_weight
        cw      = compute_class_weight("balanced", classes=np.array([0,1,2]), y=yt.numpy())
        cw_t    = torch.FloatTensor(cw)
        criterion = nn.CrossEntropyLoss(weight=cw_t)

        model = _build_tcn(self._n_features)
        if model is None:
            return {"error": "build_failed"}

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        loader    = DataLoader(TensorDataset(Xt, yt), batch_size=self.batch_size, shuffle=False)

        best_loss = float("inf")
        patience  = 0

        for epoch in range(self.epochs):
            model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(Xv), yv).item()

            if val_loss < best_loss:
                best_loss = val_loss
                patience  = 0
            else:
                patience += 1
                if patience >= 15:
                    break

        self._model = model
        logger.info("tcn_trained", best_val_loss=f"{best_loss:.4f}")
        return {"best_val_loss": best_loss}

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            return np.ones((max(1, len(X) - self.seq_len), 3)) / 3
        if self._model is None:
            return np.ones((max(1, len(X) - self.seq_len), 3)) / 3
        Xt, _ = self._make_sequences(X)
        self._model.eval()
        with torch.no_grad():
            return F.softmax(self._model(Xt), dim=-1).numpy()

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
        try:
            import torch
        except ImportError:
            return
        if self._model is None:
            return
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), p / "model.pt")
        (p / "meta.json").write_text(json.dumps({
            "feature_names": self._feature_names,
            "n_features":    self._n_features,
            "seq_len":       self.seq_len,
        }))

    @classmethod
    def load(cls, path: str) -> "TCNModel":
        p    = Path(path)
        meta = json.loads((p / "meta.json").read_text())
        obj  = cls(seq_len=meta["seq_len"])
        obj._feature_names = meta["feature_names"]
        obj._n_features    = meta["n_features"]
        try:
            import torch
            model = _build_tcn(obj._n_features)
            if model is not None:
                model.load_state_dict(torch.load(p / "model.pt", weights_only=True))
                obj._model = model
        except ImportError:
            pass
        return obj
