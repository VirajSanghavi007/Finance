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
SEQ_LEN     = 60
PATCH_LEN   = 16
STRIDE      = 8
D_MODEL     = 128


def _n_patches(seq_len: int = SEQ_LEN) -> int:
    return (seq_len - PATCH_LEN) // STRIDE + 1


def _build_patchtst(n_features: int):
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return None

    n_patches = _n_patches()

    class PatchTST(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Channel-independent: one transformer per feature
            self.patch_proj = nn.Linear(PATCH_LEN, D_MODEL)
            self.pos_emb    = nn.Parameter(torch.randn(1, n_patches, D_MODEL))
            enc_layer = nn.TransformerEncoderLayer(
                d_model=D_MODEL, nhead=8, dim_feedforward=256,
                dropout=0.1, batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=4)
            self.head = nn.Sequential(
                nn.Linear(n_features * D_MODEL, 128),
                nn.GELU(),
                nn.Linear(128, 3),
            )

        def forward(self, x):  # x: (B, T, F)
            B, T, F = x.shape
            channel_outs = []
            for f in range(F):
                ch = x[:, :, f]  # (B, T)
                patches = []
                for i in range(0, T - PATCH_LEN + 1, STRIDE):
                    patches.append(ch[:, i:i + PATCH_LEN])
                if not patches:
                    continue
                p = torch.stack(patches, dim=1)      # (B, N_patches, PATCH_LEN)
                p = self.patch_proj(p) + self.pos_emb[:, :p.shape[1], :]
                out = self.encoder(p)                  # (B, N_patches, D_MODEL)
                channel_outs.append(out.mean(dim=1))  # (B, D_MODEL)
            combined = torch.cat(channel_outs, dim=-1)  # (B, F * D_MODEL)
            return self.head(combined)

    return PatchTST()


class PatchTSTModel(BaseModel):
    """Patch-based Time Series Transformer (PatchTST, 2023)."""

    def __init__(self, seq_len: int = SEQ_LEN, epochs: int = 50,
                 batch_size: int = 32, lr: float = 1e-3) -> None:
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
            logger.warning("pytorch_not_available_skipping_patchtst")
            return {"error": "pytorch_not_available"}

        self._feature_names = list(X_train.columns)
        self._n_features    = len(self._feature_names)

        Xt, yt = self._make_sequences(X_train, y_train)
        Xv, yv = self._make_sequences(X_val,   y_val)

        from sklearn.utils.class_weight import compute_class_weight
        cw = compute_class_weight("balanced", classes=np.array([0,1,2]), y=yt.numpy())
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(cw))

        model = _build_patchtst(self._n_features)
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
        logger.info("patchtst_trained", best_val_loss=f"{best_loss:.4f}")
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
    def load(cls, path: str) -> "PatchTSTModel":
        p    = Path(path)
        meta = json.loads((p / "meta.json").read_text())
        obj  = cls(seq_len=meta["seq_len"])
        obj._feature_names = meta["feature_names"]
        obj._n_features    = meta["n_features"]
        try:
            import torch
            model = _build_patchtst(obj._n_features)
            if model is not None:
                model.load_state_dict(torch.load(p / "model.pt", weights_only=True))
                obj._model = model
        except ImportError:
            pass
        return obj
