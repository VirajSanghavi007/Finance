from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.base import BaseModel
from src.config.logging_config import get_logger

logger = get_logger(__name__)

LABEL_UNMAP = {0: -1, 1: 0, 2: 1}
SEQ_LEN = 60


def _build_nbeats(input_size: int, forecast_size: int = 5):
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return None

    class NBeatsBlock(nn.Module):
        def __init__(self, input_size: int, theta_size: int,
                     basis_type: str = "generic") -> None:
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(input_size, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, theta_size),
            )
            self.basis_type = basis_type
            self.input_size = input_size
            self.forecast_size = forecast_size

        def forward(self, x):
            theta = self.fc(x)
            backcast = theta[:, :self.input_size]
            forecast = theta[:, self.input_size:]
            return backcast, forecast

    class NBeats(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Trend, seasonality, generic stacks
            self.blocks = nn.ModuleList([
                NBeatsBlock(input_size, input_size + forecast_size, "trend"),
                NBeatsBlock(input_size, input_size + forecast_size, "seasonal"),
                NBeatsBlock(input_size, input_size + forecast_size, "generic"),
            ])
            self.fc_out = nn.Linear(forecast_size, 3)  # convert forecast to class

        def forward(self, x):
            residual = x
            forecasts = []
            for block in self.blocks:
                backcast, forecast = block(residual)
                residual = residual - backcast
                forecasts.append(forecast)
            total_forecast = sum(forecasts)
            return self.fc_out(total_forecast)

    return NBeats()


class NBeatsModel(BaseModel):
    """N-BEATS: Neural Basis Expansion Analysis for Time Series."""

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
        # N-BEATS takes univariate close (last column 'close' or uses PCA)
        # Here we flatten the last SEQ_LEN rows as a 1D input per sample
        close_idx = list(X.columns).index("close") if "close" in X.columns else 0
        vals = X.fillna(0).iloc[:, close_idx].values.astype(np.float32)
        seqs, labels = [], []
        for i in range(self.seq_len, len(vals)):
            seqs.append(vals[i - self.seq_len:i])
            if y is not None:
                from src.models.deep.lstm_model import LABEL_MAP
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
            logger.warning("pytorch_not_available_skipping_nbeats")
            return {"error": "pytorch_not_available"}

        self._feature_names = list(X_train.columns)
        self._n_features    = len(self._feature_names)

        Xt, yt = self._make_sequences(X_train, y_train)
        Xv, yv = self._make_sequences(X_val,   y_val)

        from sklearn.utils.class_weight import compute_class_weight
        cw = compute_class_weight("balanced", classes=np.array([0,1,2]), y=yt.numpy())
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(cw))

        model = _build_nbeats(self.seq_len)
        if model is None:
            return {"error": "build_failed"}

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
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
        logger.info("nbeats_trained", best_val_loss=f"{best_loss:.4f}")
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
            "n_features": self._n_features,
            "seq_len": self.seq_len,
        }))

    @classmethod
    def load(cls, path: str) -> "NBeatsModel":
        p    = Path(path)
        meta = json.loads((p / "meta.json").read_text())
        obj  = cls(seq_len=meta["seq_len"])
        obj._feature_names = meta["feature_names"]
        obj._n_features    = meta["n_features"]
        try:
            import torch
            model = _build_nbeats(obj.seq_len)
            if model is not None:
                model.load_state_dict(torch.load(p / "model.pt", weights_only=True))
                obj._model = model
        except ImportError:
            pass
        return obj
