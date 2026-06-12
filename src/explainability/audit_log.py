from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.constants import PROJECT_ROOT, LOGS_DIR
from src.config.logging_config import get_logger
from src.explainability.attribution import SignalAttribution

logger = get_logger(__name__)

_AUDIT_DIR = LOGS_DIR / "audit"


class AuditLog:
    """
    Append-only JSONL audit trail of all trading signals with their attribution.
    One record per signal — enables post-hoc review and regulatory compliance.
    """

    def __init__(self, log_dir: Path | None = None) -> None:
        self._dir = log_dir or _AUDIT_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

    def _log_path(self, date_str: str) -> Path:
        return self._dir / f"audit_{date_str}.jsonl"

    def record(self, attribution: SignalAttribution, extra: dict | None = None) -> None:
        """Append one signal attribution record to today's audit log."""
        record: dict[str, Any] = {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "timestamp":   str(attribution.timestamp),
            "ticker":      attribution.ticker,
            "signal":      attribution.signal,
            "confidence":  round(attribution.confidence, 4),
            "regime":      attribution.regime,
            "top_features": {k: round(float(v), 6) for k, v in attribution.top_features.items()},
            "model_votes": attribution.model_votes,
        }
        if extra:
            record.update(extra)

        date_str  = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_path  = self._log_path(date_str)
        with log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def read(self, date_str: str | None = None) -> pd.DataFrame:
        """Load audit records for a given date (YYYY-MM-DD), or today if None."""
        if date_str is None:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_path = self._log_path(date_str)
        if not log_path.exists():
            return pd.DataFrame()
        records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
        return pd.DataFrame(records)

    def summary(self, date_str: str | None = None) -> dict:
        """Return signal count statistics for a given day."""
        df = self.read(date_str)
        if df.empty:
            return {"total": 0}
        return {
            "total": len(df),
            "long":  int((df["signal"] == 1).sum()),
            "flat":  int((df["signal"] == 0).sum()),
            "short": int((df["signal"] == -1).sum()),
            "avg_confidence": float(df["confidence"].mean()),
        }
