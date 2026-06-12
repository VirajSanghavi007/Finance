from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.config.logging_config import get_logger

logger = get_logger(__name__)

_KS_ALPHA = 0.05   # significance level for drift detection


class DriftMonitor:
    """
    Detects covariate shift between training and live data.

    Uses Kolmogorov-Smirnov test per feature.  Flags features where the
    distribution has shifted significantly.
    """

    def __init__(
        self,
        significance: float = _KS_ALPHA,
        min_samples: int = 30,
    ) -> None:
        self.significance = significance
        self.min_samples  = min_samples
        self._reference: pd.DataFrame | None = None

    def fit(self, reference_data: pd.DataFrame) -> None:
        """Store reference distribution (training data statistics)."""
        self._reference = reference_data.fillna(0)
        logger.info("drift_monitor_fitted", n_features=len(reference_data.columns))

    def check(self, live_data: pd.DataFrame) -> dict[str, dict]:
        """
        Compare live_data against reference.
        Returns dict: {feature → {"ks_stat": float, "p_value": float, "drifted": bool}}
        """
        if self._reference is None:
            return {}

        results: dict[str, dict] = {}
        live_clean = live_data.fillna(0)

        for col in self._reference.columns:
            if col not in live_clean.columns:
                continue
            ref_vals  = self._reference[col].dropna().values
            live_vals = live_clean[col].dropna().values

            if len(ref_vals) < self.min_samples or len(live_vals) < self.min_samples:
                continue

            ks_stat, p_val = stats.ks_2samp(ref_vals, live_vals)
            drifted = bool(p_val < self.significance)
            results[col] = {
                "ks_stat":  float(ks_stat),
                "p_value":  float(p_val),
                "drifted":  drifted,
            }
            if drifted:
                logger.warning("feature_drift_detected", feature=col,
                               ks_stat=f"{ks_stat:.4f}", p_val=f"{p_val:.4f}")

        return results

    def drifted_features(self, live_data: pd.DataFrame) -> list[str]:
        return [f for f, r in self.check(live_data).items() if r["drifted"]]

    def drift_score(self, live_data: pd.DataFrame) -> float:
        """Fraction of features that have drifted (0 = none, 1 = all)."""
        results = self.check(live_data)
        if not results:
            return 0.0
        return sum(1 for r in results.values() if r["drifted"]) / len(results)
