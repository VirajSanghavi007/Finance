from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.config.constants import TRADING_DAYS
from src.config.logging_config import get_logger

logger = get_logger(__name__)

_DEFAULT_CONFIDENCE = 0.99
_DEFAULT_HORIZON    = 1  # days


class VaRCalculator:
    """
    Value-at-Risk and Conditional Value-at-Risk (CVaR/Expected Shortfall).

    Supports:
      - Historical simulation (non-parametric)
      - Parametric (Gaussian)
      - Cornish-Fisher expansion (handles skew + kurtosis)
    """

    def __init__(
        self,
        confidence: float = _DEFAULT_CONFIDENCE,
        horizon: int = _DEFAULT_HORIZON,
    ) -> None:
        self.confidence = confidence
        self.horizon    = horizon

    def historical_var(self, returns: pd.Series | np.ndarray) -> float:
        """1-day VaR via historical simulation (negative number = loss)."""
        r = np.asarray(returns)
        r = r[~np.isnan(r)]
        if len(r) < 30:
            return 0.0
        scaled = r * np.sqrt(self.horizon)
        return float(np.percentile(scaled, (1 - self.confidence) * 100))

    def parametric_var(self, returns: pd.Series | np.ndarray) -> float:
        """1-day VaR assuming normal distribution."""
        r = np.asarray(returns)
        r = r[~np.isnan(r)]
        if len(r) < 30:
            return 0.0
        mu  = np.mean(r)
        sig = np.std(r)
        z   = stats.norm.ppf(1 - self.confidence)
        return float((mu + z * sig) * np.sqrt(self.horizon))

    def cornish_fisher_var(self, returns: pd.Series | np.ndarray) -> float:
        """VaR adjusted for skewness and excess kurtosis via Cornish-Fisher."""
        r = np.asarray(returns)
        r = r[~np.isnan(r)]
        if len(r) < 30:
            return 0.0
        mu   = np.mean(r)
        sig  = np.std(r)
        sk   = float(stats.skew(r))
        ku   = float(stats.kurtosis(r))  # excess kurtosis
        z    = stats.norm.ppf(1 - self.confidence)
        # CF adjustment
        z_cf = (z
                + (z**2 - 1) * sk / 6
                + (z**3 - 3*z) * ku / 24
                - (2*z**3 - 5*z) * sk**2 / 36)
        return float((mu + z_cf * sig) * np.sqrt(self.horizon))

    def cvar(self, returns: pd.Series | np.ndarray) -> float:
        """
        Conditional VaR (Expected Shortfall): mean of losses beyond VaR.
        Returns a negative number (magnitude of expected loss).
        """
        r = np.asarray(returns)
        r = r[~np.isnan(r)]
        if len(r) < 30:
            return 0.0
        var = self.historical_var(r)
        tail = r[r <= var]
        return float(np.mean(tail)) if len(tail) else var

    def compute_all(self, returns: pd.Series | np.ndarray) -> dict[str, float]:
        return {
            "var_historical":    self.historical_var(returns),
            "var_parametric":    self.parametric_var(returns),
            "var_cornish_fisher": self.cornish_fisher_var(returns),
            "cvar":              self.cvar(returns),
            "confidence":        self.confidence,
            "horizon_days":      self.horizon,
        }
