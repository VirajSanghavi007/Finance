from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.constants import HMM_N_STATES, HMM_STATE_BEAR, HMM_STATE_BULL, VIX_LOW, VIX_MID, VIX_HIGH
from src.config.logging_config import get_logger

logger = get_logger(__name__)


def _fit_hmm(returns: pd.Series, n_states: int = 3):
    try:
        from hmmlearn import hmm  # type: ignore
        X = returns.values.reshape(-1, 1)
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=100,
            random_state=42,
        )
        model.fit(X)
        return model
    except ImportError:
        logger.warning("hmmlearn_not_installed_using_vol_regime")
        return None


def _vol_regime_fallback(returns: pd.Series, n_states: int = 3) -> np.ndarray:
    """Simple volatility-based regime if hmmlearn not available.
    Uses expanding-window quantiles to avoid lookahead bias."""
    vol = returns.rolling(21).std()
    # Expanding quantiles: only use data up to each point in time
    q33 = vol.expanding(min_periods=21).quantile(0.33)
    q67 = vol.expanding(min_periods=21).quantile(0.67)
    states = np.where(vol <= q33, HMM_STATE_BULL,
             np.where(vol <= q67, 1,  # sideways
                      HMM_STATE_BEAR))
    return states


def compute_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    close   = df["close"]
    returns = np.log(close / close.shift(1)).fillna(0)
    out     = pd.DataFrame(index=df.index)

    # HMM regime detection
    model = _fit_hmm(returns, HMM_N_STATES)
    if model is not None:
        X = returns.values.reshape(-1, 1)
        states     = model.predict(X)
        state_prob = model.predict_proba(X)

        # Re-order states by mean return (bear=0, sideways=1, bull=2)
        means = [model.means_[i, 0] for i in range(HMM_N_STATES)]
        order = np.argsort(means)  # ascending: bear -> sideways -> bull
        remap = {old: new for new, old in enumerate(order)}
        states = np.array([remap[s] for s in states])

        out["reg_hmm_state"]     = states
        out["reg_hmm_bear_prob"] = state_prob[:, order[0]]
        out["reg_hmm_side_prob"] = state_prob[:, order[1]]
        out["reg_hmm_bull_prob"] = state_prob[:, order[2]]
    else:
        states = _vol_regime_fallback(returns)
        out["reg_hmm_state"]     = states
        out["reg_hmm_bear_prob"] = (states == HMM_STATE_BEAR).astype(float)
        out["reg_hmm_side_prob"] = (states == 1).astype(float)
        out["reg_hmm_bull_prob"] = (states == HMM_STATE_BULL).astype(float)

    # Trend strength: ADX(14)
    high = df["high"]
    low  = df["low"]
    tr   = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    dm_pos = (high.diff()).clip(lower=0)
    dm_neg = (-low.diff()).clip(lower=0)
    dm_pos = dm_pos.where(dm_pos > dm_neg, 0)
    dm_neg = dm_neg.where(dm_neg > dm_pos, 0)
    atr14  = tr.ewm(alpha=1/14, adjust=False).mean()
    di_pos = 100 * dm_pos.ewm(alpha=1/14, adjust=False).mean() / atr14.replace(0, np.nan)
    di_neg = 100 * dm_neg.ewm(alpha=1/14, adjust=False).mean() / atr14.replace(0, np.nan)
    dx     = 100 * (di_pos - di_neg).abs() / (di_pos + di_neg).replace(0, np.nan)
    out["reg_trend_strength"] = dx.ewm(alpha=1/14, adjust=False).mean()

    # Trend direction: 50d vs 200d MA crossover
    ma50  = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    out["reg_trend_direction"] = np.where(ma50 > ma200, 1.0, -1.0)

    return out
