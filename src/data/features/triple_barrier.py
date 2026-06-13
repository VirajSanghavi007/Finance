"""
Triple Barrier Labeling (López de Prado, AFML Ch.3)

Labels each bar as:
  +1  → price hit the upper (profit-take) barrier first
  -1  → price hit the lower (stop-loss) barrier first
   0  → neither barrier hit within the time window (timeout)

This replaces the naive sign(next_close - close) label used previously.
The key insight: a trade is only profitable if it reaches your profit target
before it hits your stop-loss, within a maximum holding window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def triple_barrier_labels(
    close: pd.Series,
    daily_vol: pd.Series,
    pt_sl: tuple[float, float] = (1.5, 1.0),
    max_holding: int = 10,
    min_ret: float = 0.0,
) -> pd.DataFrame:
    """
    Compute triple barrier labels for every bar.

    Args:
        close:       Closing price series.
        daily_vol:   Daily volatility (e.g. 21-day rolling std of log returns).
                     Used to set adaptive barrier widths.
        pt_sl:       (profit_take_multiplier, stop_loss_multiplier) × daily_vol.
                     e.g. (1.5, 1.0) means PT = 1.5σ above entry, SL = 1.0σ below.
        max_holding: Maximum number of bars to hold before timeout (vertical barrier).
        min_ret:     Minimum absolute return to count as a signal (filter noise).

    Returns:
        DataFrame with columns:
          label       — final label: +1, -1, 0
          ret         — actual return over the holding period
          t_exit      — timestamp when barrier was hit
          barrier_hit — 'pt', 'sl', or 'timeout'
    """
    pt_mult, sl_mult = pt_sl
    out_label   : list[int]   = []
    out_ret     : list[float] = []
    out_t_exit  : list        = []
    out_barrier : list[str]   = []

    close_arr   = close.values
    vol_arr     = daily_vol.values
    n           = len(close_arr)

    for i in range(n):
        sigma = vol_arr[i]
        if np.isnan(sigma) or sigma <= 0:
            out_label.append(0)
            out_ret.append(0.0)
            out_t_exit.append(close.index[i])
            out_barrier.append("nan_vol")
            continue

        entry  = close_arr[i]
        pt     = entry * (1 + pt_mult  * sigma)   # upper barrier
        sl     = entry * (1 - sl_mult  * sigma)   # lower barrier
        horizon = min(i + max_holding, n - 1)

        label   = 0
        ret     = 0.0
        t_exit  = close.index[min(horizon, n - 1)]
        barrier = "timeout"

        for j in range(i + 1, horizon + 1):
            p = close_arr[j]
            if p >= pt:
                label   = 1
                ret     = (p - entry) / entry
                t_exit  = close.index[j]
                barrier = "pt"
                break
            if p <= sl:
                label   = -1
                ret     = (p - entry) / entry
                t_exit  = close.index[j]
                barrier = "sl"
                break
        else:
            # vertical barrier: use actual return at horizon
            if horizon < n:
                ret = (close_arr[horizon] - entry) / entry
                label = int(np.sign(ret)) if abs(ret) > min_ret else 0

        out_label.append(label)
        out_ret.append(ret)
        out_t_exit.append(t_exit)
        out_barrier.append(barrier)

    return pd.DataFrame(
        {
            "label":       out_label,
            "ret":         out_ret,
            "t_exit":      out_t_exit,
            "barrier_hit": out_barrier,
        },
        index=close.index,
    )


def compute_triple_barrier_targets(
    df: pd.DataFrame,
    pt_sl: tuple[float, float] = (1.5, 1.0),
    max_holding: int = 10,
    vol_window: int  = 21,
) -> pd.DataFrame:
    """
    Compute all target columns using triple barrier labeling.
    Drops the old naive sign(forward return) approach.

    Returns a DataFrame with:
      target_1d       — next-day log return sign (kept for compatibility)
      target_5d       — 5-day log return sign (kept for compatibility)
      target_ret_1d   — raw next-day log return
      target_ret_5d   — raw 5-day log return
      target_vol_adj_1d — vol-adjusted 1d return
      target_tb       — triple barrier label (+1/-1/0) ← new primary target
      target_tb_ret   — actual return until barrier hit  ← new
      target_tb_barrier — which barrier was hit          ← new (for analysis)
    """
    close = df["close"]

    # Daily vol: rolling std of log returns (no lookahead — uses only past)
    log_ret   = np.log(close / close.shift(1))
    daily_vol = log_ret.rolling(vol_window, min_periods=5).std()

    # Triple barrier
    tb = triple_barrier_labels(close, daily_vol, pt_sl=pt_sl, max_holding=max_holding)

    # Keep original targets for backward compatibility
    log_ret_1d = np.log(close.shift(-1) / close)
    log_ret_5d = np.log(close.shift(-5) / close)
    past_vol   = log_ret.rolling(5).std()

    tgt = pd.DataFrame(index=df.index)
    tgt["target_1d"]          = np.sign(log_ret_1d).fillna(0).astype(int)
    tgt["target_5d"]          = np.sign(log_ret_5d).fillna(0).astype(int)
    tgt["target_ret_1d"]      = log_ret_1d
    tgt["target_ret_5d"]      = log_ret_5d
    tgt["target_vol_adj_1d"]  = log_ret_1d / past_vol.replace(0, np.nan)
    tgt["target_tb"]          = tb["label"]
    tgt["target_tb_ret"]      = tb["ret"]
    tgt["target_tb_barrier"]  = tb["barrier_hit"]

    return tgt


# Extended TARGET_COLS that includes triple barrier targets
TB_TARGET_COLS = [
    "target_1d", "target_5d",
    "target_ret_1d", "target_ret_5d", "target_vol_adj_1d",
    "target_tb", "target_tb_ret", "target_tb_barrier",
]
