from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: int = 1,
    benchmark_sharpe: float = 0.0,
) -> float:
    """
    Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

    Corrects for:
      1. Multiple-testing bias — penalises strategies selected from N_trials
      2. Non-normality — skewness and excess kurtosis of returns

    Returns probability ∈ [0,1] that the true SR exceeds the expected
    maximum under the null of N_trials independent draws.
    Values > 0.95 indicate statistical significance.
    """
    try:
        from scipy.stats import norm
    except ImportError:
        return 0.0

    T = len(returns)
    if T < 30:
        return 0.0

    mean_r = float(returns.mean())
    std_r  = float(returns.std())
    if std_r == 0:
        return 0.0

    sr_hat = mean_r / std_r          # non-annualized SR
    skew   = float(returns.skew())
    kurt   = float(returns.kurt())   # excess kurtosis

    # Variance of SR estimator under non-normality
    var_sr = (1.0 - skew * sr_hat + (kurt / 4.0) * sr_hat ** 2) / T
    if var_sr <= 0:
        return 0.0
    std_sr = math.sqrt(var_sr)

    # Expected maximum SR under N_trials (from extreme value theory)
    if n_trials > 1:
        expected_max_z = norm.ppf(1.0 - 1.0 / n_trials)
        sr_expected    = expected_max_z * std_sr
    else:
        # Compare against benchmark Sharpe (converted to daily)
        sr_expected = benchmark_sharpe / math.sqrt(TRADING_DAYS)

    z_score = (sr_hat - sr_expected) / std_sr
    return float(norm.cdf(z_score))


def _sharpe(returns: pd.Series, rf_daily: float) -> float:
    excess = returns - rf_daily
    std = excess.std()
    if std == 0 or math.isnan(std):
        return 0.0
    return float(excess.mean() / std * math.sqrt(TRADING_DAYS))


def _sortino(returns: pd.Series, rf_daily: float) -> float:
    excess = returns - rf_daily
    downside = excess[excess < 0]
    if len(downside) < 2:
        return 0.0
    ds_std = downside.std()
    if ds_std == 0:
        return 0.0
    return float(excess.mean() / ds_std * math.sqrt(TRADING_DAYS))


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    rolling_max = equity.cummax()
    dd = (equity - rolling_max) / rolling_max.replace(0, np.nan)
    return float(dd.min())


def _drawdown_duration(equity: pd.Series) -> int:
    if equity.empty:
        return 0
    rolling_max = equity.cummax()
    in_dd = equity < rolling_max
    if not in_dd.any():
        return 0
    # Count longest consecutive drawdown period
    max_dur = 0
    cur_dur = 0
    for v in in_dd:
        if v:
            cur_dur += 1
            max_dur = max(max_dur, cur_dur)
        else:
            cur_dur = 0
    return max_dur


def _omega_ratio(returns: pd.Series, threshold: float) -> float:
    above = (returns[returns > threshold] - threshold).sum()
    below = (threshold - returns[returns < threshold]).sum()
    if below == 0:
        return float("inf")
    return float(above / below)


def _cagr(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] == 0:
        return 0.0
    years = len(equity) / TRADING_DAYS
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / max(years, 0.01)) - 1)


def compute_all_metrics(
    equity_curve: pd.Series,
    trades_df: pd.DataFrame,
    risk_free_rate: float = 0.045,
    regime_series: Optional[pd.Series] = None,
    benchmark_returns: Optional[pd.Series] = None,
) -> dict:
    returns = equity_curve.pct_change().dropna()
    rf_daily = risk_free_rate / TRADING_DAYS

    if len(returns) < 5:
        return {"error": "insufficient_data"}

    dd = (equity_curve / equity_curve.cummax() - 1)
    max_dd = float(dd.min())

    total_return = float((equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1)
    ann_return   = float((1 + returns.mean()) ** TRADING_DAYS - 1)
    cagr         = _cagr(equity_curve)
    sharpe       = _sharpe(returns, rf_daily)
    sortino      = _sortino(returns, rf_daily)
    calmar       = ann_return / abs(max_dd) if max_dd != 0 else 0.0
    omega        = _omega_ratio(returns, rf_daily)

    # Information ratio vs benchmark
    info_ratio = 0.0
    if benchmark_returns is not None:
        aligned_bench = benchmark_returns.reindex(returns.index).fillna(0)
        active = returns - aligned_bench
        ir_std = active.std()
        info_ratio = float(active.mean() / ir_std * math.sqrt(TRADING_DAYS)) if ir_std > 0 else 0.0

    # Trade stats
    n_trades = len(trades_df) if not trades_df.empty else 0
    if n_trades > 0 and "pnl" in trades_df.columns:
        wins  = trades_df[trades_df["pnl"] > 0]
        loses = trades_df[trades_df["pnl"] < 0]
        win_rate      = len(wins) / n_trades
        profit_factor = (wins["pnl"].sum() / abs(loses["pnl"].sum())
                         if len(loses) > 0 and loses["pnl"].sum() != 0 else 0.0)
        avg_win  = float(wins["pnl"].mean())  if len(wins)  > 0 else 0.0
        avg_loss = float(loses["pnl"].mean()) if len(loses) > 0 else 0.0
        largest_win  = float(wins["pnl"].max())  if len(wins)  > 0 else 0.0
        largest_loss = float(loses["pnl"].min()) if len(loses) > 0 else 0.0
        avg_hold = float(trades_df["holding_days"].mean()) if "holding_days" in trades_df else 0.0
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
    else:
        win_rate = profit_factor = avg_win = avg_loss = 0.0
        largest_win = largest_loss = avg_hold = expectancy = 0.0

    trades_per_yr = (n_trades / (len(returns) / TRADING_DAYS)) if len(returns) > 0 else 0.0

    # Distribution stats
    skew = float(returns.skew())
    kurt = float(returns.kurt())
    var_95  = float(returns.quantile(0.05))
    cvar_95_mask = returns < returns.quantile(0.05)
    cvar_95 = float(returns[cvar_95_mask].mean()) if cvar_95_mask.any() else var_95

    # Regime-specific Sharpe
    sharpe_bull = sharpe_bear = sharpe_side = None
    if regime_series is not None:
        reg = regime_series.reindex(returns.index)
        for label, key in [("bull", "sharpe_bull"), ("bear", "sharpe_bear"), ("sideways", "sharpe_sideways")]:
            mask = reg == label
            if mask.sum() > 20:
                sub = returns[mask]
                val = _sharpe(sub, rf_daily)
                if key == "sharpe_bull":
                    sharpe_bull = val
                elif key == "sharpe_bear":
                    sharpe_bear = val
                else:
                    sharpe_side = val

    dsr = deflated_sharpe_ratio(returns, n_trials=1)

    return {
        "total_return":          total_return,
        "deflated_sharpe_ratio": dsr,
        "annualized_return":     ann_return,
        "cagr":                  cagr,
        "sharpe_ratio":          sharpe,
        "sortino_ratio":         sortino,
        "calmar_ratio":          calmar,
        "omega_ratio":           omega,
        "information_ratio":     info_ratio,
        "max_drawdown":          max_dd,
        "max_drawdown_duration": _drawdown_duration(equity_curve),
        "avg_drawdown":          float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0,
        "recovery_factor":       total_return / abs(max_dd) if max_dd != 0 else 0.0,
        "total_trades":          n_trades,
        "trades_per_year":       trades_per_yr,
        "win_rate":              win_rate,
        "profit_factor":         profit_factor,
        "avg_trade_return":      (trades_df["pnl"].mean() / equity_curve.iloc[0]
                                   if n_trades > 0 else 0.0),
        "avg_win":               avg_win,
        "avg_loss":              avg_loss,
        "largest_win":           largest_win,
        "largest_loss":          largest_loss,
        "avg_holding_days":      avg_hold,
        "expectancy":            expectancy,
        "return_skewness":       skew,
        "return_kurtosis":       kurt,
        "var_95":                var_95,
        "cvar_95":               cvar_95,
        "sharpe_bull":           sharpe_bull,
        "sharpe_bear":           sharpe_bear,
        "sharpe_sideways":       sharpe_side,
    }
