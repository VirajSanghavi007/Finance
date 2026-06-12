from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.metrics import compute_all_metrics
from src.config.logging_config import get_logger

logger = get_logger(__name__)


def monte_carlo_simulation(
    trades_df: pd.DataFrame,
    initial_capital: float = 100_000.0,
    n_simulations: int = 10_000,
    risk_free_rate: float = 0.045,
    random_state: int = 42,
) -> dict:
    """
    Bootstrap resample trades to estimate distribution of outcomes.
    Returns confidence intervals for Sharpe, max drawdown, and ruin probability.
    """
    if trades_df.empty or "pnl" not in trades_df.columns:
        return {}

    rng = np.random.default_rng(random_state)
    pnl_arr = trades_df["pnl"].values
    n_trades = len(pnl_arr)

    sharpes:  list[float] = []
    max_dds:  list[float] = []
    end_vals: list[float] = []

    for _ in range(n_simulations):
        sampled = rng.choice(pnl_arr, size=n_trades, replace=True)
        equity = np.concatenate([[initial_capital], initial_capital + np.cumsum(sampled)])
        equity_s = pd.Series(equity)
        returns  = equity_s.pct_change().dropna()
        if len(returns) < 2:
            continue

        rf_daily = risk_free_rate / 252
        excess   = returns - rf_daily
        std      = excess.std()
        sharpe   = float(excess.mean() / std * np.sqrt(252)) if std > 0 else 0.0
        max_dd   = float((equity_s / equity_s.cummax() - 1).min())

        sharpes.append(sharpe)
        max_dds.append(max_dd)
        end_vals.append(equity[-1])

    if not sharpes:
        return {}

    sharpes_arr  = np.array(sharpes)
    max_dds_arr  = np.array(max_dds)
    end_vals_arr = np.array(end_vals)

    return {
        "sharpe_p5":     float(np.percentile(sharpes_arr, 5)),
        "sharpe_p50":    float(np.percentile(sharpes_arr, 50)),
        "sharpe_p95":    float(np.percentile(sharpes_arr, 95)),
        "max_dd_p50":    float(np.percentile(max_dds_arr, 50)),
        "max_dd_p95":    float(np.percentile(max_dds_arr, 95)),  # worst-case 95th
        "prob_20pct_loss": float(np.mean(end_vals_arr < initial_capital * 0.8)),
        "prob_double":     float(np.mean(end_vals_arr > initial_capital * 2.0)),
        "n_simulations": n_simulations,
    }
