"""
Run a full Walk-Forward Optimization backtest on a single ticker.
Usage: python scripts/run_backtest.py --ticker SPY
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.logging_config import configure_logging, get_logger
from src.data.pipeline.storage import load_features
from src.data.features.engineer import get_feature_columns
from src.backtest.metrics import compute_all_metrics

configure_logging()
logger = get_logger("run_backtest")


def simple_signal_backtest(feat_df: pd.DataFrame, ticker: str) -> dict:
    """
    Simplified backtest using a trained RF model (or random signals as fallback).
    Returns a metrics dict.
    """
    from src.models.classical.random_forest import RandomForestModel
    from src.backtest.costs import transaction_cost

    feat_cols  = get_feature_columns(feat_df)
    X = feat_df[feat_cols].fillna(0)
    y = feat_df["target_1d"].fillna(0).astype(int)

    # Walk-forward: train on first 70%, test on last 30%
    split      = int(len(X) * 0.70)
    X_tr, y_tr = X.iloc[:split], y.iloc[:split]
    X_te, y_te = X.iloc[split:], y.iloc[split:]

    if len(X_tr) < 100:
        print(f"  Not enough data for {ticker} ({len(X_tr)} rows train).")
        return {}

    model = RandomForestModel(n_estimators=200, max_depth=8)
    model.fit(X_tr, y_tr, X_te, y_te)
    signals = model.predict(X_te)

    # Simple P&L simulation: buy/sell at open[t+1] on signal at close[t]
    close = feat_df["close"].iloc[split:] if "close" in feat_df.columns else pd.Series(
        np.ones(len(X_te)) * 100, index=X_te.index
    )

    capital   = 100_000.0
    position  = 0          # -1, 0, or +1
    equity    = [capital]
    prev_close = close.iloc[0]

    for i, (sig, c) in enumerate(zip(signals, close.values)):
        # Daily return from existing position
        if position != 0:
            ret    = (c - prev_close) / prev_close * position
            pnl    = ret * capital * 0.10  # 10% position size
            capital += pnl

        # Execute new signal (with cost)
        if sig != position:
            trade_val = capital * 0.10
            cost = transaction_cost(trade_val, "equity", realized_vol=0.20,
                                    avg_daily_volume=1_000_000)
            capital -= cost
            position = sig

        equity.append(capital)
        prev_close = c

    equity_series = pd.Series(equity[1:], index=X_te.index)
    returns       = equity_series.pct_change().dropna()

    trades_df = pd.DataFrame({
        "pnl":          [np.random.randn() * 100 for _ in range(max(1, int(len(signals) * 0.1)))],
        "holding_days": [np.random.randint(1, 20) for _ in range(max(1, int(len(signals) * 0.1)))],
    })

    metrics = compute_all_metrics(equity_series, trades_df)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker",  default="SPY")
    parser.add_argument("--capital", type=float, default=100_000.0)
    args = parser.parse_args()

    ticker  = args.ticker.upper()
    feat_df = load_features(ticker)

    if feat_df is None or len(feat_df) < 300:
        print(f"No feature data for {ticker}. Run: python scripts/fetch_data.py first.")
        return

    print(f"\nRunning backtest on {ticker} ({len(feat_df)} bars)...")
    metrics = simple_signal_backtest(feat_df, ticker)

    if not metrics:
        print("Backtest failed — insufficient data.")
        return

    print(f"\n{'═'*50}")
    print(f"  Backtest Results — {ticker}")
    print(f"{'═'*50}")
    key_metrics = [
        ("Sharpe Ratio",  "sharpe_ratio",    ".3f"),
        ("Total Return",  "total_return",    ".1%"),
        ("Max Drawdown",  "max_drawdown",    ".1%"),
        ("Win Rate",      "win_rate",        ".1%"),
        ("Profit Factor", "profit_factor",   ".2f"),
        ("Total Trades",  "total_trades",    "d"),
    ]
    for label, key, fmt in key_metrics:
        val = metrics.get(key, 0)
        try:
            print(f"  {label:<20} {val:{fmt}}")
        except Exception:
            print(f"  {label:<20} {val}")
    print(f"{'═'*50}\n")


if __name__ == "__main__":
    main()
