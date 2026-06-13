"""
Run Walk-Forward Optimization backtest across all trained tickers.

Usage:
    python scripts/run_full_backtest.py
    python scripts/run_full_backtest.py --tickers SPY QQQ AAPL
    python scripts/run_full_backtest.py --ticker SPY  (single ticker)

Saves JSON results to data/backtest_results/wfo_{timestamp}.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.logging_config import configure_logging, get_logger
from src.data.features.engineer import get_feature_columns
from src.data.pipeline.storage import load_features
from src.backtest.metrics import compute_all_metrics
from src.backtest.costs import transaction_cost
from src.models.registry import ModelRegistry

configure_logging()
logger = get_logger("run_full_backtest")

TRAIN_FRAC  = 0.70
MIN_ROWS    = 300
CAPITAL     = 100_000.0
POSITION_SZ = 1.00   # full capital per trade (single-asset backtest)


def backtest_ticker(
    ticker: str,
    feat_df: pd.DataFrame,
    registry: ModelRegistry,
) -> dict:
    """
    Simple 70/30 walk-forward backtest using best available model from registry.
    Returns metrics dict.
    """
    feat_cols = get_feature_columns(feat_df)
    X = feat_df[feat_cols].select_dtypes(include=[np.number]).fillna(0)
    target_col = "target_tb" if "target_tb" in feat_df.columns else "target_1d"
    y = feat_df[target_col].fillna(0).astype(int)

    split = int(len(X) * TRAIN_FRAC)
    X_te = X.iloc[split:]
    y_te = y.iloc[split:]

    if len(X_te) < 50:
        return {"error": "insufficient_test_data", "ticker": ticker}

    # Try models in order of expected quality
    model = None
    model_name_used = None
    for suffix in ["xgb", "lgbm", "rf"]:
        name = f"{ticker}_{suffix}"
        try:
            m = registry.load_model(name)
            if m is not None:
                model = m
                model_name_used = name
                break
        except Exception:
            pass

    if model is None:
        logger.warning("no_model_found", ticker=ticker)
        return {"error": "no_model", "ticker": ticker}

    # Generate signals on test set
    try:
        signals = model.predict(X_te)
    except Exception as e:
        logger.error("predict_failed", ticker=ticker, error=str(e))
        return {"error": str(e), "ticker": ticker}

    # Simulate P&L: use target_ret_1d as forward return proxy
    if "target_ret_1d" in feat_df.columns:
        fwd_rets = feat_df["target_ret_1d"].iloc[split:].reindex(X_te.index).fillna(0)
    else:
        fwd_rets = y_te.astype(float) * 0.005  # rough proxy

    # Shift fwd_rets so signal[i] (set at close[i]) earns fwd_ret[i]
    # (return from close[i] to close[i+1]).
    # In the loop we apply fwd_r BEFORE updating position, so without
    # shift we'd earn fwd_ret[i] with signal[i-1].  Shifting forward
    # by 1 corrects this: at iteration i we now hold signal[i-1] and
    # apply fwd_ret[i-1], which is exactly the return of bar i.
    fwd_rets_arr = np.concatenate([[0.0], fwd_rets.values[:-1]])

    capital   = CAPITAL
    position  = 0
    equity    = []
    trade_records = []
    entry_cap = CAPITAL
    entry_sig = 0

    for sig, fwd_r in zip(signals, fwd_rets_arr):
        # Apply existing position P&L
        if position != 0:
            pnl = position * fwd_r * capital * POSITION_SZ
            capital += pnl

        # Trade on signal change
        if sig != position:
            trade_val = capital * POSITION_SZ
            cost = transaction_cost(trade_val, "equity",
                                    realized_vol=0.20, avg_daily_volume=1_000_000)
            capital -= cost

            # Record closed trade
            if entry_sig != 0:
                trade_pnl = capital - entry_cap
                trade_records.append({"pnl": trade_pnl, "holding_days": 1})

            position  = int(sig)
            entry_cap = capital
            entry_sig = int(sig)

        equity.append(capital)

    equity_series = pd.Series(equity, index=X_te.index)
    trades_df = pd.DataFrame(trade_records) if trade_records else pd.DataFrame(
        {"pnl": [0.0], "holding_days": [1]}
    )

    metrics = compute_all_metrics(equity_series, trades_df)
    metrics["ticker"]      = ticker
    metrics["model_used"]  = model_name_used
    metrics["test_bars"]   = len(X_te)
    metrics["train_bars"]  = split
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--ticker",  default=None, help="Single ticker shortcut")
    args = parser.parse_args()

    registry = ModelRegistry()
    trained  = registry.list_models()

    # Determine which tickers to run
    if args.ticker:
        tickers_to_run = [args.ticker.upper()]
    elif args.tickers:
        tickers_to_run = [t.upper() for t in args.tickers]
    else:
        # Run all tickers that have a registered model
        tickers_to_run = sorted({n.rsplit("_", 1)[0] for n in trained})

    if not tickers_to_run:
        print("No trained models found. Run scripts/train_models.py --register first.")
        return

    print(f"\nRunning WFO backtest on {len(tickers_to_run)} tickers: {tickers_to_run}\n")

    results = []
    for ticker in tickers_to_run:
        feat_df = load_features(ticker)
        if feat_df is None or len(feat_df) < MIN_ROWS:
            print(f"  {ticker}: no features data, skipping.")
            continue

        print(f"  {ticker}: {len(feat_df)} bars ... ", end="", flush=True)
        metrics = backtest_ticker(ticker, feat_df, registry)

        if "error" in metrics:
            print(f"SKIPPED ({metrics['error']})")
        else:
            sh = metrics.get("sharpe_ratio", 0)
            ret = metrics.get("total_return", 0)
            dd  = metrics.get("max_drawdown", 0)
            print(f"Sharpe={sh:.2f}  Ret={ret:.1%}  MaxDD={dd:.1%}  Model={metrics.get('model_used')}")
            results.append(metrics)

    if not results:
        print("No results generated.")
        return

    # Aggregate
    sharpes = [r["sharpe_ratio"] for r in results if "sharpe_ratio" in r]
    agg = {
        "n_tickers":     len(results),
        "mean_sharpe":   float(np.mean(sharpes)),
        "median_sharpe": float(np.median(sharpes)),
        "pct_positive":  float(sum(s > 0 for s in sharpes) / len(sharpes)),
        "pct_above_1":   float(sum(s > 1 for s in sharpes) / len(sharpes)),
        "run_at":        datetime.now(timezone.utc).isoformat(),
    }

    print(f"\n{'='*55}")
    print(f"  Aggregate Results ({len(results)} tickers)")
    print(f"{'='*55}")
    print(f"  Mean Sharpe:         {agg['mean_sharpe']:.3f}")
    print(f"  Median Sharpe:       {agg['median_sharpe']:.3f}")
    print(f"  % Positive Sharpe:   {agg['pct_positive']:.0%}")
    print(f"  % Sharpe > 1.0:      {agg['pct_above_1']:.0%}")
    print(f"{'='*55}\n")

    # Top 5 by Sharpe
    top = sorted(results, key=lambda r: r.get("sharpe_ratio", -999), reverse=True)[:5]
    print("  Top 5 by Sharpe:")
    for r in top:
        print(f"    {r['ticker']:<8}  Sharpe={r.get('sharpe_ratio',0):.2f}  "
              f"CAGR={r.get('cagr',0):.1%}  MaxDD={r.get('max_drawdown',0):.1%}")

    # Save JSON results
    out_dir = Path("data/backtest_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"wfo_{ts}.json"

    # Convert numpy types for JSON serialization
    def _jsonify(obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    output = {
        "aggregated": agg,
        "tickers": [
            {k: _jsonify(v) for k, v in r.items()}
            for r in results
        ],
        "folds": [
            {
                "Fold":     i + 1,
                "Ticker":   r.get("ticker", "?"),
                "Sharpe":   round(r.get("sharpe_ratio", 0), 3),
                "CAGR %":   round(r.get("cagr", 0) * 100, 1),
                "Max DD %": round(abs(r.get("max_drawdown", 0)) * 100, 1),
                "Win Rate": round(r.get("win_rate", 0) * 100, 1),
                "N Trades": r.get("total_trades", 0),
                "Model":    r.get("model_used", "?"),
            }
            for i, r in enumerate(results)
        ],
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=_jsonify)

    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
