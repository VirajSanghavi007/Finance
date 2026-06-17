"""
Start live paper trading loop using Alpaca paper account.
Loads trained models from the registry, fetches current market data,
generates ensemble signals, and submits orders.

Usage: python scripts/start_paper_trading.py [--tickers SPY QQQ AAPL] [--interval 60]
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.logging_config import configure_logging, get_logger
from src.config.settings import get_settings

configure_logging()
logger = get_logger("paper_trader")

DEFAULT_TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]


def _load_latest_models(tickers: list[str]) -> dict:
    """Load the latest registered model for each ticker (tries xgb, lgbm, rf in order)."""
    from src.models.registry import ModelRegistry

    registry = ModelRegistry()
    models = {}
    for ticker in tickers:
        for model_type in ("xgb", "lgbm", "rf"):
            name = f"{ticker}_{model_type}"
            try:
                model = registry.load_model(name)
                models[ticker] = model
                logger.info("model_loaded", ticker=ticker, model=model_type)
                break
            except (KeyError, Exception):
                continue
        if ticker not in models:
            logger.warning("no_model_found", ticker=ticker)
    return models


def _fetch_current_features(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Load latest feature rows from processed parquet files."""
    from src.data.pipeline.storage import load_features
    from src.data.features.engineer import get_feature_columns

    feature_rows = {}
    for ticker in tickers:
        try:
            df = load_features(ticker)
            if df is not None and len(df) > 0:
                feat_cols = get_feature_columns(df)
                X = df[feat_cols].fillna(0).select_dtypes(include=[np.number])
                feature_rows[ticker] = X.iloc[[-1]]  # most recent bar
        except Exception as e:
            logger.warning("feature_load_failed", ticker=ticker, error=str(e))
    return feature_rows


def _fetch_current_prices(tickers: list[str]) -> dict[str, float]:
    """Fetch latest available close prices via yfinance (works on weekends too)."""
    try:
        import yfinance as yf
        prices = {}
        for ticker in tickers:
            try:
                t = yf.Ticker(ticker)
                hist = t.history(period="5d")
                if len(hist) > 0:
                    prices[ticker] = float(hist["Close"].iloc[-1])
            except Exception:
                pass
        return prices
    except ImportError:
        return {}


def _generate_signals(
    models: dict,
    feature_rows: dict[str, pd.DataFrame],
) -> tuple[dict[str, int], dict[str, float]]:
    """Generate signals and confidences for each ticker."""
    signals: dict[str, int] = {}
    confidences: dict[str, float] = {}

    for ticker, model in models.items():
        if ticker not in feature_rows:
            continue
        try:
            X = feature_rows[ticker]
            proba = model.predict_proba(X)[0]  # shape (3,) for [-1, 0, 1]
            signal = int(np.argmax(proba)) - 1  # [0,1,2] -> [-1,0,1]
            confidence = float(proba.max())
            signals[ticker] = signal
            confidences[ticker] = confidence
            logger.info(
                "signal_generated",
                ticker=ticker,
                signal=signal,
                confidence=round(confidence, 3),
            )
        except Exception as e:
            logger.warning("signal_failed", ticker=ticker, error=str(e))

    return signals, confidences


def _submit_to_alpaca(
    signals: dict[str, int],
    confidences: dict[str, float],
    prices: dict[str, float],
    api_key: str,
    secret_key: str,
    min_confidence: float = 0.55,
    capital_per_trade: float = 5_000.0,
) -> None:
    """Submit buy/sell orders to Alpaca paper account."""
    try:
        import alpaca_trade_api as tradeapi
    except ImportError:
        logger.error("alpaca_sdk_not_installed", msg="pip install alpaca-trade-api")
        return

    api = tradeapi.REST(
        api_key, secret_key, "https://paper-api.alpaca.markets"
    )

    try:
        account = api.get_account()
        logger.info(
            "alpaca_account",
            portfolio_value=account.portfolio_value,
            cash=account.cash,
            status=account.status,
        )
    except Exception as e:
        logger.error("alpaca_account_failed", error=str(e))
        return

    # Check market is open
    clock = api.get_clock()
    if not clock.is_open:
        logger.info("market_closed", next_open=str(clock.next_open))
        return

    for ticker, signal in signals.items():
        conf = confidences.get(ticker, 0.0)
        price = prices.get(ticker, 0.0)

        if conf < min_confidence:
            logger.info("signal_skipped_low_confidence", ticker=ticker, confidence=conf)
            continue

        if signal == 0 or price <= 0:
            continue

        qty = max(1, int(capital_per_trade / price))
        side = "buy" if signal == 1 else "sell"

        try:
            order = api.submit_order(
                symbol=ticker,
                qty=qty,
                side=side,
                type="market",
                time_in_force="day",
            )
            logger.info(
                "order_submitted",
                ticker=ticker,
                side=side,
                qty=qty,
                price=price,
                order_id=order.id,
            )
        except Exception as e:
            logger.warning("order_failed", ticker=ticker, error=str(e))


def run_once(tickers: list[str], settings) -> None:
    """Run one iteration of the trading loop."""
    logger.info("trading_loop_start", tickers=tickers, time=datetime.now(timezone.utc).isoformat())

    models = _load_latest_models(tickers)
    if not models:
        logger.warning("no_models_loaded", msg="Train models first: python scripts/train_models.py")
        return

    feature_rows = _fetch_current_features(tickers)
    prices = _fetch_current_prices(tickers)
    signals, confidences = _generate_signals(models, feature_rows)

    if not signals:
        logger.info("no_signals_generated")
        return

    print("\n--- Signals ---")
    for ticker in sorted(signals):
        sig_str = {1: "LONG", -1: "SHORT", 0: "FLAT"}.get(signals[ticker], "?")
        conf = confidences.get(ticker, 0.0)
        price = prices.get(ticker, 0.0)
        print(f"  {ticker:<8} {sig_str:<6}  conf={conf:.2f}  price=${price:.2f}")
    print()

    if settings.alpaca_api_key and settings.alpaca_secret_key:
        _submit_to_alpaca(
            signals=signals,
            confidences=confidences,
            prices=prices,
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
        )
    else:
        logger.warning("alpaca_keys_missing", msg="Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--interval", type=int, default=300, help="Seconds between loop iterations")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    args = parser.parse_args()

    settings = get_settings()
    tickers = [t.upper() for t in args.tickers]

    print(f"\nAlgoTrade Paper Trader")
    print(f"  Tickers : {', '.join(tickers)}")
    print(f"  Interval: {args.interval}s")
    print(f"  Alpaca  : {'connected' if settings.alpaca_api_key else 'no key -- signals only'}")
    print()

    if args.once:
        run_once(tickers, settings)
        return

    while True:
        try:
            run_once(tickers, settings)
        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            logger.error("loop_error", error=str(e))

        logger.info("sleeping", seconds=args.interval)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
