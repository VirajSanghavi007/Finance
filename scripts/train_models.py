"""
Train all ensemble models on historical feature data.
Usage: python scripts/train_models.py [--tickers SPY QQQ] [--register]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.logging_config import configure_logging, get_logger
from src.data.features.engineer import get_feature_columns, TARGET_COLS
from src.data.pipeline.storage import load_features
from src.models.registry import ModelRegistry

configure_logging()
logger = get_logger("train_models")


def _get_target(feat_df: pd.DataFrame) -> pd.Series:
    """Prefer triple-barrier label; fall back to naive sign(return)."""
    if "target_tb" in feat_df.columns:
        return feat_df["target_tb"].fillna(0).astype(int)
    return feat_df["target_1d"].fillna(0).astype(int)


def _split(feat_df: pd.DataFrame, test_frac: float = 0.15):
    split = int(len(feat_df) * (1 - test_frac))
    feat_cols = get_feature_columns(feat_df)
    X = feat_df[feat_cols].fillna(0)
    # Drop any remaining non-numeric columns (e.g. string categoricals)
    X = X.select_dtypes(include=[np.number])
    y = _get_target(feat_df)
    return X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:]


def train_ticker(ticker: str, register: bool, registry: ModelRegistry) -> dict:
    feat_df = load_features(ticker)
    if feat_df is None or len(feat_df) < 252:
        logger.warning("insufficient_data", ticker=ticker, rows=0 if feat_df is None else len(feat_df))
        return {}

    if "target_tb" not in feat_df.columns and "target_1d" not in feat_df.columns:
        logger.warning("no_target_col", ticker=ticker)
        return {}

    X_tr, y_tr, X_va, y_va = _split(feat_df)
    metrics_all: dict = {}

    # ── RandomForest (always available) ────────────────────────────────────────
    try:
        from src.models.classical.random_forest import RandomForestModel
        rf = RandomForestModel(n_estimators=200, max_depth=8)
        m  = rf.fit(X_tr, y_tr, X_va, y_va)
        if register:
            registry.register(rf, f"{ticker}_rf", metrics={"val_acc": m.get("val_acc", 0)})
        metrics_all["rf"] = m
        logger.info("rf_trained", ticker=ticker, **m)
    except Exception as e:
        logger.error("rf_failed", ticker=ticker, error=str(e))

    # ── XGBoost ────────────────────────────────────────────────────────────────
    try:
        from src.models.classical.xgb_model import XGBoostModel
        xgb = XGBoostModel(n_trials=20)
        m   = xgb.fit(X_tr, y_tr, X_va, y_va)
        if register:
            registry.register(xgb, f"{ticker}_xgb", metrics=m)
        metrics_all["xgb"] = m
        logger.info("xgb_trained", ticker=ticker)
    except ImportError:
        logger.warning("xgboost_not_installed_skipping")
    except Exception as e:
        logger.error("xgb_failed", ticker=ticker, error=str(e))

    # ── LightGBM ───────────────────────────────────────────────────────────────
    try:
        from src.models.classical.lgbm_model import LightGBMModel
        lgbm = LightGBMModel(n_trials=20)
        m    = lgbm.fit(X_tr, y_tr, X_va, y_va)
        if register:
            registry.register(lgbm, f"{ticker}_lgbm", metrics=m)
        metrics_all["lgbm"] = m
        logger.info("lgbm_trained", ticker=ticker)
    except ImportError:
        logger.warning("lightgbm_not_installed_skipping")
    except Exception as e:
        logger.error("lgbm_failed", ticker=ticker, error=str(e))

    return metrics_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Tickers to train on (default: top 5 from EQUITY_UNIVERSE)")
    parser.add_argument("--register", action="store_true",
                        help="Register trained models in ModelRegistry")
    args = parser.parse_args()

    from src.config.assets import EQUITY_UNIVERSE
    tickers = args.tickers or list(EQUITY_UNIVERSE.keys())[:5]

    registry = ModelRegistry()
    total_trained = 0

    for ticker in tickers:
        print(f"\nTraining on {ticker}...")
        metrics = train_ticker(ticker, register=args.register, registry=registry)
        if metrics:
            total_trained += 1
            print(f"  {ticker}: {list(metrics.keys())} trained")
        else:
            print(f"  {ticker}: skipped (no data or insufficient rows)")

    print(f"\nDone. Trained {total_trained}/{len(tickers)} tickers.")
    if args.register:
        print(f"Registered in ModelRegistry: {registry.list_models()}")


if __name__ == "__main__":
    main()
