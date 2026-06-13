"""
Build and register a stacked ensemble model after all base models are trained.

Usage: python scripts/build_ensemble.py [--tickers SPY QQQ AAPL]

1. Loads all registered models for each ticker
2. Generates OOF (out-of-fold) predictions from base models
3. Trains a meta-learner (LogisticRegression) on OOF predictions
4. Saves the ensemble as '{ticker}_ensemble' in the registry

Requires: at least 2 base models per ticker (rf + xgb or rf + lgbm)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.logging_config import configure_logging, get_logger
from src.data.features.engineer import get_feature_columns
from src.data.pipeline.storage import load_features
from src.models.ensemble.stacker import StackingEnsemble, LABEL_MAP
from src.models.registry import ModelRegistry

configure_logging()
logger = get_logger("build_ensemble")

N_SPLITS  = 5
MIN_ROWS  = 500


def _get_oof_predictions(
    models: dict,
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Generate out-of-fold predictions from each base model using TimeSeriesSplit.
    Returns (oof_probas, y_mapped) where oof_probas is model_name -> (N, 3) array.
    """
    n = len(X)
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    oof: dict[str, np.ndarray] = {
        name: np.full((n, 3), 1/3) for name in models
    }

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr = X.iloc[train_idx].fillna(0)
        y_tr = y.iloc[train_idx]
        X_va = X.iloc[val_idx].fillna(0)

        for model_name, model in models.items():
            try:
                model.fit(X_tr, y_tr, X_va, y.iloc[val_idx])
                probas = model.predict_proba(X_va)
                if probas.ndim == 1:
                    probas = probas.reshape(1, -1)
                oof[model_name][val_idx] = probas
            except Exception as e:
                logger.warning("oof_fold_failed",
                               model=model_name, fold=fold_idx, error=str(e))

    y_mapped = np.array([LABEL_MAP.get(int(v), 1) for v in y.values])
    return oof, y_mapped


def build_ensemble_for_ticker(
    ticker: str,
    registry: ModelRegistry,
) -> bool:
    """Build and register ensemble for one ticker. Returns True on success."""
    # Load feature data
    feat_df = load_features(ticker)
    if feat_df is None or len(feat_df) < MIN_ROWS:
        print(f"  {ticker}: insufficient data ({0 if feat_df is None else len(feat_df)} rows)")
        return False

    feat_cols = get_feature_columns(feat_df)
    X = feat_df[feat_cols].select_dtypes(include=[np.number]).fillna(0)
    target_col = "target_tb" if "target_tb" in feat_df.columns else "target_1d"
    y = feat_df[target_col].fillna(0).astype(int)

    # Use training portion only for ensemble fitting (avoid test contamination)
    split = int(len(X) * 0.70)
    X_tr = X.iloc[:split]
    y_tr = y.iloc[:split]

    # Load base models (we'll refit them on OOF folds)
    model_classes = {}
    for suffix in ["rf", "xgb", "lgbm"]:
        name = f"{ticker}_{suffix}"
        try:
            model = registry.load_model(name)
            if model is not None:
                model_classes[name] = model
        except Exception:
            pass

    if len(model_classes) < 2:
        print(f"  {ticker}: only {len(model_classes)} base models, need >= 2")
        return False

    print(f"  {ticker}: {len(model_classes)} base models, generating OOF predictions...")

    # Generate OOF predictions
    try:
        oof_probas, y_mapped = _get_oof_predictions(model_classes, X_tr, y_tr)
    except Exception as e:
        print(f"  {ticker}: OOF generation failed: {e}")
        return False

    # Train meta-learner
    stacker = StackingEnsemble(n_splits=N_SPLITS)
    metrics = stacker.fit(oof_probas, y_mapped)
    print(f"  {ticker}: meta-learner trained, train_acc={metrics.get('train_acc', 0):.3f}")

    # Register stacker under a simple key
    try:
        import joblib
        from src.config.constants import PROJECT_ROOT
        model_dir = PROJECT_ROOT / "data" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        stacker_path = model_dir / f"{ticker}_stacker.pkl"
        joblib.dump(stacker, stacker_path)
        logger.info("stacker_saved", ticker=ticker, path=str(stacker_path))
        print(f"  {ticker}: ensemble saved to {stacker_path}")
        return True
    except Exception as e:
        print(f"  {ticker}: save failed: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Tickers to build ensemble for (default: all trained)")
    args = parser.parse_args()

    registry = ModelRegistry()
    trained  = registry.list_models()

    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        # All tickers that have at least 2 base models
        from collections import defaultdict
        by_ticker: dict = defaultdict(list)
        for name in trained:
            parts = name.rsplit("_", 1)
            if len(parts) == 2:
                by_ticker[parts[0]].append(parts[1])
        tickers = [t for t, models in by_ticker.items() if len(models) >= 2]

    if not tickers:
        print("No tickers with >= 2 base models. Train models first.")
        return

    print(f"\nBuilding ensemble for {len(tickers)} tickers: {tickers}\n")

    success = 0
    for ticker in sorted(tickers):
        ok = build_ensemble_for_ticker(ticker, registry)
        if ok:
            success += 1

    print(f"\nDone. Built ensemble for {success}/{len(tickers)} tickers.")


if __name__ == "__main__":
    main()
