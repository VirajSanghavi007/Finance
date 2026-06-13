"""
Build and register a stacked ensemble model after all base models are trained.

Usage: python scripts/build_ensemble.py [--tickers SPY QQQ AAPL]

Strategy: use the pre-trained base models to predict on the training data
(temporal holdout subsets), then train a meta-learner on those predictions.
This is faster than retraining with Optuna on every OOF fold.

Saves: data/models/{ticker}_stacker.pkl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.logging_config import configure_logging, get_logger
from src.data.features.engineer import get_feature_columns
from src.data.pipeline.storage import load_features
from src.models.ensemble.stacker import StackingEnsemble, LABEL_MAP
from src.models.registry import ModelRegistry

configure_logging()
logger = get_logger("build_ensemble")

N_TEMPORAL_SPLITS = 5   # number of temporal holdout splits
MIN_ROWS          = 300


def _pseudo_oof(
    models: dict,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Generate pseudo-OOF predictions using temporal splits of the TRAINING data.
    Uses ALREADY-TRAINED models (no Optuna re-tuning per fold), which is much
    faster.  The stacker still learns useful model combinations.

    Returns:
        oof_probas: model_name -> (N_tr, 3) probability array
        y_mapped:   mapped target labels 0/1/2 for N_tr rows
    """
    n   = len(X_tr)
    oof = {name: np.full((n, 3), 1/3) for name in models}

    # Temporal splits: earlier portion trains, later portion is OOF
    split_size = n // (N_TEMPORAL_SPLITS + 1)
    for fold in range(N_TEMPORAL_SPLITS):
        # val: the (fold+1)th temporal block
        val_start = (fold + 1) * split_size
        val_end   = min(val_start + split_size, n)
        val_idx   = list(range(val_start, val_end))
        X_val     = X_tr.iloc[val_idx].fillna(0)

        for model_name, model in models.items():
            try:
                probas = model.predict_proba(X_val)
                if probas.ndim == 1:
                    probas = probas.reshape(1, -1)
                oof[model_name][val_idx] = probas
            except Exception as e:
                logger.warning("oof_failed",
                               model=model_name, fold=fold, error=str(e))

    y_mapped = np.array([LABEL_MAP.get(int(v), 1) for v in y_tr.values])
    return oof, y_mapped


def build_ensemble_for_ticker(
    ticker: str,
    registry: ModelRegistry,
) -> bool:
    """Build and save stacker ensemble for one ticker. Returns True on success."""
    feat_df = load_features(ticker)
    if feat_df is None or len(feat_df) < MIN_ROWS:
        print(f"  {ticker}: insufficient data")
        return False

    feat_cols  = get_feature_columns(feat_df)
    X          = feat_df[feat_cols].select_dtypes(include=[np.number]).fillna(0)
    target_col = "target_tb" if "target_tb" in feat_df.columns else "target_1d"
    y          = feat_df[target_col].fillna(0).astype(int)

    # Use training portion only
    split = int(len(X) * 0.70)
    X_tr  = X.iloc[:split]
    y_tr  = y.iloc[:split]

    # Load pre-trained base models
    models = {}
    for suffix in ["rf", "xgb", "lgbm"]:
        name = f"{ticker}_{suffix}"
        try:
            m = registry.load_model(name)
            if m is not None:
                models[name] = m
        except Exception:
            pass

    if len(models) < 2:
        print(f"  {ticker}: only {len(models)} base models, need >= 2")
        return False

    print(f"  {ticker}: {len(models)} models, building pseudo-OOF predictions...", end="", flush=True)

    try:
        oof_probas, y_mapped = _pseudo_oof(models, X_tr, y_tr)
    except Exception as e:
        print(f" FAILED: {e}")
        return False

    # Train meta-learner
    stacker = StackingEnsemble(n_splits=5)
    metrics = stacker.fit(oof_probas, y_mapped)
    print(f" acc={metrics.get('train_acc', 0):.3f}")

    # Save to disk
    try:
        import joblib
        from src.config.constants import PROJECT_ROOT
        model_dir = PROJECT_ROOT / "data" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        out_path  = model_dir / f"{ticker}_stacker.pkl"
        joblib.dump(stacker, out_path)
        logger.info("stacker_saved", ticker=ticker, path=str(out_path))
        return True
    except Exception as e:
        print(f"  {ticker}: save failed: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=None)
    args = parser.parse_args()

    registry = ModelRegistry()
    trained  = registry.list_models()

    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        from collections import defaultdict
        by_ticker: dict = defaultdict(list)
        for name in trained:
            parts = name.rsplit("_", 1)
            if len(parts) == 2:
                by_ticker[parts[0]].append(parts[1])
        tickers = sorted(t for t, models in by_ticker.items() if len(models) >= 2)

    if not tickers:
        print("No tickers with >= 2 base models. Train models first.")
        return

    print(f"\nBuilding stacker ensemble for {len(tickers)} tickers: {tickers}\n")

    success = 0
    for ticker in tickers:
        ok = build_ensemble_for_ticker(ticker, registry)
        if ok:
            success += 1

    print(f"\nDone. Ensemble built for {success}/{len(tickers)} tickers.")
    if success > 0:
        from src.config.constants import PROJECT_ROOT
        print(f"Stacker files: data/models/*_stacker.pkl")


if __name__ == "__main__":
    main()
