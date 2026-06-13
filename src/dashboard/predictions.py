"""
Shared prediction utilities for dashboard pages.
Loads trained models from registry and generates real predictions.
"""
from __future__ import annotations

import joblib
from pathlib import Path
from collections import defaultdict
from functools import lru_cache

import numpy as np
import pandas as pd

_LABEL_UNMAP = {0: -1, 1: 0, 2: 1}


def get_trained_tickers() -> list[str]:
    """Return all tickers with at least one trained model in the registry."""
    try:
        from src.models.registry import ModelRegistry
        reg = ModelRegistry()
        names = reg.list_models()
        by_ticker: dict = defaultdict(list)
        for name in names:
            parts = name.rsplit("_", 1)
            if len(parts) == 2:
                by_ticker[parts[0]].append(parts[1])
        return sorted(by_ticker.keys())
    except Exception:
        return []


def get_fully_trained_tickers() -> list[str]:
    """Return tickers with all 3 base models (rf + xgb + lgbm) registered."""
    try:
        from src.models.registry import ModelRegistry
        reg = ModelRegistry()
        names = reg.list_models()
        by_ticker: dict = defaultdict(set)
        for name in names:
            parts = name.rsplit("_", 1)
            if len(parts) == 2:
                by_ticker[parts[0]].add(parts[1])
        return sorted(t for t, v in by_ticker.items() if len(v) >= 3)
    except Exception:
        return []


def predict_ticker(
    ticker: str,
    n_rows: int = 504,
) -> pd.DataFrame | None:
    """
    Generate model predictions for a ticker using the best available model.

    Preference order: stacker ensemble > xgb > lgbm > rf

    Returns DataFrame with columns: date, signal, confidence, source
    or None if no model or data available.
    """
    try:
        from src.models.registry import ModelRegistry
        from src.data.pipeline.storage import load_features
        from src.data.features.engineer import get_feature_columns
        from src.config.constants import PROJECT_ROOT

        df = load_features(ticker)
        if df is None or len(df) < 60:
            return None

        feat_cols = get_feature_columns(df)
        X = df[feat_cols].select_dtypes(include=[np.number]).fillna(0)
        X_window = X.tail(n_rows)

        # Try stacker first
        stacker_path = PROJECT_ROOT / "data" / "models" / f"{ticker}_stacker.pkl"
        if stacker_path.exists():
            try:
                stacker = joblib.load(stacker_path)
                reg = ModelRegistry()
                models_map = {}
                for suffix in ["rf", "xgb", "lgbm"]:
                    m = reg.load_model(f"{ticker}_{suffix}")
                    if m is not None:
                        models_map[f"{ticker}_{suffix}"] = m

                if len(models_map) >= 2:
                    probas = {name: m.predict_proba(X_window)
                              for name, m in models_map.items()}
                    ensemble_proba = stacker.predict_proba(probas)
                    signals = stacker.predict(probas)
                    confidence = ensemble_proba.max(axis=1)
                    return pd.DataFrame({
                        "signal":     signals,
                        "confidence": confidence,
                        "source":     "ensemble",
                    }, index=X_window.index)
            except Exception:
                pass

        # Fall back to best single model
        reg = ModelRegistry()
        for suffix in ["xgb", "lgbm", "rf"]:
            try:
                m = reg.load_model(f"{ticker}_{suffix}")
                if m is None:
                    continue
                proba = m.predict_proba(X_window)
                signals = m.predict(X_window)
                confidence = proba.max(axis=1)
                return pd.DataFrame({
                    "signal":     signals,
                    "confidence": confidence,
                    "source":     suffix,
                }, index=X_window.index)
            except Exception:
                continue

    except Exception:
        pass
    return None


def get_latest_signal(ticker: str) -> dict:
    """
    Get the most recent signal for a ticker.

    Returns dict with keys: signal, confidence, source
    or neutral defaults if not available.
    """
    preds = predict_ticker(ticker, n_rows=60)
    if preds is not None and len(preds) > 0:
        last = preds.iloc[-1]
        return {
            "ticker":     ticker,
            "signal":     int(last["signal"]),
            "confidence": float(last["confidence"]),
            "source":     str(last["source"]),
        }
    return {"ticker": ticker, "signal": 0, "confidence": 0.0, "source": "none"}


def get_all_latest_signals(tickers: list[str] | None = None) -> list[dict]:
    """Get the latest signal for each trained ticker."""
    if tickers is None:
        tickers = get_trained_tickers()
    return [get_latest_signal(t) for t in tickers]
