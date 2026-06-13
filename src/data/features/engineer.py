from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.config.logging_config import get_logger
from src.config.constants import RAW_EQUITIES_DIR, RAW_CRYPTO_DIR
from src.data.features.momentum       import compute_momentum_features
from src.data.features.volatility     import compute_volatility_features
from src.data.features.volume         import compute_volume_features
from src.data.features.statistical    import compute_statistical_features
from src.data.features.cross_asset    import compute_cross_asset_features
from src.data.features.microstructure import compute_microstructure_features
from src.data.features.sentiment      import compute_sentiment_features
from src.data.features.fundamental    import compute_fundamental_features
from src.data.features.regime             import compute_regime_features
from src.data.features.triple_barrier     import compute_triple_barrier_targets, TB_TARGET_COLS
from src.data.features.options            import compute_options_features
from src.data.pipeline.storage            import save_features

logger = get_logger(__name__)

# Columns that are TARGETS -- never used as model inputs
TARGET_COLS = TB_TARGET_COLS


def engineer_features(
    df: pd.DataFrame,
    ticker: str,
    include_fundamentals: bool = True,
    include_sentiment: bool = True,
) -> pd.DataFrame:
    """
    Main feature engineering pipeline for a single asset.

    STRICT NO-LOOKAHEAD RULE:
      All features at time T use only data available at end-of-day T.
      Target variables are shifted forward (use data[T+1]) -- NEVER input features.
    """
    logger.info("engineering_features", ticker=ticker, rows=len(df))

    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns for {ticker}: {required - set(df.columns)}")

    # Sort and clean index
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    frames: list[pd.DataFrame] = []

    # ── Feature groups (all strictly use only past/current data) ──────────────
    frames.append(compute_momentum_features(df))
    frames.append(compute_volatility_features(df))
    frames.append(compute_volume_features(df))
    frames.append(compute_statistical_features(df))
    frames.append(compute_cross_asset_features(df, ticker))
    frames.append(compute_microstructure_features(df))
    frames.append(compute_regime_features(df))

    if include_sentiment:
        try:
            frames.append(compute_sentiment_features(df, ticker))
        except Exception as exc:
            logger.warning("sentiment_failed", ticker=ticker, error=str(exc))

    if include_fundamentals:
        try:
            frames.append(compute_fundamental_features(df, ticker))
        except Exception as exc:
            logger.warning("fundamentals_failed", ticker=ticker, error=str(exc))

    # ── Options features (equities only, NaN for crypto/macro) ───────────────
    try:
        frames.append(compute_options_features(df, ticker))
    except Exception as exc:
        logger.warning("options_failed", ticker=ticker, error=str(exc))

    # ── Combine features ──────────────────────────────────────────────────────
    feature_df = pd.concat(frames, axis=1)

    # ── Sentiment divergence (needs price trend from close) ───────────────────
    if "sent_score_1d" in feature_df.columns:
        price_trend = df["close"].pct_change(5).apply(np.sign)
        sent_trend  = feature_df["sent_score_1d"].apply(np.sign)
        feature_df["sent_diverge"] = (price_trend != sent_trend).astype(float)

    # ── Targets -- triple barrier labels (MUST NOT be used as features) ────────
    targets = compute_triple_barrier_targets(df)
    feature_df = pd.concat([feature_df, targets], axis=1)

    # ── Final cleanup ─────────────────────────────────────────────────────────
    # Replace inf/-inf with NaN
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)

    # Drop rows where close doesn't exist (shouldn't happen but safety net)
    feature_df = feature_df[df["close"].notna()]

    logger.info(
        "features_complete",
        ticker=ticker,
        rows=len(feature_df),
        feature_cols=len([c for c in feature_df.columns if c not in TARGET_COLS]),
        target_cols=len(TARGET_COLS),
    )

    return feature_df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return only feature columns (exclude targets)."""
    return [c for c in df.columns if c not in TARGET_COLS]


def run_all_features(
    equity_tickers: Optional[list[str]] = None,
    crypto_symbols: Optional[list[str]] = None,
    save: bool = True,
) -> dict[str, pd.DataFrame]:
    from src.config.assets import EQUITY_UNIVERSE, CRYPTO_UNIVERSE
    from src.data.bootstrap.equity_loader import load_equity
    from src.data.bootstrap.crypto_loader import load_crypto
    from src.data.pipeline.cleaner import clean_ohlcv

    tickers  = equity_tickers or list(EQUITY_UNIVERSE.keys())
    c_syms   = crypto_symbols or list(CRYPTO_UNIVERSE.keys())
    results: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        df = load_equity(ticker)
        if df is None or df.empty:
            logger.warning("equity_not_found_skipping", ticker=ticker)
            continue
        try:
            df = clean_ohlcv(df, ticker=ticker)
            feat_df = engineer_features(df, ticker)
            if save:
                save_features(feat_df, ticker)
            results[ticker] = feat_df
        except Exception as exc:
            logger.error("feature_engineering_failed", ticker=ticker, error=str(exc))

    for symbol in c_syms:
        df = load_crypto(symbol)
        if df is None or df.empty:
            continue
        try:
            df = clean_ohlcv(df, ticker=symbol)
            feat_df = engineer_features(df, symbol, include_fundamentals=False)
            safe_name = symbol.replace("/", "_")
            if save:
                save_features(feat_df, safe_name)
            results[symbol] = feat_df
        except Exception as exc:
            logger.error("crypto_feature_failed", symbol=symbol, error=str(exc))

    logger.info("all_features_done", tickers=len(results))
    return results
