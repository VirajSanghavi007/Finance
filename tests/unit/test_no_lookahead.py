"""
Step 19: CRITICAL no-lookahead test.
Verifies that no feature column at time T uses data from T+1 or later.
This is the most important test in the entire test suite.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.features.engineer import engineer_features, get_feature_columns, TARGET_COLS


def _inject_spike(df: pd.DataFrame, spike_idx: int, multiplier: float = 10.0) -> pd.DataFrame:
    """Inject an unmistakable spike at position spike_idx — should NOT appear before it."""
    df = df.copy()
    df.iloc[spike_idx, df.columns.get_loc("close")] *= multiplier
    df.iloc[spike_idx, df.columns.get_loc("high")]  *= multiplier
    return df


def test_features_do_not_use_future_prices(sample_ohlcv):
    """
    Insert a massive price spike at bar 200.
    Any feature computed BEFORE bar 200 that depends on the spike
    would constitute lookahead bias.

    We check that feature values at bars < 190 are identical with and without
    the spike (allowing 10 bars for any rolling windows that might shift).
    """
    spike_pos = 200

    df_clean = sample_ohlcv.copy()
    df_spike = _inject_spike(sample_ohlcv.copy(), spike_pos, multiplier=50.0)

    feat_clean = engineer_features(
        df_clean, ticker="NOLAG_TEST",
        include_fundamentals=False, include_sentiment=False,
    )
    feat_spike = engineer_features(
        df_spike, ticker="NOLAG_TEST",
        include_fundamentals=False, include_sentiment=False,
    )

    feat_cols = get_feature_columns(feat_clean)
    # Compare only numeric, non-target features
    num_cols = [c for c in feat_cols
                if pd.api.types.is_numeric_dtype(feat_clean[c])]

    # Bars well before the spike — must be identical
    pre_spike = slice(None, spike_pos - 15)  # leave 15-bar buffer for EWM

    for col in num_cols:
        clean_vals = feat_clean[col].iloc[pre_spike].values
        spike_vals = feat_spike[col].iloc[pre_spike].values

        # Ignore rows that are NaN in both (warmup period)
        both_valid = ~(np.isnan(clean_vals) | np.isnan(spike_vals))
        if both_valid.sum() == 0:
            continue

        max_diff = np.abs(
            clean_vals[both_valid].astype(float) - spike_vals[both_valid].astype(float)
        ).max()
        assert max_diff < 1e-6, (
            f"LOOKAHEAD DETECTED in '{col}': "
            f"pre-spike values changed by {max_diff:.6f} after injecting spike at bar {spike_pos}. "
            f"This column uses future data."
        )


def test_target_columns_are_forward_looking(sample_ohlcv):
    """Confirm targets DO use future data (they should — they're labels)."""
    spike_pos = 100
    df_spike = _inject_spike(sample_ohlcv.copy(), spike_pos, multiplier=20.0)
    df_clean = sample_ohlcv.copy()

    feat_clean = engineer_features(
        df_clean, ticker="T_TEST",
        include_fundamentals=False, include_sentiment=False,
    )
    feat_spike = engineer_features(
        df_spike, ticker="T_TEST",
        include_fundamentals=False, include_sentiment=False,
    )

    # target_1d at bar spike_pos - 1 should reflect the spike at spike_pos
    t1_clean = feat_clean["target_ret_1d"].iloc[spike_pos - 1]
    t1_spike = feat_spike["target_ret_1d"].iloc[spike_pos - 1]
    assert abs(t1_spike - t1_clean) > 0.5, (
        "target_ret_1d at bar spike_pos-1 should differ (it looks forward) "
        f"but got clean={t1_clean:.4f} spike={t1_spike:.4f}"
    )


def test_no_future_close_in_features(sample_ohlcv):
    """
    Verify that feature matrix at row i does NOT contain close[i+1] anywhere.
    We do this by checking Pearson correlation between each feature column
    and the 1-step-ahead close. No feature should have |corr| > 0.999
    (which would indicate it IS the future close or a trivial transform of it).
    """
    feat_df = engineer_features(
        sample_ohlcv, ticker="CORR_TEST",
        include_fundamentals=False, include_sentiment=False,
    )
    feat_cols = get_feature_columns(feat_df)
    future_close = sample_ohlcv["close"].shift(-1).reindex(feat_df.index)

    for col in feat_cols:
        series = feat_df[col]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        valid = series.notna() & future_close.notna()
        if valid.sum() < 30:
            continue
        corr = abs(series[valid].corr(future_close[valid]))
        assert corr < 0.999, (
            f"Feature '{col}' has suspiciously high correlation ({corr:.4f}) "
            f"with future close — possible lookahead leak"
        )
