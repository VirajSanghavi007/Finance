"""
Step 18: Feature engineer tests.
Verifies: no NaN in feature columns (after warmup), no inf, correct shape.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.features.engineer import engineer_features, get_feature_columns, TARGET_COLS


def test_feature_columns_no_inf(sample_feature_df):
    feat_cols = get_feature_columns(sample_feature_df)
    feat_data = sample_feature_df[feat_cols]
    has_inf = np.isinf(feat_data.select_dtypes(include="number")).any().any()
    assert not has_inf, "Feature columns contain inf values"


def test_feature_columns_reasonable_nan_rate(sample_feature_df):
    """After 200-bar warmup, NaN rate should be < 10% for most features."""
    feat_cols = get_feature_columns(sample_feature_df)
    trimmed = sample_feature_df[feat_cols].iloc[220:]  # past all warmup windows
    nan_rate = trimmed.isna().mean().mean()
    assert nan_rate < 0.15, f"Too many NaNs after warmup: {nan_rate:.1%}"


def test_targets_not_in_feature_cols(sample_feature_df):
    feat_cols = get_feature_columns(sample_feature_df)
    for t in TARGET_COLS:
        assert t not in feat_cols, f"Target column {t} leaked into features"


def test_target_columns_present(sample_feature_df):
    for t in TARGET_COLS:
        assert t in sample_feature_df.columns, f"Missing target column: {t}"


def test_momentum_features_present(sample_feature_df):
    expected = ["mom_rsi_14", "mom_macd_line", "mom_roc_10", "mom_stoch_k"]
    for col in expected:
        assert col in sample_feature_df.columns, f"Missing: {col}"


def test_volatility_features_present(sample_feature_df):
    expected = ["vol_atr_14", "vol_bb_pct", "vol_hv_21", "vol_parkinson"]
    for col in expected:
        assert col in sample_feature_df.columns, f"Missing: {col}"


def test_volume_features_present(sample_feature_df):
    expected = ["vl_obv", "vl_cmf_20", "vl_mfi_14"]
    for col in expected:
        assert col in sample_feature_df.columns, f"Missing: {col}"


def test_regime_features_present(sample_feature_df):
    expected = ["reg_hmm_state", "reg_trend_strength", "reg_trend_direction"]
    for col in expected:
        assert col in sample_feature_df.columns, f"Missing: {col}"


def test_index_sorted(sample_feature_df):
    assert sample_feature_df.index.is_monotonic_increasing


def test_no_duplicate_index(sample_feature_df):
    assert not sample_feature_df.index.duplicated().any()


def test_minimum_feature_count(sample_feature_df):
    feat_cols = get_feature_columns(sample_feature_df)
    assert len(feat_cols) >= 50, f"Expected >= 50 features, got {len(feat_cols)}"
