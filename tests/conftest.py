from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """100 days of synthetic OHLCV data — no real market data needed for unit tests."""
    np.random.seed(42)
    n = 300
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, n)))
    high  = close * np.exp(np.random.uniform(0, 0.02, n))
    low   = close * np.exp(-np.random.uniform(0, 0.02, n))
    open_ = close * np.exp(np.random.normal(0, 0.005, n))
    high  = np.maximum(high, np.maximum(open_, close))
    low   = np.minimum(low,  np.minimum(open_, close))
    vol   = np.random.randint(1_000_000, 50_000_000, n).astype(float)

    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": vol
    }, index=dates)


@pytest.fixture
def sample_feature_df(sample_ohlcv) -> pd.DataFrame:
    from src.data.features.engineer import engineer_features
    return engineer_features(
        sample_ohlcv, ticker="TEST",
        include_fundamentals=False,
        include_sentiment=False,
    )
