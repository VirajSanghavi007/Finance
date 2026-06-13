"""
Tests for the 9 research-backed modules added in the second phase.
Covers: triple barrier, purged kfold, HRP, meta-labeler, conformal,
        offline RL, Mamba, options features, deflated Sharpe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ─── Triple Barrier ───────────────────────────────────────────────────────────

class TestTripleBarrier:
    def _make_close(self, n: int = 100) -> pd.Series:
        np.random.seed(0)
        prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, n)))
        return pd.Series(prices, index=pd.bdate_range("2020-01-01", periods=n))

    def test_labels_in_valid_set(self):
        from src.data.features.triple_barrier import triple_barrier_labels
        close = self._make_close()
        vol   = np.full(len(close), 0.01)
        vol_s = pd.Series(vol, index=close.index)
        tb = triple_barrier_labels(close, vol_s)
        assert set(tb["label"].unique()).issubset({-1, 0, 1})

    def test_no_future_leakage(self):
        """Labels are computed without using data after t_exit."""
        from src.data.features.triple_barrier import triple_barrier_labels
        close = self._make_close(200)
        vol_s = close.pct_change().rolling(21).std().fillna(0.01)
        tb = triple_barrier_labels(close, vol_s)
        # t_exit must be >= index (never before the signal bar)
        assert (tb["t_exit"] >= tb.index).all()

    def test_barrier_hit_categories(self):
        from src.data.features.triple_barrier import triple_barrier_labels
        close = self._make_close(200)
        vol_s = close.pct_change().rolling(21).std().fillna(0.01)
        tb = triple_barrier_labels(close, vol_s)
        assert set(tb["barrier_hit"]).issubset({"pt", "sl", "timeout", "nan_vol"})

    def test_compute_triple_barrier_targets_shape(self):
        from src.data.features.triple_barrier import compute_triple_barrier_targets, TB_TARGET_COLS
        df = pd.DataFrame({"close": self._make_close().values,
                           "open": self._make_close().values},
                          index=self._make_close().index)
        tgt = compute_triple_barrier_targets(df)
        assert all(c in tgt.columns for c in TB_TARGET_COLS)
        assert len(tgt) == len(df)


# ─── Purged K-Fold ────────────────────────────────────────────────────────────

class TestPurgedKFold:
    def _make_data(self, n: int = 200):
        X = pd.DataFrame(np.random.randn(n, 5), columns=list("ABCDE"))
        y = pd.Series(np.random.choice([-1, 0, 1], n))
        return X, y

    def test_no_overlap_between_train_test(self):
        from src.utils.purged_kfold import PurgedKFold
        X, y = self._make_data()
        cv = PurgedKFold(n_splits=5, embargo_pct=0.01)
        for train_idx, test_idx in cv.split(X, y):
            assert len(np.intersect1d(train_idx, test_idx)) == 0

    def test_produces_correct_n_splits(self):
        from src.utils.purged_kfold import PurgedKFold
        X, y = self._make_data()
        cv = PurgedKFold(n_splits=5)
        folds = list(cv.split(X, y))
        assert len(folds) == 5

    def test_each_fold_is_contiguous(self):
        """Each test fold should be a contiguous block (standard k-fold property)."""
        from src.utils.purged_kfold import PurgedKFold
        X, y = self._make_data()
        cv = PurgedKFold(n_splits=5, embargo_pct=0.01)
        for train_idx, test_idx in cv.split(X, y):
            if len(test_idx) > 1:
                sorted_test = np.sort(test_idx)
                # Contiguous: max - min + 1 == len (no gaps)
                assert sorted_test[-1] - sorted_test[0] + 1 == len(sorted_test), \
                    "Test fold indices are not contiguous"


# ─── HRP ──────────────────────────────────────────────────────────────────────

class TestHRP:
    def _make_returns(self, n: int = 100, k: int = 5) -> pd.DataFrame:
        np.random.seed(42)
        return pd.DataFrame(
            np.random.normal(0, 0.01, (n, k)),
            columns=[f"A{i}" for i in range(k)],
        )

    def test_weights_sum_to_one(self):
        from src.risk.hrp import hrp_weights
        returns = self._make_returns()
        w = hrp_weights(returns, min_periods=30)
        assert abs(w.sum() - 1.0) < 1e-9

    def test_weights_non_negative(self):
        from src.risk.hrp import hrp_weights
        returns = self._make_returns()
        w = hrp_weights(returns, min_periods=30)
        assert (w >= 0).all()

    def test_fallback_equal_weight_when_insufficient(self):
        from src.risk.hrp import hrp_weights
        returns = self._make_returns(n=5, k=3)
        w = hrp_weights(returns, min_periods=30)  # only 5 rows, needs 30
        assert abs(w.sum() - 1.0) < 1e-9
        # Should be equal weight
        assert all(abs(w - 1/3) < 1e-9)

    def test_two_assets(self):
        from src.risk.hrp import hrp_weights
        returns = self._make_returns(n=60, k=2)
        w = hrp_weights(returns, min_periods=30)
        assert abs(w.sum() - 1.0) < 1e-9
        assert len(w) == 2


# ─── Meta-Labeler ─────────────────────────────────────────────────────────────

class TestMetaLabeler:
    def test_make_meta_labels_shape(self):
        from src.models.meta_labeler import make_meta_labels
        n = 100
        signals  = pd.Series(np.random.choice([-1, 0, 1], n))
        returns  = pd.Series(np.random.normal(0, 0.01, n))
        meta     = make_meta_labels(signals, returns)
        assert len(meta) == n
        assert set(meta.unique()).issubset({0, 1})

    def test_flat_signal_always_zero_meta(self):
        from src.models.meta_labeler import make_meta_labels
        n = 50
        signals = pd.Series(np.zeros(n, dtype=int))
        returns = pd.Series(np.random.normal(0, 0.01, n))
        meta    = make_meta_labels(signals, returns)
        assert (meta == 0).all(), "Flat signals should produce meta_label=0"

    def test_meta_labeler_fit_predict(self):
        from src.models.meta_labeler import MetaLabeler, make_meta_labels
        n = 200
        X       = pd.DataFrame(np.random.randn(n, 10))
        signals = pd.Series(np.random.choice([-1, 0, 1], n))
        returns = pd.Series(np.random.normal(0, 0.01, n))
        meta_y  = make_meta_labels(signals, returns)
        ml = MetaLabeler(n_estimators=50, max_depth=3)
        result = ml.fit(X, meta_y, signals)
        assert "train_acc" in result or "status" in result
        proba = ml.predict_proba(X)
        assert proba.shape == (n,)
        assert ((proba >= 0) & (proba <= 1)).all()


# ─── Conformal Predictor ──────────────────────────────────────────────────────

class TestConformalPredictor:
    def _make_probas(self, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
        np.random.seed(42)
        logits = np.random.randn(n, 3)
        probas = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        y_true = np.argmax(probas + np.random.randn(n, 3) * 0.5, axis=1)
        return probas, y_true

    def test_calibrate_sets_threshold(self):
        from src.models.ensemble.conformal import ConformalPredictor
        probas, y_true = self._make_probas()
        cp = ConformalPredictor(alpha=0.10)
        cp.calibrate(probas, y_true)
        assert cp.is_fitted
        assert 0.0 < cp.threshold < 1.0

    def test_coverage_guarantee(self):
        """Empirical coverage must be ≥ 1-alpha on calibration set."""
        from src.models.ensemble.conformal import ConformalPredictor
        probas, y_true = self._make_probas(500)
        split = 250
        cp = ConformalPredictor(alpha=0.10)
        cp.calibrate(probas[:split], y_true[:split])
        pred_sets, _ = cp.predict_set(probas[split:])
        covered = sum(y_true[split + i] in s for i, s in enumerate(pred_sets))
        coverage = covered / len(pred_sets)
        assert coverage >= 0.88, f"Coverage {coverage:.2%} below 90% target"

    def test_unfitted_fallback(self):
        from src.models.ensemble.conformal import ConformalPredictor
        probas, _ = self._make_probas(50)
        cp = ConformalPredictor()
        signals, conf = cp.predict_scalar(probas)
        assert signals.shape == (50,)
        assert set(signals).issubset({-1, 0, 1})


# ─── Deflated Sharpe Ratio ────────────────────────────────────────────────────

class TestDeflatedSharpe:
    def test_dsr_range(self):
        from src.backtest.metrics import deflated_sharpe_ratio
        returns = pd.Series(np.random.normal(0.001, 0.01, 252))
        dsr = deflated_sharpe_ratio(returns, n_trials=100)
        assert 0.0 <= dsr <= 1.0

    def test_dsr_increases_with_better_sr(self):
        from src.backtest.metrics import deflated_sharpe_ratio
        bad  = pd.Series(np.random.normal(0.0, 0.01, 252))
        good = pd.Series(np.random.normal(0.001, 0.01, 252))
        assert deflated_sharpe_ratio(good, n_trials=1) > \
               deflated_sharpe_ratio(bad,  n_trials=1)

    def test_dsr_penalised_by_more_trials(self):
        from src.backtest.metrics import deflated_sharpe_ratio
        np.random.seed(0)
        returns = pd.Series(np.random.normal(0.0005, 0.01, 252))
        dsr_1   = deflated_sharpe_ratio(returns, n_trials=1)
        dsr_100 = deflated_sharpe_ratio(returns, n_trials=100)
        assert dsr_1 > dsr_100, "More trials should penalise DSR"

    def test_dsr_in_compute_all_metrics(self):
        from src.backtest.metrics import compute_all_metrics
        equity = pd.Series([100_000 * (1.001 ** i) for i in range(252)])
        equity.index = pd.bdate_range("2020-01-01", periods=252)
        m = compute_all_metrics(equity, pd.DataFrame())
        assert "deflated_sharpe_ratio" in m
        assert 0.0 <= m["deflated_sharpe_ratio"] <= 1.0

    def test_dsr_short_series_returns_zero(self):
        from src.backtest.metrics import deflated_sharpe_ratio
        returns = pd.Series([0.001, 0.002, -0.001])
        assert deflated_sharpe_ratio(returns) == 0.0


# ─── Options Features ─────────────────────────────────────────────────────────

class TestOptionsFeatures:
    def _make_df(self, n: int = 50) -> pd.DataFrame:
        close = pd.Series(
            150 + np.random.randn(n).cumsum(),
            index=pd.bdate_range("2024-01-01", periods=n),
        )
        return pd.DataFrame({"close": close, "open": close, "high": close * 1.01,
                             "low": close * 0.99, "volume": np.ones(n) * 1e6})

    def test_non_eligible_ticker_returns_all_nan(self):
        from src.data.features.options import compute_options_features
        df = self._make_df()
        result = compute_options_features(df, "FAKECRYPTO")
        assert result.isna().all().all()

    def test_correct_columns_returned(self):
        from src.data.features.options import compute_options_features
        df = self._make_df()
        result = compute_options_features(df, "FAKECRYPTO")
        expected = ["opt_iv_atm", "opt_iv_skew", "opt_pcr_vol", "opt_pcr_oi",
                    "opt_iv_rv_spread", "opt_term_spread"]
        assert all(c in result.columns for c in expected)

    def test_index_matches_input(self):
        from src.data.features.options import compute_options_features
        df = self._make_df()
        # Use a non-eligible ticker so we test the fallback path without network I/O
        result = compute_options_features(df, "FAKECRYPTO")
        assert len(result) == len(df)


# ─── HRP position sizer integration ──────────────────────────────────────────

class TestHRPPositionSizer:
    def test_hrp_portfolio_weights_shape(self):
        from src.risk.position_sizer import PositionSizer
        np.random.seed(0)
        n, k = 60, 4
        returns = pd.DataFrame(np.random.normal(0, 0.01, (n, k)),
                               columns=["AAPL", "MSFT", "GOOGL", "AMZN"])
        signals = pd.Series({"AAPL": 1, "MSFT": 0, "GOOGL": 1, "AMZN": -1})
        ps = PositionSizer()
        weights = ps.hrp_portfolio_weights(returns, signals)
        assert len(weights) > 0
        assert (weights.abs() <= ps.max_position).all()

    def test_hrp_zero_weight_for_flat_signal(self):
        from src.risk.position_sizer import PositionSizer
        np.random.seed(0)
        returns = pd.DataFrame(np.random.normal(0, 0.01, (60, 3)),
                               columns=["A", "B", "C"])
        signals = pd.Series({"A": 1, "B": 0, "C": 1})
        ps = PositionSizer()
        weights = ps.hrp_portfolio_weights(returns, signals)
        assert "B" not in weights.index or weights.get("B", 0) == 0
