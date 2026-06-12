"""Step 27: Walk-forward no train/test overlap test."""
from __future__ import annotations

import pandas as pd
import pytest

from src.backtest.walk_forward import _generate_folds


def test_folds_no_overlap():
    dates = pd.bdate_range("2015-01-01", "2023-12-31")
    folds = _generate_folds(dates, train_days=730, test_days=91, step_days=91, min_history=1000)

    assert len(folds) > 0, "Should generate at least one fold"
    for i, (train_s, train_e, test_s, test_e) in enumerate(folds):
        # Train end must be before test start
        assert train_e < test_s, (
            f"Fold {i}: train end {train_e.date()} >= test start {test_s.date()}"
        )
        # No date can be in both train and test
        assert test_s > train_e


def test_folds_sorted():
    dates = pd.bdate_range("2015-01-01", "2023-12-31")
    folds = _generate_folds(dates, train_days=730, test_days=91, step_days=91, min_history=1000)

    for i in range(len(folds) - 1):
        assert folds[i][2] < folds[i + 1][2], "Fold test starts should be increasing"


def test_insufficient_history_raises():
    dates = pd.bdate_range("2020-01-01", "2020-06-01")  # ~100 days — too short
    folds = _generate_folds(dates, train_days=730, test_days=91, step_days=91, min_history=1000)
    assert len(folds) == 0, "Should not generate folds with insufficient history"


def test_train_window_size():
    dates = pd.bdate_range("2010-01-01", "2023-12-31")
    folds = _generate_folds(dates, train_days=730, test_days=91, step_days=91, min_history=1000)

    for i, (train_s, train_e, test_s, test_e) in enumerate(folds):
        train_days = (train_e - train_s).days
        # Allow some slack for business day alignment
        assert train_days >= 600, f"Fold {i} train window too short: {train_days} days"
