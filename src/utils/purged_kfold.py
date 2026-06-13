"""
Purged K-Fold Cross-Validation for time series (López de Prado, AFML Ch.7)

Standard TimeSeriesSplit still leaks when labels overlap adjacent folds
(e.g. a 5-day forward-return label computed on bar T overlaps with train
 data that includes bars T+1 through T+4 in the next fold).

PurgedKFold fixes this by:
  1. Purging: removing training samples whose labels overlap with the test window.
  2. Embargo: removing a buffer of samples immediately after the test window
     to prevent information from leaking via serial correlation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


class PurgedKFold:
    """
    Cross-validation splitter for financial time series.

    Parameters
    ----------
    n_splits      : number of folds (default 5)
    embargo_pct   : fraction of total samples to embargo after each test fold.
                    0.01 = 1% of total rows embargoed (covers serial correlation).
    """

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01) -> None:
        self.n_splits    = n_splits
        self.embargo_pct = embargo_pct

    def split(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
        groups: np.ndarray | None = None,
        pred_times: pd.Series | None = None,
        eval_times: pd.Series | None = None,
    ):
        """
        Yield (train_indices, test_indices) tuples.

        If pred_times and eval_times are provided they are used to compute
        label overlap precisely. Otherwise a simple forward-purge based on
        index position is used (assumes labels don't span more than one bar).
        """
        n = len(X) if hasattr(X, '__len__') else X.shape[0]
        embargo_size = max(1, int(n * self.embargo_pct))

        indices = np.arange(n)
        fold_size = n // self.n_splits

        for fold in range(self.n_splits):
            test_start = fold * fold_size
            test_end   = test_start + fold_size if fold < self.n_splits - 1 else n

            test_idx  = indices[test_start:test_end]

            # Purge: remove train samples whose prediction/label window
            # overlaps with the test window.
            if pred_times is not None and eval_times is not None:
                # Precise purge: any train sample whose eval_time falls within
                # the test prediction window is removed.
                test_pred_start = pred_times.iloc[test_start]
                test_eval_end   = eval_times.iloc[test_end - 1]

                keep_mask = (eval_times <= test_pred_start) | (pred_times > test_eval_end)
                train_idx = indices[keep_mask.values]
            else:
                # Simple purge: exclude test window + embargo buffer from train
                purge_end = min(test_end + embargo_size, n)
                train_idx = np.concatenate([
                    indices[:test_start],
                    indices[purge_end:],
                ])

            # Embargo: additionally remove the buffer immediately after test
            embargo_end = min(test_end + embargo_size, n)
            embargo_idx = indices[test_end:embargo_end]
            train_idx   = np.setdiff1d(train_idx, embargo_idx)

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


def purged_cv_score(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    scoring_fn,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
) -> list[float]:
    """
    Convenience wrapper: run PurgedKFold and return list of scores.

    scoring_fn(y_true, y_pred) → float
    """
    cv     = PurgedKFold(n_splits=n_splits, embargo_pct=embargo_pct)
    scores = []
    for train_idx, test_idx in cv.split(X, y):
        X_tr = X.iloc[train_idx]
        y_tr = y.iloc[train_idx]
        X_te = X.iloc[test_idx]
        y_te = y.iloc[test_idx]
        estimator.fit(X_tr, y_tr)
        preds = estimator.predict(X_te)
        scores.append(scoring_fn(y_te, preds))
    return scores
