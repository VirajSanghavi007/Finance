"""
Split Conformal Prediction for ensemble signals.
(Angelopoulos & Bates, 2022 — distribution-free coverage guarantees)

Problem: the softmax confidence from our ensemble (e.g. 0.72) is not a
real probability. It might actually be right only 55% of the time.
Conformal prediction wraps any model and gives statistically valid coverage:
"this prediction set contains the true label with probability ≥ 1−α".

How it works (Split Conformal):
  1. Calibration phase (one-time, on a held-out set):
     For each calibration sample, compute the non-conformity score:
       score[i] = 1 − P(true_class[i])   (softmax probability of the correct class)
     Store these N calibration scores.

  2. Prediction phase:
     For a new sample, include class c in the prediction set if:
       1 − P(c) ≤ quantile(calibration_scores, level = ceil((N+1)(1−α)) / N)
     Equivalently, the threshold τ = quantile at level (1−α) adjusted for finite N.

  3. The prediction set size tells you uncertainty:
     Size 1 → confident single prediction
     Size 2 → between two classes
     Size 3 → complete uncertainty

The conformal confidence we report is:
  conf = 1 − non_conformity_score_of_predicted_class
adjusted to be a calibrated probability.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class ConformalPredictor:
    """
    Wraps the ensemble's predict_proba output with conformal calibration.

    Usage:
        cp = ConformalPredictor(alpha=0.10)  # 90% coverage
        cp.calibrate(calib_probas, calib_true_labels)

        # At prediction time:
        pred_sets, conf_scores = cp.predict_set(test_probas)
        # pred_sets: list of lists of classes in prediction set
        # conf_scores: calibrated confidence per sample
    """

    def __init__(self, alpha: float = 0.10) -> None:
        """
        Args:
            alpha: miscoverage rate. alpha=0.10 → 90% coverage guarantee.
        """
        self.alpha         = alpha
        self._threshold: float | None = None
        self._n_calib: int = 0
        self._fitted = False

    # ── Calibration ───────────────────────────────────────────────────────────

    def calibrate(
        self,
        probas: np.ndarray,
        y_true: np.ndarray,
    ) -> "ConformalPredictor":
        """
        Compute the conformal threshold from calibration data.

        Args:
            probas  : (N, 3) softmax probabilities from ensemble.
            y_true  : (N,)   true class labels in {0, 1, 2}
                      (mapped from {-1, 0, +1} via +1 offset)
        """
        n = len(y_true)
        if n < 10:
            return self  # not enough data

        # Non-conformity score = 1 − P(true class)
        y_int = np.asarray(y_true, dtype=int)
        scores = 1.0 - probas[np.arange(n), y_int]

        # Finite-sample correction: quantile level = ceil((N+1)(1−α)) / N
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        level = float(np.clip(level, 0.0, 1.0))
        self._threshold = float(np.quantile(scores, level))
        self._n_calib   = n
        self._fitted    = True
        return self

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_set(
        self,
        probas: np.ndarray,
    ) -> tuple[list[list[int]], np.ndarray]:
        """
        Return prediction sets and calibrated confidence scores.

        Args:
            probas : (N, 3) softmax from ensemble.

        Returns:
            pred_sets     : list[list[int]] — classes included in prediction set
            conf_scores   : (N,) calibrated confidence ∈ [0, 1]
        """
        if not self._fitted or self._threshold is None:
            # Fallback: return argmax with raw softmax confidence
            best = np.argmax(probas, axis=1)
            conf = probas[np.arange(len(probas)), best]
            return [[int(b)] for b in best], conf

        threshold = self._threshold
        pred_sets  = []
        conf_scores = np.zeros(len(probas))

        for i, row in enumerate(probas):
            # Include class c if 1 − P(c) ≤ threshold
            included = [c for c in range(3) if (1.0 - row[c]) <= threshold]
            if not included:
                # Never return empty set — include the most probable class
                included = [int(np.argmax(row))]
            pred_sets.append(included)

            # Calibrated confidence: if set size=1, use its probability;
            # if size>1, use 1/(set_size) as effective uncertainty measure
            if len(included) == 1:
                conf_scores[i] = float(row[included[0]])
            else:
                conf_scores[i] = 1.0 / len(included)

        return pred_sets, conf_scores

    def predict_scalar(
        self,
        probas: np.ndarray,
        abstain_on_uncertainty: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convenience: return scalar signals and calibrated confidences.

        When prediction set size > 1 (uncertain):
          - if abstain_on_uncertainty=True  → signal = 0 (flat)
          - if abstain_on_uncertainty=False → signal = argmax (most probable)

        Returns:
            signals    : (N,) in {-1, 0, +1}
            conf       : (N,) calibrated confidence
        """
        LABEL_UNMAP = {0: -1, 1: 0, 2: 1}
        pred_sets, conf = self.predict_set(probas)

        signals = np.zeros(len(pred_sets), dtype=int)
        for i, s in enumerate(pred_sets):
            if len(s) == 1:
                signals[i] = LABEL_UNMAP[s[0]]
            elif not abstain_on_uncertainty:
                # Use the class with highest raw probability
                best = int(np.argmax(probas[i]))
                signals[i] = LABEL_UNMAP[best]
            # else: signal stays 0 (abstain)

        return signals, conf

    @property
    def threshold(self) -> float | None:
        return self._threshold

    @property
    def is_fitted(self) -> bool:
        return self._fitted
