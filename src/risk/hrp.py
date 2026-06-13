"""
Hierarchical Risk Parity (López de Prado, Journal of Portfolio Management 2016)

HRP builds portfolios via three steps:
  1. Tree clustering   — single-linkage hierarchical clustering on correlation matrix
  2. Quasi-diagonalisation — reorder assets so similar ones are adjacent
  3. Recursive bisection   — allocate weight by inverse variance, recursively

Advantages over Markowitz:
  - No matrix inversion (numerically stable even when assets are correlated)
  - No expected returns needed (pure risk-based)
  - Empirically outperforms equal-weight and Markowitz out-of-sample
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform


def _corr_to_distance(corr: np.ndarray) -> np.ndarray:
    """Convert correlation matrix to distance matrix: d = sqrt(0.5 * (1 - ρ))."""
    dist = np.sqrt(0.5 * (1.0 - corr))
    np.fill_diagonal(dist, 0.0)
    return dist


def _get_quasi_diagonal(link: np.ndarray, n: int) -> list[int]:
    """
    Reorder leaf indices so that similar assets are adjacent
    (quasi-diagonalises the correlation matrix).
    """
    root, nodes = to_tree(link, rd=True)

    def _recurse(node) -> list[int]:
        if node.is_leaf():
            return [node.id]
        return _recurse(node.left) + _recurse(node.right)

    return _recurse(root)


def _recursive_bisection(
    cov: pd.DataFrame,
    sorted_items: list[str],
) -> pd.Series:
    """
    Allocate weights by recursive bisection:
    at each split, scale each sub-cluster's weight by its inverse variance.
    """
    weights = pd.Series(1.0, index=sorted_items)

    def _bisect(items: list[str]) -> None:
        if len(items) <= 1:
            return
        mid   = len(items) // 2
        left  = items[:mid]
        right = items[mid:]

        def _cluster_var(cluster: list[str]) -> float:
            sub_cov = cov.loc[cluster, cluster].values
            w_inv   = np.ones(len(cluster)) / len(cluster)  # equal-weight within sub-cluster
            return float(w_inv @ sub_cov @ w_inv)

        var_l = _cluster_var(left)
        var_r = _cluster_var(right)

        total = var_l + var_r
        if total == 0:
            alpha_l = 0.5
        else:
            alpha_l = 1.0 - var_l / total  # higher var → smaller weight

        weights[left]  *= alpha_l
        weights[right] *= (1.0 - alpha_l)

        _bisect(left)
        _bisect(right)

    _bisect(sorted_items)
    return weights


def hrp_weights(
    returns: pd.DataFrame,
    min_periods: int = 30,
) -> pd.Series:
    """
    Compute HRP portfolio weights from a returns DataFrame.

    Parameters
    ----------
    returns     : DataFrame where columns = assets, rows = daily returns.
    min_periods : Minimum observations required; returns equal weights if not met.

    Returns
    -------
    pd.Series of weights indexed by asset name, summing to 1.
    """
    assets = list(returns.columns)
    n      = len(assets)

    if len(returns) < min_periods or n < 2:
        return pd.Series(1.0 / n, index=assets)

    # ── Step 1: Covariance and correlation ────────────────────────────────────
    cov  = returns.cov()
    corr = returns.corr().fillna(0).values

    # ── Step 2: Distance matrix + hierarchical clustering ────────────────────
    dist       = _corr_to_distance(corr)
    condensed  = squareform(dist, checks=False)
    link       = linkage(condensed, method="single")

    # ── Step 3: Quasi-diagonalisation ────────────────────────────────────────
    sorted_idx   = _get_quasi_diagonal(link, n)
    sorted_names = [assets[i] for i in sorted_idx]

    # ── Step 4: Recursive bisection ──────────────────────────────────────────
    weights = _recursive_bisection(cov, sorted_names)
    weights = weights / weights.sum()   # normalise to 1
    return weights.reindex(assets).fillna(0.0)
