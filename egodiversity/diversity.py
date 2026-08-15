"""Diversity scores and subset selectors over cached episode features.

The headline metric is the Vendi score (Friedman & Dieng, 2023): the
exponential of the Shannon entropy of the normalized eigenvalue spectrum of a
similarity kernel. It reads as the "effective number of distinct behaviors" in
a subset — a set of k identical episodes scores ~1 no matter how large k is,
while k mutually dissimilar episodes score ~k.

Kernel choice: RBF on standardized features with the median heuristic for the
bandwidth. Standardization puts the (heterogeneously scaled) feature blocks on
comparable footing; the median heuristic adapts the bandwidth to the data's
own scale, and Experiment D in validate.py shows subset rankings are stable
across a 16x range of bandwidths around it.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigvalsh
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def _standardize(X: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(np.asarray(X, dtype=np.float64))


def _rbf_kernel(X: np.ndarray, sigma_mult: float = 1.0) -> np.ndarray:
    D = squareform(pdist(X, metric="euclidean"))
    off_diag = D[np.triu_indices_from(D, k=1)]
    sigma = float(np.median(off_diag[off_diag > 0])) if np.any(off_diag > 0) else 1.0
    sigma *= sigma_mult
    return np.exp(-(D**2) / (2.0 * sigma**2))


def vendi_score(X: np.ndarray, sigma_mult: float = 1.0) -> float:
    """Vendi score of the episode set whose feature rows are X."""
    X = np.asarray(X, dtype=np.float64)
    if len(X) < 2:
        return 1.0
    K = _rbf_kernel(_standardize(X), sigma_mult)
    lam = eigvalsh(K)
    lam = np.clip(lam, 0.0, None)
    p = lam / lam.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def mean_pairwise_distance(X: np.ndarray) -> float:
    """Mean euclidean distance between standardized feature vectors."""
    X = _standardize(X)
    if len(X) < 2:
        return 0.0
    return float(pdist(X, metric="euclidean").mean())


def coverage_radius(X: np.ndarray, k: int = 3) -> float:
    """Mean distance to the k nearest neighbors (lower = denser coverage)."""
    X = _standardize(X)
    k = min(k, len(X) - 1)
    if k < 1:
        return 0.0
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dists, _ = nn.kneighbors(X)
    return float(dists[:, 1:].mean())


def _pairwise_dist(X: np.ndarray) -> np.ndarray:
    return squareform(pdist(_standardize(X), metric="euclidean"))


def greedy_max_diversity(X: np.ndarray, k: int) -> list[int]:
    """Farthest-point (k-center style) greedy: maximally spread subset.

    Seeds with the most distant pair, then repeatedly adds the candidate that
    most increases the subset's Vendi score.

    Note: we optimize the Vendi score directly rather than using farthest-point
    (k-center) traversal. In ~740-dim standardized space, pairwise distances
    concentrate, so farthest-point subsets score no better than random ones;
    direct greedy optimization of the metric is both simpler to defend and
    empirically stronger (see validate.py, Experiment C).
    """
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    k = min(k, n)
    if k <= 1:
        return [0] if k == 1 else []
    # Seed: most distant pair (Vendi of a pair is distance-monotone).
    D = _pairwise_dist(X)
    i, j = np.unravel_index(np.argmax(D), D.shape)
    selected = [int(i), int(j)]
    while len(selected) < k:
        remaining = (c for c in range(n) if c not in selected)
        nxt = max(remaining, key=lambda c: vendi_score(X[selected + [c]]))
        selected.append(nxt)
    return selected


def greedy_min_diversity(X: np.ndarray, k: int) -> list[int]:
    """Greedy tight-cluster subset: minimally diverse k episodes.

    Seeds with the point closest to the centroid, then repeatedly adds the
    unselected point closest to the selected set (single-linkage growth).

    Note: this is distance-based rather than Vendi-greedy on purpose. Greedy
    minimization of Vendi is too myopic here — the median heuristic re-scales
    the kernel as the subset grows, so early picks never commit to the dense
    region. Growing a cluster by distance finds the genuinely tight subset
    (Vendi ~1.6 vs ~4.5 for Vendi-greedy; see validate.py, Experiment C).
    """
    D = _pairwise_dist(X)
    n = len(X)
    k = min(k, n)
    Z = _standardize(X)
    first = int(np.argmin(np.linalg.norm(Z - Z.mean(axis=0), axis=1)))
    selected = [first]
    min_d = D[first].copy()
    min_d[first] = np.inf
    for _ in range(k - 1):
        nxt = int(np.argmin(min_d))
        selected.append(nxt)
        min_d = np.minimum(min_d, D[nxt])
        min_d[nxt] = np.inf
    return selected
