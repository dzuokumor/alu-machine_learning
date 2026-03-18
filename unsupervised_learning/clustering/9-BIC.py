#!/usr/bin/env python3
"""Find the best number of clusters for a GMM using the Bayesian
Information Criterion."""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Find the best number of clusters for a GMM using BIC."""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    n, d = X.shape
    if kmax is None:
        kmax = n
    if not isinstance(kmin, int) or kmin <= 0 or kmin > n:
        return None, None, None, None
    if not isinstance(kmax, int) or kmax <= 0 or kmax > n:
        return None, None, None, None
    if kmax <= kmin:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    liks = np.empty(kmax - kmin + 1)
    bics = np.empty(kmax - kmin + 1)
    results = []

    for idx, k in enumerate(range(kmin, kmax + 1)):
        pi, m, S, g, li = expectation_maximization(
            X, k, iterations, tol, verbose)
        results.append((pi, m, S))
        liks[idx] = li

        # number of parameters: k*d (means) + k*d*(d+1)/2 (covariances)
        # + (k-1) (priors)
        p = k * d + k * d * (d + 1) / 2 + (k - 1)
        bics[idx] = p * np.log(n) - 2 * li

    best_idx = np.argmin(bics)
    best_k = best_idx + kmin
    best_result = results[best_idx]

    return best_k, best_result, liks, bics
