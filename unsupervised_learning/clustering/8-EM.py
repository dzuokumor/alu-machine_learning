#!/usr/bin/env python3
"""Expectation Maximization for a Gaussian Mixture Model."""

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5,
                             verbose=False):
    """Perform the expectation maximization for a GMM."""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    g, li = expectation(X, pi, m, S)

    if verbose:
        print("Log Likelihood after {} iterations: {}"
              .format(0, round(li, 5)))

    for i in range(iterations):
        prev_li = li
        pi, m, S = maximization(X, g)
        g, li = expectation(X, pi, m, S)

        if verbose and ((i + 1) % 10 == 0 or abs(li - prev_li) <= tol):
            print("Log Likelihood after {} iterations: {}"
                  .format(i + 1, round(li, 5)))

        if abs(li - prev_li) <= tol:
            break

    return pi, m, S, g, li
