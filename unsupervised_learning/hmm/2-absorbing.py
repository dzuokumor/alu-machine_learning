#!/usr/bin/env python3
"""Determines if a markov chain is absorbing."""


import numpy as np


def absorbing(P):
    """Determine if a markov chain is absorbing."""
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return False
    n1, n2 = P.shape
    if n1 != n2:
        return False

    D = np.diagonal(P)
    if not (D == 1).any():
        return False
    if (D == 1).all():
        return True

    absorb = set(np.where(D == 1)[0])
    reachable = set(np.where(D == 1)[0])

    changed = True
    while changed:
        changed = False
        for i in range(n1):
            if i not in reachable:
                for j in reachable:
                    if P[i][j] > 0:
                        reachable.add(i)
                        changed = True
                        break

    return len(reachable) == n1
