"""
Prime generation utilities.

Responsibility: prime generation only. No factorization, no sieve logic.
"""

import numpy as np


def prime_flags_upto(N: int) -> np.ndarray:
    """
    Return boolean array where flags[i] is True iff i is prime.

    Uses Sieve of Eratosthenes.

    Parameters
    ----------
    N : int
        Upper bound (inclusive).

    Returns
    -------
    np.ndarray
        Boolean array of length N+1.
    """
    flags = np.ones(N + 1, dtype=bool)
    flags[0] = flags[1] = False
    for p in range(2, int(N**0.5) + 1):
        if flags[p]:
            flags[p*p::p] = False
    return flags


def primes_upto(N: int) -> np.ndarray:
    """
    Return array of all primes <= N.

    Parameters
    ----------
    N : int
        Upper bound (inclusive).

    Returns
    -------
    np.ndarray
        Array of primes.
    """
    flags = prime_flags_upto(N)
    return np.nonzero(flags)[0]
