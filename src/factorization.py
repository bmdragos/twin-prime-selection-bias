"""
Factorization utilities.

Responsibility: factor information, cleanly separated.
This file must not know about twin primes or states.
"""

import numpy as np
from typing import Set


def spf_sieve(N: int) -> np.ndarray:
    """
    Compute smallest prime factor for all integers up to N.

    Parameters
    ----------
    N : int
        Upper bound (inclusive).

    Returns
    -------
    np.ndarray
        Array where spf[i] is the smallest prime factor of i.
        spf[0] = 0, spf[1] = 1, and spf[p] = p for primes.

    Note
    ----
    This uses the convention spf[p] = p for primes.
    The parallel_sieve module uses spf[p] = 0 (uint32, memory-efficient).
    These are NOT interchangeable in omega/omega_leq_P functions.
    """
    spf = np.arange(N + 1, dtype=np.int64)
    spf[0] = 0
    spf[1] = 1

    for p in range(2, int(N**0.5) + 1):
        if spf[p] == p:  # p is prime
            for multiple in range(p * p, N + 1, p):
                if spf[multiple] == multiple:
                    spf[multiple] = p
    return spf


def omega(n: int, spf: np.ndarray) -> int:
    """
    Count distinct prime factors of n (big omega).

    Parameters
    ----------
    n : int
        Integer to factor.
    spf : np.ndarray
        Smallest prime factor array from spf_sieve (uses spf[p]=p for primes).
        WARNING: Do NOT use with parallel_sieve SPF (uses 0 sentinel).

    Returns
    -------
    int
        Number of distinct prime factors.
    """
    if n <= 1:
        return 0

    count = 0
    prev = 0
    while n > 1:
        p = spf[n]
        if p != prev:
            count += 1
            prev = p
        n //= p
    return count


def omega_leq_P(n: int, spf: np.ndarray, P: int) -> int:
    """
    Count distinct prime factors of n that are <= P.

    Parameters
    ----------
    n : int
        Integer to factor.
    spf : np.ndarray
        Smallest prime factor array from spf_sieve (uses spf[p]=p for primes).
        WARNING: Do NOT use with parallel_sieve SPF (uses 0 sentinel).
    P : int
        Upper bound on primes to count.

    Returns
    -------
    int
        Number of distinct prime factors <= P.
    """
    if n <= 1:
        return 0

    count = 0
    prev = 0
    while n > 1:
        p = spf[n]
        if p != prev and p <= P:
            count += 1
            prev = p
        elif p != prev:
            prev = p
        n //= p
    return count


def distinct_primes_leq_P(n: int, spf: np.ndarray, P: int) -> Set[int]:
    """
    Return set of distinct prime factors of n that are <= P.

    Parameters
    ----------
    n : int
        Integer to factor.
    spf : np.ndarray
        Smallest prime factor array.
    P : int
        Upper bound on primes to count.

    Returns
    -------
    set
        Set of prime factors <= P.
    """
    if n <= 1:
        return set()

    primes = set()
    while n > 1:
        p = spf[n]
        if p <= P:
            primes.add(p)
        n //= p
    return primes


def Omega(n: int, spf: np.ndarray) -> int:
    """
    Count prime factors of n with multiplicity (big Omega).

    Parameters
    ----------
    n : int
        Integer to factor.
    spf : np.ndarray
        Smallest prime factor array.

    Returns
    -------
    int
        Total count of prime factors with multiplicity.
    """
    if n <= 1:
        return 0

    count = 0
    while n > 1:
        n //= spf[n]
        count += 1
    return count
