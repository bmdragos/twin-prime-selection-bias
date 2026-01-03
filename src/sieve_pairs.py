"""
Logic around pairs (6k-1, 6k+1).

Responsibility: definition of the objects of study.

Note: All analysis is conditioned on the 6k±1 residue classes.
These are exactly the integers coprime to 2 and 3, and include all
primes > 3. Results about composites in these classes do NOT generalize
directly to "all composites" (which would include multiples of 2 and 3).
"""

import numpy as np
from typing import Tuple, Iterator

# State labels
PP = 'PP'  # both prime
PC = 'PC'  # 6k-1 prime, 6k+1 composite
CP = 'CP'  # 6k-1 composite, 6k+1 prime
CC = 'CC'  # both composite

STATES = [PP, PC, CP, CC]


def pair_values(k: int) -> Tuple[int, int]:
    """
    Return the pair (6k-1, 6k+1) for a given k >= 1.

    Parameters
    ----------
    k : int
        Pair index (k >= 1).

    Returns
    -------
    tuple
        (6k-1, 6k+1)
    """
    return (6 * k - 1, 6 * k + 1)


def pair_state(a: int, b: int, prime_flags: np.ndarray) -> str:
    """
    Determine the state of pair (a, b) given primality flags.

    Parameters
    ----------
    a : int
        First element of pair (6k-1).
    b : int
        Second element of pair (6k+1).
    prime_flags : np.ndarray
        Boolean array where prime_flags[i] is True iff i is prime.

    Returns
    -------
    str
        One of 'PP', 'PC', 'CP', 'CC'.
    """
    a_prime = prime_flags[a]
    b_prime = prime_flags[b]

    if a_prime and b_prime:
        return PP
    elif a_prime and not b_prime:
        return PC
    elif not a_prime and b_prime:
        return CP
    else:
        return CC


def iterate_pairs(K: int) -> Iterator[Tuple[int, int, int]]:
    """
    Iterate over pairs (k, a, b) for k = 1, 2, ..., K.

    Parameters
    ----------
    K : int
        Maximum k value.

    Yields
    ------
    tuple
        (k, 6k-1, 6k+1)
    """
    for k in range(1, K + 1):
        a, b = pair_values(k)
        yield k, a, b


def compute_all_states(K: int, prime_flags: np.ndarray) -> np.ndarray:
    """
    Compute states for all pairs k = 1, ..., K.

    Parameters
    ----------
    K : int
        Maximum k value.
    prime_flags : np.ndarray
        Boolean primality flags.

    Returns
    -------
    np.ndarray
        Array of state strings, length K.
    """
    states = np.empty(K, dtype='<U2')
    for i, (k, a, b) in enumerate(iterate_pairs(K)):
        states[i] = pair_state(a, b, prime_flags)
    return states
