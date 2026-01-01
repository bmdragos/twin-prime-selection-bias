"""
Transfer-matrix model (theoretical spine).

Responsibility: the mean-field model only. No empirical data here.
This file IS the model.
"""

import numpy as np
from typing import Dict, List, Tuple

# State indexing
STATE_INDEX = {'PP': 0, 'PC': 1, 'CP': 2, 'CC': 3}
INDEX_STATE = {0: 'PP', 1: 'PC', 2: 'CP', 3: 'CC'}
NUM_STATES = 4


def T_p(p: int) -> np.ndarray:
    """
    Compute the ungraded transfer matrix for prime p.

    For a prime p >= 5, at most one of 6k-1 or 6k+1 can be divisible
    (they differ by 2, so can't both be 0 mod p).

    Probabilities:
        - (p-2)/p: neither divisible (no hit)
        - 1/p: only 6k-1 divisible (hit a)
        - 1/p: only 6k+1 divisible (hit b)

    Parameters
    ----------
    p : int
        A prime >= 5.

    Returns
    -------
    np.ndarray
        4x4 transfer matrix.
    """
    p_no_hit = (p - 2) / p
    p_hit = 1 / p

    T = np.zeros((4, 4))

    # From PP: both prime candidates
    T[0, 0] = p_no_hit    # no hit -> stay PP
    T[1, 0] = p_hit       # hit b -> PC (b becomes composite)
    T[2, 0] = p_hit       # hit a -> CP (a becomes composite)

    # From PC: a=prime, b=composite
    T[1, 1] = p_no_hit + p_hit  # no hit OR hit b -> stay PC
    T[3, 1] = p_hit             # hit a -> CC

    # From CP: a=composite, b=prime
    T[2, 2] = p_no_hit + p_hit  # no hit OR hit a -> stay CP
    T[3, 2] = p_hit             # hit b -> CC

    # From CC: both composite, absorbing in state
    T[3, 3] = 1.0

    return T


def T_p_graded(p: int, A: int, B: int) -> np.ndarray:
    """
    Compute graded transfer matrix tracking factor counts.

    The graded version tracks not just state but also the
    accumulated factor counts for components.

    Parameters
    ----------
    p : int
        A prime >= 5.
    A : int
        Maximum tracked factors for 6k-1 component.
    B : int
        Maximum tracked factors for 6k+1 component.

    Returns
    -------
    np.ndarray
        (4 * A * B) x (4 * A * B) transfer matrix.
    """
    dim = 4 * A * B
    T = np.zeros((dim, dim))

    prob_div = 2 / p
    surv = 1 - prob_div

    def idx(state: int, a: int, b: int) -> int:
        """Convert (state, a_count, b_count) to linear index."""
        return state * A * B + a * B + b

    # For p >= 5, at most one of 6k-1 or 6k+1 is divisible by p
    # (they differ by 2, so can't both be 0 mod p for p >= 5)
    # Probabilities: no hit = (p-2)/p, hit a = 1/p, hit b = 1/p
    p_no_hit = (p - 2) / p
    p_hit = 1 / p

    for state in range(4):
        for a in range(A):
            for b in range(B):
                src = idx(state, a, b)
                a_new = min(a + 1, A - 1)
                b_new = min(b + 1, B - 1)

                if state == 0:  # PP: both prime candidates
                    # no hit: stay PP
                    T[idx(0, a, b), src] += p_no_hit
                    # hit a: go to CP (a becomes composite), increment a
                    T[idx(2, a_new, b), src] += p_hit
                    # hit b: go to PC (b becomes composite), increment b
                    T[idx(1, a, b_new), src] += p_hit

                elif state == 1:  # PC: a=prime, b=composite
                    # no hit: stay PC
                    T[idx(1, a, b), src] += p_no_hit
                    # hit a: go to CC (a becomes composite), increment a
                    T[idx(3, a_new, b), src] += p_hit
                    # hit b: stay PC (b already composite), increment b
                    T[idx(1, a, b_new), src] += p_hit

                elif state == 2:  # CP: a=composite, b=prime
                    # no hit: stay CP
                    T[idx(2, a, b), src] += p_no_hit
                    # hit a: stay CP (a already composite), increment a
                    T[idx(2, a_new, b), src] += p_hit
                    # hit b: go to CC (b becomes composite), increment b
                    T[idx(3, a, b_new), src] += p_hit

                else:  # CC: both composite
                    # no hit: stay CC
                    T[idx(3, a, b), src] += p_no_hit
                    # hit a: stay CC, increment a
                    T[idx(3, a_new, b), src] += p_hit
                    # hit b: stay CC, increment b
                    T[idx(3, a, b_new), src] += p_hit

    return T


def matrix_product(matrices: List[np.ndarray]) -> np.ndarray:
    """
    Compute product of transfer matrices.

    Parameters
    ----------
    matrices : list of np.ndarray
        List of matrices to multiply (left to right).

    Returns
    -------
    np.ndarray
        Product matrix.
    """
    result = matrices[0].copy()
    for M in matrices[1:]:
        result = M @ result
    return result


def apply_primes(primes: np.ndarray, graded: bool = False,
                 A: int = 10, B: int = 10) -> np.ndarray:
    """
    Compute cumulative transfer matrix for a sequence of primes.

    Parameters
    ----------
    primes : np.ndarray
        Array of primes to apply (should start >= 5).
    graded : bool
        If True, use graded matrices.
    A, B : int
        Maximum tracked factor counts (only if graded=True).

    Returns
    -------
    np.ndarray
        Product of all transfer matrices.
    """
    if len(primes) == 0:
        if graded:
            return np.eye(4 * A * B)
        return np.eye(4)

    if graded:
        matrices = [T_p_graded(p, A, B) for p in primes]
    else:
        matrices = [T_p(p) for p in primes]

    return matrix_product(matrices)


def initial_distribution() -> np.ndarray:
    """
    Return initial state distribution (all pairs start as PP).

    Returns
    -------
    np.ndarray
        Initial probability vector.
    """
    v = np.zeros(4)
    v[0] = 1.0  # All start as PP
    return v
