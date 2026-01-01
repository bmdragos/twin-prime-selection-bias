"""
Extracting predictions from the transfer-matrix model.

Responsibility: producing model predictions like tilt_model(P).
"""

import numpy as np
from typing import Dict, Tuple

from .transfer_matrix import (
    T_p, T_p_graded, apply_primes, initial_distribution,
    STATE_INDEX, INDEX_STATE, NUM_STATES
)
from .primes import primes_upto


def state_generating_functions(P: int, A: int = 15, B: int = 15) -> Dict[str, np.ndarray]:
    """
    Compute generating functions for factor counts in each state.

    Uses graded transfer matrices up to prime P.

    Parameters
    ----------
    P : int
        Maximum prime to include.
    A, B : int
        Maximum tracked factor counts.

    Returns
    -------
    dict
        Dictionary mapping state names to (A, B) arrays of probabilities.
    """
    primes = primes_upto(P)
    primes = primes[primes >= 5]  # Skip 2 and 3

    T = apply_primes(primes, graded=True, A=A, B=B)

    # Initial distribution: all probability in PP with (0,0) factors
    dim = 4 * A * B
    v0 = np.zeros(dim)
    v0[0] = 1.0  # PP, 0 factors each

    # Final distribution
    v_final = T @ v0

    # Extract by state
    result = {}
    for state_name, state_idx in STATE_INDEX.items():
        dist = np.zeros((A, B))
        for a in range(A):
            for b in range(B):
                idx = state_idx * A * B + a * B + b
                dist[a, b] = v_final[idx]
        result[state_name] = dist

    return result


def conditional_hit_distribution(state_dist: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute marginal distributions of factor counts given state.

    Parameters
    ----------
    state_dist : np.ndarray
        (A, B) array of joint probabilities for a state.

    Returns
    -------
    tuple
        (marginal_a, marginal_b) distributions, normalized.
    """
    total = state_dist.sum()
    if total == 0:
        A, B = state_dist.shape
        return np.zeros(A), np.zeros(B)

    # Normalize
    normalized = state_dist / total

    # Marginals
    marginal_a = normalized.sum(axis=1)
    marginal_b = normalized.sum(axis=0)

    return marginal_a, marginal_b


def model_mean_omega(P: int, state: str, component: str = 'a',
                     A: int = 15, B: int = 15) -> float:
    """
    Compute model prediction for mean omega of a component in a state.

    Parameters
    ----------
    P : int
        Maximum prime.
    state : str
        State name ('PP', 'PC', 'CP', 'CC').
    component : str
        'a' for 6k-1 or 'b' for 6k+1.
    A, B : int
        Maximum tracked factor counts.

    Returns
    -------
    float
        Model prediction for E[omega | state].
    """
    distributions = state_generating_functions(P, A, B)
    marginal_a, marginal_b = conditional_hit_distribution(distributions[state])

    if component == 'a':
        return np.sum(np.arange(A) * marginal_a)
    else:
        return np.sum(np.arange(B) * marginal_b)


def model_tilt(P: int, threshold: int = 4, base: int = 2,
               state: str = 'PP', component: str = 'a',
               A: int = 20, B: int = 20) -> float:
    """
    Compute model prediction for tilt ratio.

    Parameters
    ----------
    P : int
        Maximum prime.
    threshold : int
        Upper threshold for tilt.
    base : int
        Lower threshold for tilt.
    state : str
        State to condition on.
    component : str
        'a' or 'b'.
    A, B : int
        Maximum tracked factor counts.

    Returns
    -------
    float
        Model tilt ratio.
    """
    distributions = state_generating_functions(P, A, B)
    marginal_a, marginal_b = conditional_hit_distribution(distributions[state])

    if component == 'a':
        marginal = marginal_a
    else:
        marginal = marginal_b

    high = marginal[threshold:].sum()
    low = marginal[:base].sum()

    if low == 0:
        return np.inf

    return high / low


def state_probabilities(P: int) -> Dict[str, float]:
    """
    Compute model probabilities for each state at prime cutoff P.

    Parameters
    ----------
    P : int
        Maximum prime.

    Returns
    -------
    dict
        Mapping from state name to probability.
    """
    primes = primes_upto(P)
    primes = primes[primes >= 5]

    T = apply_primes(primes, graded=False)
    v0 = initial_distribution()
    v_final = T @ v0

    return {INDEX_STATE[i]: v_final[i] for i in range(NUM_STATES)}
