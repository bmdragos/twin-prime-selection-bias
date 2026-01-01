"""
Null models for destroying structure in controlled ways.

Responsibility: skeptic-proofing. Create comparison distributions
that break different types of correlations.
"""

import numpy as np
from typing import Optional


def shuffle_labels(states: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    """
    Randomly permute state labels (destroys all spatial structure).

    Parameters
    ----------
    states : np.ndarray
        Array of state labels.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Shuffled copy of states.
    """
    rng = np.random.default_rng(seed)
    shuffled = states.copy()
    rng.shuffle(shuffled)
    return shuffled


def block_shuffle(states: np.ndarray, block_size: int,
                  seed: Optional[int] = None) -> np.ndarray:
    """
    Shuffle blocks of states (preserves local structure, destroys global).

    Parameters
    ----------
    states : np.ndarray
        Array of state labels.
    block_size : int
        Size of blocks to shuffle (e.g., 2310 = 2*3*5*7*11).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Block-shuffled copy of states.
    """
    rng = np.random.default_rng(seed)
    n = len(states)
    n_blocks = n // block_size
    remainder = n % block_size

    # Reshape into blocks
    if n_blocks > 0:
        blocks = states[:n_blocks * block_size].reshape(n_blocks, block_size)
        # Shuffle block order
        block_order = rng.permutation(n_blocks)
        shuffled_blocks = blocks[block_order].flatten()

        # Handle remainder
        if remainder > 0:
            result = np.concatenate([shuffled_blocks, states[-remainder:]])
        else:
            result = shuffled_blocks
    else:
        result = states.copy()

    return result


def slot_matched_null(omega_values: np.ndarray, states: np.ndarray,
                      target_state: str, seed: Optional[int] = None) -> np.ndarray:
    """
    Create null distribution by shuffling omega values within each state class.

    This preserves marginal distribution of omega within each state,
    but destroys any ordering/correlation structure.

    Parameters
    ----------
    omega_values : np.ndarray
        Array of omega values.
    states : np.ndarray
        Array of state labels.
    target_state : str
        State to select from after shuffling.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Omega values for target_state after within-state shuffling.
    """
    rng = np.random.default_rng(seed)
    shuffled = omega_values.copy()

    # Shuffle within each state class
    unique_states = np.unique(states)
    for state in unique_states:
        mask = states == state
        indices = np.where(mask)[0]
        values = shuffled[mask]
        rng.shuffle(values)
        shuffled[indices] = values

    # Return values for target state
    return shuffled[states == target_state]


def permute_within_primorial(states: np.ndarray, primorial: int,
                             seed: Optional[int] = None) -> np.ndarray:
    """
    Permute states within each primorial-length window.

    Preserves the residue-class structure imposed by small primes.

    Parameters
    ----------
    states : np.ndarray
        Array of state labels.
    primorial : int
        Window size (e.g., 30 = 2*3*5 or 2310 = 2*3*5*7*11).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Permuted copy of states.
    """
    rng = np.random.default_rng(seed)
    result = states.copy()
    n = len(states)

    for start in range(0, n, primorial):
        end = min(start + primorial, n)
        window = result[start:end].copy()
        rng.shuffle(window)
        result[start:end] = window

    return result
