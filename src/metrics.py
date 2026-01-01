"""
Definitions of all reported statistics.

Responsibility: paper-facing quantities. Guarantees that the paper's
definitions match the code exactly.
"""

import numpy as np
from typing import Tuple, Dict


def mean_omega(values: np.ndarray) -> float:
    """
    Compute mean of omega values.

    Parameters
    ----------
    values : np.ndarray
        Array of omega counts.

    Returns
    -------
    float
        Mean value.
    """
    return np.mean(values)


def tilt_ratio(values: np.ndarray, threshold: int = 4, base: int = 2) -> float:
    """
    Compute tilt ratio: P(omega >= threshold) / P(omega < base).

    Parameters
    ----------
    values : np.ndarray
        Array of omega counts.
    threshold : int
        Upper threshold (default 4).
    base : int
        Lower threshold (default 2).

    Returns
    -------
    float
        Ratio of probabilities, or inf if denominator is 0.
    """
    n = len(values)
    if n == 0:
        return np.nan

    high = np.sum(values >= threshold)
    low = np.sum(values < base)

    if low == 0:
        return np.inf

    return (high / n) / (low / n)


def run_lengths(sign_array: np.ndarray) -> np.ndarray:
    """
    Compute lengths of consecutive runs of the same sign.

    Parameters
    ----------
    sign_array : np.ndarray
        Array of +1 / -1 values (or boolean).

    Returns
    -------
    np.ndarray
        Array of run lengths.
    """
    if len(sign_array) == 0:
        return np.array([], dtype=int)

    # Find where values change
    changes = np.where(np.diff(sign_array) != 0)[0] + 1
    # Add boundaries
    boundaries = np.concatenate([[0], changes, [len(sign_array)]])
    # Compute lengths
    lengths = np.diff(boundaries)

    return lengths


def summarize_runs(runs: np.ndarray) -> Dict[str, float]:
    """
    Compute summary statistics for run lengths.

    Parameters
    ----------
    runs : np.ndarray
        Array of run lengths.

    Returns
    -------
    dict
        Dictionary with mean, median, max, std.
    """
    if len(runs) == 0:
        return {
            'mean': np.nan,
            'median': np.nan,
            'max': np.nan,
            'std': np.nan,
            'count': 0
        }

    return {
        'mean': np.mean(runs),
        'median': np.median(runs),
        'max': np.max(runs),
        'std': np.std(runs),
        'count': len(runs)
    }


def empirical_tilt(omega_a: np.ndarray, omega_b: np.ndarray,
                   state_mask: np.ndarray) -> Tuple[float, float]:
    """
    Compute empirical mean omega for each component of pairs in a given state.

    Parameters
    ----------
    omega_a : np.ndarray
        Omega values for 6k-1 elements.
    omega_b : np.ndarray
        Omega values for 6k+1 elements.
    state_mask : np.ndarray
        Boolean mask selecting pairs in the state of interest.

    Returns
    -------
    tuple
        (mean_omega_a, mean_omega_b) for selected pairs.
    """
    return (np.mean(omega_a[state_mask]), np.mean(omega_b[state_mask]))
