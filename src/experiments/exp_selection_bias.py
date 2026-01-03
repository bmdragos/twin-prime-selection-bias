"""
Experiment: Selection Bias in Factor Counts (Paper Section 3)

Computes empirical omega/Omega statistics and applies null models.
Outputs tables and CSVs.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any

from ..primes import prime_flags_upto
from ..sieve_pairs import compute_all_states, pair_values, STATES
from ..factorization import spf_sieve, omega, omega_leq_P, Omega
from ..metrics import mean_omega, tilt_ratio
from ..null_models import shuffle_labels, block_shuffle, slot_matched_null


def compute_omega_by_state(K: int, P: int = None) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute omega values for each component partitioned by state.

    Parameters
    ----------
    K : int
        Number of pairs to analyze.
    P : int, optional
        Maximum prime for omega_leq_P. If None, computes full omega.

    Returns
    -------
    dict
        Nested dict: {state: {'a': omega_values, 'b': omega_values}}
    """
    N = 6 * K + 1
    prime_flags = prime_flags_upto(N)
    spf = spf_sieve(N)
    states = compute_all_states(K, prime_flags)

    # Compute omega for all a and b values
    omega_a = np.zeros(K, dtype=int)
    omega_b = np.zeros(K, dtype=int)

    for i in range(K):
        a, b = pair_values(i + 1)
        if P is None:
            omega_a[i] = omega(a, spf)
            omega_b[i] = omega(b, spf)
        else:
            omega_a[i] = omega_leq_P(a, spf, P)
            omega_b[i] = omega_leq_P(b, spf, P)

    # Partition by state
    result = {}
    for state in STATES:
        mask = states == state
        result[state] = {
            'a': omega_a[mask],
            'b': omega_b[mask]
        }

    return result


def run_selection_bias_experiment(K: int, output_dir: Path,
                                  seed: int = 123) -> pd.DataFrame:
    """
    Run the full selection bias experiment.

    Parameters
    ----------
    K : int
        Number of pairs.
    output_dir : Path
        Directory for output files.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Summary statistics table.
    """
    print(f"Running selection bias experiment with K={K:,}")

    N = 6 * K + 1
    prime_flags = prime_flags_upto(N)
    spf = spf_sieve(N)
    states = compute_all_states(K, prime_flags)

    # Compute omega for all pairs
    omega_a = np.zeros(K, dtype=int)
    omega_b = np.zeros(K, dtype=int)

    for i in range(K):
        a, b = pair_values(i + 1)
        omega_a[i] = omega(a, spf)
        omega_b[i] = omega(b, spf)

    # Summary statistics by state
    rows = []
    for state in STATES:
        mask = states == state
        count = np.sum(mask)

        if count > 0:
            mean_a = mean_omega(omega_a[mask])
            mean_b = mean_omega(omega_b[mask])
            tilt_a = tilt_ratio(omega_a[mask])
            tilt_b = tilt_ratio(omega_b[mask])
        else:
            mean_a = mean_b = tilt_a = tilt_b = np.nan

        rows.append({
            'state': state,
            'count': count,
            'fraction': count / K,
            'mean_omega_a': mean_a,
            'mean_omega_b': mean_b,
            'tilt_a': tilt_a,
            'tilt_b': tilt_b
        })

    df = pd.DataFrame(rows)

    # Null model comparisons
    print("  Computing null model comparisons...")
    null_rows = []

    # Shuffled null
    shuffled_states = shuffle_labels(states, seed=seed)
    for state in STATES:
        mask = shuffled_states == state
        count = np.sum(mask)
        if count > 0:
            null_rows.append({
                'null_model': 'shuffled',
                'state': state,
                'mean_omega_a': mean_omega(omega_a[mask]),
                'mean_omega_b': mean_omega(omega_b[mask])
            })

    # Block shuffled null (block_size = 2310)
    block_shuffled = block_shuffle(states, block_size=2310, seed=seed)
    for state in STATES:
        mask = block_shuffled == state
        count = np.sum(mask)
        if count > 0:
            null_rows.append({
                'null_model': 'block_shuffled',
                'state': state,
                'mean_omega_a': mean_omega(omega_a[mask]),
                'mean_omega_b': mean_omega(omega_b[mask])
            })

    df_null = pd.DataFrame(null_rows)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / 'selection_bias_summary.csv', index=False)
    df_null.to_csv(output_dir / 'selection_bias_null_models.csv', index=False)

    print(f"  Results saved to {output_dir}")

    return df


if __name__ == '__main__':
    import yaml

    with open('config/default.yaml') as f:
        config = yaml.safe_load(f)

    output_dir = Path('data/results')
    df = run_selection_bias_experiment(config['K'], output_dir, config['seed'])
    print("\nSummary:")
    print(df.to_string(index=False))
