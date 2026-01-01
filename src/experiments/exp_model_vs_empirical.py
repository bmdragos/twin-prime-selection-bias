"""
Experiment: Model vs Empirical Comparison (Paper Sections 5-6)

Computes empirical tilt and compares to transfer-matrix model predictions.
This is the crux experiment.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

from ..primes import prime_flags_upto, primes_upto
from ..sieve_pairs import compute_all_states, pair_values, STATES
from ..factorization import spf_sieve, omega_leq_P
from ..metrics import mean_omega, tilt_ratio
from ..coefficient_extraction import model_mean_omega, model_tilt, state_probabilities


def compute_empirical_stats(K: int, P: int) -> Dict[str, Any]:
    """
    Compute empirical statistics for prime cutoff P.

    Parameters
    ----------
    K : int
        Number of pairs.
    P : int
        Maximum prime for omega computation.

    Returns
    -------
    dict
        Dictionary of statistics by state.
    """
    N = 6 * K + 1
    prime_flags = prime_flags_upto(N)
    spf = spf_sieve(N)
    states = compute_all_states(K, prime_flags)

    # Compute omega_leq_P for all pairs
    omega_a = np.zeros(K, dtype=int)
    omega_b = np.zeros(K, dtype=int)

    for i in range(K):
        a, b = pair_values(i + 1)
        omega_a[i] = omega_leq_P(a, spf, P)
        omega_b[i] = omega_leq_P(b, spf, P)

    results = {}
    for state in STATES:
        mask = states == state
        count = np.sum(mask)

        if count > 0:
            results[state] = {
                'count': count,
                'mean_omega_a': mean_omega(omega_a[mask]),
                'mean_omega_b': mean_omega(omega_b[mask]),
                'tilt_a': tilt_ratio(omega_a[mask]),
                'tilt_b': tilt_ratio(omega_b[mask])
            }
        else:
            results[state] = {
                'count': 0,
                'mean_omega_a': np.nan,
                'mean_omega_b': np.nan,
                'tilt_a': np.nan,
                'tilt_b': np.nan
            }

    return results


def compute_model_stats(P: int) -> Dict[str, Any]:
    """
    Compute model predictions for prime cutoff P.

    Parameters
    ----------
    P : int
        Maximum prime.

    Returns
    -------
    dict
        Dictionary of model predictions by state.
    """
    probs = state_probabilities(P)

    results = {}
    for state in STATES:
        results[state] = {
            'probability': probs[state],
            'mean_omega_a': model_mean_omega(P, state, 'a'),
            'mean_omega_b': model_mean_omega(P, state, 'b'),
            'tilt_a': model_tilt(P, state=state, component='a'),
            'tilt_b': model_tilt(P, state=state, component='b')
        }

    return results


def run_model_comparison_experiment(K: int, P_grid: List[int],
                                    output_dir: Path) -> pd.DataFrame:
    """
    Run the model vs empirical comparison across P values.

    Parameters
    ----------
    K : int
        Number of pairs for empirical computation.
    P_grid : list of int
        Prime cutoff values to test.
    output_dir : Path
        Directory for output files.

    Returns
    -------
    pd.DataFrame
        Comparison table.
    """
    print(f"Running model vs empirical comparison")
    print(f"  K = {K:,}, P_grid = {P_grid}")

    rows = []

    for P in P_grid:
        print(f"  Processing P = {P}...")

        # Empirical
        emp = compute_empirical_stats(K, P)

        # Model
        mod = compute_model_stats(P)

        for state in STATES:
            rows.append({
                'P': P,
                'state': state,
                'emp_count': emp[state]['count'],
                'emp_mean_omega_a': emp[state]['mean_omega_a'],
                'emp_mean_omega_b': emp[state]['mean_omega_b'],
                'emp_tilt_a': emp[state]['tilt_a'],
                'emp_tilt_b': emp[state]['tilt_b'],
                'mod_probability': mod[state]['probability'],
                'mod_mean_omega_a': mod[state]['mean_omega_a'],
                'mod_mean_omega_b': mod[state]['mean_omega_b'],
                'mod_tilt_a': mod[state]['tilt_a'],
                'mod_tilt_b': mod[state]['tilt_b'],
            })

    df = pd.DataFrame(rows)

    # Compute deltas
    df['delta_mean_a'] = df['emp_mean_omega_a'] - df['mod_mean_omega_a']
    df['delta_mean_b'] = df['emp_mean_omega_b'] - df['mod_mean_omega_b']

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / 'model_vs_empirical.csv', index=False)

    print(f"  Results saved to {output_dir}")

    return df


if __name__ == '__main__':
    import yaml

    with open('config/default.yaml') as f:
        config = yaml.safe_load(f)

    output_dir = Path('data/results')
    df = run_model_comparison_experiment(
        config['K'],
        config['P_grid'],
        output_dir
    )
    print("\nComparison (PP state only):")
    print(df[df['state'] == 'PP'].to_string(index=False))
