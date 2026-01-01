"""
Experiment: Run Length Analysis (Appendix / Intuition Support)

Computes S_P(k), run-length distributions, and comparisons
between real, permuted, and block-shuffled sequences.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

from ..primes import prime_flags_upto, primes_upto
from ..sieve_pairs import compute_all_states, pair_values, STATES, PP
from ..factorization import spf_sieve, omega_leq_P
from ..metrics import run_lengths, summarize_runs
from ..null_models import shuffle_labels, block_shuffle


def compute_sign_sequence(K: int, P: int) -> np.ndarray:
    """
    Compute the sign sequence S_P(k) = omega_a - omega_b for each pair.

    Parameters
    ----------
    K : int
        Number of pairs.
    P : int
        Prime cutoff for omega.

    Returns
    -------
    np.ndarray
        Array of signs (-1, 0, +1).
    """
    N = 6 * K + 1
    spf = spf_sieve(N)

    signs = np.zeros(K, dtype=int)
    for i in range(K):
        a, b = pair_values(i + 1)
        omega_a = omega_leq_P(a, spf, P)
        omega_b = omega_leq_P(b, spf, P)
        diff = omega_a - omega_b
        signs[i] = np.sign(diff)

    return signs


def compute_state_run_lengths(K: int) -> Dict[str, np.ndarray]:
    """
    Compute run lengths for each state.

    Parameters
    ----------
    K : int
        Number of pairs.

    Returns
    -------
    dict
        Mapping from state to run length array.
    """
    N = 6 * K + 1
    prime_flags = prime_flags_upto(N)
    states = compute_all_states(K, prime_flags)

    result = {}
    for state in STATES:
        # Convert to binary: 1 if this state, 0 otherwise
        binary = (states == state).astype(int)
        runs = run_lengths(binary)
        # Only keep runs of 1s (runs of the target state)
        # Runs alternate between 0-runs and 1-runs
        # If binary starts with target state, 1-runs are at even indices
        # Otherwise at odd indices
        if len(binary) > 0 and binary[0] == 1:
            result[state] = runs[::2]  # Even indices
        else:
            result[state] = runs[1::2]  # Odd indices

    return result


def run_run_length_experiment(K: int, P_grid: List[int],
                              output_dir: Path, seed: int = 123) -> Dict[str, pd.DataFrame]:
    """
    Run the full run-length analysis experiment.

    Parameters
    ----------
    K : int
        Number of pairs.
    P_grid : list of int
        Prime cutoffs to analyze.
    output_dir : Path
        Output directory.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary of result DataFrames.
    """
    print(f"Running run-length experiment with K={K:,}")

    N = 6 * K + 1
    prime_flags = prime_flags_upto(N)
    states = compute_all_states(K, prime_flags)

    # State run lengths
    print("  Computing state run lengths...")
    state_runs = compute_state_run_lengths(K)

    state_rows = []
    for state in STATES:
        summary = summarize_runs(state_runs[state])
        state_rows.append({
            'state': state,
            **summary
        })
    df_state_runs = pd.DataFrame(state_rows)

    # Sign sequence run lengths for various P
    print("  Computing sign sequence run lengths...")
    sign_rows = []
    for P in P_grid:
        print(f"    P = {P}")
        signs = compute_sign_sequence(K, P)

        # Only consider non-zero signs
        nonzero_mask = signs != 0
        nonzero_signs = signs[nonzero_mask]

        if len(nonzero_signs) > 0:
            runs = run_lengths(nonzero_signs)
            summary = summarize_runs(runs)
            sign_rows.append({
                'P': P,
                'type': 'empirical',
                **summary
            })

            # Shuffled comparison
            shuffled = shuffle_labels(nonzero_signs, seed=seed)
            runs_shuffled = run_lengths(shuffled)
            summary_shuffled = summarize_runs(runs_shuffled)
            sign_rows.append({
                'P': P,
                'type': 'shuffled',
                **summary_shuffled
            })

    df_sign_runs = pd.DataFrame(sign_rows)

    # PP run comparison: empirical vs null models
    print("  Computing PP run comparisons...")
    pp_mask = states == PP
    pp_binary = pp_mask.astype(int)

    comparison_rows = []

    # Empirical
    emp_runs = run_lengths(pp_binary)
    emp_pp_runs = emp_runs[::2] if pp_binary[0] == 1 else emp_runs[1::2]
    comparison_rows.append({
        'model': 'empirical',
        **summarize_runs(emp_pp_runs)
    })

    # Shuffled
    shuffled_states = shuffle_labels(states, seed=seed)
    shuf_binary = (shuffled_states == PP).astype(int)
    shuf_runs = run_lengths(shuf_binary)
    shuf_pp_runs = shuf_runs[::2] if shuf_binary[0] == 1 else shuf_runs[1::2]
    comparison_rows.append({
        'model': 'shuffled',
        **summarize_runs(shuf_pp_runs)
    })

    # Block shuffled
    block_states = block_shuffle(states, block_size=2310, seed=seed)
    block_binary = (block_states == PP).astype(int)
    block_runs = run_lengths(block_binary)
    block_pp_runs = block_runs[::2] if block_binary[0] == 1 else block_runs[1::2]
    comparison_rows.append({
        'model': 'block_shuffled',
        **summarize_runs(block_pp_runs)
    })

    df_comparison = pd.DataFrame(comparison_rows)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    df_state_runs.to_csv(output_dir / 'run_lengths_by_state.csv', index=False)
    df_sign_runs.to_csv(output_dir / 'run_lengths_sign_sequence.csv', index=False)
    df_comparison.to_csv(output_dir / 'run_lengths_pp_comparison.csv', index=False)

    print(f"  Results saved to {output_dir}")

    return {
        'state_runs': df_state_runs,
        'sign_runs': df_sign_runs,
        'comparison': df_comparison
    }


if __name__ == '__main__':
    import yaml

    with open('config/default.yaml') as f:
        config = yaml.safe_load(f)

    output_dir = Path('data/results')
    results = run_run_length_experiment(
        config['K'],
        config['P_grid'],
        output_dir,
        config['seed']
    )

    print("\nState run lengths:")
    print(results['state_runs'].to_string(index=False))
