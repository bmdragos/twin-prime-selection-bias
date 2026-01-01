"""
GPU-accelerated Model vs Empirical Comparison.

Falls back to CPU if GPU not available.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import time

from ..primes import prime_flags_upto, primes_upto
from ..sieve_pairs import STATES
from ..factorization import spf_sieve
from ..metrics import mean_omega, tilt_ratio
from ..coefficient_extraction import model_mean_omega, model_tilt, state_probabilities

# Try GPU imports
try:
    from ..gpu_factorization import (
        HAS_GPU, omega_leq_P_gpu, omega_gpu, compute_states_gpu, check_gpu
    )
except ImportError:
    HAS_GPU = False


def compute_empirical_stats_gpu(K: int, P: int, prime_flags: np.ndarray,
                                 spf: np.ndarray) -> Dict[str, Any]:
    """
    Compute empirical statistics using GPU acceleration.
    """
    # Generate all a, b values
    k_vals = np.arange(1, K + 1, dtype=np.int64)
    a_vals = 6 * k_vals - 1
    b_vals = 6 * k_vals + 1

    # Compute states on GPU
    state_codes = compute_states_gpu(K, prime_flags)
    # Convert codes to strings for compatibility
    code_to_state = {0: 'PP', 1: 'PC', 2: 'CP', 3: 'CC'}

    # Compute omega on GPU
    omega_a = omega_leq_P_gpu(a_vals, spf, P)
    omega_b = omega_leq_P_gpu(b_vals, spf, P)

    results = {}
    for code, state in code_to_state.items():
        mask = state_codes == code
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
    """Compute model predictions for prime cutoff P."""
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


def run_model_comparison_gpu(K: int, P_grid: List[int],
                              output_dir: Path) -> pd.DataFrame:
    """
    Run the model vs empirical comparison with GPU acceleration.
    """
    print(f"Running GPU-accelerated model vs empirical comparison")
    print(f"  K = {K:,}, P_grid = {P_grid}")

    if HAS_GPU:
        check_gpu()
    else:
        print("  WARNING: GPU not available, using CPU fallback")
        # Fall back to CPU implementation
        from .exp_model_vs_empirical import run_model_comparison_experiment
        return run_model_comparison_experiment(K, P_grid, output_dir)

    N = 6 * K + 1

    # Pre-compute sieves (CPU, done once)
    print("  Building sieves...")
    t0 = time.time()
    prime_flags = prime_flags_upto(N)
    spf = spf_sieve(N)
    print(f"    Sieves built in {time.time() - t0:.1f}s")

    rows = []

    for P in P_grid:
        print(f"  Processing P = {P}...", end=" ", flush=True)
        t0 = time.time()

        # Empirical (GPU)
        emp = compute_empirical_stats_gpu(K, P, prime_flags, spf)

        # Model (CPU, fast)
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

        print(f"{time.time() - t0:.2f}s")

    df = pd.DataFrame(rows)
    df['delta_mean_a'] = df['emp_mean_omega_a'] - df['mod_mean_omega_a']
    df['delta_mean_b'] = df['emp_mean_omega_b'] - df['mod_mean_omega_b']

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / 'model_vs_empirical.csv', index=False)

    print(f"  Results saved to {output_dir}")

    return df


if __name__ == '__main__':
    import yaml

    with open('config/default.yaml') as f:
        config = yaml.safe_load(f)

    output_dir = Path('data/results')
    df = run_model_comparison_gpu(
        config['K'],
        config['P_grid'],
        output_dir
    )
    print("\nComparison (PP state only):")
    print(df[df['state'] == 'PP'].to_string(index=False))
