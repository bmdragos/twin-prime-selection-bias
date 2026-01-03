"""
Experiment: Cousin Primes (n, n+4) among 6k±1 candidates

Verifies the selection bias for cousin prime pairs.
For mutual exclusivity: if q | n and q | (n+4), then q | 4, so only q=2 fails.
When restricted to 6k±1 candidates (coprime to 2 and 3), neither member
is divisible by 2 or 3, so all primes q >= 5 satisfy mutual exclusivity.

Predicted shift: sum_{p >= 5} 1/[p(p-1)] ≈ 0.1065
(Same as twin primes, since the sum starts at p=5.)

Usage:
    python -m src.experiments.exp_cousin_primes --K 1e7 --save
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from ..factorization import spf_sieve, omega
from ..primes import prime_flags_upto


def compute_cousin_prime_bias(K: int, verbose: bool = True) -> dict:
    """
    Compute omega bias for cousin prime pairs (n, n+4) among 6k±1 candidates.

    We consider pairs where n = 6k-1 or n = 6k+1 (i.e., n coprime to 6),
    and compute ω(n+4) conditioned on whether n is prime.

    Parameters
    ----------
    K : int
        Number of k values to consider (k = 1, 2, ..., K).
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Results including means, counts, and predicted values.
    """
    if verbose:
        print(f"Cousin prime bias computation for K = {K:,}")

    # Maximum value we need: for k up to K, we have n up to 6K+1,
    # and n+4 up to 6K+5
    max_val = 6 * K + 5

    if verbose:
        print(f"  Building SPF sieve up to {max_val:,}...")
    spf = spf_sieve(max_val)

    if verbose:
        print(f"  Building prime flags up to {max_val:,}...")
    prime_flags = prime_flags_upto(max_val)

    if verbose:
        print("  Computing omega values for cousin pairs...")

    # Collect omega(n+4) for n prime vs n composite
    # where n is of the form 6k±1
    omega_when_prime = []
    omega_when_composite = []

    for k in range(1, K + 1):
        # Consider both 6k-1 and 6k+1
        for n in [6*k - 1, 6*k + 1]:
            if n + 4 > max_val:
                continue

            cousin = n + 4
            om = omega(cousin, spf)

            if prime_flags[n]:
                omega_when_prime.append(om)
            else:
                omega_when_composite.append(om)

    omega_when_prime = np.array(omega_when_prime)
    omega_when_composite = np.array(omega_when_composite)

    mean_prime = np.mean(omega_when_prime)
    mean_composite = np.mean(omega_when_composite)
    diff = mean_prime - mean_composite

    # Predicted: sum_{p >= 5} 1/[p(p-1)] (same as twin primes)
    predicted = sum(1 / (p * (p - 1)) for p in _primes_up_to(1000) if p >= 5)

    results = {
        'K': K,
        'n_prime_count': len(omega_when_prime),
        'n_composite_count': len(omega_when_composite),
        'mean_omega_when_n_prime': mean_prime,
        'mean_omega_when_n_composite': mean_composite,
        'difference': diff,
        'predicted': predicted,
        'relative_error': abs(diff - predicted) / predicted,
    }

    if verbose:
        print(f"\nResults for cousin pairs (n, n+4) among 6k±1, K = {K:,}:")
        print(f"  Sample sizes: {results['n_prime_count']:,} (n prime), {results['n_composite_count']:,} (n composite)")
        print(f"  E[ω(n+4) | n prime]:     {mean_prime:.4f}")
        print(f"  E[ω(n+4) | n composite]: {mean_composite:.4f}")
        print(f"  Difference:              {diff:.4f}")
        print(f"  Predicted (Σ 1/[p(p-1)], p≥5): {predicted:.4f}")
        print(f"  Relative error:          {results['relative_error']:.2%}")

    return results


def _primes_up_to(n: int) -> list:
    """Simple sieve for small n."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    return [i for i in range(n + 1) if sieve[i]]


def main():
    parser = argparse.ArgumentParser(description="Cousin prime pair bias verification")
    parser.add_argument('--K', type=float, default=1e7, help='Number of k values (default: 1e7)')
    parser.add_argument('--save', action='store_true', help='Save results to data/reference/')
    args = parser.parse_args()

    K = int(args.K)
    results = compute_cousin_prime_bias(K)

    if args.save:
        outdir = Path('data/reference/cousin_primes')
        outdir.mkdir(parents=True, exist_ok=True)

        # Save as CSV
        df = pd.DataFrame([results])
        csv_path = outdir / f'cousin_primes_K{K:.0e}.csv'.replace('+', '')
        df.to_csv(csv_path, index=False)
        print(f"\nSaved to {csv_path}")

        # Save as markdown table
        md_path = outdir / 'table.md'
        with open(md_path, 'w') as f:
            f.write(f"# Cousin Primes (n, n+4) among 6k±1 candidates\n\n")
            f.write(f"$K = {K:,}$\n\n")
            f.write("| Population | Mean ω(n+4) | Sample size |\n")
            f.write("|------------|-------------|-------------|\n")
            f.write(f"| n prime | {results['mean_omega_when_n_prime']:.3f} | {results['n_prime_count']:,} |\n")
            f.write(f"| n composite | {results['mean_omega_when_n_composite']:.3f} | {results['n_composite_count']:,} |\n")
            f.write(f"| **Difference** | **{results['difference']:.3f}** | — |\n")
            f.write(f"\nPredicted: Σ 1/[p(p-1)] for p ≥ 5 = {results['predicted']:.4f}\n")
            f.write(f"\nRelative error: {results['relative_error']:.2%}\n")
        print(f"Saved to {md_path}")


if __name__ == '__main__':
    main()
