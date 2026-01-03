"""
Experiment: Sophie Germain Pairs (n, 2n+1)

Verifies the selection bias for Sophie Germain pairs.
For mutual exclusivity: if q | n and q | (2n+1), then q | 1, impossible.
So all odd primes q >= 3 satisfy mutual exclusivity.

Predicted shift: sum_{q >= 3} 1/[q(q-1)] ≈ 0.273
(Sum starts at q=3 because 2 | (2n+1) is impossible when n is odd.)

Usage:
    python -m src.experiments.exp_sophie_germain --N 1e7 --save
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from ..factorization import spf_sieve, omega
from ..primes import prime_flags_upto


def compute_sophie_germain_bias(N: int, verbose: bool = True) -> dict:
    """
    Compute omega bias for Sophie Germain pairs (n, 2n+1).

    Parameters
    ----------
    N : int
        Maximum value of n to consider.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Results including means, counts, and predicted values.
    """
    if verbose:
        print(f"Sophie Germain bias computation for n <= {N:,}")

    # We need SPF up to 2N+1 for computing omega(2n+1)
    max_val = 2 * N + 1

    if verbose:
        print(f"  Building SPF sieve up to {max_val:,}...")
    spf = spf_sieve(max_val)

    if verbose:
        print(f"  Building prime flags up to {N:,}...")
    prime_flags = prime_flags_upto(N)

    # We only consider odd n (since 2n+1 must be odd for Sophie Germain)
    # Actually, for n even, 2n+1 is odd. For n odd, 2n+1 is also odd.
    # The standard Sophie Germain definition uses odd primes p.
    # Let's consider all n >= 2 and compare prime vs composite.

    if verbose:
        print("  Computing omega values...")

    # Collect omega(2n+1) for n prime vs n composite
    omega_when_prime = []
    omega_when_composite = []

    for n in range(2, N + 1):
        val = 2 * n + 1
        om = omega(val, spf)

        if prime_flags[n]:
            omega_when_prime.append(om)
        else:
            omega_when_composite.append(om)

    omega_when_prime = np.array(omega_when_prime)
    omega_when_composite = np.array(omega_when_composite)

    mean_prime = np.mean(omega_when_prime)
    mean_composite = np.mean(omega_when_composite)
    diff = mean_prime - mean_composite

    # Predicted: sum_{q >= 3} 1/[q(q-1)]
    # = 1/(3*2) + 1/(5*4) + 1/(7*6) + ...
    # = 1/6 + 1/20 + 1/42 + 1/72 + ...
    # ≈ 0.2728...
    predicted = sum(1 / (p * (p - 1)) for p in _primes_up_to(1000) if p >= 3)

    results = {
        'N': N,
        'n_prime_count': len(omega_when_prime),
        'n_composite_count': len(omega_when_composite),
        'mean_omega_when_n_prime': mean_prime,
        'mean_omega_when_n_composite': mean_composite,
        'difference': diff,
        'predicted': predicted,
        'relative_error': abs(diff - predicted) / predicted,
    }

    if verbose:
        print(f"\nResults for Sophie Germain pairs (n, 2n+1), n <= {N:,}:")
        print(f"  Sample sizes: {results['n_prime_count']:,} (n prime), {results['n_composite_count']:,} (n composite)")
        print(f"  E[ω(2n+1) | n prime]:     {mean_prime:.4f}")
        print(f"  E[ω(2n+1) | n composite]: {mean_composite:.4f}")
        print(f"  Difference:               {diff:.4f}")
        print(f"  Predicted (Σ 1/[q(q-1)], q≥3): {predicted:.4f}")
        print(f"  Relative error:           {results['relative_error']:.2%}")

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
    parser = argparse.ArgumentParser(description="Sophie Germain pair bias verification")
    parser.add_argument('--N', type=float, default=1e7, help='Maximum n value (default: 1e7)')
    parser.add_argument('--save', action='store_true', help='Save results to data/reference/')
    args = parser.parse_args()

    N = int(args.N)
    results = compute_sophie_germain_bias(N)

    if args.save:
        outdir = Path('data/reference/sophie_germain')
        outdir.mkdir(parents=True, exist_ok=True)

        # Save as CSV
        df = pd.DataFrame([results])
        csv_path = outdir / f'sophie_germain_N{N:.0e}.csv'.replace('+', '')
        df.to_csv(csv_path, index=False)
        print(f"\nSaved to {csv_path}")

        # Save as markdown table
        md_path = outdir / 'table.md'
        with open(md_path, 'w') as f:
            f.write(f"# Sophie Germain Pairs (n, 2n+1)\n\n")
            f.write(f"$N = {N:,}$\n\n")
            f.write("| Population | Mean ω(2n+1) | Sample size |\n")
            f.write("|------------|--------------|-------------|\n")
            f.write(f"| n prime | {results['mean_omega_when_n_prime']:.3f} | {results['n_prime_count']:,} |\n")
            f.write(f"| n composite | {results['mean_omega_when_n_composite']:.3f} | {results['n_composite_count']:,} |\n")
            f.write(f"| **Difference** | **{results['difference']:.3f}** | — |\n")
            f.write(f"\nPredicted: Σ 1/[q(q-1)] for q ≥ 3 = {results['predicted']:.4f}\n")
            f.write(f"\nRelative error: {results['relative_error']:.2%}\n")
        print(f"Saved to {md_path}")


if __name__ == '__main__':
    main()
