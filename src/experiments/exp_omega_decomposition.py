#!/usr/bin/env python3
"""
Experiment: Omega Decomposition (Small vs Large Prime Factors)

Decomposes the selection bias into:
1. Small prime factors (p <= sqrt(N))
2. Large prime cofactors (p > sqrt(N))

This addresses the "sieve endgame" observation: CC composites are more likely
to have a large prime cofactor, which partially offsets the small-prime bias.

At K=10^7:
- Small omega bias: ~4.9%
- Large prime effect: CC has 68.5% vs PC 66.7%
- Net full omega bias: ~2.96%
"""

import numpy as np
from math import isqrt
import time
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.wheel_sieve import wheel_spf_sieve, wheel_spf_lookup
from src.primes import prime_flags_upto


def compute_omega_decomposition(K: int, verbose: bool = True):
    """
    Compute omega decomposition into small and large prime components.

    Parameters
    ----------
    K : int
        Number of pairs to analyze
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Results dictionary with all statistics
    """
    N = 6 * K + 1
    sqrt_N = isqrt(N)

    if verbose:
        print(f"Omega Decomposition Analysis")
        print(f"K = {K:,}, N = {N:,}, sqrt(N) = {sqrt_N:,}")
        print("=" * 60)

    # Generate wheel SPF
    t0 = time.time()
    if verbose:
        print("Generating wheel SPF sieve...")
    spf_wheel = wheel_spf_sieve(K)
    if verbose:
        print(f"  Done in {time.time()-t0:.1f}s")

    # Generate prime flags
    if verbose:
        print("Generating prime flags...")
    t0 = time.time()
    prime_flags = prime_flags_upto(N)
    if verbose:
        print(f"  Done in {time.time()-t0:.1f}s")

    # Counters for PC (a prime, b composite)
    pc_omega_small_sum = 0
    pc_omega_full_sum = 0
    pc_has_big_sum = 0
    pc_count = 0

    # Counters for CC (both composite)
    cc_omega_small_sum = 0
    cc_omega_full_sum = 0
    cc_has_big_sum = 0
    cc_count = 0

    if verbose:
        print("Processing pairs...")
    t0 = time.time()

    report_interval = max(K // 10, 1000000)

    for k in range(1, K + 1):
        a = 6 * k - 1
        b = 6 * k + 1

        a_prime = prime_flags[a]
        b_prime = prime_flags[b]

        if b_prime:
            continue  # Skip PP and CP (b is prime)

        # b is composite - compute omega decomposition
        n = b
        omega_small = 0
        omega_full = 0
        has_big = 0
        prev_p = 0

        while n > 1:
            p = wheel_spf_lookup(n, spf_wheel)
            if p != prev_p:
                omega_full += 1
                if p <= sqrt_N:
                    omega_small += 1
                else:
                    has_big = 1
                prev_p = p
            n //= p

        if a_prime:  # PC
            pc_omega_small_sum += omega_small
            pc_omega_full_sum += omega_full
            pc_has_big_sum += has_big
            pc_count += 1
        else:  # CC
            cc_omega_small_sum += omega_small
            cc_omega_full_sum += omega_full
            cc_has_big_sum += has_big
            cc_count += 1

        if verbose and k % report_interval == 0:
            elapsed = time.time() - t0
            rate = k / elapsed
            eta = (K - k) / rate
            print(f"  {k:,} / {K:,} ({k/K*100:.0f}%) - {rate/1e6:.2f}M/s - ETA {eta:.0f}s")

    if verbose:
        print(f"  Done in {time.time()-t0:.1f}s")

    # Compute means
    pc_omega_small = pc_omega_small_sum / pc_count
    pc_omega_full = pc_omega_full_sum / pc_count
    pc_has_big = pc_has_big_sum / pc_count

    cc_omega_small = cc_omega_small_sum / cc_count
    cc_omega_full = cc_omega_full_sum / cc_count
    cc_has_big = cc_has_big_sum / cc_count

    # Compute differences
    diff_small = pc_omega_small - cc_omega_small
    diff_big = pc_has_big - cc_has_big
    diff_full = pc_omega_full - cc_omega_full

    results = {
        'K': K,
        'N': N,
        'sqrt_N': sqrt_N,
        'pc_count': pc_count,
        'cc_count': cc_count,
        'pc_omega_small': pc_omega_small,
        'pc_omega_full': pc_omega_full,
        'pc_has_big': pc_has_big,
        'cc_omega_small': cc_omega_small,
        'cc_omega_full': cc_omega_full,
        'cc_has_big': cc_has_big,
        'diff_small': diff_small,
        'diff_big': diff_big,
        'diff_full': diff_full,
        'bias_small_pct': diff_small / cc_omega_small * 100,
        'bias_full_pct': diff_full / cc_omega_full * 100,
        'reduction_pct': -diff_big / diff_small * 100 if diff_small != 0 else 0,
    }

    return results


def print_results(results: dict):
    """Print results in formatted table."""
    print()
    print("=" * 70)
    print(f"RESULTS: K = {results['K']:,}")
    print("=" * 70)
    print(f"PC count: {results['pc_count']:,}")
    print(f"CC count: {results['cc_count']:,}")
    print(f"sqrt(N) = {results['sqrt_N']:,}")
    print()

    print(f"{'Component':<35} {'PC':>12} {'CC':>12} {'Diff':>12}")
    print("-" * 70)
    print(f"{'omega_small (p <= sqrt(N))':<35} {results['pc_omega_small']:>12.6f} {results['cc_omega_small']:>12.6f} {results['diff_small']:>+12.6f}")
    print(f"{'Has large prime (p > sqrt(N))':<35} {results['pc_has_big']*100:>11.2f}% {results['cc_has_big']*100:>11.2f}% {results['diff_big']:>+12.6f}")
    print(f"{'omega_full':<35} {results['pc_omega_full']:>12.6f} {results['cc_omega_full']:>12.6f} {results['diff_full']:>+12.6f}")
    print("-" * 70)
    print()

    print("BIAS ANALYSIS:")
    print(f"  Small-prime bias: {results['bias_small_pct']:.2f}%")
    print(f"  Full omega bias:  {results['bias_full_pct']:.2f}%")
    print(f"  Large-prime reduction: {results['reduction_pct']:.1f}%")
    print()

    print("DECOMPOSITION CHECK:")
    check = results['diff_small'] + results['diff_big']
    print(f"  diff_small + diff_big = {results['diff_small']:.6f} + {results['diff_big']:.6f} = {check:.6f}")
    print(f"  diff_full = {results['diff_full']:.6f}")
    print(f"  Match: {'YES' if abs(check - results['diff_full']) < 1e-9 else 'NO'}")


def print_markdown_table(results: dict):
    """Print results as markdown for paper inclusion."""
    print()
    print("### Markdown Table for Paper")
    print()
    print(f"At $K = {results['K']:,}$ (where $\\sqrt{{N}} = {results['sqrt_N']:,}$):")
    print()
    print("| Component | PC | CC | Difference |")
    print("|-----------|-----|-----|------------|")
    print(f"| $\\omega_{{\\text{{small}}}}$ (factors $\\leq \\sqrt{{N}}$) | {results['pc_omega_small']:.3f} | {results['cc_omega_small']:.3f} | $+{results['diff_small']:.3f}$ |")
    print(f"| Has large prime factor $(> \\sqrt{{N}})$ | {results['pc_has_big']*100:.1f}% | {results['cc_has_big']*100:.1f}% | ${results['diff_big']:+.3f}$ |")
    print(f"| **Full $\\omega$** | {results['pc_omega_full']:.3f} | {results['cc_omega_full']:.3f} | $+{results['diff_full']:.3f}$ |")
    print()
    print(f"Small-prime bias: **{results['bias_small_pct']:.1f}%**, reduced to **{results['bias_full_pct']:.2f}%** by large-prime effect ({results['reduction_pct']:.0f}% reduction).")


def save_results(results: dict, output_dir: Path):
    """Save results to CSV."""
    import csv
    from datetime import datetime

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = output_dir / f"omega_decomposition_K{results['K']}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save main results
    csv_path = run_dir / 'results.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for key, value in results.items():
            writer.writerow([key, value])

    # Save markdown
    md_path = run_dir / 'table.md'
    with open(md_path, 'w') as f:
        f.write(f"# Omega Decomposition at K={results['K']:,}\n\n")
        f.write(f"$\\sqrt{{N}} = {results['sqrt_N']:,}$\n\n")
        f.write("| Component | PC | CC | Difference |\n")
        f.write("|-----------|-----|-----|------------|\n")
        f.write(f"| $\\omega_{{\\text{{small}}}}$ | {results['pc_omega_small']:.4f} | {results['cc_omega_small']:.4f} | {results['diff_small']:+.4f} |\n")
        f.write(f"| Has large prime | {results['pc_has_big']*100:.2f}% | {results['cc_has_big']*100:.2f}% | {results['diff_big']:+.4f} |\n")
        f.write(f"| **Full $\\omega$** | {results['pc_omega_full']:.4f} | {results['cc_omega_full']:.4f} | {results['diff_full']:+.4f} |\n")
        f.write(f"\nSmall-prime bias: {results['bias_small_pct']:.2f}%\n")
        f.write(f"Full omega bias: {results['bias_full_pct']:.2f}%\n")
        f.write(f"Large-prime reduction: {results['reduction_pct']:.1f}%\n")

    print(f"\nResults saved to {run_dir}/")
    return run_dir


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Omega decomposition analysis')
    parser.add_argument('--K', type=float, default=1e7, help='Number of pairs (default: 1e7)')
    parser.add_argument('--save', action='store_true', help='Save results to data/results/')
    args = parser.parse_args()

    K = int(args.K)

    results = compute_omega_decomposition(K, verbose=True)
    print_results(results)
    print_markdown_table(results)

    if args.save:
        output_dir = Path('data/results')
        save_results(results, output_dir)
