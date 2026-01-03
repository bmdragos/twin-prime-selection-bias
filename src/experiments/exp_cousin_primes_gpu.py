#!/usr/bin/env python3
"""
Experiment: Cousin Primes (n, n+4) among 6k±1 - GPU Accelerated

GPU-optimized version for DGX Spark.

Predicted shift: sum_{p >= 5} 1/[p(p-1)] ≈ 0.1065 (same as twin primes)

Note on p=3: For n = 6k-1, n+4 = 6k+3 is always divisible by 3.
For n = 6k+1, n+4 = 6k+5 is never divisible by 3.
Since 3|n+4 is determined by residue class (not primality of n),
the p=3 term contributes zero to the DIFFERENCE. The sum starts at p=5.

Usage:
    python -m src.experiments.exp_cousin_primes_gpu --K 1e8 --save
"""

import numpy as np
import time
import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.parallel_sieve import spf_sieve_parallel


def _primes_up_to(n: int) -> list:
    """Simple sieve for computing predicted constant."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    return [i for i in range(n + 1) if sieve[i]]


def compute_cousin_primes_gpu(K: int, verbose: bool = True) -> dict:
    """
    Compute omega bias for cousin prime pairs using GPU.
    Analyzes (n, n+4) where n is of the form 6k±1.
    """
    try:
        from numba import cuda
        if not cuda.is_available():
            raise ImportError("No GPU")
    except ImportError:
        if verbose:
            print("GPU not available, using CPU...")
        from src.experiments.exp_cousin_primes import compute_cousin_prime_bias
        return compute_cousin_prime_bias(K, verbose)

    max_val = 6 * K + 5  # Maximum n+4 value

    if verbose:
        print(f"Cousin Primes GPU Analysis")
        print(f"K = {K:,}, max value = {max_val:,}")
        print("=" * 60)

    # Build SPF sieve (parallel, uses 0=prime sentinel)
    t0 = time.time()
    if verbose:
        print("Building parallel SPF sieve...")
    spf = spf_sieve_parallel(max_val)
    if verbose:
        print(f"  Done in {time.time()-t0:.1f}s, {spf.nbytes/1e9:.2f}GB")

    # Create arrays of n values (6k-1 and 6k+1)
    t0 = time.time()
    if verbose:
        print("Creating value arrays...")

    # Generate all 6k±1 values up to 6K+1
    k_vals = np.arange(1, K + 1, dtype=np.int64)
    n_minus = 6 * k_vals - 1  # 6k-1
    n_plus = 6 * k_vals + 1   # 6k+1
    n_vals = np.concatenate([n_minus, n_plus])
    target_vals = n_vals + 4  # n+4

    # Identify primes using SPF (0 = prime)
    is_n_prime = (spf[n_vals] == 0)
    if verbose:
        print(f"  {np.sum(is_n_prime):,} primes, {np.sum(~is_n_prime):,} composites")

    # GPU omega computation
    if verbose:
        print("Computing omega on GPU...")

    @cuda.jit
    def omega_kernel(numbers, spf, results):
        idx = cuda.grid(1)
        if idx >= numbers.shape[0]:
            return
        n = numbers[idx]
        if n <= 1:
            results[idx] = 0
            return
        count = 0
        prev = 0
        while n > 1:
            p = spf[n]
            if p == 0:
                p = n
            if p != prev:
                count += 1
            prev = p
            n //= p
        results[idx] = count

    # Transfer to GPU
    d_targets = cuda.to_device(target_vals)
    d_spf = cuda.to_device(spf)
    d_omega = cuda.device_array(len(target_vals), dtype=np.int32)

    threads = 256
    blocks = (len(target_vals) + threads - 1) // threads
    omega_kernel[blocks, threads](d_targets, d_spf, d_omega)
    cuda.synchronize()

    omega_vals = d_omega.copy_to_host()
    if verbose:
        print(f"  Done in {time.time()-t0:.1f}s")

    # Compute means
    omega_prime = omega_vals[is_n_prime]
    omega_composite = omega_vals[~is_n_prime]

    mean_prime = np.mean(omega_prime)
    mean_composite = np.mean(omega_composite)
    diff = mean_prime - mean_composite

    predicted = sum(1 / (p * (p - 1)) for p in _primes_up_to(1000) if p >= 5)

    results = {
        'K': K,
        'n_prime_count': len(omega_prime),
        'n_composite_count': len(omega_composite),
        'mean_omega_when_n_prime': float(mean_prime),
        'mean_omega_when_n_composite': float(mean_composite),
        'difference': float(diff),
        'predicted': predicted,
        'relative_error': abs(diff - predicted) / predicted,
    }

    if verbose:
        print(f"\nResults for cousin pairs (n, n+4) among 6k±1, K = {K:,}:")
        print(f"  E[ω(n+4) | n prime]:     {mean_prime:.4f}")
        print(f"  E[ω(n+4) | n composite]: {mean_composite:.4f}")
        print(f"  Difference:              {diff:.4f}")
        print(f"  Predicted:               {predicted:.4f}")
        print(f"  Relative error:          {results['relative_error']:.2%}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=float, default=1e8)
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    K = int(args.K)
    results = compute_cousin_primes_gpu(K)

    if args.save:
        import pandas as pd
        outdir = Path('data/reference/cousin_primes')
        outdir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame([results])
        csv_path = outdir / f'cousin_primes_K{K:.0e}.csv'.replace('+', '')
        df.to_csv(csv_path, index=False)
        print(f"\nSaved to {csv_path}")

        md_path = outdir / 'table.md'
        with open(md_path, 'w') as f:
            f.write(f"# Cousin Primes (n, n+4) among 6k±1 candidates\n\n")
            f.write(f"$K = {K:,}$\n\n")
            f.write("| Population | Mean ω(n+4) | Sample size |\n")
            f.write("|------------|-------------|-------------|\n")
            f.write(f"| n prime | {results['mean_omega_when_n_prime']:.4f} | {results['n_prime_count']:,} |\n")
            f.write(f"| n composite | {results['mean_omega_when_n_composite']:.4f} | {results['n_composite_count']:,} |\n")
            f.write(f"| **Difference** | **{results['difference']:.4f}** | — |\n")
            f.write(f"\nPredicted: Σ 1/[p(p-1)] for p ≥ 5 = {results['predicted']:.4f}\n")
            f.write(f"\nRelative error: {results['relative_error']:.2%}\n")
        print(f"Saved to {md_path}")


if __name__ == '__main__':
    main()
