#!/usr/bin/env python3
"""
Experiment: Sophie Germain Pairs (n, 2n+1) - GPU Accelerated

GPU-optimized version for DGX Spark.

Predicted shift: sum_{q >= 3} 1/[q(q-1)] ≈ 0.273

Usage:
    python -m src.experiments.exp_sophie_germain_gpu --N 1e8 --save
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


def compute_sophie_germain_gpu(N: int, verbose: bool = True) -> dict:
    """
    Compute omega bias for Sophie Germain pairs using GPU.
    """
    try:
        from numba import cuda
        if not cuda.is_available():
            raise ImportError("No GPU")
    except ImportError:
        if verbose:
            print("GPU not available, using CPU...")
        from src.experiments.exp_sophie_germain import compute_sophie_germain_bias
        return compute_sophie_germain_bias(N, verbose)

    max_val = 2 * N + 1

    if verbose:
        print(f"Sophie Germain GPU Analysis")
        print(f"N = {N:,}, max value = {max_val:,}")
        print("=" * 60)

    # Build SPF sieve (parallel, uses 0=prime sentinel)
    t0 = time.time()
    if verbose:
        print("Building parallel SPF sieve...")
    spf = spf_sieve_parallel(max_val)
    if verbose:
        print(f"  Done in {time.time()-t0:.1f}s, {spf.nbytes/1e9:.2f}GB")

    # Create arrays of n values and 2n+1 values
    t0 = time.time()
    if verbose:
        print("Creating value arrays...")
    n_vals = np.arange(2, N + 1, dtype=np.int64)
    target_vals = 2 * n_vals + 1  # 2n+1

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

    predicted = sum(1 / (p * (p - 1)) for p in _primes_up_to(1000) if p >= 3)

    results = {
        'N': N,
        'n_prime_count': len(omega_prime),
        'n_composite_count': len(omega_composite),
        'mean_omega_when_n_prime': float(mean_prime),
        'mean_omega_when_n_composite': float(mean_composite),
        'difference': float(diff),
        'predicted': predicted,
        'relative_error': abs(diff - predicted) / predicted,
    }

    if verbose:
        print(f"\nResults for Sophie Germain pairs (n, 2n+1), n <= {N:,}:")
        print(f"  E[ω(2n+1) | n prime]:     {mean_prime:.4f}")
        print(f"  E[ω(2n+1) | n composite]: {mean_composite:.4f}")
        print(f"  Difference:               {diff:.4f}")
        print(f"  Predicted:                {predicted:.4f}")
        print(f"  Relative error:           {results['relative_error']:.2%}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=float, default=1e8)
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    N = int(args.N)
    results = compute_sophie_germain_gpu(N)

    if args.save:
        import pandas as pd
        outdir = Path('data/reference/sophie_germain')
        outdir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame([results])
        csv_path = outdir / f'sophie_germain_N{N:.0e}.csv'.replace('+', '')
        df.to_csv(csv_path, index=False)
        print(f"\nSaved to {csv_path}")

        md_path = outdir / 'table.md'
        with open(md_path, 'w') as f:
            f.write(f"# Sophie Germain Pairs (n, 2n+1)\n\n")
            f.write(f"$N = {N:,}$\n\n")
            f.write("| Population | Mean ω(2n+1) | Sample size |\n")
            f.write("|------------|--------------|-------------|\n")
            f.write(f"| n prime | {results['mean_omega_when_n_prime']:.4f} | {results['n_prime_count']:,} |\n")
            f.write(f"| n composite | {results['mean_omega_when_n_composite']:.4f} | {results['n_composite_count']:,} |\n")
            f.write(f"| **Difference** | **{results['difference']:.4f}** | — |\n")
            f.write(f"\nPredicted: Σ 1/[q(q-1)] for q ≥ 3 = {results['predicted']:.4f}\n")
            f.write(f"\nRelative error: {results['relative_error']:.2%}\n")
        print(f"Saved to {md_path}")


if __name__ == '__main__':
    main()
