#!/usr/bin/env python3
"""
Compute scaled residual ε_p for many primes and plot ε_p vs p.

For twin primes (6k-1, 6k+1), the local density model predicts:
    P(p | b | a prime) = 1/(p-1)  for all p >= 5

The scaled residual is:
    ε_p = (p-1) × P̂(p|b|a prime) - 1

If the model is correct, ε_p → 0 as K → ∞.

This script uses GPU acceleration via Numba CUDA when available,
falling back to CPU for systems without GPU support.
"""

import numpy as np
import time
import json
import csv
import argparse
from pathlib import Path
from typing import List, Dict

# Import project infrastructure
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.primes import prime_flags_upto

# GPU support
try:
    from numba import cuda
    import numba
    HAS_GPU = cuda.is_available()
except ImportError:
    HAS_GPU = False
    print("Numba CUDA not available, using CPU-only mode")


def get_primes_up_to(n: int) -> np.ndarray:
    """Sieve of Eratosthenes to get all primes up to n."""
    sieve = np.ones(n + 1, dtype=bool)
    sieve[0:2] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    return np.where(sieve)[0]


# =============================================================================
# GPU Implementation (Numba CUDA)
# =============================================================================

if HAS_GPU:
    # Kernel for processing a batch of primes (up to 256)
    MAX_PRIMES_PER_BATCH = 256

    @cuda.jit
    def _epsilon_kernel(b_vals, a_is_prime, primes, n_primes, n,
                        block_counts_a_prime, block_primality_counts):
        """
        CUDA kernel: count p|b occurrences when a is prime.
        Uses shared memory block reduction, then writes partial sums.
        """
        # Shared memory for this block's partial sums
        shared_counts = cuda.shared.array(256, dtype=numba.int64)  # up to 256 primes
        shared_n_prime = cuda.shared.array(1, dtype=numba.int64)

        tid = cuda.threadIdx.x
        bid = cuda.blockIdx.x
        idx = cuda.grid(1)

        # Initialize shared memory
        if tid < n_primes:
            shared_counts[tid] = 0
        if tid == 0:
            shared_n_prime[0] = 0
        cuda.syncthreads()

        # Each thread processes its element
        if idx < n:
            b = b_vals[idx]
            is_a_prime = a_is_prime[idx]

            if is_a_prime:
                cuda.atomic.add(shared_n_prime, 0, 1)
                # Check divisibility by each prime
                for p_idx in range(n_primes):
                    p = primes[p_idx]
                    if b % p == 0:
                        cuda.atomic.add(shared_counts, p_idx, 1)

        cuda.syncthreads()

        # Write block results
        if tid < n_primes:
            block_counts_a_prime[bid, tid] = shared_counts[tid]
        if tid == 0:
            block_primality_counts[bid] = shared_n_prime[0]

    @cuda.jit
    def _reduce_counts_kernel(block_counts, n_blocks, n_primes, final_counts):
        """Reduce block partial sums to final totals."""
        p_idx = cuda.grid(1)
        if p_idx >= n_primes:
            return
        total = 0
        for i in range(n_blocks):
            total += block_counts[i, p_idx]
        final_counts[p_idx] = total

    @cuda.jit
    def _reduce_primality_kernel(block_primality, n_blocks, final_primality):
        """Reduce primality counts."""
        total = 0
        for i in range(n_blocks):
            total += block_primality[i]
        final_primality[0] = total


def compute_epsilon_gpu(K: int, primes_to_test: np.ndarray) -> tuple:
    """
    Compute ε_p for each prime using GPU.

    Processes primes in batches to fit shared memory constraints.
    Returns (n_a_prime, dict of counts per prime).
    """
    print(f"GPU mode: Computing for K={K:,}, {len(primes_to_test)} primes")

    N = 6 * K + 1
    print(f"  Building prime sieve up to {N:,}...")
    t0 = time.time()
    prime_flags = prime_flags_upto(N)
    print(f"    Sieve completed in {time.time() - t0:.1f}s")

    # Build arrays
    print(f"  Building pair arrays...")
    t0 = time.time()
    k_vals = np.arange(1, K + 1, dtype=np.int64)
    b_vals = 6 * k_vals + 1
    a_vals = 6 * k_vals - 1
    a_is_prime = prime_flags[a_vals].astype(np.uint8)
    print(f"    Arrays built in {time.time() - t0:.1f}s")

    # GPU setup
    threads_per_block = 256
    n_blocks = (K + threads_per_block - 1) // threads_per_block

    # Transfer input arrays once
    print(f"  Transferring to GPU...")
    t0 = time.time()
    d_b_vals = cuda.to_device(b_vals)
    d_a_is_prime = cuda.to_device(a_is_prime)
    cuda.synchronize()
    print(f"    Transfer completed in {time.time() - t0:.1f}s")

    # Process primes in batches
    all_counts = {}
    n_a_prime = None

    n_batches = (len(primes_to_test) + MAX_PRIMES_PER_BATCH - 1) // MAX_PRIMES_PER_BATCH
    print(f"  Processing {n_batches} batch(es) of primes...")

    for batch_idx in range(n_batches):
        start = batch_idx * MAX_PRIMES_PER_BATCH
        end = min(start + MAX_PRIMES_PER_BATCH, len(primes_to_test))
        batch_primes = primes_to_test[start:end]
        n_primes = len(batch_primes)

        print(f"    Batch {batch_idx+1}/{n_batches}: primes {batch_primes[0]} to {batch_primes[-1]}")

        # Allocate arrays for this batch
        d_primes = cuda.to_device(batch_primes.astype(np.int64))
        d_block_counts = cuda.device_array((n_blocks, n_primes), dtype=np.int64)
        d_block_primality = cuda.device_array(n_blocks, dtype=np.int64)
        d_final_counts = cuda.device_array(n_primes, dtype=np.int64)
        d_final_primality = cuda.device_array(1, dtype=np.int64)

        # Run kernel
        t0 = time.time()
        _epsilon_kernel[n_blocks, threads_per_block](
            d_b_vals, d_a_is_prime, d_primes, n_primes, K,
            d_block_counts, d_block_primality
        )

        # Reduce
        reduce_blocks = (n_primes + 31) // 32
        _reduce_counts_kernel[reduce_blocks, 32](
            d_block_counts, n_blocks, n_primes, d_final_counts
        )
        _reduce_primality_kernel[1, 1](d_block_primality, n_blocks, d_final_primality)

        cuda.synchronize()

        # Get results
        counts = d_final_counts.copy_to_host()
        if n_a_prime is None:
            n_a_prime = int(d_final_primality.copy_to_host()[0])

        for i, p in enumerate(batch_primes):
            all_counts[int(p)] = int(counts[i])

        print(f"      Completed in {time.time() - t0:.1f}s")

    print(f"  Total primes with a prime: {n_a_prime:,}")
    return n_a_prime, all_counts


# =============================================================================
# CPU Implementation (fallback)
# =============================================================================

def compute_epsilon_cpu(K: int, primes_to_test: np.ndarray) -> tuple:
    """
    Compute ε_p for each prime using CPU.

    Returns (n_a_prime, dict of counts per prime).
    """
    print(f"CPU mode: Computing for K={K:,}, {len(primes_to_test)} primes")

    N = 6 * K + 1
    print(f"  Building prime sieve up to {N:,}...")
    t0 = time.time()
    prime_flags = prime_flags_upto(N)
    print(f"    Sieve completed in {time.time() - t0:.1f}s")

    # Build arrays
    print(f"  Building pair arrays...")
    t0 = time.time()
    k_vals = np.arange(1, K + 1, dtype=np.int64)
    b_vals = 6 * k_vals + 1
    a_vals = 6 * k_vals - 1
    a_is_prime = prime_flags[a_vals]
    print(f"    Arrays built in {time.time() - t0:.1f}s")

    # Count
    n_a_prime = int(a_is_prime.sum())
    b_prime = b_vals[a_is_prime]

    print(f"  Counting divisibility for {len(primes_to_test)} primes...")
    t0 = time.time()
    counts = {}
    for i, p in enumerate(primes_to_test):
        if i % 50 == 0:
            print(f"    Processing prime {i+1}/{len(primes_to_test)} (p={p})...")
        counts[int(p)] = int((b_prime % p == 0).sum())
    print(f"    Counting completed in {time.time() - t0:.1f}s")

    print(f"  Total primes with a prime: {n_a_prime:,}")
    return n_a_prime, counts


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compute ε_p for many primes")
    parser.add_argument("--K", type=float, default=1e8, help="Number of pairs (default: 10^8)")
    parser.add_argument("--max_prime", type=int, default=1000, help="Max prime to test (default: 1000)")
    parser.add_argument("--output", type=str, default="data/reference/epsilon_vs_p", help="Output directory")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    args = parser.parse_args()

    K = int(args.K)
    max_prime = args.max_prime
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Computing ε_p for primes up to {max_prime} with K = {K:,}")
    print(f"Output: {output_dir}")
    print(f"GPU available: {HAS_GPU and not args.cpu}")

    # Get primes to test (skip 2 and 3 which are handled by the 6k±1 wheel)
    all_primes = get_primes_up_to(max_prime)
    primes_to_test = all_primes[all_primes >= 5]
    print(f"Testing {len(primes_to_test)} primes: {primes_to_test[:5]}...{primes_to_test[-3:]}")

    # Compute
    t0 = time.time()

    if HAS_GPU and not args.cpu:
        n_a_prime, counts = compute_epsilon_gpu(K, primes_to_test)
    else:
        n_a_prime, counts = compute_epsilon_cpu(K, primes_to_test)

    elapsed = time.time() - t0
    print(f"\nTotal computation took {elapsed:.1f}s")

    # Compute ε_p for each prime
    results = []
    for p in primes_to_test:
        p = int(p)
        count = counts[p]
        p_hat = count / n_a_prime
        predicted = 1.0 / (p - 1)
        epsilon_p = (p - 1) * p_hat - 1
        se = np.sqrt(p_hat * (1 - p_hat) / n_a_prime)
        z = (p_hat - predicted) / se if se > 0 else 0

        results.append({
            'p': p,
            'count': count,
            'p_hat': p_hat,
            'predicted': predicted,
            'epsilon_p': epsilon_p,
            'se': se,
            'z_score': z
        })

    # Save CSV
    csv_path = output_dir / "epsilon_vs_p.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['p', 'count', 'p_hat', 'predicted', 'epsilon_p', 'se', 'z_score'])
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {csv_path}")

    # Save metadata
    meta = {
        'description': 'Scaled residual ε_p vs prime p for twin prime pattern',
        'K': K,
        'n_a_prime': n_a_prime,
        'n_primes_tested': len(primes_to_test),
        'max_prime': int(max_prime),
        'epsilon_definition': 'ε_p = (p-1) × P̂(p|b | a prime) - 1',
        'expected_value': 'ε_p → 0 if local density model is correct',
        'computation_time_seconds': elapsed,
        'gpu_used': HAS_GPU and not args.cpu
    }
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(meta, f, indent=2)

    # Statistics
    epsilons = np.array([r['epsilon_p'] for r in results])
    print(f"\nε_p statistics:")
    print(f"  Mean: {np.mean(epsilons):.2e}")
    print(f"  Std:  {np.std(epsilons):.2e}")
    print(f"  Max |ε_p|: {np.max(np.abs(epsilons)):.2e}")
    print(f"  Range: [{np.min(epsilons):.2e}, {np.max(epsilons):.2e}]")

    # Plot if requested
    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Left: ε_p vs p
            ax1 = axes[0]
            primes_arr = np.array([r['p'] for r in results])
            ax1.scatter(primes_arr, epsilons, s=8, alpha=0.6, c='steelblue')
            ax1.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='Expected (ε=0)')
            ax1.set_xlabel('Prime p', fontsize=12)
            ax1.set_ylabel('Scaled residual ε_p', fontsize=12)
            ax1.set_title(f'Local Density Verification: ε_p vs p\n(K = {K:.0e}, {len(primes_to_test)} primes)', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Right: histogram of ε_p
            ax2 = axes[1]
            ax2.hist(epsilons, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
            ax2.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Expected (ε=0)')
            ax2.set_xlabel('Scaled residual ε_p', fontsize=12)
            ax2.set_ylabel('Count', fontsize=12)
            ax2.set_title(f'Distribution of ε_p\nMean = {np.mean(epsilons):.2e}, Std = {np.std(epsilons):.2e}', fontsize=12)
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            fig_path = output_dir / "epsilon_vs_p.png"
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            print(f"Saved {fig_path}")
            plt.close()

        except ImportError:
            print("matplotlib not available, skipping plot")

    print("\nDone!")


if __name__ == "__main__":
    main()
