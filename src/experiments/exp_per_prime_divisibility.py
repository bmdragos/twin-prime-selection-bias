"""
Experiment: Per-Prime Divisibility Statistics (Table 3.1 verification)

Computes empirical P(p|b | a prime) and P(p|b | a composite) for small primes.
This directly verifies the local mechanism: the heuristic predicts these should be
approximately 1/(p-1) and 1/p respectively.

Supports both CPU and GPU (CUDA) execution paths.
GPU path uses optimizations from OPTIMIZATIONS.md:
- GPU-side aggregation with two-phase reduction
- Unified memory context with array reuse
- Efficient data types (uint8 for flags, int32 for counts)
"""

import numpy as np
from pathlib import Path
from typing import Dict, List
import time

from ..primes import prime_flags_upto
from ..sieve_pairs import pair_values

# GPU support
try:
    from numba import cuda
    import numba
    HAS_GPU = cuda.is_available()
except ImportError:
    HAS_GPU = False


def has_unified_memory() -> bool:
    """Detect if GPU has unified memory (Grace Hopper/Blackwell)."""
    if not HAS_GPU:
        return False
    try:
        device = cuda.get_current_device()
        cc = device.compute_capability
        return cc[0] >= 9  # sm_90+ (Hopper) or sm_120+ (Blackwell)
    except Exception:
        return False


if HAS_GPU:
    @cuda.jit
    def _per_prime_divisibility_kernel(b_vals, a_is_prime, primes, n_primes, n,
                                       block_counts_a_prime, block_counts_a_composite,
                                       block_primality_counts):
        """
        CUDA kernel: count p|b occurrences with two-phase block reduction.

        Uses shared memory to reduce atomic contention, then writes block-level
        partial sums. A second kernel reduces block sums to final totals.

        block_counts_a_prime[block_id, p_idx] = count for this block
        block_counts_a_composite[block_id, p_idx] = count for this block
        block_primality_counts[block_id, 0/1] = a_prime/a_composite count for this block
        """
        # Shared memory for this block's partial sums
        # Shape: [n_primes] for a_prime counts, [n_primes] for a_composite counts, [2] for primality
        shared_a_prime = cuda.shared.array(16, dtype=numba.int64)  # max 16 primes
        shared_a_comp = cuda.shared.array(16, dtype=numba.int64)
        shared_primality = cuda.shared.array(2, dtype=numba.int64)

        tid = cuda.threadIdx.x
        bid = cuda.blockIdx.x
        idx = cuda.grid(1)

        # Initialize shared memory (first n_primes+2 threads)
        if tid < n_primes:
            shared_a_prime[tid] = 0
            shared_a_comp[tid] = 0
        if tid < 2:
            shared_primality[tid] = 0
        cuda.syncthreads()

        # Each thread processes its element
        if idx < n:
            b = b_vals[idx]
            is_a_prime = a_is_prime[idx]

            # Count primality
            if is_a_prime:
                cuda.atomic.add(shared_primality, 0, 1)
            else:
                cuda.atomic.add(shared_primality, 1, 1)

            # Check divisibility by each small prime
            for p_idx in range(n_primes):
                p = primes[p_idx]
                if b % p == 0:
                    if is_a_prime:
                        cuda.atomic.add(shared_a_prime, p_idx, 1)
                    else:
                        cuda.atomic.add(shared_a_comp, p_idx, 1)

        cuda.syncthreads()

        # First n_primes threads write block results
        if tid < n_primes:
            block_counts_a_prime[bid, tid] = shared_a_prime[tid]
            block_counts_a_composite[bid, tid] = shared_a_comp[tid]
        if tid < 2:
            block_primality_counts[bid, tid] = shared_primality[tid]

    @cuda.jit
    def _reduce_block_counts_kernel(block_counts, n_blocks, n_primes, final_counts):
        """
        Final reduction: sum all block partial sums.
        One thread per prime.
        """
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
        idx = cuda.grid(1)
        if idx >= 2:
            return

        total = 0
        for i in range(n_blocks):
            total += block_primality[i, idx]
        final_primality[idx] = total


def compute_per_prime_stats_gpu(K: int, primes: List[int] = [5, 7, 11, 13]) -> Dict:
    """
    Compute per-prime divisibility statistics using GPU with optimizations.

    Uses:
    - Two-phase block reduction (from OPTIMIZATIONS.md)
    - Efficient data types
    - Single kernel launch for all primes
    """
    print(f"Computing per-prime divisibility stats (GPU optimized) for K={K:,}")

    N = 6 * K + 1
    primes_arr = np.array(primes, dtype=np.int64)
    n_primes = len(primes)

    print(f"  Generating prime flags up to {N:,}...")
    t0 = time.time()
    prime_flags = prime_flags_upto(N)
    print(f"    Sieve completed in {time.time() - t0:.1f}s")

    # Build arrays with efficient types
    print(f"  Building pair arrays...")
    t0 = time.time()
    # Use vectorized numpy operations for speed
    k_vals = np.arange(1, K + 1, dtype=np.int64)
    b_vals = 6 * k_vals + 1
    a_vals = 6 * k_vals - 1
    # Use uint8 for boolean flags (saves memory)
    a_is_prime = prime_flags[a_vals].astype(np.uint8)
    print(f"    Arrays built in {time.time() - t0:.1f}s")

    # GPU setup
    print(f"  Transferring to GPU (unified memory context)...")
    t0 = time.time()

    threads_per_block = 256
    n_blocks = (K + threads_per_block - 1) // threads_per_block

    # Transfer input arrays once
    d_b_vals = cuda.to_device(b_vals)
    d_a_is_prime = cuda.to_device(a_is_prime)
    d_primes = cuda.to_device(primes_arr)

    # Allocate block-level partial sum arrays
    d_block_counts_a_prime = cuda.device_array((n_blocks, n_primes), dtype=np.int64)
    d_block_counts_a_comp = cuda.device_array((n_blocks, n_primes), dtype=np.int64)
    d_block_primality = cuda.device_array((n_blocks, 2), dtype=np.int64)

    # Allocate final result arrays
    d_final_a_prime = cuda.device_array(n_primes, dtype=np.int64)
    d_final_a_comp = cuda.device_array(n_primes, dtype=np.int64)
    d_final_primality = cuda.device_array(2, dtype=np.int64)

    cuda.synchronize()
    print(f"    Transfer completed in {time.time() - t0:.1f}s")

    # Launch main kernel
    print(f"  Running GPU kernels ({n_blocks:,} blocks × {threads_per_block} threads)...")
    t0 = time.time()

    _per_prime_divisibility_kernel[n_blocks, threads_per_block](
        d_b_vals, d_a_is_prime, d_primes, n_primes, K,
        d_block_counts_a_prime, d_block_counts_a_comp, d_block_primality
    )

    # Reduce block sums
    reduce_blocks = (n_primes + 31) // 32
    _reduce_block_counts_kernel[reduce_blocks, 32](
        d_block_counts_a_prime, n_blocks, n_primes, d_final_a_prime
    )
    _reduce_block_counts_kernel[reduce_blocks, 32](
        d_block_counts_a_comp, n_blocks, n_primes, d_final_a_comp
    )
    _reduce_primality_kernel[1, 2](
        d_block_primality, n_blocks, d_final_primality
    )

    cuda.synchronize()
    print(f"    Kernels completed in {time.time() - t0:.1f}s")

    # Transfer only final results (not arrays!)
    counts_a_prime = d_final_a_prime.copy_to_host()
    counts_a_comp = d_final_a_comp.copy_to_host()
    primality = d_final_primality.copy_to_host()

    n_a_prime = int(primality[0])
    n_a_composite = int(primality[1])

    print(f"  Results: {n_a_prime:,} pairs with a prime, {n_a_composite:,} with a composite")

    # Compute probabilities
    results = {
        'K': K,
        'n_a_prime': n_a_prime,
        'n_a_composite': n_a_composite,
        'primes': {}
    }

    for i, p in enumerate(primes):
        p_b_given_a_prime = counts_a_prime[i] / n_a_prime if n_a_prime > 0 else 0
        p_b_given_a_comp = counts_a_comp[i] / n_a_composite if n_a_composite > 0 else 0

        results['primes'][p] = {
            'count_a_prime_and_p_div_b': int(counts_a_prime[i]),
            'count_a_comp_and_p_div_b': int(counts_a_comp[i]),
            'empirical_given_a_prime': p_b_given_a_prime,
            'empirical_given_a_composite': p_b_given_a_comp,
            'predicted_given_a_prime': 1 / (p - 1),
            'predicted_given_a_composite': 1 / p,
            'increment': p_b_given_a_prime - p_b_given_a_comp,
            'predicted_increment': 1 / (p * (p - 1))
        }

    return results


def compute_per_prime_stats_cpu(K: int, primes: List[int] = [5, 7, 11, 13]) -> Dict:
    """
    Compute per-prime divisibility statistics using CPU.
    """
    print(f"Computing per-prime divisibility stats (CPU) for K={K:,}")

    N = 6 * K + 1
    print(f"  Generating prime flags up to {N:,}...")
    t0 = time.time()
    prime_flags = prime_flags_upto(N)
    print(f"    Sieve completed in {time.time() - t0:.1f}s")

    # Vectorized approach for CPU
    print(f"  Building arrays and computing stats...")
    t0 = time.time()

    k_vals = np.arange(1, K + 1, dtype=np.int64)
    a_vals = 6 * k_vals - 1
    b_vals = 6 * k_vals + 1
    a_is_prime = prime_flags[a_vals]

    n_a_prime = np.sum(a_is_prime)
    n_a_composite = K - n_a_prime

    results = {
        'K': K,
        'n_a_prime': int(n_a_prime),
        'n_a_composite': int(n_a_composite),
        'primes': {}
    }

    for p in primes:
        p_divides_b = (b_vals % p == 0)

        count_a_prime = np.sum(a_is_prime & p_divides_b)
        count_a_comp = np.sum(~a_is_prime & p_divides_b)

        p_b_given_a_prime = count_a_prime / n_a_prime if n_a_prime > 0 else 0
        p_b_given_a_comp = count_a_comp / n_a_composite if n_a_composite > 0 else 0

        results['primes'][p] = {
            'count_a_prime_and_p_div_b': int(count_a_prime),
            'count_a_comp_and_p_div_b': int(count_a_comp),
            'empirical_given_a_prime': float(p_b_given_a_prime),
            'empirical_given_a_composite': float(p_b_given_a_comp),
            'predicted_given_a_prime': 1 / (p - 1),
            'predicted_given_a_composite': 1 / p,
            'increment': float(p_b_given_a_prime - p_b_given_a_comp),
            'predicted_increment': 1 / (p * (p - 1))
        }

    print(f"    Completed in {time.time() - t0:.1f}s")
    return results


def compute_per_prime_stats(K: int, primes: List[int] = [5, 7, 11, 13],
                            force_cpu: bool = False) -> Dict:
    """
    Compute per-prime divisibility statistics (auto-selects GPU or CPU).
    """
    if HAS_GPU and not force_cpu:
        print(f"Using GPU acceleration (optimized)")
        return compute_per_prime_stats_gpu(K, primes)
    else:
        if force_cpu:
            print("Forced CPU execution")
        else:
            print("GPU not available, using CPU")
        return compute_per_prime_stats_cpu(K, primes)


def print_table(results: Dict):
    """Print results as a formatted table."""
    print("\n" + "="*100)
    print("Table 3.1: Per-Prime Verification")
    print("="*100)
    print(f"K = {results['K']:,}")
    print(f"Pairs with a prime: {results['n_a_prime']:,}")
    print(f"Pairs with a composite: {results['n_a_composite']:,}")
    print()

    header = "| p  | P(p|b|a prime) emp | pred 1/(p-1) | P(p|b|a comp) emp | pred ~1/p | Increment emp | pred 1/[p(p-1)] |"
    sep =    "|----|--------------------|--------------|-------------------|-----------|---------------|-----------------|"
    print(header)
    print(sep)

    for p, data in results['primes'].items():
        emp_ap = data['empirical_given_a_prime']
        pred_ap = data['predicted_given_a_prime']
        emp_ac = data['empirical_given_a_composite']
        pred_ac = data['predicted_given_a_composite']
        incr = data['increment']
        pred_incr = data['predicted_increment']
        print(f"| {p:2d} | {emp_ap:.6f}           | {pred_ap:.6f}     | {emp_ac:.6f}          | {pred_ac:.6f}  | {incr:.6f}      | {pred_incr:.6f}        |")


def print_markdown_table(results: Dict):
    """Print results as markdown table for paper inclusion."""
    print("\n### Empirical Per-Prime Verification")
    print()
    print(f"At $K = {results['K']:,}$:")
    print()
    print("| $p$ | $\\mathbb{P}(p \\mid b \\mid a \\text{ prime})$ | Predicted $1/(p-1)$ | $\\mathbb{P}(p \\mid b \\mid a \\text{ comp})$ | Increment | Predicted $1/[p(p-1)]$ |")
    print("|-----|------------------------------------------------|---------------------|-----------------------------------------------|-----------|------------------------|")

    for p, data in results['primes'].items():
        emp_prime = data['empirical_given_a_prime']
        pred_prime = data['predicted_given_a_prime']
        emp_comp = data['empirical_given_a_composite']
        increment = data['increment']
        pred_incr = data['predicted_increment']
        print(f"| {p} | {emp_prime:.4f} | {pred_prime:.4f} | {emp_comp:.4f} | {increment:.4f} | {pred_incr:.4f} |")


def save_results(results: Dict, output_dir: Path, use_timestamp: bool = True):
    """Save results to CSV file with optional timestamped folder."""
    import csv
    from datetime import datetime

    if use_timestamp:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = output_dir / f"per_prime_K{results['K']}_{timestamp}"
    else:
        run_dir = output_dir

    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / 'per_prime_divisibility.csv'

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['K', 'n_a_prime', 'n_a_composite'])
        writer.writerow([results['K'], results['n_a_prime'], results['n_a_composite']])
        writer.writerow([])
        writer.writerow(['p', 'count_a_prime_p_div_b', 'count_a_comp_p_div_b',
                        'emp_given_a_prime', 'emp_given_a_comp',
                        'pred_given_a_prime', 'pred_given_a_comp',
                        'increment', 'pred_increment'])
        for p, data in results['primes'].items():
            writer.writerow([
                p,
                data['count_a_prime_and_p_div_b'],
                data['count_a_comp_and_p_div_b'],
                data['empirical_given_a_prime'],
                data['empirical_given_a_composite'],
                data['predicted_given_a_prime'],
                data['predicted_given_a_composite'],
                data['increment'],
                data['predicted_increment']
            ])

    # Also save a markdown version for easy paper inclusion
    md_path = run_dir / 'table_3_1.md'
    with open(md_path, 'w') as f:
        f.write(f"# Table 3.1: Per-Prime Verification\n\n")
        f.write(f"$K = {results['K']:,}$\n\n")
        f.write("| $p$ | $\\mathbb{P}(p \\mid b \\mid a \\text{ prime})$ | Predicted $1/(p-1)$ | $\\mathbb{P}(p \\mid b \\mid a \\text{ comp})$ | Increment | Predicted $1/[p(p-1)]$ |\n")
        f.write("|-----|------------------------------------------------|---------------------|-----------------------------------------------|-----------|------------------------|\n")
        for p, data in results['primes'].items():
            f.write(f"| {p} | {data['empirical_given_a_prime']:.4f} | {data['predicted_given_a_prime']:.4f} | {data['empirical_given_a_composite']:.4f} | {data['increment']:.4f} | {data['predicted_increment']:.4f} |\n")

    print(f"\nResults saved to {run_dir}/")
    print(f"  - per_prime_divisibility.csv")
    print(f"  - table_3_1.md")


if __name__ == '__main__':
    import sys

    # Parse arguments
    K = int(float(sys.argv[1])) if len(sys.argv) > 1 else 10_000_000
    force_cpu = '--cpu' in sys.argv
    no_timestamp = '--no-timestamp' in sys.argv

    print(f"GPU available: {HAS_GPU}")
    if HAS_GPU:
        device = cuda.get_current_device()
        print(f"GPU: {device.name}, compute capability {device.compute_capability}")
        print(f"Unified memory: {has_unified_memory()}")

    results = compute_per_prime_stats(K, force_cpu=force_cpu)
    print_table(results)
    print()
    print_markdown_table(results)

    # Save to timestamped folder
    output_dir = Path('data/results')
    output_dir.mkdir(parents=True, exist_ok=True)
    save_results(results, output_dir, use_timestamp=not no_timestamp)
