"""
Experiment: Per-Prime Divisibility for Sophie Germain Pairs (n, 2n+1)

Verifies the local mechanism: P(p | 2n+1 | n prime) = 1/(p-1) for each prime p.

For Sophie Germain pairs:
- If p | n and p | (2n+1), then p | (2n+1 - 2n) = 1, impossible.
- Mutual exclusivity holds for ALL primes p >= 2.
- For p = 2: 2n+1 is always odd, so 2 never divides b. Skip p=2.
- Sum starts at p = 3: Σ_{p>=3} 1/[p(p-1)] ≈ 0.273

Population: odd n from 3 to N (even n are never prime except n=2).
"""

import numpy as np
from pathlib import Path
from typing import Dict, List
import time

from ..primes import prime_flags_upto

# GPU support
try:
    from numba import cuda
    import numba
    HAS_GPU = cuda.is_available()
except ImportError:
    HAS_GPU = False


if HAS_GPU:
    @cuda.jit
    def _sophie_germain_per_prime_kernel(n_vals, b_vals, n_is_prime, primes, n_primes, count,
                                          block_counts_n_prime, block_counts_n_composite,
                                          block_primality_counts):
        """
        CUDA kernel for Sophie Germain per-prime statistics.
        """
        shared_n_prime = cuda.shared.array(16, dtype=numba.int64)
        shared_n_comp = cuda.shared.array(16, dtype=numba.int64)
        shared_primality = cuda.shared.array(2, dtype=numba.int64)

        tid = cuda.threadIdx.x
        bid = cuda.blockIdx.x
        idx = cuda.grid(1)

        if tid < n_primes:
            shared_n_prime[tid] = 0
            shared_n_comp[tid] = 0
        if tid < 2:
            shared_primality[tid] = 0
        cuda.syncthreads()

        if idx < count:
            b = b_vals[idx]
            is_n_prime = n_is_prime[idx]

            if is_n_prime:
                cuda.atomic.add(shared_primality, 0, 1)
            else:
                cuda.atomic.add(shared_primality, 1, 1)

            for p_idx in range(n_primes):
                p = primes[p_idx]
                if b % p == 0:
                    if is_n_prime:
                        cuda.atomic.add(shared_n_prime, p_idx, 1)
                    else:
                        cuda.atomic.add(shared_n_comp, p_idx, 1)

        cuda.syncthreads()

        if tid < n_primes:
            block_counts_n_prime[bid, tid] = shared_n_prime[tid]
            block_counts_n_composite[bid, tid] = shared_n_comp[tid]
        if tid < 2:
            block_primality_counts[bid, tid] = shared_primality[tid]

    @cuda.jit
    def _reduce_block_counts_kernel(block_counts, n_blocks, n_primes, final_counts):
        p_idx = cuda.grid(1)
        if p_idx >= n_primes:
            return
        total = 0
        for i in range(n_blocks):
            total += block_counts[i, p_idx]
        final_counts[p_idx] = total

    @cuda.jit
    def _reduce_primality_kernel(block_primality, n_blocks, final_primality):
        idx = cuda.grid(1)
        if idx >= 2:
            return
        total = 0
        for i in range(n_blocks):
            total += block_primality[i, idx]
        final_primality[idx] = total


def compute_per_prime_stats_gpu(N: int, primes: List[int] = [3, 5, 7, 11, 13]) -> Dict:
    """
    Compute per-prime divisibility stats for Sophie Germain pairs using GPU.
    """
    print(f"Computing Sophie Germain per-prime stats (GPU) for N={N:,}")

    # Sophie Germain: (n, 2n+1) for odd n
    # b = 2n+1 can be up to 2N+1
    max_val = 2 * N + 1
    primes_arr = np.array(primes, dtype=np.int64)
    n_primes = len(primes)

    print(f"  Generating prime flags up to {max_val:,}...")
    t0 = time.time()
    prime_flags = prime_flags_upto(max_val)
    print(f"    Sieve completed in {time.time() - t0:.1f}s")

    # Build arrays for odd n only (even n are never prime except n=2)
    print(f"  Building pair arrays for odd n in [3, {N}]...")
    t0 = time.time()
    n_vals = np.arange(3, N + 1, 2, dtype=np.int64)  # odd n: 3, 5, 7, ...
    b_vals = 2 * n_vals + 1
    n_is_prime = prime_flags[n_vals].astype(np.uint8)
    count = len(n_vals)
    print(f"    {count:,} pairs built in {time.time() - t0:.1f}s")

    # GPU execution
    print(f"  Running GPU kernels...")
    t0 = time.time()

    threads_per_block = 256
    n_blocks = (count + threads_per_block - 1) // threads_per_block

    d_n_vals = cuda.to_device(n_vals)
    d_b_vals = cuda.to_device(b_vals)
    d_n_is_prime = cuda.to_device(n_is_prime)
    d_primes = cuda.to_device(primes_arr)

    d_block_counts_n_prime = cuda.device_array((n_blocks, n_primes), dtype=np.int64)
    d_block_counts_n_comp = cuda.device_array((n_blocks, n_primes), dtype=np.int64)
    d_block_primality = cuda.device_array((n_blocks, 2), dtype=np.int64)

    d_final_n_prime = cuda.device_array(n_primes, dtype=np.int64)
    d_final_n_comp = cuda.device_array(n_primes, dtype=np.int64)
    d_final_primality = cuda.device_array(2, dtype=np.int64)

    _sophie_germain_per_prime_kernel[n_blocks, threads_per_block](
        d_n_vals, d_b_vals, d_n_is_prime, d_primes, n_primes, count,
        d_block_counts_n_prime, d_block_counts_n_comp, d_block_primality
    )

    reduce_blocks = (n_primes + 31) // 32
    _reduce_block_counts_kernel[reduce_blocks, 32](
        d_block_counts_n_prime, n_blocks, n_primes, d_final_n_prime
    )
    _reduce_block_counts_kernel[reduce_blocks, 32](
        d_block_counts_n_comp, n_blocks, n_primes, d_final_n_comp
    )
    _reduce_primality_kernel[1, 2](
        d_block_primality, n_blocks, d_final_primality
    )

    cuda.synchronize()
    print(f"    Kernels completed in {time.time() - t0:.1f}s")

    counts_n_prime = d_final_n_prime.copy_to_host()
    counts_n_comp = d_final_n_comp.copy_to_host()
    primality = d_final_primality.copy_to_host()

    n_prime = int(primality[0])
    n_composite = int(primality[1])

    print(f"  Results: {n_prime:,} odd n prime, {n_composite:,} odd n composite")

    return _build_results(N, n_prime, n_composite, primes, counts_n_prime, counts_n_comp)


def compute_per_prime_stats_cpu(N: int, primes: List[int] = [3, 5, 7, 11, 13]) -> Dict:
    """
    Compute per-prime divisibility stats for Sophie Germain pairs using CPU.
    """
    print(f"Computing Sophie Germain per-prime stats (CPU) for N={N:,}")

    max_val = 2 * N + 1
    print(f"  Generating prime flags up to {max_val:,}...")
    t0 = time.time()
    prime_flags = prime_flags_upto(max_val)
    print(f"    Sieve completed in {time.time() - t0:.1f}s")

    print(f"  Computing stats...")
    t0 = time.time()

    n_vals = np.arange(3, N + 1, 2, dtype=np.int64)
    b_vals = 2 * n_vals + 1
    n_is_prime = prime_flags[n_vals]

    n_prime = np.sum(n_is_prime)
    n_composite = len(n_vals) - n_prime

    counts_n_prime = []
    counts_n_comp = []
    for p in primes:
        p_divides_b = (b_vals % p == 0)
        counts_n_prime.append(np.sum(n_is_prime & p_divides_b))
        counts_n_comp.append(np.sum(~n_is_prime & p_divides_b))

    print(f"    Completed in {time.time() - t0:.1f}s")
    print(f"  Results: {n_prime:,} odd n prime, {n_composite:,} odd n composite")

    return _build_results(N, int(n_prime), int(n_composite), primes,
                          np.array(counts_n_prime), np.array(counts_n_comp))


def _build_results(N, n_prime, n_composite, primes, counts_n_prime, counts_n_comp) -> Dict:
    """Build results dictionary from counts."""
    results = {
        'N': N,
        'pattern': 'Sophie Germain (n, 2n+1)',
        'population': 'odd n in [3, N]',
        'n_prime': n_prime,
        'n_composite': n_composite,
        'predicted_sum': sum(1/(p*(p-1)) for p in primes if p >= 3),
        'primes': {}
    }

    for i, p in enumerate(primes):
        p_b_given_n_prime = counts_n_prime[i] / n_prime if n_prime > 0 else 0
        p_b_given_n_comp = counts_n_comp[i] / n_composite if n_composite > 0 else 0

        results['primes'][p] = {
            'count_n_prime_and_p_div_b': int(counts_n_prime[i]),
            'count_n_comp_and_p_div_b': int(counts_n_comp[i]),
            'empirical_given_n_prime': float(p_b_given_n_prime),
            'empirical_given_n_composite': float(p_b_given_n_comp),
            'predicted_given_n_prime': 1 / (p - 1),
            'predicted_given_n_composite': 1 / p,
            'increment': float(p_b_given_n_prime - p_b_given_n_comp),
            'predicted_increment': 1 / (p * (p - 1))
        }

    return results


def compute_per_prime_stats(N: int, primes: List[int] = [3, 5, 7, 11, 13],
                            force_cpu: bool = False) -> Dict:
    """Compute per-prime stats (auto-selects GPU or CPU)."""
    if HAS_GPU and not force_cpu:
        return compute_per_prime_stats_gpu(N, primes)
    else:
        return compute_per_prime_stats_cpu(N, primes)


def print_table(results: Dict):
    """Print formatted table."""
    print("\n" + "="*100)
    print(f"Per-Prime Verification: {results['pattern']}")
    print("="*100)
    print(f"N = {results['N']:,}, Population: {results['population']}")
    print(f"n prime: {results['n_prime']:,}, n composite: {results['n_composite']:,}")
    print(f"Predicted Σ 1/[p(p-1)] for p >= 3: {results['predicted_sum']:.4f}")
    print()

    header = "| p  | P(p|b|n prime) emp | pred 1/(p-1) | P(p|b|n comp) emp | pred ~1/p | Increment emp | pred 1/[p(p-1)] |"
    sep =    "|----|--------------------|--------------|-------------------|-----------|---------------|-----------------|"
    print(header)
    print(sep)

    for p, data in results['primes'].items():
        emp_np = data['empirical_given_n_prime']
        pred_np = data['predicted_given_n_prime']
        emp_nc = data['empirical_given_n_composite']
        pred_nc = data['predicted_given_n_composite']
        incr = data['increment']
        pred_incr = data['predicted_increment']
        print(f"| {p:2d} | {emp_np:.6f}           | {pred_np:.6f}     | {emp_nc:.6f}          | {pred_nc:.6f}  | {incr:.6f}      | {pred_incr:.6f}        |")


def save_results(results: Dict, output_dir: Path):
    """Save results to reference directory."""
    import csv

    ref_dir = output_dir / 'sophie_germain'
    ref_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = ref_dir / 'per_prime_table.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['N', 'n_prime', 'n_composite', 'predicted_sum'])
        writer.writerow([results['N'], results['n_prime'], results['n_composite'],
                        f"{results['predicted_sum']:.6f}"])
        writer.writerow([])
        writer.writerow(['p', 'emp_given_n_prime', 'pred_1/(p-1)',
                        'emp_given_n_comp', 'pred_1/p', 'increment', 'pred_increment'])
        for p, data in results['primes'].items():
            writer.writerow([
                p,
                f"{data['empirical_given_n_prime']:.6f}",
                f"{data['predicted_given_n_prime']:.6f}",
                f"{data['empirical_given_n_composite']:.6f}",
                f"{data['predicted_given_n_composite']:.6f}",
                f"{data['increment']:.6f}",
                f"{data['predicted_increment']:.6f}"
            ])

    # Markdown
    md_path = ref_dir / 'per_prime_table.md'
    with open(md_path, 'w') as f:
        f.write(f"# Sophie Germain Per-Prime Verification\n\n")
        f.write(f"Pattern: $(n, 2n+1)$ for odd $n$\n\n")
        f.write(f"$N = {results['N']:,}$\n\n")
        f.write(f"Predicted: $\\sum_{{p \\geq 3}} 1/[p(p-1)] = {results['predicted_sum']:.4f}$\n\n")
        f.write("| $p$ | $\\mathbb{P}(p \\mid b \\mid n \\text{ prime})$ | Predicted $1/(p-1)$ | $\\mathbb{P}(p \\mid b \\mid n \\text{ comp})$ | Increment |\n")
        f.write("|-----|------------------------------------------------|---------------------|-----------------------------------------------|----------|\n")
        for p, data in results['primes'].items():
            f.write(f"| {p} | **{data['empirical_given_n_prime']:.4f}** | {data['predicted_given_n_prime']:.4f} | {data['empirical_given_n_composite']:.4f} | {data['increment']:.4f} |\n")

    print(f"\nResults saved to {ref_dir}/")


if __name__ == '__main__':
    import sys

    N = int(float(sys.argv[1])) if len(sys.argv) > 1 else 100_000_000
    force_cpu = '--cpu' in sys.argv

    print(f"GPU available: {HAS_GPU}")
    if HAS_GPU and not force_cpu:
        device = cuda.get_current_device()
        print(f"GPU: {device.name}")

    results = compute_per_prime_stats(N, force_cpu=force_cpu)
    print_table(results)

    output_dir = Path('data/reference')
    save_results(results, output_dir)
