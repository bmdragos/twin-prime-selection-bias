"""
Experiment: Per-Prime Divisibility for Cousin Primes (n, n+4)

Verifies the local mechanism: P(p | n+4 | n prime) = 1/(p-1) for each prime p >= 5.

For cousin primes among 6k±1 candidates:
- If p | n and p | (n+4), then p | 4, impossible for p >= 5.
- For p = 3:
  - If n = 6k-1, then n+4 = 6k+3 ≡ 0 (mod 3) ALWAYS
  - If n = 6k+1, then n+4 = 6k+5 ≡ 2 (mod 3) NEVER
  - Since 3|(n+4) depends on residue class (not primality of n), and
    prime/composite n have the same residue-class mix among 6k±1,
    the p=3 term contributes ZERO to the difference.
- Sum starts at p = 5: Σ_{p>=5} 1/[p(p-1)] ≈ 0.1065

Population: n = 6k±1 for k in [1, K]
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
    def _cousin_per_prime_kernel(n_vals, b_vals, n_is_prime, primes, n_primes, count,
                                  block_counts_n_prime, block_counts_n_composite,
                                  block_primality_counts):
        """
        CUDA kernel for cousin primes per-prime statistics.
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


def compute_per_prime_stats_gpu(K: int, primes: List[int] = [3, 5, 7, 11, 13]) -> Dict:
    """
    Compute per-prime divisibility stats for cousin primes using GPU.

    Note: p=3 is included for completeness but its contribution to the
    difference should be ~0 (depends on residue class, not primality).
    """
    print(f"Computing cousin primes per-prime stats (GPU) for K={K:,}")

    # Cousin primes: (n, n+4) for n in 6k±1
    # Max value is (6K+1) + 4 = 6K+5
    max_val = 6 * K + 5
    primes_arr = np.array(primes, dtype=np.int64)
    n_primes = len(primes)

    print(f"  Generating prime flags up to {max_val:,}...")
    t0 = time.time()
    prime_flags = prime_flags_upto(max_val)
    print(f"    Sieve completed in {time.time() - t0:.1f}s")

    # Build arrays for n = 6k-1 and 6k+1
    print(f"  Building pair arrays for 6k±1 candidates...")
    t0 = time.time()
    k_vals = np.arange(1, K + 1, dtype=np.int64)
    n_minus = 6 * k_vals - 1  # 6k-1
    n_plus = 6 * k_vals + 1   # 6k+1

    # Combine both residue classes
    n_vals = np.concatenate([n_minus, n_plus])
    b_vals = n_vals + 4
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

    _cousin_per_prime_kernel[n_blocks, threads_per_block](
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

    print(f"  Results: {n_prime:,} n prime, {n_composite:,} n composite")

    return _build_results(K, n_prime, n_composite, primes, counts_n_prime, counts_n_comp)


def compute_per_prime_stats_cpu(K: int, primes: List[int] = [3, 5, 7, 11, 13]) -> Dict:
    """
    Compute per-prime divisibility stats for cousin primes using CPU.
    """
    print(f"Computing cousin primes per-prime stats (CPU) for K={K:,}")

    max_val = 6 * K + 5
    print(f"  Generating prime flags up to {max_val:,}...")
    t0 = time.time()
    prime_flags = prime_flags_upto(max_val)
    print(f"    Sieve completed in {time.time() - t0:.1f}s")

    print(f"  Computing stats...")
    t0 = time.time()

    k_vals = np.arange(1, K + 1, dtype=np.int64)
    n_minus = 6 * k_vals - 1
    n_plus = 6 * k_vals + 1
    n_vals = np.concatenate([n_minus, n_plus])
    b_vals = n_vals + 4
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
    print(f"  Results: {n_prime:,} n prime, {n_composite:,} n composite")

    return _build_results(K, int(n_prime), int(n_composite), primes,
                          np.array(counts_n_prime), np.array(counts_n_comp))


def _build_results(K, n_prime, n_composite, primes, counts_n_prime, counts_n_comp) -> Dict:
    """Build results dictionary from counts."""
    # Sum only for p >= 5 (p=3 cancels)
    predicted_sum = sum(1/(p*(p-1)) for p in primes if p >= 5)

    results = {
        'K': K,
        'pattern': 'Cousin primes (n, n+4)',
        'population': '6k±1 candidates',
        'n_prime': n_prime,
        'n_composite': n_composite,
        'predicted_sum': predicted_sum,
        'note': 'p=3 contributes 0 to difference (depends on residue class, not primality)',
        'primes': {}
    }

    for i, p in enumerate(primes):
        p_b_given_n_prime = counts_n_prime[i] / n_prime if n_prime > 0 else 0
        p_b_given_n_comp = counts_n_comp[i] / n_composite if n_composite > 0 else 0

        # For p=3, the "naive" prediction is 1/3, but actual depends on residue class mix
        # The INCREMENT prediction is still 1/[p(p-1)] for p >= 5, but ~0 for p=3
        if p == 3:
            note = "Residue-class determined (should cancel)"
        else:
            note = None

        results['primes'][p] = {
            'count_n_prime_and_p_div_b': int(counts_n_prime[i]),
            'count_n_comp_and_p_div_b': int(counts_n_comp[i]),
            'empirical_given_n_prime': float(p_b_given_n_prime),
            'empirical_given_n_composite': float(p_b_given_n_comp),
            'predicted_given_n_prime': 1 / (p - 1) if p >= 5 else None,  # p=3 is special
            'predicted_given_n_composite': 1 / p if p >= 5 else None,
            'increment': float(p_b_given_n_prime - p_b_given_n_comp),
            'predicted_increment': 1 / (p * (p - 1)) if p >= 5 else 0,  # p=3 should be ~0
            'note': note
        }

    return results


def compute_per_prime_stats(K: int, primes: List[int] = [3, 5, 7, 11, 13],
                            force_cpu: bool = False) -> Dict:
    """Compute per-prime stats (auto-selects GPU or CPU)."""
    if HAS_GPU and not force_cpu:
        return compute_per_prime_stats_gpu(K, primes)
    else:
        return compute_per_prime_stats_cpu(K, primes)


def print_table(results: Dict):
    """Print formatted table."""
    print("\n" + "="*110)
    print(f"Per-Prime Verification: {results['pattern']}")
    print("="*110)
    print(f"K = {results['K']:,}, Population: {results['population']}")
    print(f"n prime: {results['n_prime']:,}, n composite: {results['n_composite']:,}")
    print(f"Predicted Σ 1/[p(p-1)] for p >= 5: {results['predicted_sum']:.4f}")
    print(f"Note: {results['note']}")
    print()

    header = "| p  | P(p|b|n prime) emp | pred 1/(p-1) | P(p|b|n comp) emp | pred ~1/p | Increment emp | pred         | Note |"
    sep =    "|----|--------------------|--------------|--------------------|-----------|---------------|--------------|------|"
    print(header)
    print(sep)

    for p, data in results['primes'].items():
        emp_np = data['empirical_given_n_prime']
        pred_np = data['predicted_given_n_prime']
        emp_nc = data['empirical_given_n_composite']
        pred_nc = data['predicted_given_n_composite']
        incr = data['increment']
        pred_incr = data['predicted_increment']
        note = data.get('note', '')

        pred_np_str = f"{pred_np:.6f}" if pred_np is not None else "N/A"
        pred_nc_str = f"{pred_nc:.6f}" if pred_nc is not None else "N/A"

        print(f"| {p:2d} | {emp_np:.6f}           | {pred_np_str:>12} | {emp_nc:.6f}           | {pred_nc_str:>9} | {incr:+.6f}     | {pred_incr:.6f}     | {note or ''} |")


def save_results(results: Dict, output_dir: Path):
    """Save results to reference directory."""
    import csv

    ref_dir = output_dir / 'cousin_primes'
    ref_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = ref_dir / 'per_prime_table.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['K', 'n_prime', 'n_composite', 'predicted_sum', 'note'])
        writer.writerow([results['K'], results['n_prime'], results['n_composite'],
                        f"{results['predicted_sum']:.6f}", results['note']])
        writer.writerow([])
        writer.writerow(['p', 'emp_given_n_prime', 'pred_1/(p-1)',
                        'emp_given_n_comp', 'pred_1/p', 'increment', 'pred_increment', 'note'])
        for p, data in results['primes'].items():
            writer.writerow([
                p,
                f"{data['empirical_given_n_prime']:.6f}",
                data['predicted_given_n_prime'] if data['predicted_given_n_prime'] else 'N/A',
                f"{data['empirical_given_n_composite']:.6f}",
                data['predicted_given_n_composite'] if data['predicted_given_n_composite'] else 'N/A',
                f"{data['increment']:.6f}",
                f"{data['predicted_increment']:.6f}",
                data.get('note', '')
            ])

    # Markdown
    md_path = ref_dir / 'per_prime_table.md'
    with open(md_path, 'w') as f:
        f.write(f"# Cousin Primes Per-Prime Verification\n\n")
        f.write(f"Pattern: $(n, n+4)$ among $6k \\pm 1$ candidates\n\n")
        f.write(f"$K = {results['K']:,}$\n\n")
        f.write(f"**Note:** For $p=3$, divisibility of $n+4$ depends on residue class:\n")
        f.write(f"- $n = 6k-1 \\Rightarrow n+4 = 6k+3 \\equiv 0 \\pmod 3$ (always)\n")
        f.write(f"- $n = 6k+1 \\Rightarrow n+4 = 6k+5 \\equiv 2 \\pmod 3$ (never)\n\n")
        f.write(f"Since prime/composite $n$ have the same residue-class mix, $p=3$ contributes zero to the difference.\n\n")
        f.write(f"Predicted: $\\sum_{{p \\geq 5}} 1/[p(p-1)] = {results['predicted_sum']:.4f}$\n\n")
        f.write("| $p$ | $\\mathbb{P}(p \\mid b \\mid n \\text{ prime})$ | Predicted | $\\mathbb{P}(p \\mid b \\mid n \\text{ comp})$ | Increment | Note |\n")
        f.write("|-----|------------------------------------------------|-----------|-----------------------------------------------|-----------|------|\n")
        for p, data in results['primes'].items():
            pred = f"{data['predicted_given_n_prime']:.4f}" if data['predicted_given_n_prime'] else "—"
            note = data.get('note', '') or ''
            f.write(f"| {p} | **{data['empirical_given_n_prime']:.4f}** | {pred} | {data['empirical_given_n_composite']:.4f} | {data['increment']:+.4f} | {note} |\n")

    print(f"\nResults saved to {ref_dir}/")


if __name__ == '__main__':
    import sys

    K = int(float(sys.argv[1])) if len(sys.argv) > 1 else 100_000_000
    force_cpu = '--cpu' in sys.argv

    print(f"GPU available: {HAS_GPU}")
    if HAS_GPU and not force_cpu:
        device = cuda.get_current_device()
        print(f"GPU: {device.name}")

    results = compute_per_prime_stats(K, force_cpu=force_cpu)
    print_table(results)

    output_dir = Path('data/reference')
    save_results(results, output_dir)
