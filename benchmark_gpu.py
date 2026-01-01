#!/usr/bin/env python3
"""
Benchmark GPU optimizations for twin prime experiment.

Compares:
1. Current approach: transfer full arrays, aggregate on CPU
2. Optimized: aggregate on GPU, transfer only sums

Run at K=10^7 or 10^8 for quick comparison.
"""

import argparse
import time
import numpy as np

from src.parallel_sieve import spf_sieve_parallel

try:
    from numba import cuda
    from src.gpu_factorization import (
        HAS_GPU, HAS_UNIFIED_MEMORY, UnifiedMemoryGPUContext,
        compute_states_gpu, check_gpu
    )
except ImportError:
    HAS_GPU = False
    HAS_UNIFIED_MEMORY = False


def benchmark(K: int):
    """Run benchmark comparing GPU approaches."""
    print("=" * 60)
    print(f"GPU Benchmark: K = {K:,}")
    print("=" * 60)

    if not HAS_GPU:
        print("ERROR: GPU not available")
        return

    check_gpu()
    print()

    N = 6 * K + 1
    P_test = [50, 997]  # Just two P values for quick test

    # Build SPF
    print("Building SPF sieve...", end=" ", flush=True)
    t0 = time.time()
    spf = spf_sieve_parallel(N)
    print(f"{time.time() - t0:.1f}s")

    # Prime flags
    prime_flags = (spf == 0)
    prime_flags[0] = prime_flags[1] = False

    # State codes
    print("Computing states...", end=" ", flush=True)
    t0 = time.time()
    state_codes = compute_states_gpu(K, prime_flags)
    print(f"{time.time() - t0:.1f}s")

    # State counts
    counts = {i: np.sum(state_codes == i) for i in range(4)}
    print(f"State counts: PP={counts[0]:,}, PC={counts[1]:,}, CP={counts[2]:,}, CC={counts[3]:,}")
    print()

    # Generate a/b values
    k_vals = np.arange(1, K + 1, dtype=np.int64)
    a_vals = 6 * k_vals - 1
    b_vals = 6 * k_vals + 1

    # Create GPU context
    print("Creating GPU context...")
    gpu_ctx = UnifiedMemoryGPUContext(a_vals, b_vals, spf)

    # Transfer state_codes to GPU for aggregated approach
    d_state_codes = cuda.to_device(state_codes)
    print()

    # ============================================================
    # Benchmark 1: Current approach (transfer full arrays)
    # ============================================================
    print("-" * 60)
    print("Approach 1: Transfer full arrays, CPU aggregation")
    print("-" * 60)

    times_transfer = []
    for P in P_test:
        t0 = time.time()
        omega_a, omega_b = gpu_ctx.compute_omega_leq_P(P)

        # CPU aggregation
        mean_a = {i: np.mean(omega_a[state_codes == i]) for i in range(4)}
        mean_b = {i: np.mean(omega_b[state_codes == i]) for i in range(4)}

        elapsed = time.time() - t0
        times_transfer.append(elapsed)
        print(f"  P={P}: {elapsed:.2f}s  (PC mean_b = {mean_b[1]:.4f})")

    print(f"  Total: {sum(times_transfer):.2f}s")
    print()

    # ============================================================
    # Benchmark 2: GPU aggregation (transfer only sums)
    # ============================================================
    print("-" * 60)
    print("Approach 2: GPU aggregation, transfer only sums")
    print("-" * 60)

    times_aggregated = []
    for P in P_test:
        t0 = time.time()
        sums_a, sums_b = gpu_ctx.compute_omega_leq_P_aggregated(P, d_state_codes)

        # Compute means from sums
        mean_a = {i: sums_a[i] / counts[i] if counts[i] > 0 else 0 for i in range(4)}
        mean_b = {i: sums_b[i] / counts[i] if counts[i] > 0 else 0 for i in range(4)}

        elapsed = time.time() - t0
        times_aggregated.append(elapsed)
        print(f"  P={P}: {elapsed:.2f}s  (PC mean_b = {mean_b[1]:.4f})")

    print(f"  Total: {sum(times_aggregated):.2f}s")
    print()

    # ============================================================
    # Summary
    # ============================================================
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    speedup = sum(times_transfer) / sum(times_aggregated)
    print(f"Transfer full arrays: {sum(times_transfer):.2f}s")
    print(f"GPU aggregation:      {sum(times_aggregated):.2f}s")
    print(f"Speedup: {speedup:.2f}x")

    # Verify results match
    print()
    print("Verifying correctness...")
    omega_a, omega_b = gpu_ctx.compute_omega_leq_P(P_test[0])
    sums_a, sums_b = gpu_ctx.compute_omega_leq_P_aggregated(P_test[0], d_state_codes)

    for i in range(4):
        expected_sum = np.sum(omega_a[state_codes == i])
        actual_sum = sums_a[i]
        if abs(expected_sum - actual_sum) < 1:
            print(f"  State {i}: OK (sum_a = {actual_sum:.0f})")
        else:
            print(f"  State {i}: MISMATCH! expected={expected_sum:.0f}, got={actual_sum:.0f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark GPU optimizations')
    parser.add_argument('--K', type=float, default=1e7, help='Number of pairs')
    args = parser.parse_args()

    benchmark(int(args.K))
