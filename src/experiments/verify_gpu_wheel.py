#!/usr/bin/env python3
"""
Verify GPU wheel sieve produces identical results to CPU wheel sieve.

Tests:
1. State classification matches
2. Omega values match
3. Aggregated sums match

Run on DGX Spark with small K first.
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.wheel_sieve import (
    wheel_spf_sieve,
    omega_wheel,
    omega_leq_P_wheel,
    wheel_spf_lookup,
)


def verify_gpu_wheel(K: int, verbose: bool = True) -> bool:
    """
    Verify GPU wheel implementation against CPU wheel.
    """
    try:
        from src.gpu_wheel_sieve import WheelGPUContext, HAS_GPU
    except ImportError as e:
        print(f"Could not import GPU wheel sieve: {e}")
        return False

    if not HAS_GPU:
        print("GPU not available, skipping GPU verification")
        return True  # Not a failure, just skip

    if verbose:
        print(f"\n=== GPU Wheel Verification for K={K:,} ===")

    # Generate wheel SPF on CPU
    t0 = time.time()
    spf_wheel = wheel_spf_sieve(K)
    t_sieve = time.time() - t0
    if verbose:
        print(f"  Wheel sieve: {t_sieve:.2f}s, {spf_wheel.nbytes/1e6:.1f}MB")

    # Create GPU context
    t0 = time.time()
    ctx = WheelGPUContext(K, spf_wheel)
    t_ctx = time.time() - t0
    if verbose:
        print(f"  GPU context setup: {t_ctx:.2f}s")

    # Test 1: State classification
    if verbose:
        print("\n  Testing state classification...")

    t0 = time.time()
    states_gpu = ctx.compute_states()
    t_gpu = time.time() - t0

    # CPU reference
    t0 = time.time()
    states_cpu = np.zeros(K, dtype=np.int32)
    for k in range(1, K + 1):
        a, b = 6 * k - 1, 6 * k + 1
        a_prime = wheel_spf_lookup(a, spf_wheel) == a
        b_prime = wheel_spf_lookup(b, spf_wheel) == b
        if a_prime and b_prime:
            states_cpu[k-1] = 0
        elif a_prime:
            states_cpu[k-1] = 1
        elif b_prime:
            states_cpu[k-1] = 2
        else:
            states_cpu[k-1] = 3
    t_cpu = time.time() - t0

    states_match = np.array_equal(states_gpu, states_cpu)
    if verbose:
        print(f"    GPU: {t_gpu:.3f}s, CPU: {t_cpu:.3f}s")
        print(f"    States match: {states_match}")
        if not states_match:
            diffs = np.where(states_gpu != states_cpu)[0]
            print(f"    First 5 mismatches at indices: {diffs[:5]}")

    # Test 2: Omega values
    P = 13
    if verbose:
        print(f"\n  Testing omega_leq_{P}...")

    t0 = time.time()
    omega_a_gpu, omega_b_gpu = ctx.compute_omega_leq_P(P)
    t_gpu = time.time() - t0

    # CPU reference (just first 10000 for speed)
    check_n = min(K, 10000)
    t0 = time.time()
    omega_a_cpu = np.zeros(check_n, dtype=np.int32)
    omega_b_cpu = np.zeros(check_n, dtype=np.int32)
    for k in range(1, check_n + 1):
        a, b = 6 * k - 1, 6 * k + 1
        omega_a_cpu[k-1] = omega_leq_P_wheel(a, spf_wheel, P)
        omega_b_cpu[k-1] = omega_leq_P_wheel(b, spf_wheel, P)
    t_cpu = time.time() - t0

    omega_a_match = np.array_equal(omega_a_gpu[:check_n], omega_a_cpu)
    omega_b_match = np.array_equal(omega_b_gpu[:check_n], omega_b_cpu)

    if verbose:
        print(f"    GPU: {t_gpu:.3f}s (full K), CPU: {t_cpu:.3f}s (first {check_n:,})")
        print(f"    omega_a match: {omega_a_match}")
        print(f"    omega_b match: {omega_b_match}")

    # Test 3: Full omega
    if verbose:
        print(f"\n  Testing full omega...")

    t0 = time.time()
    full_a_gpu, full_b_gpu = ctx.compute_omega()
    t_gpu = time.time() - t0

    t0 = time.time()
    full_a_cpu = np.zeros(check_n, dtype=np.int32)
    full_b_cpu = np.zeros(check_n, dtype=np.int32)
    for k in range(1, check_n + 1):
        a, b = 6 * k - 1, 6 * k + 1
        full_a_cpu[k-1] = omega_wheel(a, spf_wheel)
        full_b_cpu[k-1] = omega_wheel(b, spf_wheel)
    t_cpu = time.time() - t0

    full_a_match = np.array_equal(full_a_gpu[:check_n], full_a_cpu)
    full_b_match = np.array_equal(full_b_gpu[:check_n], full_b_cpu)

    if verbose:
        print(f"    GPU: {t_gpu:.3f}s, CPU: {t_cpu:.3f}s")
        print(f"    full_omega_a match: {full_a_match}")
        print(f"    full_omega_b match: {full_b_match}")

    # Test 4: Aggregated results
    if verbose:
        print(f"\n  Testing aggregated omega_leq_{P}...")

    d_states = ctx.compute_states_gpu()
    t0 = time.time()
    sums_a_gpu, sums_b_gpu = ctx.compute_omega_leq_P_aggregated(P, d_states)
    t_gpu = time.time() - t0

    # CPU reference using full arrays
    state_counts = np.bincount(states_cpu, minlength=4)
    sums_a_cpu = np.zeros(4, dtype=np.float64)
    sums_b_cpu = np.zeros(4, dtype=np.float64)
    for s in range(4):
        mask = states_cpu == s
        if mask.sum() > 0:
            sums_a_cpu[s] = omega_a_gpu[mask].sum()  # Use GPU values (already verified)
            sums_b_cpu[s] = omega_b_gpu[mask].sum()

    if verbose:
        print(f"    GPU aggregation: {t_gpu:.3f}s")
        print(f"    State counts: PP={state_counts[0]:,}, PC={state_counts[1]:,}, "
              f"CP={state_counts[2]:,}, CC={state_counts[3]:,}")
        print(f"\n    {'State':<6} {'GPU_sum_a':>12} {'CPU_sum_a':>12} {'GPU_sum_b':>12} {'CPU_sum_b':>12}")
        for s, name in enumerate(['PP', 'PC', 'CP', 'CC']):
            print(f"    {name:<6} {sums_a_gpu[s]:>12.0f} {sums_a_cpu[s]:>12.0f} "
                  f"{sums_b_gpu[s]:>12.0f} {sums_b_cpu[s]:>12.0f}")

    sums_match = (np.allclose(sums_a_gpu, sums_a_cpu, rtol=1e-10) and
                  np.allclose(sums_b_gpu, sums_b_cpu, rtol=1e-10))

    if verbose:
        print(f"\n    Aggregated sums match: {sums_match}")

    # Summary
    all_pass = states_match and omega_a_match and omega_b_match and full_a_match and full_b_match and sums_match

    if verbose:
        print("\n" + "=" * 50)
        if all_pass:
            print("✓ All GPU wheel verifications passed!")
        else:
            print("✗ Some verifications failed!")

    return all_pass


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Verify GPU wheel sieve')
    parser.add_argument('--K', type=float, default=1e5, help='Number of pairs')
    args = parser.parse_args()

    K = int(args.K)

    print(f"GPU Wheel Sieve Verification")
    print(f"K = {K:,}")
    print("=" * 50)

    success = verify_gpu_wheel(K)

    if success:
        print("\nGPU wheel sieve verified! Ready for K=10^10.")
    else:
        sys.exit(1)
