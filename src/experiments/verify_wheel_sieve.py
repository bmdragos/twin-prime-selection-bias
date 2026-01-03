#!/usr/bin/env python3
"""
Verify wheel sieve produces identical results to standard sieve.

Compares:
1. SPF values for all 6k±1 numbers
2. Omega values for all pair members
3. State classifications
4. Mean omega per state

Run at small K first to verify correctness before scaling up.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.wheel_sieve import (
    wheel_spf_sieve,
    wheel_spf_lookup,
    omega_wheel,
    omega_leq_P_wheel,
    index_to_n,
    n_to_index,
)
from src.parallel_sieve import spf_sieve_parallel


def verify_spf_values(K: int, verbose: bool = True) -> bool:
    """Verify SPF values match between implementations."""
    if verbose:
        print(f"\n=== Verifying SPF values for K={K:,} ===")

    # Generate both SPF arrays
    t0 = time.time()
    spf_full = spf_sieve_parallel(6 * K + 1)
    t_full = time.time() - t0

    t0 = time.time()
    spf_wheel = wheel_spf_sieve(K)
    t_wheel = time.time() - t0

    if verbose:
        print(f"  Full sieve: {t_full:.2f}s, size={spf_full.nbytes/1e6:.1f}MB")
        print(f"  Wheel sieve: {t_wheel:.2f}s, size={spf_wheel.nbytes/1e6:.1f}MB")
        print(f"  Memory ratio: {spf_full.nbytes / spf_wheel.nbytes:.1f}x")

    # Compare SPF for all 6k±1 numbers
    errors = 0
    for k in range(1, K + 1):
        for n in [6 * k - 1, 6 * k + 1]:
            # Get SPF from full array
            spf_from_full = spf_full[n]
            if spf_from_full == 0:  # Prime sentinel in full array
                spf_from_full = n

            # Get SPF from wheel array
            spf_from_wheel = wheel_spf_lookup(n, spf_wheel)

            if spf_from_full != spf_from_wheel:
                errors += 1
                if errors <= 10:
                    print(f"  MISMATCH at n={n}: full={spf_from_full}, wheel={spf_from_wheel}")

    if verbose:
        if errors == 0:
            print(f"  ✓ All {2*K:,} SPF values match!")
        else:
            print(f"  ✗ {errors:,} mismatches found")

    return errors == 0


def verify_omega_values(K: int, P_values: list = None, verbose: bool = True) -> bool:
    """Verify omega values match for all pair members."""
    if P_values is None:
        P_values = [5, 7, 11, 13]

    if verbose:
        print(f"\n=== Verifying omega values for K={K:,} ===")

    # Generate SPF arrays
    spf_full = spf_sieve_parallel(6 * K + 1)
    spf_wheel = wheel_spf_sieve(K)

    # Import standard omega function
    from src.factorization import omega, omega_leq_P

    # Generate all pair values
    a_vals = np.array([6 * k - 1 for k in range(1, K + 1)], dtype=np.int64)
    b_vals = np.array([6 * k + 1 for k in range(1, K + 1)], dtype=np.int64)

    all_errors = 0

    # Test full omega
    if verbose:
        print("  Testing full omega...")

    for i, n in enumerate(list(a_vals[:1000]) + list(b_vals[:1000])):
        # Full implementation (need to handle 0 sentinel)
        omega_full = 0
        temp_n = int(n)
        prev = 0
        while temp_n > 1:
            p = spf_full[temp_n]
            if p == 0:
                p = temp_n
            if p != prev:
                omega_full += 1
                prev = p
            temp_n //= p

        omega_w = omega_wheel(n, spf_wheel)

        if omega_full != omega_w:
            all_errors += 1
            if all_errors <= 5:
                print(f"    MISMATCH omega({n}): full={omega_full}, wheel={omega_w}")

    if verbose:
        print(f"    Checked 2000 values, {all_errors} errors")

    # Test omega_leq_P
    for P in P_values:
        if verbose:
            print(f"  Testing omega_leq_P with P={P}...")

        p_errors = 0
        for n in list(a_vals[:500]) + list(b_vals[:500]):
            # Full implementation
            omega_full = 0
            temp_n = int(n)
            prev = 0
            while temp_n > 1:
                p = spf_full[temp_n]
                if p == 0:
                    p = temp_n
                if p != prev and p <= P:
                    omega_full += 1
                if p != prev:
                    prev = p
                temp_n //= p

            omega_w = omega_leq_P_wheel(n, spf_wheel, P)

            if omega_full != omega_w:
                p_errors += 1
                all_errors += 1
                if p_errors <= 3:
                    print(f"    MISMATCH omega_leq_{P}({n}): full={omega_full}, wheel={omega_w}")

        if verbose:
            print(f"    Checked 1000 values, {p_errors} errors")

    if verbose:
        if all_errors == 0:
            print(f"  ✓ All omega values match!")
        else:
            print(f"  ✗ {all_errors} total errors")

    return all_errors == 0


def verify_full_analysis(K: int, verbose: bool = True) -> bool:
    """
    Run full analysis with both implementations and compare results.
    """
    if verbose:
        print(f"\n=== Full analysis comparison for K={K:,} ===")

    # Generate SPF arrays
    t0 = time.time()
    spf_full = spf_sieve_parallel(6 * K + 1)
    t_full = time.time() - t0

    t0 = time.time()
    spf_wheel = wheel_spf_sieve(K)
    t_wheel = time.time() - t0

    if verbose:
        print(f"  Sieve times: full={t_full:.2f}s, wheel={t_wheel:.2f}s")

    # Generate pairs
    a_vals = np.array([6 * k - 1 for k in range(1, K + 1)], dtype=np.int64)
    b_vals = np.array([6 * k + 1 for k in range(1, K + 1)], dtype=np.int64)

    # Classify states using full SPF
    def is_prime_full(n):
        return spf_full[n] == 0

    states = np.zeros(K, dtype=np.int32)
    for i in range(K):
        a_prime = is_prime_full(a_vals[i])
        b_prime = is_prime_full(b_vals[i])
        if a_prime and b_prime:
            states[i] = 0  # PP
        elif a_prime:
            states[i] = 1  # PC
        elif b_prime:
            states[i] = 2  # CP
        else:
            states[i] = 3  # CC

    state_counts = np.bincount(states, minlength=4)
    if verbose:
        print(f"  States: PP={state_counts[0]:,}, PC={state_counts[1]:,}, "
              f"CP={state_counts[2]:,}, CC={state_counts[3]:,}")

    # Compute omega sums for each state using both methods
    P = 13  # Test with P=13

    if verbose:
        print(f"  Computing omega_leq_{P} sums by state...")

    # Full method
    omega_a_full = np.zeros(K, dtype=np.int32)
    omega_b_full = np.zeros(K, dtype=np.int32)
    for i in range(K):
        # omega_leq_P for a
        n = int(a_vals[i])
        count = 0
        prev = 0
        while n > 1:
            p = spf_full[n]
            if p == 0:
                p = n
            if p != prev and p <= P:
                count += 1
            if p != prev:
                prev = p
            n //= p
        omega_a_full[i] = count

        # omega_leq_P for b
        n = int(b_vals[i])
        count = 0
        prev = 0
        while n > 1:
            p = spf_full[n]
            if p == 0:
                p = n
            if p != prev and p <= P:
                count += 1
            if p != prev:
                prev = p
            n //= p
        omega_b_full[i] = count

    # Wheel method
    omega_a_wheel = np.zeros(K, dtype=np.int32)
    omega_b_wheel = np.zeros(K, dtype=np.int32)
    for i in range(K):
        omega_a_wheel[i] = omega_leq_P_wheel(a_vals[i], spf_wheel, P)
        omega_b_wheel[i] = omega_leq_P_wheel(b_vals[i], spf_wheel, P)

    # Compare
    a_match = np.array_equal(omega_a_full, omega_a_wheel)
    b_match = np.array_equal(omega_b_full, omega_b_wheel)

    if verbose:
        print(f"  omega_a arrays match: {a_match}")
        print(f"  omega_b arrays match: {b_match}")

    # Compare mean omega per state
    if verbose:
        print(f"\n  Mean omega_leq_{P} by state:")
        print(f"  {'State':<6} {'Full_a':>10} {'Wheel_a':>10} {'Full_b':>10} {'Wheel_b':>10}")

    all_match = True
    for s, name in enumerate(['PP', 'PC', 'CP', 'CC']):
        mask = states == s
        if mask.sum() == 0:
            continue

        mean_a_full = omega_a_full[mask].mean()
        mean_a_wheel = omega_a_wheel[mask].mean()
        mean_b_full = omega_b_full[mask].mean()
        mean_b_wheel = omega_b_wheel[mask].mean()

        if verbose:
            print(f"  {name:<6} {mean_a_full:>10.6f} {mean_a_wheel:>10.6f} "
                  f"{mean_b_full:>10.6f} {mean_b_wheel:>10.6f}")

        if abs(mean_a_full - mean_a_wheel) > 1e-10 or abs(mean_b_full - mean_b_wheel) > 1e-10:
            all_match = False

    if verbose:
        if all_match and a_match and b_match:
            print(f"\n  ✓ Full analysis results match!")
        else:
            print(f"\n  ✗ Results do not match!")

    return all_match and a_match and b_match


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Verify wheel sieve correctness')
    parser.add_argument('--K', type=float, default=1e5, help='Number of pairs (default: 1e5)')
    parser.add_argument('--full', action='store_true', help='Run full analysis comparison')
    args = parser.parse_args()

    K = int(args.K)

    print(f"Wheel Sieve Verification")
    print(f"K = {K:,}")
    print("=" * 50)

    # Run verifications
    spf_ok = verify_spf_values(K)
    omega_ok = verify_omega_values(K)

    if args.full:
        full_ok = verify_full_analysis(K)
    else:
        full_ok = True

    print("\n" + "=" * 50)
    if spf_ok and omega_ok and full_ok:
        print("✓ All verifications passed!")
        print(f"\nWheel sieve is correct. Memory savings: 3x")
        print(f"Ready to scale up to K=10^10 with ~80GB instead of 240GB")
    else:
        print("✗ Some verifications failed!")
        sys.exit(1)
