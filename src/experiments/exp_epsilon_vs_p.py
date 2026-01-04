#!/usr/bin/env python3
"""
Compute scaled residual ε_p for many primes and plot ε_p vs p.

For twin primes (6k-1, 6k+1), the local density model predicts:
    P(p | b | a prime) = 1/(p-1)  for all p >= 5

The scaled residual is:
    ε_p = (p-1) × P̂(p|b|a prime) - 1

If the model is correct, ε_p → 0 as K → ∞.

This script:
1. Tests many primes (up to a few thousand)
2. Computes ε_p for each
3. Saves data and generates a plot showing ε_p vs p

The resulting plot should show all ε_p clustered near zero,
providing visual confirmation that the mechanism is correct.
"""

import numpy as np
import time
import json
import csv
import argparse
from pathlib import Path

# Check for GPU
try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    print("CuPy not available, using CPU-only mode")


def get_primes_up_to(n: int) -> np.ndarray:
    """Sieve of Eratosthenes to get all primes up to n."""
    sieve = np.ones(n + 1, dtype=bool)
    sieve[0:2] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    return np.where(sieve)[0]


def compute_epsilon_gpu(K: int, primes_to_test: np.ndarray, block_size: int = 1_000_000):
    """
    Compute ε_p for each prime using GPU.

    Uses blocked processing to handle large K values.
    """
    import cupy as cp

    # Total pairs
    n_pairs = K

    # Initialize counters
    n_a_prime = 0
    count_p_div_b_given_a_prime = {p: 0 for p in primes_to_test}

    # Process in blocks
    for k_start in range(1, K + 1, block_size):
        k_end = min(k_start + block_size, K + 1)
        k_block = cp.arange(k_start, k_end, dtype=cp.int64)

        # a = 6k - 1, b = 6k + 1
        a = 6 * k_block - 1
        b = 6 * k_block + 1

        # Check primality (trial division for now)
        a_is_prime = cp.ones(len(k_block), dtype=bool)

        # Quick composite check using small primes
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
            if p * p > int(a.max()):
                break
            a_is_prime &= (a % p != 0) | (a == p)

        # For larger factors, we need proper primality test
        # Use trial division with remaining primes
        max_a = int(a.max())
        small_primes = get_primes_up_to(min(int(max_a**0.5) + 1, 50000))
        for p in small_primes:
            if p > 31:
                a_is_prime &= (a % p != 0) | (a == p)

        # Count primes in a
        n_a_prime += int(a_is_prime.sum())

        # For each test prime, count divisibility of b when a is prime
        for p in primes_to_test:
            mask = a_is_prime & (b % p == 0)
            count_p_div_b_given_a_prime[p] += int(mask.sum())

    return n_a_prime, count_p_div_b_given_a_prime


def compute_epsilon_cpu(K: int, primes_to_test: np.ndarray, block_size: int = 1_000_000):
    """
    Compute ε_p for each prime using CPU with numpy and SPF sieve.
    """
    # Build SPF sieve for primality testing
    max_val = 6 * K + 1
    print(f"Building SPF sieve up to {max_val:,}...")

    # For very large K, use blocked primality check
    if max_val > 1e9:
        return compute_epsilon_cpu_blocked(K, primes_to_test, block_size)

    # SPF sieve (0 = prime)
    spf = np.zeros(max_val + 1, dtype=np.uint32)
    for p in range(2, int(max_val**0.5) + 1):
        if spf[p] == 0:  # p is prime
            for m in range(p*p, max_val + 1, p):
                if spf[m] == 0:
                    spf[m] = p

    # Process all pairs
    k = np.arange(1, K + 1, dtype=np.int64)
    a = 6 * k - 1
    b = 6 * k + 1

    # a is prime iff spf[a] == 0
    a_is_prime = spf[a] == 0
    n_a_prime = int(a_is_prime.sum())

    # Extract b values where a is prime
    b_prime = b[a_is_prime]

    # Count p|b for each test prime
    results = {}
    for p in primes_to_test:
        count = int((b_prime % p == 0).sum())
        results[p] = count

    return n_a_prime, results


def compute_epsilon_cpu_blocked(K: int, primes_to_test: np.ndarray, block_size: int = 10_000_000):
    """Blocked CPU implementation for very large K."""
    from multiprocessing import Pool, cpu_count

    n_a_prime = 0
    count_p_div_b_given_a_prime = {p: 0 for p in primes_to_test}

    # For each block, we need a local sieve
    for k_start in range(1, K + 1, block_size):
        k_end = min(k_start + block_size, K + 1)
        print(f"Processing k = {k_start:,} to {k_end:,}...")

        # Local block
        k_block = np.arange(k_start, k_end, dtype=np.int64)
        a = 6 * k_block - 1
        b = 6 * k_block + 1

        max_val = int(b.max())
        min_val = int(a.min())

        # Segmented sieve for this range
        is_prime_a = segmented_primality(a)

        n_a_prime += int(is_prime_a.sum())

        b_prime = b[is_prime_a]
        for p in primes_to_test:
            count_p_div_b_given_a_prime[p] += int((b_prime % p == 0).sum())

    return n_a_prime, count_p_div_b_given_a_prime


def segmented_primality(values: np.ndarray) -> np.ndarray:
    """Check primality for an array of values using trial division."""
    max_val = int(values.max())
    sqrt_max = int(max_val**0.5) + 1

    # Get primes up to sqrt(max)
    small_primes = get_primes_up_to(min(sqrt_max, 100000))

    is_prime = np.ones(len(values), dtype=bool)

    for p in small_primes:
        if p * p > max_val:
            break
        is_prime &= (values % p != 0) | (values == p)

    return is_prime


def main():
    parser = argparse.ArgumentParser(description="Compute ε_p for many primes")
    parser.add_argument("--K", type=float, default=1e8, help="Number of pairs (default: 10^8)")
    parser.add_argument("--max_prime", type=int, default=1000, help="Max prime to test (default: 1000)")
    parser.add_argument("--output", type=str, default="data/reference/epsilon_vs_p", help="Output directory")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    args = parser.parse_args()

    K = int(args.K)
    max_prime = args.max_prime
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Computing ε_p for primes up to {max_prime} with K = {K:,}")
    print(f"Output: {output_dir}")

    # Get primes to test (skip 2 and 3 which are handled by the 6k±1 wheel)
    all_primes = get_primes_up_to(max_prime)
    primes_to_test = all_primes[all_primes >= 5]
    print(f"Testing {len(primes_to_test)} primes: {primes_to_test[:10]}...{primes_to_test[-5:]}")

    # Compute
    t0 = time.time()

    if HAS_GPU:
        print("Using GPU...")
        n_a_prime, counts = compute_epsilon_gpu(K, primes_to_test)
    else:
        print("Using CPU...")
        n_a_prime, counts = compute_epsilon_cpu(K, primes_to_test)

    elapsed = time.time() - t0
    print(f"Computation took {elapsed:.1f}s")
    print(f"n_a_prime = {n_a_prime:,}")

    # Compute ε_p for each prime
    results = []
    for p in primes_to_test:
        count = counts[p]
        p_hat = count / n_a_prime
        predicted = 1.0 / (p - 1)
        epsilon_p = (p - 1) * p_hat - 1
        se = np.sqrt(p_hat * (1 - p_hat) / n_a_prime)
        z = (p_hat - predicted) / se if se > 0 else 0

        results.append({
            'p': int(p),
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
        'computation_time_seconds': elapsed
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
