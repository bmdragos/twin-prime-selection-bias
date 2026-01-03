#!/usr/bin/env python3
"""
Stability and baseline analysis for twin prime selection bias.

Two critical analyses for skeptic-proofing:
1. Tail-window stability: Is bias stable in [K/2,K], [0.9K,K], log bins?
2. Explicit baselines: PC vs CC vs unconditional composites

This addresses the concern that prefix averages might hide slowly drifting behavior.
"""

import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path

from src.primes import prime_flags_upto
from src.factorization import spf_sieve

try:
    from src.gpu_factorization import (
        HAS_GPU, HAS_UNIFIED_MEMORY, compute_states_gpu, check_gpu
    )
    from src.parallel_sieve import spf_sieve_parallel
except ImportError:
    HAS_GPU = False
    HAS_UNIFIED_MEMORY = False


def compute_omega_array(numbers: np.ndarray, spf: np.ndarray) -> np.ndarray:
    """Compute omega (distinct prime factors) for array of numbers."""
    results = np.zeros(len(numbers), dtype=np.int32)
    for i, n in enumerate(numbers):
        if n <= 1:
            continue
        count = 0
        prev = 0
        while n > 1:
            p = spf[n]
            if p == 0:  # 0 = prime sentinel
                p = n
            if p != prev:
                count += 1
            prev = p
            n //= p
        results[i] = count
    return results


def run_window_analysis(K: int, output_dir: Path):
    """
    Analyze selection bias stability across different windows.

    Windows tested:
    - Full prefix [1, K]
    - Second half [K/2, K]
    - Last 10% [0.9K, K]
    - Logarithmic bins
    """
    print("=" * 70)
    print("Twin Prime Selection Bias - Stability Analysis")
    print("=" * 70)
    print(f"K = {K:,}")
    print()

    N = 6 * K + 1

    # Build sieves
    print("Building sieves...")
    t0 = time.time()
    if K > 10**7:
        spf = spf_sieve_parallel(N)
        # parallel_sieve uses 0 as prime sentinel
        prime_flags = (spf == 0)
    else:
        spf = spf_sieve(N)
        # factorization.spf_sieve uses spf[n] = n for primes
        prime_flags = (spf == np.arange(N + 1))
    prime_flags[0] = prime_flags[1] = False
    print(f"  Completed in {time.time() - t0:.1f}s")
    print()

    # Compute states
    print("Computing pair states...")
    t0 = time.time()
    if HAS_GPU:
        state_codes = compute_states_gpu(K, prime_flags)
    else:
        k_vals = np.arange(1, K + 1)
        a_vals = 6 * k_vals - 1
        b_vals = 6 * k_vals + 1
        a_prime = prime_flags[a_vals]
        b_prime = prime_flags[b_vals]
        state_codes = np.where(
            a_prime & b_prime, 0,  # PP
            np.where(a_prime & ~b_prime, 1,  # PC
            np.where(~a_prime & b_prime, 2,  # CP
            3)))  # CC
    print(f"  Completed in {time.time() - t0:.1f}s")
    print()

    # Compute omega for all pairs
    print("Computing omega for all pairs...")
    t0 = time.time()
    k_vals = np.arange(1, K + 1, dtype=np.int64)
    a_vals = 6 * k_vals - 1
    b_vals = 6 * k_vals + 1

    # Use parallel if available
    if K > 10**7:
        from src.parallel_sieve import omega_parallel
        omega_a = omega_parallel(a_vals, spf)
        omega_b = omega_parallel(b_vals, spf)
    else:
        omega_a = compute_omega_array(a_vals, spf)
        omega_b = compute_omega_array(b_vals, spf)
    print(f"  Completed in {time.time() - t0:.1f}s")
    print()

    # ================================================================
    # Analysis 1: Window stability
    # ================================================================
    print("-" * 70)
    print("ANALYSIS 1: Window Stability")
    print("-" * 70)
    print()

    windows = [
        ("Full [1, K]", 0, K),
        ("Second half [K/2, K]", K // 2, K),
        ("Last 10% [0.9K, K]", int(0.9 * K), K),
        ("Last 1% [0.99K, K]", int(0.99 * K), K),
    ]

    window_results = []
    for name, start, end in windows:
        mask_window = (np.arange(K) >= start) & (np.arange(K) < end)

        # PC composites in window
        pc_mask = (state_codes == 1) & mask_window
        pc_omega_b = omega_b[pc_mask]

        # CC composites in window (use omega_a as reference)
        cc_mask = (state_codes == 3) & mask_window
        cc_omega = omega_a[cc_mask]  # Either a or b, they're symmetric

        if len(pc_omega_b) > 0 and len(cc_omega) > 0:
            mean_pc = np.mean(pc_omega_b)
            mean_cc = np.mean(cc_omega)
            bias_pct = 100 * (mean_pc - mean_cc) / mean_cc
        else:
            mean_pc = mean_cc = bias_pct = np.nan

        window_results.append({
            'window': name,
            'k_start': start + 1,
            'k_end': end,
            'n_pairs': end - start,
            'n_pc': np.sum(pc_mask),
            'n_cc': np.sum(cc_mask),
            'mean_omega_pc': mean_pc,
            'mean_omega_cc': mean_cc,
            'bias_pct': bias_pct
        })

        print(f"{name}:")
        print(f"  PC composite omega: {mean_pc:.6f} (n={np.sum(pc_mask):,})")
        print(f"  CC composite omega: {mean_cc:.6f} (n={np.sum(cc_mask):,})")
        print(f"  Selection bias: {bias_pct:.4f}%")
        print()

    # Logarithmic bins
    print("Logarithmic bins:")
    log_bins = []
    bin_edges = [10**i for i in range(4, int(np.log10(K)) + 1)]
    bin_edges.append(K)

    for i in range(len(bin_edges) - 1):
        start, end = bin_edges[i], bin_edges[i + 1]
        mask_window = (np.arange(K) >= start) & (np.arange(K) < end)

        pc_mask = (state_codes == 1) & mask_window
        cc_mask = (state_codes == 3) & mask_window

        if np.sum(pc_mask) > 0 and np.sum(cc_mask) > 0:
            mean_pc = np.mean(omega_b[pc_mask])
            mean_cc = np.mean(omega_a[cc_mask])
            bias_pct = 100 * (mean_pc - mean_cc) / mean_cc

            log_bins.append({
                'bin': f"[10^{int(np.log10(start))}, 10^{int(np.log10(end))})" if end < K else f"[10^{int(np.log10(start))}, K]",
                'k_start': start,
                'k_end': end,
                'mean_omega_pc': mean_pc,
                'mean_omega_cc': mean_cc,
                'bias_pct': bias_pct
            })

            print(f"  [{start:,}, {end:,}): bias = {bias_pct:.4f}%")

    print()

    # ================================================================
    # Analysis 2: Baseline comparisons
    # ================================================================
    print("-" * 70)
    print("ANALYSIS 2: Baseline Comparisons")
    print("-" * 70)
    print()

    # Baseline 1: CC composites (current reference)
    cc_mask = state_codes == 3
    baseline_cc = np.mean(omega_a[cc_mask])

    # Baseline 2: All composites at 6k-1 positions (unconditional on pair state)
    # A composite at 6k-1 regardless of what 6k+1 is
    a_composite = ~prime_flags[a_vals]
    baseline_all_a = np.mean(omega_a[a_composite])

    # Baseline 3: All composites at 6k+1 positions
    b_composite = ~prime_flags[b_vals]
    baseline_all_b = np.mean(omega_b[b_composite])

    # PC composite mean
    pc_mask = state_codes == 1
    mean_pc_b = np.mean(omega_b[pc_mask])

    # CP composite mean
    cp_mask = state_codes == 2
    mean_cp_a = np.mean(omega_a[cp_mask])

    print("Mean omega for different populations:")
    print()
    print(f"  PC composite (b):        {mean_pc_b:.6f}  (n={np.sum(pc_mask):,})")
    print(f"  CP composite (a):        {mean_cp_a:.6f}  (n={np.sum(cp_mask):,})")
    print(f"  CC composite (a or b):   {baseline_cc:.6f}  (n={np.sum(cc_mask):,})")
    print(f"  All composites at 6k-1:  {baseline_all_a:.6f}  (n={np.sum(a_composite):,})")
    print(f"  All composites at 6k+1:  {baseline_all_b:.6f}  (n={np.sum(b_composite):,})")
    print()

    print("Selection bias against different baselines:")
    print()

    bias_vs_cc = 100 * (mean_pc_b - baseline_cc) / baseline_cc
    bias_vs_all_b = 100 * (mean_pc_b - baseline_all_b) / baseline_all_b

    print(f"  PC vs CC:              {bias_vs_cc:.4f}%")
    print(f"  PC vs all composites:  {bias_vs_all_b:.4f}%")
    print()

    baseline_results = {
        'mean_pc_composite': mean_pc_b,
        'mean_cp_composite': mean_cp_a,
        'mean_cc_composite': baseline_cc,
        'mean_all_composites_a': baseline_all_a,
        'mean_all_composites_b': baseline_all_b,
        'bias_pc_vs_cc': bias_vs_cc,
        'bias_pc_vs_unconditional': bias_vs_all_b,
        'n_pc': int(np.sum(pc_mask)),
        'n_cp': int(np.sum(cp_mask)),
        'n_cc': int(np.sum(cc_mask)),
        'n_composites_a': int(np.sum(a_composite)),
        'n_composites_b': int(np.sum(b_composite)),
    }

    # ================================================================
    # Analysis 3: Decomposition of CC baseline
    # ================================================================
    print("-" * 70)
    print("ANALYSIS 3: Why CC differs from unconditional")
    print("-" * 70)
    print()

    # CC is conditioned on "other side also composite"
    # This is itself a non-trivial condition

    # Compare: composites at 6k+1 where 6k-1 is also composite (= CC)
    #      vs: composites at 6k+1 where 6k-1 is prime (= PC's b)
    #      vs: all composites at 6k+1

    print("Decomposition of 6k+1 composites:")
    print()
    print(f"  When 6k-1 is prime (PC):     omega = {mean_pc_b:.6f}")
    print(f"  When 6k-1 is composite (CC): omega = {baseline_cc:.6f}")
    print(f"  Unconditional:               omega = {baseline_all_b:.6f}")
    print()

    # The unconditional should be a mixture:
    # P(6k-1 prime) * E[omega | PC] + P(6k-1 composite) * E[omega | CC-like]
    frac_a_prime = np.mean(prime_flags[a_vals])
    frac_a_composite = 1 - frac_a_prime

    expected_mixture = frac_a_prime * mean_pc_b + frac_a_composite * baseline_cc
    print(f"  Mixture prediction: {frac_a_prime:.4f} * {mean_pc_b:.4f} + {frac_a_composite:.4f} * {baseline_cc:.4f}")
    print(f"                    = {expected_mixture:.6f}")
    print(f"  Actual unconditional: {baseline_all_b:.6f}")
    print(f"  Difference: {baseline_all_b - expected_mixture:.6f}")
    print()

    # ================================================================
    # Save results
    # ================================================================
    output_dir.mkdir(parents=True, exist_ok=True)

    df_windows = pd.DataFrame(window_results)
    df_windows.to_csv(output_dir / 'window_stability.csv', index=False)

    df_logbins = pd.DataFrame(log_bins)
    df_logbins.to_csv(output_dir / 'logarithmic_bins.csv', index=False)

    df_baselines = pd.DataFrame([baseline_results])
    df_baselines.to_csv(output_dir / 'baseline_comparisons.csv', index=False)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Window stability:")
    for r in window_results:
        print(f"  {r['window']}: {r['bias_pct']:.4f}%")
    print()
    print("Baseline comparisons:")
    print(f"  PC vs CC:              {bias_vs_cc:.4f}%")
    print(f"  PC vs unconditional:   {bias_vs_all_b:.4f}%")
    print()
    print(f"Results saved to {output_dir.absolute()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stability analysis for selection bias')
    parser.add_argument('--K', type=float, default=1e8, help='Number of pairs')
    parser.add_argument('--output', type=str, default='data/stability', help='Output directory')
    args = parser.parse_args()

    K = int(args.K)
    output_dir = Path(args.output)

    total_start = time.time()
    run_window_analysis(K, output_dir)
    print(f"\nTotal runtime: {time.time() - total_start:.1f}s")
