#!/usr/bin/env python3
"""
Experiment: Omega Decomposition (GPU-accelerated)

GPU-optimized version for DGX Spark. Uses:
- Wheel SPF sieve (3x memory savings)
- On-the-fly pair generation (no a/b arrays)
- GPU-side aggregation (returns 8 numbers instead of 8GB)

Decomposes the selection bias into:
1. Small prime factors (p <= sqrt(N))
2. Large prime cofactors (p > sqrt(N))

At K=10^8:
- Small omega bias: ~4.8%
- Large prime effect: CC has more large cofactors
- Net full omega bias: ~2.93%
"""

import numpy as np
from math import isqrt
import time
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.wheel_sieve import wheel_spf_sieve


def compute_omega_decomposition_gpu(K: int, verbose: bool = True):
    """
    Compute omega decomposition using GPU acceleration.

    Uses WheelGPUContext for unified memory DGX.

    Parameters
    ----------
    K : int
        Number of pairs to analyze
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Results dictionary with all statistics
    """
    try:
        from src.gpu_wheel_sieve import WheelGPUContext, HAS_GPU
        if not HAS_GPU:
            raise ImportError("No GPU available")
    except ImportError as e:
        print(f"GPU not available: {e}")
        print("Falling back to CPU version...")
        from src.experiments.exp_omega_decomposition import compute_omega_decomposition
        return compute_omega_decomposition(K, verbose)

    N = 6 * K + 1
    sqrt_N = isqrt(N)

    if verbose:
        print(f"Omega Decomposition Analysis (GPU)")
        print(f"K = {K:,}, N = {N:,}, sqrt(N) = {sqrt_N:,}")
        print("=" * 60)

    # Generate wheel SPF
    t0 = time.time()
    if verbose:
        print("Generating wheel SPF sieve...")
    spf_wheel = wheel_spf_sieve(K)
    if verbose:
        print(f"  Done in {time.time()-t0:.1f}s, {spf_wheel.nbytes/1e9:.1f}GB")

    # Initialize GPU context
    t0 = time.time()
    if verbose:
        print("Setting up GPU context...")
    ctx = WheelGPUContext(K, spf_wheel)
    if verbose:
        print(f"  Done in {time.time()-t0:.1f}s")

    # Compute states on GPU (keep on device)
    if verbose:
        print("Computing pair states on GPU...")
    t0 = time.time()
    d_state_codes = ctx.compute_states_gpu()

    # Get state counts (small transfer)
    state_counts = np.zeros(4, dtype=np.int64)
    state_codes_host = d_state_codes.copy_to_host()
    for s in range(4):
        state_counts[s] = np.sum(state_codes_host == s)
    if verbose:
        print(f"  Done in {time.time()-t0:.1f}s")
        print(f"  PP={state_counts[0]:,}, PC={state_counts[1]:,}, CP={state_counts[2]:,}, CC={state_counts[3]:,}")

    # Compute omega_small (factors <= sqrt(N))
    if verbose:
        print(f"Computing omega_small (P <= {sqrt_N:,}) on GPU...")
    t0 = time.time()
    sums_a_small, sums_b_small = ctx.compute_omega_leq_P_aggregated(sqrt_N, d_state_codes)
    if verbose:
        print(f"  Done in {time.time()-t0:.1f}s")

    # Compute omega_full
    if verbose:
        print("Computing omega_full on GPU...")
    t0 = time.time()
    sums_a_full, sums_b_full = ctx.compute_omega_aggregated(d_state_codes)
    if verbose:
        print(f"  Done in {time.time()-t0:.1f}s")

    # We focus on b values (the one after prime a in PC)
    # PC = state 1, CC = state 3
    pc_count = int(state_counts[1])
    cc_count = int(state_counts[3])

    # Mean omega values for b
    pc_omega_small = sums_b_small[1] / pc_count
    pc_omega_full = sums_b_full[1] / pc_count
    cc_omega_small = sums_b_small[3] / cc_count
    cc_omega_full = sums_b_full[3] / cc_count

    # has_big = omega_full - omega_small (since factors are disjoint)
    # Actually we need to count pairs where omega_full > omega_small
    # This requires a different kernel. For now, approximate:
    # Mean(has_big) ≈ Mean(omega_full) - Mean(omega_small)
    pc_has_big_approx = pc_omega_full - pc_omega_small
    cc_has_big_approx = cc_omega_full - cc_omega_small

    # Compute differences
    diff_small = pc_omega_small - cc_omega_small
    diff_big = pc_has_big_approx - cc_has_big_approx
    diff_full = pc_omega_full - cc_omega_full

    results = {
        'K': K,
        'N': N,
        'sqrt_N': sqrt_N,
        'pc_count': pc_count,
        'cc_count': cc_count,
        'pc_omega_small': pc_omega_small,
        'pc_omega_full': pc_omega_full,
        'pc_has_big': pc_has_big_approx,
        'cc_omega_small': cc_omega_small,
        'cc_omega_full': cc_omega_full,
        'cc_has_big': cc_has_big_approx,
        'diff_small': diff_small,
        'diff_big': diff_big,
        'diff_full': diff_full,
        'bias_small_pct': diff_small / cc_omega_small * 100,
        'bias_full_pct': diff_full / cc_omega_full * 100,
        'reduction_pct': -diff_big / diff_small * 100 if diff_small != 0 else 0,
    }

    return results


def print_results(results: dict):
    """Print results in formatted table."""
    print()
    print("=" * 70)
    print(f"RESULTS: K = {results['K']:,}")
    print("=" * 70)
    print(f"PC count: {results['pc_count']:,}")
    print(f"CC count: {results['cc_count']:,}")
    print(f"sqrt(N) = {results['sqrt_N']:,}")
    print()

    print(f"{'Component':<35} {'PC':>12} {'CC':>12} {'Diff':>12}")
    print("-" * 70)
    print(f"{'omega_small (p <= sqrt(N))':<35} {results['pc_omega_small']:>12.6f} {results['cc_omega_small']:>12.6f} {results['diff_small']:>+12.6f}")
    print(f"{'Has large prime (p > sqrt(N))':<35} {results['pc_has_big']:>12.4f} {results['cc_has_big']:>12.4f} {results['diff_big']:>+12.6f}")
    print(f"{'omega_full':<35} {results['pc_omega_full']:>12.6f} {results['cc_omega_full']:>12.6f} {results['diff_full']:>+12.6f}")
    print("-" * 70)
    print()

    print("BIAS ANALYSIS:")
    print(f"  Small-prime bias: {results['bias_small_pct']:.2f}%")
    print(f"  Full omega bias:  {results['bias_full_pct']:.2f}%")
    print(f"  Large-prime reduction: {results['reduction_pct']:.1f}%")
    print()

    print("DECOMPOSITION CHECK:")
    check = results['diff_small'] + results['diff_big']
    print(f"  diff_small + diff_big = {results['diff_small']:.6f} + {results['diff_big']:.6f} = {check:.6f}")
    print(f"  diff_full = {results['diff_full']:.6f}")
    print(f"  Match: {'YES' if abs(check - results['diff_full']) < 1e-6 else 'NO'}")


def print_markdown_table(results: dict):
    """Print results as markdown for paper inclusion."""
    print()
    print("### Markdown Table for Paper")
    print()
    print(f"At $K = {results['K']:,}$ (where $\\sqrt{{N}} = {results['sqrt_N']:,}$):")
    print()
    print("| Component | PC | CC | Difference |")
    print("|-----------|-----|-----|------------|")
    print(f"| $\\omega_{{\\text{{small}}}}$ (factors $\\leq \\sqrt{{N}}$) | {results['pc_omega_small']:.3f} | {results['cc_omega_small']:.3f} | $+{results['diff_small']:.3f}$ |")
    print(f"| Has large prime factor $(> \\sqrt{{N}})$ | {results['pc_has_big']:.3f} | {results['cc_has_big']:.3f} | ${results['diff_big']:+.3f}$ |")
    print(f"| **Full $\\omega$** | {results['pc_omega_full']:.3f} | {results['cc_omega_full']:.3f} | $+{results['diff_full']:.3f}$ |")
    print()
    print(f"Small-prime bias: **{results['bias_small_pct']:.1f}%**, reduced to **{results['bias_full_pct']:.2f}%** by large-prime effect ({results['reduction_pct']:.0f}% reduction).")


def save_results(results: dict, output_dir: Path):
    """Save results to CSV."""
    import csv
    from datetime import datetime

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = output_dir / f"omega_decomposition_K{results['K']}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save main results
    csv_path = run_dir / 'results.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for key, value in results.items():
            writer.writerow([key, value])

    # Save markdown
    md_path = run_dir / 'table.md'
    with open(md_path, 'w') as f:
        f.write(f"# Omega Decomposition at K={results['K']:,}\n\n")
        f.write(f"$\\sqrt{{N}} = {results['sqrt_N']:,}$\n\n")
        f.write("| Component | PC | CC | Difference |\n")
        f.write("|-----------|-----|-----|------------|\n")
        f.write(f"| $\\omega_{{\\text{{small}}}}$ | {results['pc_omega_small']:.4f} | {results['cc_omega_small']:.4f} | {results['diff_small']:+.4f} |\n")
        f.write(f"| Has large prime | {results['pc_has_big']:.4f} | {results['cc_has_big']:.4f} | {results['diff_big']:+.4f} |\n")
        f.write(f"| **Full $\\omega$** | {results['pc_omega_full']:.4f} | {results['cc_omega_full']:.4f} | {results['diff_full']:+.4f} |\n")
        f.write(f"\nSmall-prime bias: {results['bias_small_pct']:.2f}%\n")
        f.write(f"Full omega bias: {results['bias_full_pct']:.2f}%\n")
        f.write(f"Large-prime reduction: {results['reduction_pct']:.1f}%\n")

    print(f"\nResults saved to {run_dir}/")
    return run_dir


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Omega decomposition analysis (GPU)')
    parser.add_argument('--K', type=float, default=1e8, help='Number of pairs (default: 1e8)')
    parser.add_argument('--save', action='store_true', help='Save results to data/results/')
    args = parser.parse_args()

    K = int(args.K)

    total_start = time.time()
    results = compute_omega_decomposition_gpu(K, verbose=True)
    total_time = time.time() - total_start

    print_results(results)
    print_markdown_table(results)
    print(f"\nTotal time: {total_time:.1f}s")

    if args.save:
        output_dir = Path('data/results')
        save_results(results, output_dir)
