#!/usr/bin/env python3
"""
GPU-accelerated twin prime experiment runner for DGX Spark.

Usage:
    python run_gpu.py              # K=10^8 (default)
    python run_gpu.py --K 1e9      # K=10^9
    python run_gpu.py --K 1e7      # K=10^7 (quick test)
"""

import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path

from src.primes import prime_flags_upto
from src.sieve_pairs import STATES
from src.factorization import spf_sieve
from src.metrics import mean_omega, tilt_ratio
from src.coefficient_extraction import model_mean_omega, model_tilt, state_probabilities

# Import GPU functions
try:
    from src.gpu_factorization import (
        HAS_GPU, HAS_UNIFIED_MEMORY, omega_leq_P_gpu, compute_states_gpu, check_gpu
    )
except ImportError:
    HAS_GPU = False
    HAS_UNIFIED_MEMORY = False


def run_experiment(K: int, P_grid: list, output_dir: Path):
    """Run the full experiment with GPU acceleration."""

    print("=" * 60)
    print("Twin Prime Selection Bias - GPU Accelerated")
    print("=" * 60)
    print(f"K = {K:,}")
    print(f"P_grid = {P_grid}")
    print()

    if HAS_GPU:
        check_gpu()
    else:
        print("WARNING: GPU not available, falling back to CPU")
        print("This will be slow for large K!")
    print()

    N = 6 * K + 1

    # Phase 1: Build sieves (CPU, parallel if available)
    print("-" * 60)
    print("Phase 1: Building sieves")
    print("-" * 60)

    # Try parallel sieve for large N
    use_parallel = N > 10**7
    if use_parallel:
        try:
            from src.parallel_sieve import spf_sieve_parallel, prime_flags_parallel
            print(f"  Using parallel sieve ({N:,} elements)")

            t0 = time.time()
            print("  SPF sieve (parallel)...")
            spf = spf_sieve_parallel(N)
            print(f"    Completed in {time.time() - t0:.1f}s")

            t0 = time.time()
            print("  Prime flags...", end=" ", flush=True)
            # With uint32 SPF: 0 = prime sentinel (spf[n]==0 means n is prime)
            prime_flags = (spf == 0)
            prime_flags[0] = prime_flags[1] = False
            print(f"{time.time() - t0:.1f}s")

        except Exception as e:
            print(f"  Parallel sieve failed ({e}), falling back to sequential")
            use_parallel = False

    if not use_parallel:
        t0 = time.time()
        print("  Prime sieve...", end=" ", flush=True)
        prime_flags = prime_flags_upto(N)
        print(f"{time.time() - t0:.1f}s")

        t0 = time.time()
        print("  SPF sieve...", end=" ", flush=True)
        spf = spf_sieve(N)
        print(f"{time.time() - t0:.1f}s")
    print()

    # Phase 2: Compute states (GPU)
    print("-" * 60)
    print("Phase 2: Computing pair states")
    print("-" * 60)

    t0 = time.time()
    if HAS_GPU:
        state_codes = compute_states_gpu(K, prime_flags)
    else:
        from src.sieve_pairs import compute_all_states
        states_str = compute_all_states(K, prime_flags)
        state_map = {'PP': 0, 'PC': 1, 'CP': 2, 'CC': 3}
        state_codes = np.array([state_map[s] for s in states_str], dtype=np.int32)
    print(f"  States computed in {time.time() - t0:.1f}s")

    # State distribution
    code_to_state = {0: 'PP', 1: 'PC', 2: 'CP', 3: 'CC'}
    print("  Distribution:")
    for code in range(4):
        count = np.sum(state_codes == code)
        print(f"    {code_to_state[code]}: {count:,} ({100*count/K:.2f}%)")
    print()

    # Phase 3: Compute omega for each P (GPU)
    print("-" * 60)
    print("Phase 3: Model vs Empirical comparison")
    print("-" * 60)

    # Pre-generate all a, b values
    k_vals = np.arange(1, K + 1, dtype=np.int64)
    a_vals = 6 * k_vals - 1
    b_vals = 6 * k_vals + 1

    # Create GPU context
    # On unified memory systems (Grace Hopper/Blackwell), GPU can access all system RAM
    # On discrete GPUs, limited by VRAM so fall back to CPU for large K
    if HAS_UNIFIED_MEMORY:
        use_gpu = True
        print("    Unified memory detected - GPU can access full 128GB RAM")
    else:
        use_gpu = HAS_GPU and K < 5 * 10**8

    d_state_codes = None  # GPU state codes for aggregation
    if use_gpu:
        try:
            if HAS_UNIFIED_MEMORY:
                from numba import cuda
                from src.gpu_factorization import UnifiedMemoryGPUContext
                gpu_ctx = UnifiedMemoryGPUContext(a_vals, b_vals, spf)
                # Transfer state codes to GPU once for aggregation
                d_state_codes = cuda.to_device(state_codes)
                print("    Using GPU-side aggregation (avoids 8GB transfers per P)")
            else:
                from src.gpu_factorization import GPUContext
                gpu_ctx = GPUContext(a_vals, b_vals, spf)
        except Exception as e:
            print(f"    GPU init failed ({e}), using CPU-parallel")
            use_gpu = False
            gpu_ctx = None
    else:
        gpu_ctx = None
        if K >= 5 * 10**8 and not HAS_UNIFIED_MEMORY:
            print("    K >= 5×10^8: Using CPU-parallel (SPF too large for discrete GPU)")

    # Pre-compute state counts for aggregated means
    state_counts = {code: np.sum(state_codes == code) for code in range(4)}

    rows = []

    for P in P_grid:
        t0 = time.time()
        print(f"  P = {P}...", end=" ", flush=True)

        # Compute omega (GPU aggregated, GPU full, or CPU-parallel)
        if gpu_ctx is not None and d_state_codes is not None:
            # GPU-side aggregation: returns only 8 sums instead of 8GB arrays
            sums_a, sums_b = gpu_ctx.compute_omega_leq_P_aggregated(P, d_state_codes)
            emp_mean_a = {code: sums_a[code] / state_counts[code] if state_counts[code] > 0 else np.nan
                         for code in range(4)}
            emp_mean_b = {code: sums_b[code] / state_counts[code] if state_counts[code] > 0 else np.nan
                         for code in range(4)}
        elif gpu_ctx is not None:
            # GPU full transfer (discrete GPU)
            omega_a, omega_b = gpu_ctx.compute_omega_leq_P(P)
            emp_mean_a = {code: mean_omega(omega_a[state_codes == code]) if state_counts[code] > 0 else np.nan
                         for code in range(4)}
            emp_mean_b = {code: mean_omega(omega_b[state_codes == code]) if state_counts[code] > 0 else np.nan
                         for code in range(4)}
        else:
            from src.parallel_sieve import omega_leq_P_parallel
            omega_a = omega_leq_P_parallel(a_vals, spf, P)
            omega_b = omega_leq_P_parallel(b_vals, spf, P)
            emp_mean_a = {code: mean_omega(omega_a[state_codes == code]) if state_counts[code] > 0 else np.nan
                         for code in range(4)}
            emp_mean_b = {code: mean_omega(omega_b[state_codes == code]) if state_counts[code] > 0 else np.nan
                         for code in range(4)}

        # Empirical stats by state
        for code, state in code_to_state.items():
            # Model predictions
            probs = state_probabilities(P)

            rows.append({
                'P': P,
                'state': state,
                'emp_count': state_counts[code],
                'emp_mean_omega_a': emp_mean_a[code],
                'emp_mean_omega_b': emp_mean_b[code],
                'mod_probability': probs[state],
                'mod_mean_omega_a': model_mean_omega(P, state, 'a'),
                'mod_mean_omega_b': model_mean_omega(P, state, 'b'),
            })

        print(f"{time.time() - t0:.1f}s")

    # Build results DataFrame
    df = pd.DataFrame(rows)
    df['delta_mean_a'] = df['emp_mean_omega_a'] - df['mod_mean_omega_a']
    df['delta_mean_b'] = df['emp_mean_omega_b'] - df['mod_mean_omega_b']

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / 'model_vs_empirical.csv', index=False)

    # Also save selection bias summary
    # Compute full omega once using GPU context or CPU-parallel
    print("  Computing full omega for selection bias...")
    t0 = time.time()
    if gpu_ctx is not None and d_state_codes is not None:
        # GPU-side aggregation
        sums_a, sums_b = gpu_ctx.compute_omega_aggregated(d_state_codes)
        full_mean_a = {code: sums_a[code] / state_counts[code] if state_counts[code] > 0 else np.nan
                       for code in range(4)}
        full_mean_b = {code: sums_b[code] / state_counts[code] if state_counts[code] > 0 else np.nan
                       for code in range(4)}
    elif gpu_ctx is not None:
        full_omega_a, full_omega_b = gpu_ctx.compute_omega()
        full_mean_a = {code: np.mean(full_omega_a[state_codes == code]) if state_counts[code] > 0 else np.nan
                       for code in range(4)}
        full_mean_b = {code: np.mean(full_omega_b[state_codes == code]) if state_counts[code] > 0 else np.nan
                       for code in range(4)}
    else:
        from src.parallel_sieve import omega_parallel
        full_omega_a = omega_parallel(a_vals, spf)
        full_omega_b = omega_parallel(b_vals, spf)
        full_mean_a = {code: np.mean(full_omega_a[state_codes == code]) if state_counts[code] > 0 else np.nan
                       for code in range(4)}
        full_mean_b = {code: np.mean(full_omega_b[state_codes == code]) if state_counts[code] > 0 else np.nan
                       for code in range(4)}
    print(f"    Completed in {time.time() - t0:.1f}s")

    bias_rows = []
    for code, state in code_to_state.items():
        bias_rows.append({
            'state': state,
            'count': state_counts[code],
            'fraction': state_counts[code] / K,
            'mean_omega_a': full_mean_a[code],
            'mean_omega_b': full_mean_b[code],
        })

    df_bias = pd.DataFrame(bias_rows)
    df_bias.to_csv(output_dir / 'selection_bias_summary.csv', index=False)

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\nSelection Bias Summary:")
    print(df_bias.to_string(index=False))

    print("\nModel vs Empirical (PC composite):")
    pc = df[df['state'] == 'PC'][['P', 'emp_mean_omega_b', 'mod_mean_omega_b', 'delta_mean_b']]
    pc.columns = ['P', 'empirical', 'model', 'delta']
    print(pc.to_string(index=False))

    # Key metric
    pc_omega = df_bias[df_bias['state'] == 'PC']['mean_omega_b'].values[0]
    cc_omega = df_bias[df_bias['state'] == 'CC']['mean_omega_a'].values[0]
    bias_pct = 100 * (pc_omega - cc_omega) / cc_omega

    print(f"\n*** Selection bias: {bias_pct:.3f}% ***")
    print(f"    (PC composite ω = {pc_omega:.4f}, CC composite ω = {cc_omega:.4f})")

    print(f"\nResults saved to {output_dir.absolute()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPU-accelerated twin prime experiment')
    parser.add_argument('--K', type=float, default=1e8,
                        help='Number of pairs (default: 1e8)')
    parser.add_argument('--output', type=str, default='data/results',
                        help='Output directory')
    parser.add_argument('--timestamp', action='store_true',
                        help='Add timestamp subfolder to output')
    args = parser.parse_args()

    K = int(args.K)
    P_grid = [50, 97, 197, 397, 597, 797, 997]

    # Create timestamped output directory
    if args.timestamp:
        from datetime import datetime
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(args.output) / f"run_K{K:.0e}_{ts}"
    else:
        output_dir = Path(args.output)

    total_start = time.time()
    run_experiment(K, P_grid, output_dir)
    print(f"\nTotal runtime: {time.time() - total_start:.1f}s")
