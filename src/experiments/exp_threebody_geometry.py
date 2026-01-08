"""
Experiment: Three-Body Geometry of Twin Primes

Tests whether the Gaussian factorization of good primes differs
between twin-participating and non-twin primes.

Hypothesis: In the Z[i] view, twin primes are (π, π̄, q) triplets:
  - π = a + bi (Gaussian prime)
  - π̄ = a - bi (conjugate)
  - q = bad prime (inert in Z[i])

The "twinness" might correlate with the geometry of π in the complex plane.

Metrics:
1. Aspect ratio: max(a,b)/min(a,b) where p = a² + b²
2. Trace: 2 * max(a,b) (real part contribution)
3. Parity structure: (a mod 2, b mod 2)
4. Residue classes: a mod m, b mod m for small m

Control groups:
- Twin good primes: good primes that are part of a twin pair
- Non-twin good primes: good primes in PC or CP pairs (not twins)
"""

import numpy as np
from numba import njit, prange
import time
import json
import os
from datetime import datetime
from pathlib import Path

# Conditional imports for GPU
try:
    from numba import cuda
    HAS_GPU = cuda.is_available()
except ImportError:
    HAS_GPU = False

if HAS_GPU:
    from src.gpu_wheel_sieve import WheelGPUContext
from src.wheel_sieve import wheel_spf_sieve


# ========== Cornacchia's Algorithm ==========

@njit
def mod_pow(base: int, exp: int, mod: int) -> int:
    """Fast modular exponentiation."""
    result = 1
    base = base % mod
    while exp > 0:
        if exp & 1:
            result = (result * base) % mod
        exp >>= 1
        base = (base * base) % mod
    return result


@njit
def tonelli_shanks(n: int, p: int) -> int:
    """
    Find r such that r² ≡ n (mod p).
    Returns 0 if no solution exists.
    """
    if mod_pow(n, (p - 1) // 2, p) != 1:
        return 0  # n is not a quadratic residue

    # Factor p-1 = Q * 2^S
    Q = p - 1
    S = 0
    while Q % 2 == 0:
        Q //= 2
        S += 1

    if S == 1:
        # p ≡ 3 (mod 4), simple case
        return mod_pow(n, (p + 1) // 4, p)

    # Find quadratic non-residue z
    z = 2
    while mod_pow(z, (p - 1) // 2, p) != p - 1:
        z += 1

    M = S
    c = mod_pow(z, Q, p)
    t = mod_pow(n, Q, p)
    R = mod_pow(n, (Q + 1) // 2, p)

    while True:
        if t == 1:
            return R

        # Find least i such that t^(2^i) = 1
        i = 1
        temp = (t * t) % p
        while temp != 1:
            temp = (temp * temp) % p
            i += 1

        # Update
        b = mod_pow(c, 1 << (M - i - 1), p)
        M = i
        c = (b * b) % p
        t = (t * c) % p
        R = (R * b) % p


@njit
def cornacchia(p: int) -> tuple:
    """
    Find (a, b) such that a² + b² = p using Cornacchia's algorithm.

    Requires p ≡ 1 (mod 4).
    Returns (larger, smaller) for consistency.
    Returns (0, 0) if p is not representable.
    """
    if p == 2:
        return (1, 1)
    if p % 4 != 1:
        return (0, 0)

    # Find r such that r² ≡ -1 (mod p)
    r = tonelli_shanks(p - 1, p)
    if r == 0:
        return (0, 0)

    # Make r > p/2 if needed (doesn't matter, but convention)
    if r < p // 2:
        r = p - r

    # Euclidean algorithm until remainder < √p
    sqrt_p = int(np.sqrt(p)) + 1
    a, b = p, r
    while b >= sqrt_p:
        a, b = b, a % b

    # Now b² < p, check if p - b² is a perfect square
    c_sq = p - b * b
    c = int(np.sqrt(c_sq))
    if c * c != c_sq:
        return (0, 0)

    return (max(b, c), min(b, c))


@njit(parallel=True)
def batch_cornacchia(primes: np.ndarray) -> np.ndarray:
    """
    Compute Gaussian factorization for array of good primes.

    Returns array of shape (n, 2) with (a, b) for each prime.
    """
    n = len(primes)
    result = np.zeros((n, 2), dtype=np.int64)

    for i in prange(n):
        a, b = cornacchia(primes[i])
        result[i, 0] = a
        result[i, 1] = b

    return result


# ========== Main Experiment ==========

def run_experiment(K: int, output_dir: str = None, n_sample: int = 50000):
    """
    Run the three-body geometry experiment.

    Parameters
    ----------
    K : int
        Number of pairs to analyze
    output_dir : str
        Directory to save results
    n_sample : int
        Number of primes to sample from each group
    """
    print(f"=" * 70)
    print(f"THREE-BODY GEOMETRY EXPERIMENT")
    print(f"K = {K:,}, sample size = {n_sample:,}")
    print(f"=" * 70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = f"data/reference/threebody_K{K:.0e}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # ===== Phase 1: GPU - Get twin primes and their types =====
    print("\n[Phase 1] GPU: Computing pair states...")
    t0 = time.time()

    spf_wheel = wheel_spf_sieve(K)
    ctx = WheelGPUContext(K, spf_wheel)

    # Get states (PP=0, PC=1, CP=2, CC=3)
    states = ctx.compute_states()

    t1 = time.time()
    print(f"    Done in {t1-t0:.2f}s")

    # State counts
    state_counts = np.bincount(states, minlength=4)
    print(f"    PP (twins): {state_counts[0]:,}")
    print(f"    PC: {state_counts[1]:,}")
    print(f"    CP: {state_counts[2]:,}")
    print(f"    CC: {state_counts[3]:,}")

    # ===== Phase 2: Extract prime values =====
    print("\n[Phase 2] Extracting prime values...")
    t0 = time.time()

    # Twin primes (PP)
    pp_indices = np.where(states == 0)[0]
    k_twins = pp_indices + 1
    a_twins = 6 * k_twins - 1  # 6k-1
    b_twins = 6 * k_twins + 1  # 6k+1

    # Classify by good/bad
    # a = 6k-1: good if ≡1 (mod 4), i.e., if k is odd
    # b = 6k+1: good if ≡1 (mod 4), i.e., if k is even
    good_bad_mask = (a_twins % 4 == 1)  # a is good, b is bad
    bad_good_mask = ~good_bad_mask       # a is bad, b is good

    # Good primes from twin pairs
    twin_good_from_gb = a_twins[good_bad_mask]  # good prime is a
    twin_good_from_bg = b_twins[bad_good_mask]  # good prime is b
    twin_good_primes = np.concatenate([twin_good_from_gb, twin_good_from_bg])

    print(f"    Twin pairs: {len(k_twins):,}")
    print(f"    (good, bad) pairs: {good_bad_mask.sum():,}")
    print(f"    (bad, good) pairs: {bad_good_mask.sum():,}")

    # Non-twin good primes (control)
    # PC: a=6k-1 is prime, b is composite → if a≡1(mod 4), a is a non-twin good prime
    # CP: b=6k+1 is prime, a is composite → if b≡1(mod 4), b is a non-twin good prime
    pc_indices = np.where(states == 1)[0]
    cp_indices = np.where(states == 2)[0]

    k_pc = pc_indices + 1
    k_cp = cp_indices + 1

    a_pc = 6 * k_pc - 1
    b_cp = 6 * k_cp + 1

    # Filter for good primes
    nontwin_good_from_pc = a_pc[a_pc % 4 == 1]
    nontwin_good_from_cp = b_cp[b_cp % 4 == 1]
    nontwin_good_primes = np.concatenate([nontwin_good_from_pc, nontwin_good_from_cp])

    print(f"    Non-twin good primes: {len(nontwin_good_primes):,}")

    t1 = time.time()
    print(f"    Done in {t1-t0:.2f}s")

    # ===== Phase 3: Sample and compute Gaussian factorizations =====
    print(f"\n[Phase 3] Computing Gaussian factorizations (sample of {n_sample:,})...")
    t0 = time.time()

    np.random.seed(42)

    # Sample twin good primes
    n_twin_sample = min(n_sample, len(twin_good_primes))
    twin_sample_idx = np.random.choice(len(twin_good_primes), n_twin_sample, replace=False)
    twin_sample = twin_good_primes[twin_sample_idx]

    # Sample non-twin good primes
    n_nontwin_sample = min(n_sample, len(nontwin_good_primes))
    nontwin_sample_idx = np.random.choice(len(nontwin_good_primes), n_nontwin_sample, replace=False)
    nontwin_sample = nontwin_good_primes[nontwin_sample_idx]

    print(f"    Twin sample: {n_twin_sample:,}")
    print(f"    Non-twin sample: {n_nontwin_sample:,}")

    # Compute factorizations (parallel CPU)
    print("    Computing Cornacchia factorizations...")
    twin_factors = batch_cornacchia(twin_sample)
    nontwin_factors = batch_cornacchia(nontwin_sample)

    t1 = time.time()
    print(f"    Done in {t1-t0:.2f}s")

    # Verify
    twin_valid = (twin_factors[:, 0] > 0)
    nontwin_valid = (nontwin_factors[:, 0] > 0)
    print(f"    Valid factorizations: twin={twin_valid.sum():,}, nontwin={nontwin_valid.sum():,}")

    # ===== Phase 4: Compute statistics =====
    print("\n[Phase 4] Computing statistics...")

    # Filter to valid factorizations
    twin_a = twin_factors[twin_valid, 0]  # larger component
    twin_b = twin_factors[twin_valid, 1]  # smaller component
    nontwin_a = nontwin_factors[nontwin_valid, 0]
    nontwin_b = nontwin_factors[nontwin_valid, 1]

    # Aspect ratios
    twin_ratio = twin_a / np.maximum(twin_b, 1)
    nontwin_ratio = nontwin_a / np.maximum(nontwin_b, 1)

    # Traces
    twin_trace = 2 * twin_a
    nontwin_trace = 2 * nontwin_a

    # Residue analysis
    def residue_stats(a, b, mod):
        a_counts = np.bincount(a % mod, minlength=mod)
        b_counts = np.bincount(b % mod, minlength=mod)
        return a_counts / len(a), b_counts / len(b)

    results = {
        "metadata": {
            "K": K,
            "n_sample": n_sample,
            "n_twin_primes": len(twin_good_primes),
            "n_nontwin_primes": len(nontwin_good_primes),
            "n_twin_sample": int(twin_valid.sum()),
            "n_nontwin_sample": int(nontwin_valid.sum()),
            "timestamp": timestamp,
        },
        "aspect_ratio": {
            "twin_mean": float(twin_ratio.mean()),
            "twin_median": float(np.median(twin_ratio)),
            "twin_std": float(twin_ratio.std()),
            "nontwin_mean": float(nontwin_ratio.mean()),
            "nontwin_median": float(np.median(nontwin_ratio)),
            "nontwin_std": float(nontwin_ratio.std()),
        },
        "trace": {
            "twin_mean": float(twin_trace.mean()),
            "twin_std": float(twin_trace.std()),
            "nontwin_mean": float(nontwin_trace.mean()),
            "nontwin_std": float(nontwin_trace.std()),
        },
        "larger_component_a": {
            "twin_mean": float(twin_a.mean()),
            "twin_std": float(twin_a.std()),
            "nontwin_mean": float(nontwin_a.mean()),
            "nontwin_std": float(nontwin_a.std()),
        },
        "smaller_component_b": {
            "twin_mean": float(twin_b.mean()),
            "twin_std": float(twin_b.std()),
            "nontwin_mean": float(nontwin_b.mean()),
            "nontwin_std": float(nontwin_b.std()),
        }
    }

    # ===== Phase 5: Statistical tests =====
    print("\n[Phase 5] Statistical tests...")
    from scipy import stats

    # KS tests
    ks_ratio = stats.ks_2samp(twin_ratio, nontwin_ratio)
    ks_trace = stats.ks_2samp(twin_trace, nontwin_trace)
    ks_a = stats.ks_2samp(twin_a.astype(float), nontwin_a.astype(float))
    ks_b = stats.ks_2samp(twin_b.astype(float), nontwin_b.astype(float))

    # Mann-Whitney U tests (non-parametric)
    mw_ratio = stats.mannwhitneyu(twin_ratio, nontwin_ratio, alternative='two-sided')
    mw_a = stats.mannwhitneyu(twin_a, nontwin_a, alternative='two-sided')

    results["statistical_tests"] = {
        "ks_aspect_ratio": {"statistic": float(ks_ratio.statistic), "pvalue": float(ks_ratio.pvalue)},
        "ks_trace": {"statistic": float(ks_trace.statistic), "pvalue": float(ks_trace.pvalue)},
        "ks_larger_a": {"statistic": float(ks_a.statistic), "pvalue": float(ks_a.pvalue)},
        "ks_smaller_b": {"statistic": float(ks_b.statistic), "pvalue": float(ks_b.pvalue)},
        "mannwhitney_ratio": {"statistic": float(mw_ratio.statistic), "pvalue": float(mw_ratio.pvalue)},
        "mannwhitney_a": {"statistic": float(mw_a.statistic), "pvalue": float(mw_a.pvalue)},
    }

    # Residue class analysis (mod 3, 4, 5)
    for mod in [3, 4, 5, 6]:
        twin_a_res, twin_b_res = residue_stats(twin_a, twin_b, mod)
        nontwin_a_res, nontwin_b_res = residue_stats(nontwin_a, nontwin_b, mod)

        # Chi-square test for residue class distribution
        twin_a_counts = np.bincount(twin_a % mod, minlength=mod)
        nontwin_a_counts = np.bincount(nontwin_a % mod, minlength=mod)

        # Normalize to same total
        expected = (twin_a_counts + nontwin_a_counts) / 2
        chi2_a = stats.chisquare(twin_a_counts, f_exp=expected * twin_a_counts.sum() / expected.sum())

        results[f"residue_mod{mod}"] = {
            "twin_a_distribution": twin_a_res.tolist(),
            "twin_b_distribution": twin_b_res.tolist(),
            "nontwin_a_distribution": nontwin_a_res.tolist(),
            "nontwin_b_distribution": nontwin_b_res.tolist(),
            "chi2_a": {"statistic": float(chi2_a.statistic), "pvalue": float(chi2_a.pvalue)},
        }

    # ===== Print summary =====
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'Twin Good':<20} {'Non-Twin Good':<20} {'Diff %':<10}")
    print("-" * 75)

    r = results["aspect_ratio"]
    diff_pct = 100 * (r["twin_mean"] - r["nontwin_mean"]) / r["nontwin_mean"]
    print(f"{'Aspect ratio (mean)':<25} {r['twin_mean']:<20.4f} {r['nontwin_mean']:<20.4f} {diff_pct:+.2f}%")

    r = results["larger_component_a"]
    diff_pct = 100 * (r["twin_mean"] - r["nontwin_mean"]) / r["nontwin_mean"]
    print(f"{'Larger component a':<25} {r['twin_mean']:<20.1f} {r['nontwin_mean']:<20.1f} {diff_pct:+.2f}%")

    r = results["smaller_component_b"]
    diff_pct = 100 * (r["twin_mean"] - r["nontwin_mean"]) / r["nontwin_mean"]
    print(f"{'Smaller component b':<25} {r['twin_mean']:<20.1f} {r['nontwin_mean']:<20.1f} {diff_pct:+.2f}%")

    print(f"\n{'Statistical Test':<30} {'KS Statistic':<15} {'p-value':<15} {'Significant?':<12}")
    print("-" * 72)

    for test_name, test_key in [("Aspect ratio", "ks_aspect_ratio"),
                                 ("Larger component (a)", "ks_larger_a"),
                                 ("Smaller component (b)", "ks_smaller_b")]:
        t = results["statistical_tests"][test_key]
        sig = "YES" if t["pvalue"] < 0.05 else "no"
        print(f"{test_name:<30} {t['statistic']:<15.4f} {t['pvalue']:<15.4e} {sig:<12}")

    # ===== Save results =====
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save raw data for plotting
    np.savez(os.path.join(output_dir, "factorizations.npz"),
             twin_a=twin_a, twin_b=twin_b,
             nontwin_a=nontwin_a, nontwin_b=nontwin_b)

    print(f"\nResults saved to {output_dir}/")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Three-body geometry experiment")
    parser.add_argument("--K", type=float, default=1e7, help="Number of pairs")
    parser.add_argument("--sample", type=int, default=50000, help="Sample size per group")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    run_experiment(int(args.K), args.output, args.sample)


if __name__ == "__main__":
    main()
