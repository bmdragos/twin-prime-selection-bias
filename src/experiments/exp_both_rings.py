"""
Experiment: Primes by Splitting Behavior in Both Z[i] and Z[ω]

All primes p > 3 fall into exactly one of 4 classes based on p mod 12:

  p ≡ 1 (mod 12):  Splits in BOTH Z[i] and Z[ω]     ("doubly split")
  p ≡ 5 (mod 12):  Splits in Z[i] only              ("Gaussian only")
  p ≡ 7 (mod 12):  Splits in Z[ω] only              ("Eisenstein only")
  p ≡ 11 (mod 12): Splits in NEITHER                ("doubly inert")

Question: Which class creates the strongest biases in twin prime factorizations?

Hypothesis: "Doubly split" primes (≡1 mod 12) might be special since they
factor in both rings simultaneously.
"""

import numpy as np
from numba import njit, prange
import time
import json
import os
from datetime import datetime
from typing import Tuple, Dict, List

try:
    from numba import cuda
    HAS_GPU = cuda.is_available()
except:
    HAS_GPU = False

if HAS_GPU:
    from src.gpu_wheel_sieve import WheelGPUContext
from src.wheel_sieve import wheel_spf_sieve


# ========== Factorization algorithms ==========

@njit
def mod_pow(base: int, exp: int, mod: int) -> int:
    """Fast modular exponentiation for numba."""
    result = 1
    base = base % mod
    while exp > 0:
        if exp & 1:
            result = (result * base) % mod
        exp >>= 1
        base = (base * base) % mod
    return result


@njit
def cornacchia(p: int) -> Tuple[int, int]:
    """O(log p) algorithm to find a² + b² = p for p ≡ 1 (mod 4)."""
    if p == 2:
        return (1, 1)
    if p % 4 != 1:
        return (0, 0)

    # Find square root of -1 mod p
    # For p ≡ 1 (mod 4), we can use: r = g^((p-1)/4) where g is a quadratic non-residue
    r = 0
    for a in range(2, p):
        # Check if a is a quadratic non-residue
        if mod_pow(a, (p-1)//2, p) == p - 1:
            r = mod_pow(a, (p-1)//4, p)
            break

    if r == 0:
        return (0, 0)

    # Euclidean algorithm descent
    a, b = p, r
    limit = int(np.sqrt(p))
    while b > limit:
        a, b = b, a % b

    c_sq = p - b * b
    c = int(np.sqrt(c_sq))
    if c * c == c_sq:
        return (max(b, c), min(b, c))

    return (0, 0)


@njit
def eisenstein_sqrt(p: int) -> Tuple[int, int]:
    """O(√p) algorithm to find a² - ab + b² = p for p ≡ 1 (mod 3)."""
    if p == 3:
        return (1, 1)
    if p % 3 != 1:
        return (0, 0)

    limit = int(np.sqrt(4 * p / 3)) + 2

    for a in range(1, limit + 1):
        discriminant = 4 * p - 3 * a * a
        if discriminant < 0:
            break

        sqrt_disc = int(np.sqrt(discriminant))
        if sqrt_disc * sqrt_disc != discriminant:
            if (sqrt_disc + 1) * (sqrt_disc + 1) == discriminant:
                sqrt_disc += 1
            else:
                continue

        for b_num in [a + sqrt_disc, a - sqrt_disc]:
            if b_num >= 0 and b_num % 2 == 0:
                b = b_num // 2
                if b >= 0 and a * a - a * b + b * b == p:
                    return (max(a, b), min(a, b))

    return (0, 0)


@njit(parallel=True)
def batch_cornacchia(primes: np.ndarray) -> np.ndarray:
    """Compute Gaussian factorization for array of primes."""
    n = len(primes)
    result = np.zeros((n, 2), dtype=np.int64)
    for i in prange(n):
        a, b = cornacchia(primes[i])
        result[i, 0] = a
        result[i, 1] = b
    return result


@njit(parallel=True)
def batch_eisenstein(primes: np.ndarray) -> np.ndarray:
    """Compute Eisenstein factorization for array of primes."""
    n = len(primes)
    result = np.zeros((n, 2), dtype=np.int64)
    for i in prange(n):
        a, b = eisenstein_sqrt(primes[i])
        result[i, 0] = a
        result[i, 1] = b
    return result


# ========== Prime classification ==========

def classify_prime(p: int) -> str:
    """Classify prime by mod 12 residue."""
    r = p % 12
    if r == 1:
        return "both_split"    # Splits in Z[i] AND Z[ω]
    elif r == 5:
        return "gaussian_only"  # Splits in Z[i] only
    elif r == 7:
        return "eisenstein_only"  # Splits in Z[ω] only
    elif r == 11:
        return "both_inert"     # Inert in both
    else:
        return "special"  # 2, 3 - ramified


def get_primes_up_to(n: int) -> np.ndarray:
    """Simple sieve of Eratosthenes."""
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return np.where(is_prime)[0]


# ========== Main experiment ==========

def run_experiment(K: int, max_prime: int = 100, n_sample: int = 100000, output_dir: str = None):
    """
    Analyze biases by prime class (mod 12).
    """
    print("=" * 70)
    print("BOTH RINGS ANALYSIS: Primes by Splitting Behavior")
    print(f"K = {K:,}, max_prime = {max_prime}, sample = {n_sample:,}")
    print("=" * 70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = f"data/reference/both_rings_K{K:.0e}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Get test primes and classify them
    test_primes = [int(p) for p in get_primes_up_to(max_prime) if p > 3]

    classes = {
        "both_split": [],      # p ≡ 1 (mod 12)
        "gaussian_only": [],   # p ≡ 5 (mod 12)
        "eisenstein_only": [], # p ≡ 7 (mod 12)
        "both_inert": [],      # p ≡ 11 (mod 12)
    }

    for p in test_primes:
        cls = classify_prime(p)
        if cls in classes:
            classes[cls].append(p)

    print("\nPrime Classification (mod 12):")
    print("-" * 60)
    print(f"Both split (≡1 mod 12):      {classes['both_split']}")
    print(f"Gaussian only (≡5 mod 12):   {classes['gaussian_only']}")
    print(f"Eisenstein only (≡7 mod 12): {classes['eisenstein_only']}")
    print(f"Both inert (≡11 mod 12):     {classes['both_inert']}")

    # ===== Phase 1: GPU - Compute states =====
    print(f"\n[Phase 1] GPU: Computing pair states for K={K:,}...")
    t0 = time.time()

    spf_wheel = wheel_spf_sieve(K)
    ctx = WheelGPUContext(K, spf_wheel)
    states = ctx.compute_states()

    t1 = time.time()
    print(f"    Done in {t1-t0:.2f}s")

    state_counts = np.bincount(states, minlength=4)
    print(f"    PP (twins): {state_counts[0]:,}")
    print(f"    PC: {state_counts[1]:,}, CP: {state_counts[2]:,}, CC: {state_counts[3]:,}")

    # ===== Phase 2: Extract twin and non-twin primes =====
    print(f"\n[Phase 2] Extracting primes...")

    pp_indices = np.where(states == 0)[0]
    k_twins = pp_indices + 1

    # For Gaussian analysis: use 6k-1 where it's ≡ 1 (mod 4)
    # For Eisenstein analysis: use 6k+1 where it's ≡ 1 (mod 3)

    # Twin primes: 6k-1 (for Gaussian when k odd), 6k+1 (for Eisenstein, always)
    twin_a = 6 * k_twins - 1  # These are primes
    twin_b = 6 * k_twins + 1  # These are primes

    # Non-twin primes from CP pairs (a composite, b prime)
    cp_indices = np.where(states == 2)[0]
    k_cp = cp_indices + 1
    nontwin_b = 6 * k_cp + 1  # These are non-twin primes

    # Non-twin primes from PC pairs (a prime, b composite)
    pc_indices = np.where(states == 1)[0]
    k_pc = pc_indices + 1
    nontwin_a = 6 * k_pc - 1  # These are non-twin primes

    print(f"    Twin primes (6k-1): {len(twin_a):,}")
    print(f"    Twin primes (6k+1): {len(twin_b):,}")
    print(f"    Non-twin primes (6k-1): {len(nontwin_a):,}")
    print(f"    Non-twin primes (6k+1): {len(nontwin_b):,}")

    # ===== Phase 3: Compute factorizations =====
    print(f"\n[Phase 3] Computing factorizations...")
    t0 = time.time()

    np.random.seed(42)

    # Sample for analysis
    n_twin = min(n_sample, len(twin_b))
    n_nontwin = min(n_sample, len(nontwin_b))

    # For Gaussian: need primes ≡ 1 (mod 4)
    # 6k-1 ≡ 1 (mod 4) when k is odd
    # 6k+1 ≡ 1 (mod 4) when k is even

    twin_gaussian = twin_b[twin_b % 4 == 1]  # 6k+1 ≡ 1 mod 4
    nontwin_gaussian = nontwin_b[nontwin_b % 4 == 1]

    # For Eisenstein: need primes ≡ 1 (mod 3)
    # 6k+1 ≡ 1 (mod 3) always
    twin_eisenstein = twin_b  # All 6k+1 are ≡ 1 mod 3
    nontwin_eisenstein = nontwin_b

    print(f"    Gaussian-compatible twins: {len(twin_gaussian):,}")
    print(f"    Eisenstein-compatible twins: {len(twin_eisenstein):,}")

    # Sample
    n_gauss = min(n_sample, len(twin_gaussian), len(nontwin_gaussian))
    n_eisen = min(n_sample, len(twin_eisenstein), len(nontwin_eisenstein))

    twin_g_sample = twin_gaussian[np.random.choice(len(twin_gaussian), n_gauss, replace=False)]
    nontwin_g_sample = nontwin_gaussian[np.random.choice(len(nontwin_gaussian), n_gauss, replace=False)]

    twin_e_sample = twin_eisenstein[np.random.choice(len(twin_eisenstein), n_eisen, replace=False)]
    nontwin_e_sample = nontwin_eisenstein[np.random.choice(len(nontwin_eisenstein), n_eisen, replace=False)]

    print(f"    Computing {n_gauss:,} Gaussian factorizations per group...")
    twin_g_factors = batch_cornacchia(twin_g_sample)
    nontwin_g_factors = batch_cornacchia(nontwin_g_sample)

    print(f"    Computing {n_eisen:,} Eisenstein factorizations per group...")
    twin_e_factors = batch_eisenstein(twin_e_sample)
    nontwin_e_factors = batch_eisenstein(nontwin_e_sample)

    t1 = time.time()
    print(f"    Done in {t1-t0:.2f}s")

    # Filter valid factorizations
    twin_g_valid = twin_g_factors[:, 0] > 0
    nontwin_g_valid = nontwin_g_factors[:, 0] > 0
    twin_e_valid = twin_e_factors[:, 0] > 0
    nontwin_e_valid = nontwin_e_factors[:, 0] > 0

    twin_g_a = twin_g_factors[twin_g_valid, 0]
    nontwin_g_a = nontwin_g_factors[nontwin_g_valid, 0]
    twin_e_a = twin_e_factors[twin_e_valid, 0]
    nontwin_e_a = nontwin_e_factors[nontwin_e_valid, 0]

    print(f"    Valid: Gaussian twin={len(twin_g_a):,}, nontwin={len(nontwin_g_a):,}")
    print(f"    Valid: Eisenstein twin={len(twin_e_a):,}, nontwin={len(nontwin_e_a):,}")

    # ===== Phase 4: Compute biases by prime class =====
    print(f"\n[Phase 4] Computing biases by prime class...")

    from scipy import stats

    results = {
        "metadata": {
            "K": K,
            "max_prime": max_prime,
            "n_sample": n_sample,
            "timestamp": timestamp,
        },
        "classes": {k: [int(p) for p in v] for k, v in classes.items()},
        "biases": {},
    }

    # Header
    print(f"\n{'Prime':<6} {'Class':<18} {'Gauss bias':<12} {'Eisen bias':<12} {'Combined':<12}")
    print("-" * 70)

    class_biases = {cls: {"gaussian": [], "eisenstein": [], "combined": []} for cls in classes}

    for q in test_primes:
        cls = classify_prime(q)

        # Gaussian bias (mod q on the 'a' component)
        if len(twin_g_a) > 0 and len(nontwin_g_a) > 0:
            twin_g_mod = twin_g_a % q
            nontwin_g_mod = nontwin_g_a % q
            twin_g_dist = np.bincount(twin_g_mod, minlength=q) / len(twin_g_a)
            nontwin_g_dist = np.bincount(nontwin_g_mod, minlength=q) / len(nontwin_g_a)
            g_rel_diff = abs(twin_g_dist[0] - nontwin_g_dist[0]) / nontwin_g_dist[0] if nontwin_g_dist[0] > 0 else 0
        else:
            g_rel_diff = 0

        # Eisenstein bias
        if len(twin_e_a) > 0 and len(nontwin_e_a) > 0:
            twin_e_mod = twin_e_a % q
            nontwin_e_mod = nontwin_e_a % q
            twin_e_dist = np.bincount(twin_e_mod, minlength=q) / len(twin_e_a)
            nontwin_e_dist = np.bincount(nontwin_e_mod, minlength=q) / len(nontwin_e_a)
            e_rel_diff = abs(twin_e_dist[0] - nontwin_e_dist[0]) / nontwin_e_dist[0] if nontwin_e_dist[0] > 0 else 0
        else:
            e_rel_diff = 0

        combined = (g_rel_diff + e_rel_diff) / 2

        class_biases[cls]["gaussian"].append(g_rel_diff)
        class_biases[cls]["eisenstein"].append(e_rel_diff)
        class_biases[cls]["combined"].append(combined)

        results["biases"][str(q)] = {
            "class": cls,
            "gaussian_bias": float(g_rel_diff),
            "eisenstein_bias": float(e_rel_diff),
            "combined_bias": float(combined),
        }

        print(f"{q:<6} {cls:<18} {100*g_rel_diff:<12.2f}% {100*e_rel_diff:<12.2f}% {100*combined:<12.2f}%")

    # ===== Phase 5: Summary by class =====
    print(f"\n{'=' * 70}")
    print("SUMMARY BY PRIME CLASS")
    print("=" * 70)

    print(f"\n{'Class':<20} {'n':<4} {'Gauss mean':<14} {'Eisen mean':<14} {'Combined':<14}")
    print("-" * 70)

    for cls, primes in classes.items():
        if len(primes) > 0:
            g_mean = np.mean(class_biases[cls]["gaussian"])
            e_mean = np.mean(class_biases[cls]["eisenstein"])
            c_mean = np.mean(class_biases[cls]["combined"])
            print(f"{cls:<20} {len(primes):<4} {100*g_mean:<14.2f}% {100*e_mean:<14.2f}% {100*c_mean:<14.2f}%")

            results[f"{cls}_summary"] = {
                "count": len(primes),
                "gaussian_mean": float(g_mean),
                "eisenstein_mean": float(e_mean),
                "combined_mean": float(c_mean),
            }

    # ===== Phase 6: Statistical test =====
    print(f"\n{'=' * 70}")
    print("STATISTICAL COMPARISON: Both-split vs Others")
    print("=" * 70)

    both_split_combined = class_biases["both_split"]["combined"]
    others_combined = (class_biases["gaussian_only"]["combined"] +
                      class_biases["eisenstein_only"]["combined"] +
                      class_biases["both_inert"]["combined"])

    if len(both_split_combined) > 1 and len(others_combined) > 1:
        t_stat, t_pval = stats.ttest_ind(both_split_combined, others_combined)
        print(f"\nBoth-split mean combined bias: {100*np.mean(both_split_combined):.2f}%")
        print(f"Others mean combined bias:     {100*np.mean(others_combined):.2f}%")
        print(f"t-test p-value: {t_pval:.4e}")

        results["comparison"] = {
            "both_split_mean": float(np.mean(both_split_combined)),
            "others_mean": float(np.mean(others_combined)),
            "t_pvalue": float(t_pval),
        }

    # ===== Phase 7: Detailed table =====
    print(f"\n{'=' * 70}")
    print("DETAILED PRIME TABLE")
    print("=" * 70)
    print(f"\n{'p':<5} {'mod 4':<7} {'mod 3':<7} {'mod 12':<8} {'Z[i]':<8} {'Z[ω]':<10} {'Class':<18}")
    print("-" * 70)

    for p in test_primes:
        mod4 = p % 4
        mod3 = p % 3
        mod12 = p % 12
        zi = "split" if mod4 == 1 else "inert"
        zw = "split" if mod3 == 1 else "inert"
        cls = classify_prime(p)
        print(f"{p:<5} {mod4:<7} {mod3:<7} {mod12:<8} {zi:<8} {zw:<10} {cls:<18}")

    # Save results
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else float(x))

    print(f"\nResults saved to {output_dir}/")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Both rings analysis")
    parser.add_argument("--K", type=float, default=1e8, help="Number of pairs")
    parser.add_argument("--max-prime", type=int, default=100, help="Test primes up to this value")
    parser.add_argument("--sample", type=int, default=100000, help="Sample size per group")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    run_experiment(int(args.K), args.max_prime, args.sample, args.output)


if __name__ == "__main__":
    main()
