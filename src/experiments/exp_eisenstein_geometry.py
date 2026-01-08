"""
Experiment: Eisenstein Integer Geometry of Twin Primes

Analyzes twin primes in Z[ω] where ω = e^{2πi/3} (Eisenstein integers).

Splitting behavior in Z[ω]:
- p ≡ 1 (mod 3): splits as p = ππ̄ where π = a + bω
- p ≡ 2 (mod 3): stays inert
- p = 3: ramifies as 3 = -ω²(1-ω)²

For twin primes (6k-1, 6k+1):
- 6k-1 ≡ 2 (mod 3) → ALWAYS inert in Z[ω]
- 6k+1 ≡ 1 (mod 3) → ALWAYS splits in Z[ω]

This is different from Z[i] where split/inert alternates!

The Eisenstein norm is: N(a + bω) = a² - ab + b²

For p ≡ 1 (mod 3), we find (a, b) such that p = a² - ab + b².

We test: do constraining primes q that split in Z[ω] (q ≡ 1 mod 3) create
larger biases than primes that are inert (q ≡ 2 mod 3)?
"""

import numpy as np
from numba import njit, prange
import time
import json
import os
from datetime import datetime
from typing import Tuple

try:
    from numba import cuda
    HAS_GPU = cuda.is_available()
except:
    HAS_GPU = False

if HAS_GPU:
    from src.gpu_wheel_sieve import WheelGPUContext
from src.wheel_sieve import wheel_spf_sieve


# ========== Eisenstein Factorization ==========

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
def cubic_root_of_unity(p: int) -> int:
    """
    Find ω such that ω³ ≡ 1 (mod p) and ω ≠ 1.
    Requires p ≡ 1 (mod 3).
    """
    if p % 3 != 1:
        return 0

    # Find a generator g of (Z/pZ)*
    # Then ω = g^((p-1)/3)
    for g in range(2, p):
        # Check if g is a primitive root
        if mod_pow(g, (p-1)//2, p) != 1 or mod_pow(g, (p-1)//3, p) == 1:
            continue
        # g is likely a generator, compute cube root of unity
        omega = mod_pow(g, (p-1)//3, p)
        if omega != 1 and mod_pow(omega, 3, p) == 1:
            return omega
    return 0


@njit
def eisenstein_factorization(p: int) -> Tuple[int, int]:
    """
    Find (a, b) such that p = a² - ab + b² (Eisenstein norm).

    Requires p ≡ 1 (mod 3).
    Returns (a, b) with a ≥ b ≥ 0.
    Returns (0, 0) if p is not representable.
    """
    if p == 3:
        return (1, 1)  # 3 = 1 - 1 + 1 (ramified)
    if p % 3 != 1:
        return (0, 0)

    # Find cube root of unity mod p
    omega = cubic_root_of_unity(p)
    if omega == 0:
        return (0, 0)

    # Use a descent method similar to Cornacchia
    # We want a² - ab + b² = p
    # This is equivalent to (2a - b)² + 3b² = 4p
    # So we find r such that r² ≡ -3 (mod p), then descend

    # r² ≡ -3 (mod p)
    # -3 = (ω - ω²)² where ω is cube root of unity
    # So r = ω - ω² mod p
    omega_sq = (omega * omega) % p
    r = (omega - omega_sq) % p
    if r < 0:
        r += p

    # Verify r² ≡ -3 (mod p)
    if (r * r) % p != (p - 3) % p:
        # Try the other root
        r = (p - r) % p
        if (r * r) % p != (p - 3) % p:
            return (0, 0)

    # Now use Euclidean algorithm on (p, r) to find the representation
    # Similar to Cornacchia for sum of two squares
    sqrt_4p = int(np.sqrt(4 * p)) + 1

    a, b = p, r
    while b >= sqrt_4p:
        a, b = b, a % b

    # Now b² < 4p, and we need (2a-b)² + 3b² = 4p
    # So 3b² ≤ 4p, meaning b ≤ sqrt(4p/3)
    # Check if 4p - b² is divisible by 3 and a perfect square

    remainder = 4 * p - b * b
    if remainder % 3 != 0:
        return (0, 0)

    x_sq = remainder // 3
    x = int(np.sqrt(x_sq))
    if x * x != x_sq:
        return (0, 0)

    # Now x² + 3*(b/2)² = p... but b might be odd
    # Actually, (2a - b)² + 3b² = 4p means x = 2a - b
    # So a = (x + b) / 2

    if (x + b) % 2 != 0:
        x = -x  # Try negative root
        if (x + b) % 2 != 0:
            return (0, 0)

    a = abs(x + b) // 2
    b = abs(b)

    # Verify
    if a * a - a * b + b * b != p:
        # Try other combinations
        for a_try in range(int(np.sqrt(p)) + 2):
            for b_try in range(int(np.sqrt(p)) + 2):
                if a_try * a_try - a_try * b_try + b_try * b_try == p:
                    return (max(a_try, b_try), min(a_try, b_try))
        return (0, 0)

    return (max(a, b), min(a, b))


@njit
def eisenstein_sqrt(p: int) -> Tuple[int, int]:
    """
    O(√p) algorithm: iterate a, solve for b analytically.

    For a² - ab + b² = p, given a:
        b² - ab + (a² - p) = 0
        b = (a ± √(4p - 3a²)) / 2

    Only valid when discriminant 4p - 3a² ≥ 0, i.e., a ≤ √(4p/3)
    """
    if p == 3:
        return (1, 1)
    if p % 3 != 1:
        return (0, 0)

    # a can range up to √(4p/3) ≈ 1.15√p
    limit = int(np.sqrt(4 * p / 3)) + 2

    for a in range(1, limit + 1):
        discriminant = 4 * p - 3 * a * a
        if discriminant < 0:
            break  # No more valid a values

        sqrt_disc = int(np.sqrt(discriminant))
        # Check if perfect square
        if sqrt_disc * sqrt_disc != discriminant:
            # Try sqrt_disc + 1 in case of floating point error
            if (sqrt_disc + 1) * (sqrt_disc + 1) == discriminant:
                sqrt_disc += 1
            else:
                continue

        # b = (a ± sqrt_disc) / 2
        for b_num in [a + sqrt_disc, a - sqrt_disc]:
            if b_num >= 0 and b_num % 2 == 0:
                b = b_num // 2
                if b >= 0:
                    # Verify solution
                    if a * a - a * b + b * b == p:
                        return (max(a, b), min(a, b))

    return (0, 0)


@njit
def eisenstein_factor(p: int) -> Tuple[int, int]:
    """Use O(√p) method directly - much faster than Cornacchia for large primes."""
    # Skip the slow Cornacchia variant that needs primitive root search
    # eisenstein_sqrt is O(√p) and sufficient for our needs
    return eisenstein_sqrt(p)


@njit(parallel=True)
def batch_eisenstein(primes: np.ndarray) -> np.ndarray:
    """Compute Eisenstein factorization for array of primes."""
    n = len(primes)
    result = np.zeros((n, 2), dtype=np.int64)
    for i in prange(n):
        a, b = eisenstein_factor(primes[i])
        result[i, 0] = a
        result[i, 1] = b
    return result


# ========== Main Experiment ==========

def run_experiment(K: int, max_prime: int = 50, n_sample: int = 100000, output_dir: str = None):
    """
    Run the Eisenstein geometry experiment.

    Tests mod p biases in the Eisenstein factorization of twin primes.
    """
    print("=" * 70)
    print("EISENSTEIN INTEGER GEOMETRY EXPERIMENT")
    print(f"K = {K:,}, max_prime = {max_prime}, sample = {n_sample:,}")
    print("=" * 70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = f"data/reference/eisenstein_K{K:.0e}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Get test primes (excluding 2 and 3)
    def sieve_primes(n):
        is_prime = np.ones(n + 1, dtype=bool)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = False
        return np.where(is_prime)[0]

    test_primes = [p for p in sieve_primes(max_prime) if p > 3]
    print(f"Testing {len(test_primes)} primes: {test_primes}")

    # Classify primes by splitting in Z[ω]
    split_primes = [p for p in test_primes if p % 3 == 1]  # split in Z[ω]
    inert_primes = [p for p in test_primes if p % 3 == 2]  # inert in Z[ω]
    print(f"Split in Z[ω] (≡1 mod 3): {split_primes}")
    print(f"Inert in Z[ω] (≡2 mod 3): {inert_primes}")

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

    # ===== Phase 2: Extract primes that split in Z[ω] =====
    print(f"\n[Phase 2] Extracting primes...")
    t0 = time.time()

    # For twin primes (6k-1, 6k+1):
    # 6k+1 ≡ 1 (mod 3) → always splits in Z[ω]
    # So we analyze the Eisenstein factorization of 6k+1

    pp_indices = np.where(states == 0)[0]
    k_twins = pp_indices + 1
    twin_split = 6 * k_twins + 1  # These split in Z[ω]

    # Non-twin primes that split in Z[ω]
    # From PC pairs: b = 6k+1 is composite, but we need primes
    # From CP pairs: b = 6k+1 is prime and splits in Z[ω]
    cp_indices = np.where(states == 2)[0]
    k_cp = cp_indices + 1
    nontwin_split = 6 * k_cp + 1  # These are prime and split in Z[ω]

    print(f"    Twin primes splitting in Z[ω]: {len(twin_split):,}")
    print(f"    Non-twin primes splitting in Z[ω]: {len(nontwin_split):,}")

    t1 = time.time()
    print(f"    Done in {t1-t0:.2f}s")

    # ===== Phase 3: Compute Eisenstein factorizations =====
    print(f"\n[Phase 3] Computing Eisenstein factorizations...")
    t0 = time.time()

    np.random.seed(42)
    n_twin = min(n_sample, len(twin_split))
    n_nontwin = min(n_sample, len(nontwin_split))

    twin_sample = twin_split[np.random.choice(len(twin_split), n_twin, replace=False)]
    nontwin_sample = nontwin_split[np.random.choice(len(nontwin_split), n_nontwin, replace=False)]

    print(f"    Computing {n_twin:,} twin factorizations...")
    twin_factors = batch_eisenstein(twin_sample)

    print(f"    Computing {n_nontwin:,} non-twin factorizations...")
    nontwin_factors = batch_eisenstein(nontwin_sample)

    # Filter valid
    twin_valid = twin_factors[:, 0] > 0
    nontwin_valid = nontwin_factors[:, 0] > 0

    twin_a = twin_factors[twin_valid, 0]
    twin_b = twin_factors[twin_valid, 1]
    nontwin_a = nontwin_factors[nontwin_valid, 0]
    nontwin_b = nontwin_factors[nontwin_valid, 1]

    t1 = time.time()
    print(f"    Valid: twin={len(twin_a):,}, nontwin={len(nontwin_a):,}")
    print(f"    Done in {t1-t0:.2f}s")

    # ===== Phase 4: Compute mod p biases =====
    print(f"\n[Phase 4] Computing mod p biases...")

    from scipy import stats

    results = {
        "metadata": {
            "K": K,
            "max_prime": max_prime,
            "n_sample": n_sample,
            "n_twin_sample": int(len(twin_a)),
            "n_nontwin_sample": int(len(nontwin_a)),
            "timestamp": timestamp,
            "ring": "Z[ω] (Eisenstein integers)",
        },
        "primes_tested": test_primes,
        "split_primes_mod3": split_primes,
        "inert_primes_mod3": inert_primes,
        "biases": {},
    }

    print(f"\n{'Prime':<8} {'Type':<10} {'Twin a≡0':<12} {'NonTwin a≡0':<14} {'RelDiff%':<12} {'p-value':<14}")
    print("-" * 80)

    bias_data = []

    for q in test_primes:
        q = int(q)
        ptype = "split" if q % 3 == 1 else "inert"

        twin_a_mod = twin_a % q
        nontwin_a_mod = nontwin_a % q

        twin_a_counts = np.bincount(twin_a_mod, minlength=q)
        nontwin_a_counts = np.bincount(nontwin_a_mod, minlength=q)

        twin_a_dist = twin_a_counts / twin_a_counts.sum()
        nontwin_a_dist = nontwin_a_counts / nontwin_a_counts.sum()

        # Chi-square test
        total = twin_a_counts.sum()
        expected = nontwin_a_dist * total
        chi2_stat, chi2_pval = stats.chisquare(twin_a_counts, f_exp=expected)

        rel_diff = (twin_a_dist[0] - nontwin_a_dist[0]) / nontwin_a_dist[0] if nontwin_a_dist[0] > 0 else 0

        results["biases"][str(q)] = {
            "type": ptype,
            "twin_a_distribution": twin_a_dist.tolist(),
            "nontwin_a_distribution": nontwin_a_dist.tolist(),
            "chi2_pvalue": float(chi2_pval),
            "relative_diff_a0": float(rel_diff),
        }

        bias_data.append({
            "q": q,
            "type": ptype,
            "rel_diff": abs(rel_diff),
            "pvalue": chi2_pval,
        })

        sig = "***" if chi2_pval < 1e-10 else ("**" if chi2_pval < 1e-5 else ("*" if chi2_pval < 0.001 else ""))
        print(f"{q:<8} {ptype:<10} {twin_a_dist[0]:<12.4f} {nontwin_a_dist[0]:<14.4f} {100*rel_diff:<+12.1f} {chi2_pval:<14.4e} {sig}")

    # ===== Phase 5: Compare split vs inert =====
    print(f"\n{'=' * 70}")
    print("SPLIT vs INERT COMPARISON (Z[ω])")
    print("=" * 70)

    split_biases = [d["rel_diff"] for d in bias_data if d["type"] == "split"]
    inert_biases = [d["rel_diff"] for d in bias_data if d["type"] == "inert"]

    print(f"\nSplit primes (≡1 mod 3): {[d['q'] for d in bias_data if d['type'] == 'split']}")
    print(f"  Biases: {[f'{100*b:.1f}%' for b in split_biases]}")
    print(f"  Mean absolute bias: {100*np.mean(split_biases):.2f}%")

    print(f"\nInert primes (≡2 mod 3): {[d['q'] for d in bias_data if d['type'] == 'inert']}")
    print(f"  Biases: {[f'{100*b:.1f}%' for b in inert_biases]}")
    print(f"  Mean absolute bias: {100*np.mean(inert_biases):.2f}%")

    if len(split_biases) > 0 and len(inert_biases) > 0:
        ratio = np.mean(split_biases) / np.mean(inert_biases) if np.mean(inert_biases) > 0 else float('inf')
        print(f"\nRatio (split/inert): {ratio:.2f}x")

    results["summary"] = {
        "split_mean_bias": float(np.mean(split_biases)) if split_biases else 0,
        "inert_mean_bias": float(np.mean(inert_biases)) if inert_biases else 0,
        "ratio": float(ratio) if 'ratio' in dir() else 0,
    }

    # ===== Phase 6: Comparison with Z[i] =====
    print(f"\n{'=' * 70}")
    print("COMPARISON: Z[i] vs Z[ω]")
    print("=" * 70)
    print("""
In Z[i] (Gaussian integers):
  - Split: p ≡ 1 (mod 4)
  - Inert: p ≡ 3 (mod 4)
  - Finding: Split primes create ~7x larger biases

In Z[ω] (Eisenstein integers):
  - Split: p ≡ 1 (mod 3)
  - Inert: p ≡ 2 (mod 3)
  - Finding: See above

Note: Some primes split in BOTH rings (p ≡ 1 mod 12),
      some split in neither (p ≡ 11 mod 12).
""")

    # Classify primes by both
    print(f"{'Prime':<8} {'Z[i]':<10} {'Z[ω]':<10} {'Combined':<15}")
    print("-" * 50)
    for q in test_primes:
        zi = "split" if q % 4 == 1 else "inert"
        zw = "split" if q % 3 == 1 else "inert"
        combined = "both split" if (q % 4 == 1 and q % 3 == 1) else \
                   "both inert" if (q % 4 == 3 and q % 3 == 2) else \
                   "mixed"
        print(f"{q:<8} {zi:<10} {zw:<10} {combined:<15}")

    # Save results
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    np.savez(os.path.join(output_dir, "factorizations.npz"),
             twin_a=twin_a, twin_b=twin_b,
             nontwin_a=nontwin_a, nontwin_b=nontwin_b)

    print(f"\nResults saved to {output_dir}/")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Eisenstein geometry experiment")
    parser.add_argument("--K", type=float, default=1e8, help="Number of pairs")
    parser.add_argument("--max-prime", type=int, default=50, help="Test primes up to this value")
    parser.add_argument("--sample", type=int, default=100000, help="Sample size per group")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    run_experiment(int(args.K), args.max_prime, args.sample, args.output)


if __name__ == "__main__":
    main()
