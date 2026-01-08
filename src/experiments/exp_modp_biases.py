"""
Experiment: Mod p Biases in Twin Prime Gaussian Factorizations

Tests whether the Gaussian factorization (a, b) of twin-participating good primes
shows biases mod p for small primes p, and compares to Hardy-Littlewood predictions.

The three-body view predicts: constraints on the inert singlet (bad prime q = p±2)
propagate to constrain the geometry of the Gaussian prime pair (a+bi, a-bi).

For each small prime q, twin primes avoid p ≡ -2 (mod q) [so that p+2 isn't divisible by q].
This constraint on p = a² + b² creates biases in (a mod q, b mod q).

Hardy-Littlewood prediction: The bias should match the local density factor
    ν_q = (q-2)/q × q²/(q-1)² for the twin prime singular series.
"""

import numpy as np
from numba import njit, prange, cuda
import time
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

# Conditional imports
try:
    HAS_GPU = cuda.is_available()
except:
    HAS_GPU = False

if HAS_GPU:
    from src.gpu_wheel_sieve import WheelGPUContext
from src.wheel_sieve import wheel_spf_sieve


# ========== Cornacchia's Algorithm (from exp_threebody_geometry.py) ==========

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
    """Find r such that r² ≡ n (mod p). Returns 0 if no solution."""
    if mod_pow(n, (p - 1) // 2, p) != 1:
        return 0

    Q = p - 1
    S = 0
    while Q % 2 == 0:
        Q //= 2
        S += 1

    if S == 1:
        return mod_pow(n, (p + 1) // 4, p)

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
        i = 1
        temp = (t * t) % p
        while temp != 1:
            temp = (temp * temp) % p
            i += 1
        b = mod_pow(c, 1 << (M - i - 1), p)
        M = i
        c = (b * b) % p
        t = (t * c) % p
        R = (R * b) % p


@njit
def cornacchia(p: int) -> Tuple[int, int]:
    """Find (a, b) such that a² + b² = p. Returns (larger, smaller)."""
    if p == 2:
        return (1, 1)
    if p % 4 != 1:
        return (0, 0)

    r = tonelli_shanks(p - 1, p)
    if r == 0:
        return (0, 0)
    if r < p // 2:
        r = p - r

    sqrt_p = int(np.sqrt(p)) + 1
    a, b = p, r
    while b >= sqrt_p:
        a, b = b, a % b

    c_sq = p - b * b
    c = int(np.sqrt(c_sq))
    if c * c != c_sq:
        return (0, 0)

    return (max(b, c), min(b, c))


@njit(parallel=True)
def batch_cornacchia(primes: np.ndarray) -> np.ndarray:
    """Compute Gaussian factorization for array of good primes."""
    n = len(primes)
    result = np.zeros((n, 2), dtype=np.int64)
    for i in prange(n):
        a, b = cornacchia(primes[i])
        result[i, 0] = a
        result[i, 1] = b
    return result


# ========== Hardy-Littlewood Predictions ==========

def hardy_littlewood_local_factor(q: int) -> float:
    """
    Compute the Hardy-Littlewood local factor for prime q.

    ν_q = q(q-2)/(q-1)² for q ≥ 3

    This represents the "density adjustment" for twin primes mod q.
    """
    if q < 3:
        return 1.0
    return q * (q - 2) / (q - 1) ** 2


def compute_expected_avoidance(q: int) -> Dict:
    """
    Compute which (a mod q, b mod q) pairs are avoided by twin primes.

    For twin (p, p+2) where p is good:
    - p + 2 must not be divisible by q
    - So p ≢ -2 ≡ q-2 (mod q)
    - For p = a² + b², this constrains which (a², b²) sums to q-2 mod q
    """
    # Quadratic residues mod q
    qr = set()
    for x in range(q):
        qr.add((x * x) % q)

    # Find which (a mod q, b mod q) pairs give p ≡ q-2 (mod q)
    # These are AVOIDED by twin primes
    avoided_target = (q - 2) % q
    avoided_pairs = []

    for a in range(q):
        for b in range(q):
            if (a * a + b * b) % q == avoided_target:
                avoided_pairs.append((a, b))

    # All pairs
    all_pairs = [(a, b) for a in range(q) for b in range(q)]

    # Allowed pairs
    allowed_pairs = [p for p in all_pairs if p not in avoided_pairs]

    return {
        "q": q,
        "avoided_residue": avoided_target,
        "avoided_pairs": avoided_pairs,
        "n_avoided": len(avoided_pairs),
        "n_allowed": len(allowed_pairs),
        "avoidance_fraction": len(avoided_pairs) / (q * q),
        "hardy_littlewood_factor": hardy_littlewood_local_factor(q),
    }


def compute_a_mod_q_expected(q: int) -> np.ndarray:
    """
    Compute expected distribution of (a mod q) for twin good primes,
    assuming uniform distribution over allowed (a, b) pairs.
    """
    info = compute_expected_avoidance(q)

    # Count how many allowed pairs have each value of a mod q
    a_counts = np.zeros(q)
    for a, b in [(a, b) for a in range(q) for b in range(q)]:
        if (a, b) not in info["avoided_pairs"]:
            a_counts[a] += 1

    # Normalize
    return a_counts / a_counts.sum()


# ========== Main Experiment ==========

def run_experiment(K: int, max_prime: int = 50, n_sample: int = 100000, output_dir: str = None):
    """
    Run the mod p bias experiment.

    Parameters
    ----------
    K : int
        Number of pairs to analyze
    max_prime : int
        Test biases for all primes up to this value
    n_sample : int
        Number of primes to sample from each group
    output_dir : str
        Directory to save results
    """
    print("=" * 70)
    print("MOD P BIAS EXPERIMENT")
    print(f"K = {K:,}, max_prime = {max_prime}, sample = {n_sample:,}")
    print("=" * 70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = f"data/reference/modp_biases_K{K:.0e}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Get list of primes to test
    def sieve_primes(n):
        is_prime = np.ones(n + 1, dtype=bool)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = False
        return np.where(is_prime)[0]

    test_primes = sieve_primes(max_prime)
    print(f"Testing {len(test_primes)} primes: {list(test_primes)}")

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

    # ===== Phase 2: Extract good primes =====
    print(f"\n[Phase 2] Extracting good primes...")
    t0 = time.time()

    # Twin primes
    pp_indices = np.where(states == 0)[0]
    k_twins = pp_indices + 1
    a_twins = 6 * k_twins - 1
    b_twins = 6 * k_twins + 1

    # Good primes from twins
    good_bad_mask = (a_twins % 4 == 1)
    twin_good = np.concatenate([a_twins[good_bad_mask], b_twins[~good_bad_mask]])

    # Non-twin good primes
    pc_indices = np.where(states == 1)[0]
    cp_indices = np.where(states == 2)[0]
    k_pc, k_cp = pc_indices + 1, cp_indices + 1
    a_pc, b_cp = 6 * k_pc - 1, 6 * k_cp + 1
    nontwin_good = np.concatenate([a_pc[a_pc % 4 == 1], b_cp[b_cp % 4 == 1]])

    print(f"    Twin good primes: {len(twin_good):,}")
    print(f"    Non-twin good primes: {len(nontwin_good):,}")

    t1 = time.time()
    print(f"    Done in {t1-t0:.2f}s")

    # ===== Phase 3: Sample and compute Gaussian factorizations =====
    print(f"\n[Phase 3] Computing Gaussian factorizations...")
    t0 = time.time()

    np.random.seed(42)
    n_twin = min(n_sample, len(twin_good))
    n_nontwin = min(n_sample, len(nontwin_good))

    twin_sample = twin_good[np.random.choice(len(twin_good), n_twin, replace=False)]
    nontwin_sample = nontwin_good[np.random.choice(len(nontwin_good), n_nontwin, replace=False)]

    twin_factors = batch_cornacchia(twin_sample)
    nontwin_factors = batch_cornacchia(nontwin_sample)

    # Filter valid
    twin_valid = twin_factors[:, 0] > 0
    nontwin_valid = nontwin_factors[:, 0] > 0

    twin_a = twin_factors[twin_valid, 0]
    twin_b = twin_factors[twin_valid, 1]
    nontwin_a = nontwin_factors[nontwin_valid, 0]
    nontwin_b = nontwin_factors[nontwin_valid, 1]

    t1 = time.time()
    print(f"    Twin sample: {len(twin_a):,}, Non-twin sample: {len(nontwin_a):,}")
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
        },
        "primes_tested": list(map(int, test_primes)),
        "biases": {},
    }

    significant_biases = []

    print(f"\n{'Prime':<8} {'HL Factor':<12} {'Twin a≡0':<12} {'NonTwin a≡0':<14} {'Chi² p-val':<14} {'Significant':<12}")
    print("-" * 80)

    for q in test_primes:
        q = int(q)

        # Empirical distributions
        twin_a_mod = twin_a % q
        twin_b_mod = twin_b % q
        nontwin_a_mod = nontwin_a % q
        nontwin_b_mod = nontwin_b % q

        # Counts
        twin_a_counts = np.bincount(twin_a_mod, minlength=q)
        nontwin_a_counts = np.bincount(nontwin_a_mod, minlength=q)

        # Distributions
        twin_a_dist = twin_a_counts / twin_a_counts.sum()
        nontwin_a_dist = nontwin_a_counts / nontwin_a_counts.sum()

        # Hardy-Littlewood prediction
        hl_info = compute_expected_avoidance(q)
        hl_expected = compute_a_mod_q_expected(q)

        # Chi-square test: twin vs non-twin
        # Normalize counts to same total for comparison
        total = twin_a_counts.sum()
        expected_from_nontwin = nontwin_a_dist * total
        chi2_result = stats.chisquare(twin_a_counts, f_exp=expected_from_nontwin)

        # KS test
        ks_result = stats.ks_2samp(twin_a_mod, nontwin_a_mod)

        # Store results
        results["biases"][str(q)] = {
            "hardy_littlewood_factor": hl_info["hardy_littlewood_factor"],
            "avoided_residue": hl_info["avoided_residue"],
            "n_avoided_pairs": hl_info["n_avoided"],
            "avoidance_fraction": hl_info["avoidance_fraction"],
            "twin_a_distribution": twin_a_dist.tolist(),
            "nontwin_a_distribution": nontwin_a_dist.tolist(),
            "hl_expected_distribution": hl_expected.tolist(),
            "chi2_twin_vs_nontwin": {
                "statistic": float(chi2_result.statistic),
                "pvalue": float(chi2_result.pvalue),
            },
            "ks_twin_vs_nontwin": {
                "statistic": float(ks_result.statistic),
                "pvalue": float(ks_result.pvalue),
            },
        }

        # Print summary
        is_sig = chi2_result.pvalue < 0.001
        sig_str = "***" if chi2_result.pvalue < 1e-10 else ("**" if chi2_result.pvalue < 1e-5 else ("*" if is_sig else ""))

        print(f"{q:<8} {hl_info['hardy_littlewood_factor']:<12.6f} {twin_a_dist[0]:<12.4f} {nontwin_a_dist[0]:<14.4f} {chi2_result.pvalue:<14.4e} {sig_str:<12}")

        if is_sig:
            significant_biases.append({
                "prime": q,
                "chi2_pvalue": float(chi2_result.pvalue),
                "twin_a0_fraction": float(twin_a_dist[0]),
                "nontwin_a0_fraction": float(nontwin_a_dist[0]),
                "difference": float(twin_a_dist[0] - nontwin_a_dist[0]),
            })

    results["significant_biases"] = significant_biases

    # ===== Phase 5: Summary =====
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)

    print(f"\nSignificant biases found at {len(significant_biases)} primes:")
    for b in significant_biases:
        diff_pct = 100 * b["difference"] / b["nontwin_a0_fraction"] if b["nontwin_a0_fraction"] > 0 else 0
        print(f"  p={b['prime']}: twin has {b['twin_a0_fraction']:.4f} vs {b['nontwin_a0_fraction']:.4f} for a≡0 ({diff_pct:+.1f}%)")

    # Check if biases match Hardy-Littlewood
    print(f"\n{'=' * 70}")
    print("HARDY-LITTLEWOOD COMPARISON")
    print("=" * 70)

    print("\nFor each significant bias, comparing empirical to H-L prediction:")
    print(f"{'Prime':<8} {'Avoided res':<12} {'Expected avoid%':<16} {'Empirical diff':<16} {'Match?':<10}")
    print("-" * 70)

    for q in [b["prime"] for b in significant_biases]:
        bias = results["biases"][str(q)]
        hl_avoid = bias["avoidance_fraction"]

        # The empirical "avoidance" is the difference between twin and non-twin
        twin_dist = np.array(bias["twin_a_distribution"])
        nontwin_dist = np.array(bias["nontwin_a_distribution"])

        # Compute divergence
        kl_div = np.sum(twin_dist * np.log(twin_dist / np.clip(nontwin_dist, 1e-10, None)))

        print(f"{q:<8} {bias['avoided_residue']:<12} {hl_avoid:<16.4f} {kl_div:<16.6f}")

    # ===== Save results =====
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save raw distributions for plotting
    np.savez(os.path.join(output_dir, "distributions.npz"),
             twin_a=twin_a, twin_b=twin_b,
             nontwin_a=nontwin_a, nontwin_b=nontwin_b,
             test_primes=test_primes)

    print(f"\nResults saved to {output_dir}/")

    return results


def main():
    parser = argparse.ArgumentParser(description="Mod p bias experiment")
    parser.add_argument("--K", type=float, default=1e8, help="Number of pairs")
    parser.add_argument("--max-prime", type=int, default=50, help="Test primes up to this value")
    parser.add_argument("--sample", type=int, default=100000, help="Sample size per group")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    run_experiment(int(args.K), args.max_prime, args.sample, args.output)


if __name__ == "__main__":
    main()
