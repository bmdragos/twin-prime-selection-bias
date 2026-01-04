"""
Experiment: Variance and Shape Analysis of ω Distribution

Tests whether conditioning on "a is prime" affects only the mean of ω(b),
or also the variance and shape of the distribution.

Key questions:
1. Does ω remain approximately normal (Erdős-Kac) under conditioning?
2. Is the variance shift consistent with per-prime Bernoulli logic?
3. Do higher moments (skewness, kurtosis) change?

Theoretical background:
- Erdős-Kac: For random integers n ≤ N, (ω(n) - log log N) / √(log log N) → N(0,1)
- Under conditioning, each prime p contributes:
  - Mean: 1/(p-1) instead of ~1/p, shift = 1/[p(p-1)]
  - Variance: Var(Bernoulli(1/(p-1))) vs Var(Bernoulli(1/p))
    = (p-2)/(p-1)² vs (p-1)/p²
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import time
from scipy import stats

from ..primes import prime_flags_upto, primes_upto
from ..factorization import spf_sieve

# GPU support
try:
    from numba import cuda
    import numba
    from ..gpu_factorization import omega_gpu
    HAS_GPU = cuda.is_available()
except ImportError:
    HAS_GPU = False
    omega_gpu = None


def omega_from_spf_vectorized(spf: np.ndarray) -> np.ndarray:
    """
    Compute omega for all integers from SPF array.
    omega[n] = number of distinct prime factors of n.

    Note: Uses spf[p] = p for primes convention (from factorization.spf_sieve).
    """
    N = len(spf) - 1
    omega = np.zeros(N + 1, dtype=np.int32)

    for n in range(2, N + 1):
        if spf[n] == n:  # n is prime (spf[p] = p)
            omega[n] = 1
        else:
            # Count distinct primes by following SPF chain
            count = 0
            m = n
            last_p = 0
            while m > 1 and spf[m] != m:  # while m is composite
                p = spf[m]
                if p != last_p:
                    count += 1
                    last_p = p
                m //= p
            if m > 1:  # m is now a prime (spf[m] == m)
                if m != last_p:
                    count += 1
            omega[n] = count

    return omega


def compute_omega_distributions_cpu(K: int) -> Dict:
    """
    Compute full ω distributions for PC and CC composites.
    Returns histograms and moments.
    """
    print(f"Computing ω distributions (CPU) for K={K:,}")

    N = 6 * K + 1
    print(f"  Generating SPF sieve up to {N:,}...")
    t0 = time.time()
    spf = spf_sieve(N)
    print(f"    Sieve completed in {time.time() - t0:.1f}s")

    print(f"  Computing ω values...")
    t0 = time.time()
    omega = omega_from_spf_vectorized(spf)
    print(f"    ω computed in {time.time() - t0:.1f}s")

    # Build arrays
    print(f"  Classifying pairs...")
    t0 = time.time()
    k_vals = np.arange(1, K + 1, dtype=np.int64)
    a_vals = 6 * k_vals - 1
    b_vals = 6 * k_vals + 1

    # spf[p] = p for primes in factorization.spf_sieve
    a_is_prime = (spf[a_vals] == a_vals)
    b_is_prime = (spf[b_vals] == b_vals)

    # States
    # PC: a prime, b composite
    # CC: a composite, b composite
    pc_mask = a_is_prime & ~b_is_prime
    cc_mask = ~a_is_prime & ~b_is_prime

    omega_pc = omega[b_vals[pc_mask]]
    omega_cc = omega[b_vals[cc_mask]]

    print(f"    Classification done in {time.time() - t0:.1f}s")
    print(f"    PC composites: {len(omega_pc):,}")
    print(f"    CC composites: {len(omega_cc):,}")

    return _analyze_distributions(omega_pc, omega_cc, K, N)


def compute_omega_distributions_gpu(K: int) -> Dict:
    """
    Compute full ω distributions using GPU for omega computation.
    """
    print(f"Computing ω distributions (GPU) for K={K:,}")

    N = 6 * K + 1
    print(f"  Generating SPF sieve up to {N:,}...")
    t0 = time.time()
    spf = spf_sieve(N)
    print(f"    Sieve completed in {time.time() - t0:.1f}s")

    # Build b_vals for PC and CC composites
    print(f"  Classifying pairs and computing ω on GPU...")
    t0 = time.time()
    k_vals = np.arange(1, K + 1, dtype=np.int64)
    a_vals = 6 * k_vals - 1
    b_vals = 6 * k_vals + 1

    # spf[p] = p for primes (not 0) in factorization.spf_sieve
    a_is_prime = (spf[a_vals] == a_vals)
    b_is_prime = (spf[b_vals] == b_vals)

    pc_mask = a_is_prime & ~b_is_prime
    cc_mask = ~a_is_prime & ~b_is_prime

    b_pc = b_vals[pc_mask]
    b_cc = b_vals[cc_mask]
    print(f"    PC composites: {len(b_pc):,}, CC composites: {len(b_cc):,}")

    # Compute omega using GPU
    print(f"  Computing ω on GPU...")
    omega_pc = omega_gpu(b_pc, spf)
    omega_cc = omega_gpu(b_cc, spf)
    print(f"    ω computed in {time.time() - t0:.1f}s")

    return _analyze_distributions(omega_pc, omega_cc, K, N)


def _analyze_distributions(omega_pc: np.ndarray, omega_cc: np.ndarray,
                           K: int, N: int) -> Dict:
    """
    Analyze the ω distributions: compute moments, histograms, normality tests.
    """
    print(f"  Analyzing distributions...")
    t0 = time.time()

    # Basic moments
    mean_pc = np.mean(omega_pc)
    mean_cc = np.mean(omega_cc)
    var_pc = np.var(omega_pc, ddof=1)
    var_cc = np.var(omega_cc, ddof=1)
    std_pc = np.std(omega_pc, ddof=1)
    std_cc = np.std(omega_cc, ddof=1)

    # Higher moments (skewness, kurtosis)
    skew_pc = stats.skew(omega_pc)
    skew_cc = stats.skew(omega_cc)
    kurt_pc = stats.kurtosis(omega_pc)  # excess kurtosis (normal = 0)
    kurt_cc = stats.kurtosis(omega_cc)

    # Erdős-Kac prediction
    log_log_N = np.log(np.log(N))

    # Histograms
    max_omega = max(omega_pc.max(), omega_cc.max())
    bins = np.arange(0, max_omega + 2)  # 0, 1, 2, ..., max+1

    hist_pc, _ = np.histogram(omega_pc, bins=bins)
    hist_cc, _ = np.histogram(omega_cc, bins=bins)

    # Normalize to probability
    prob_pc = hist_pc / len(omega_pc)
    prob_cc = hist_cc / len(omega_cc)

    # Kolmogorov-Smirnov test against normal
    # Standardize using Erdős-Kac parameters
    omega_pc_std = (omega_pc - log_log_N) / np.sqrt(log_log_N)
    omega_cc_std = (omega_cc - log_log_N) / np.sqrt(log_log_N)

    ks_pc = stats.kstest(omega_pc_std, 'norm')
    ks_cc = stats.kstest(omega_cc_std, 'norm')

    # Also test against normal with empirical parameters
    omega_pc_emp_std = (omega_pc - mean_pc) / std_pc
    omega_cc_emp_std = (omega_cc - mean_cc) / std_cc

    ks_pc_emp = stats.kstest(omega_pc_emp_std, 'norm')
    ks_cc_emp = stats.kstest(omega_cc_emp_std, 'norm')

    print(f"    Analysis done in {time.time() - t0:.1f}s")

    # Theoretical predictions
    # Variance shift from per-prime Bernoulli model
    # Var(Bernoulli(q)) = q(1-q)
    # For p >= 5:
    #   Under "a prime": q = 1/(p-1), Var = (p-2)/(p-1)²
    #   Under "a composite": q ≈ 1/p, Var = (p-1)/p²
    #   Difference = (p-2)/(p-1)² - (p-1)/p²
    primes = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    var_shift_predicted = sum(
        (p-2)/(p-1)**2 - (p-1)/p**2
        for p in primes
    )
    # This sum converges; let's compute more terms
    from ..primes import primes_upto
    all_primes = primes_upto(1000)
    var_shift_full = sum(
        (p-2)/(p-1)**2 - (p-1)/p**2
        for p in all_primes if p >= 5
    )

    results = {
        'K': K,
        'N': N,
        'log_log_N': log_log_N,

        # Sample sizes
        'n_pc': len(omega_pc),
        'n_cc': len(omega_cc),

        # Means
        'mean_pc': mean_pc,
        'mean_cc': mean_cc,
        'mean_diff': mean_pc - mean_cc,
        'mean_diff_predicted': 0.0828,  # From paper (PC vs CC)

        # Variances
        'var_pc': var_pc,
        'var_cc': var_cc,
        'var_diff': var_pc - var_cc,
        'var_ratio': var_pc / var_cc,
        'var_shift_predicted': var_shift_full,

        # Standard deviations
        'std_pc': std_pc,
        'std_cc': std_cc,

        # Higher moments
        'skew_pc': skew_pc,
        'skew_cc': skew_cc,
        'kurt_pc': kurt_pc,
        'kurt_cc': kurt_cc,

        # Erdős-Kac comparison
        'erdos_kac_mean': log_log_N,
        'erdos_kac_var': log_log_N,

        # KS tests (against standard normal with Erdős-Kac params)
        'ks_pc_stat': ks_pc.statistic,
        'ks_pc_pvalue': ks_pc.pvalue,
        'ks_cc_stat': ks_cc.statistic,
        'ks_cc_pvalue': ks_cc.pvalue,

        # KS tests (against standard normal with empirical params)
        'ks_pc_emp_stat': ks_pc_emp.statistic,
        'ks_pc_emp_pvalue': ks_pc_emp.pvalue,
        'ks_cc_emp_stat': ks_cc_emp.statistic,
        'ks_cc_emp_pvalue': ks_cc_emp.pvalue,

        # Histograms
        'bins': bins[:-1].tolist(),  # Left edges
        'prob_pc': prob_pc.tolist(),
        'prob_cc': prob_cc.tolist(),
    }

    return results


def compute_omega_distributions(K: int, force_cpu: bool = False) -> Dict:
    """Compute distributions (auto-selects GPU or CPU)."""
    if HAS_GPU and not force_cpu:
        return compute_omega_distributions_gpu(K)
    else:
        return compute_omega_distributions_cpu(K)


def print_results(results: Dict):
    """Print formatted results."""
    print("\n" + "="*80)
    print("Variance and Shape Analysis")
    print("="*80)
    print(f"K = {results['K']:,}, N = {results['N']:,}")
    print(f"log log N = {results['log_log_N']:.4f}")
    print(f"PC composites: {results['n_pc']:,}")
    print(f"CC composites: {results['n_cc']:,}")
    print()

    print("MEANS:")
    print(f"  PC mean ω: {results['mean_pc']:.4f}")
    print(f"  CC mean ω: {results['mean_cc']:.4f}")
    print(f"  Difference: {results['mean_diff']:.4f} (predicted: {results['mean_diff_predicted']:.4f})")
    print(f"  Erdős-Kac prediction (log log N): {results['erdos_kac_mean']:.4f}")
    print()

    print("VARIANCES:")
    print(f"  PC variance: {results['var_pc']:.4f}")
    print(f"  CC variance: {results['var_cc']:.4f}")
    print(f"  Difference: {results['var_diff']:.4f}")
    print(f"  Ratio (PC/CC): {results['var_ratio']:.4f} ({100*(results['var_ratio']-1):.1f}% higher)")
    print(f"  Predicted shift (Σ per-prime): {results['var_shift_predicted']:.4f}")
    print(f"  Erdős-Kac prediction (log log N): {results['erdos_kac_var']:.4f}")
    print()

    print("HIGHER MOMENTS:")
    print(f"  PC skewness: {results['skew_pc']:.4f}")
    print(f"  CC skewness: {results['skew_cc']:.4f}")
    print(f"  PC excess kurtosis: {results['kurt_pc']:.4f}")
    print(f"  CC excess kurtosis: {results['kurt_cc']:.4f}")
    print(f"  (Normal distribution: skew=0, excess kurtosis=0)")
    print()

    print("NORMALITY TESTS (KS against N(0,1) with Erdős-Kac standardization):")
    print(f"  PC: KS statistic = {results['ks_pc_stat']:.4f}, p-value = {results['ks_pc_pvalue']:.2e}")
    print(f"  CC: KS statistic = {results['ks_cc_stat']:.4f}, p-value = {results['ks_cc_pvalue']:.2e}")
    print()

    print("NORMALITY TESTS (KS against N(0,1) with empirical standardization):")
    print(f"  PC: KS statistic = {results['ks_pc_emp_stat']:.4f}, p-value = {results['ks_pc_emp_pvalue']:.2e}")
    print(f"  CC: KS statistic = {results['ks_cc_emp_stat']:.4f}, p-value = {results['ks_cc_emp_pvalue']:.2e}")


def save_results(results: Dict, output_dir: Path):
    """Save results to files."""
    import json

    ref_dir = output_dir / 'variance_analysis'
    ref_dir.mkdir(parents=True, exist_ok=True)

    # JSON for full results
    json_path = ref_dir / f'variance_K{results["K"]}.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Markdown summary
    md_path = ref_dir / 'summary.md'
    with open(md_path, 'w') as f:
        f.write("# Variance and Shape Analysis\n\n")
        f.write(f"$K = {results['K']:,}$, $N = 6K+1 = {results['N']:,}$\n\n")
        f.write(f"$\\log \\log N = {results['log_log_N']:.4f}$\n\n")

        f.write("## Means\n\n")
        f.write("| Population | Mean $\\omega$ | Sample size |\n")
        f.write("|------------|---------------|-------------|\n")
        f.write(f"| PC composites | {results['mean_pc']:.4f} | {results['n_pc']:,} |\n")
        f.write(f"| CC composites | {results['mean_cc']:.4f} | {results['n_cc']:,} |\n")
        f.write(f"| **Difference** | **{results['mean_diff']:.4f}** | — |\n\n")

        f.write("## Variances\n\n")
        f.write("| Population | Variance | Std Dev |\n")
        f.write("|------------|----------|--------|\n")
        f.write(f"| PC composites | {results['var_pc']:.4f} | {results['std_pc']:.4f} |\n")
        f.write(f"| CC composites | {results['var_cc']:.4f} | {results['std_cc']:.4f} |\n")
        f.write(f"| **Ratio (PC/CC)** | **{results['var_ratio']:.4f}** | — |\n\n")
        f.write(f"PC variance is **{100*(results['var_ratio']-1):.1f}%** higher than CC.\n\n")
        f.write(f"Predicted variance shift from per-prime Bernoulli model: {results['var_shift_predicted']:.4f}\n\n")

        f.write("## Higher Moments\n\n")
        f.write("| Moment | PC | CC | Normal |\n")
        f.write("|--------|----|----|--------|\n")
        f.write(f"| Skewness | {results['skew_pc']:.4f} | {results['skew_cc']:.4f} | 0 |\n")
        f.write(f"| Excess kurtosis | {results['kurt_pc']:.4f} | {results['kurt_cc']:.4f} | 0 |\n\n")

        f.write("## Normality (KS test with empirical standardization)\n\n")
        f.write(f"- PC: KS = {results['ks_pc_emp_stat']:.4f}, p = {results['ks_pc_emp_pvalue']:.2e}\n")
        f.write(f"- CC: KS = {results['ks_cc_emp_stat']:.4f}, p = {results['ks_cc_emp_pvalue']:.2e}\n")

    print(f"\nResults saved to {ref_dir}/")


if __name__ == '__main__':
    import sys

    K = int(float(sys.argv[1])) if len(sys.argv) > 1 else 10_000_000
    force_cpu = '--cpu' in sys.argv

    print(f"GPU available: {HAS_GPU}")

    results = compute_omega_distributions(K, force_cpu=force_cpu)
    print_results(results)

    output_dir = Path('data/reference')
    save_results(results, output_dir)
