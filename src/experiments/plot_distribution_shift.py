"""
Visualize the distribution shift: how conditioning on "a is prime" reshapes ω(b).

Creates:
1. Histogram overlay of ω(b) for PC vs CC composites
2. Difference plot showing the shift at each ω value
3. Per-prime density comparison (1/p vs 1/(p-1))
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import json
import time

from ..primes import prime_flags_upto
from ..factorization import spf_sieve


def compute_omega_from_spf(spf: np.ndarray, n: int) -> int:
    """Count distinct prime factors using SPF array."""
    if n <= 1:
        return 0
    count = 0
    while n > 1:
        p = spf[n]
        count += 1
        while n > 1 and spf[n] == p:
            n //= p
    return count


def compute_distributions(K: int):
    """Compute ω distributions for PC and CC composites."""
    print(f"Computing ω distributions for K={K:,}")

    N = 6 * K + 1
    print(f"  Generating SPF sieve up to {N:,}...")
    t0 = time.time()
    spf = spf_sieve(N)
    print(f"    Completed in {time.time() - t0:.1f}s")

    # Build pair arrays
    k_vals = np.arange(1, K + 1, dtype=np.int64)
    a_vals = 6 * k_vals - 1
    b_vals = 6 * k_vals + 1

    # Classify using SPF (spf[p] = p for primes)
    a_is_prime = (spf[a_vals] == a_vals)
    b_is_prime = (spf[b_vals] == b_vals)

    # PC: a prime, b composite
    pc_mask = a_is_prime & ~b_is_prime
    # CC: both composite
    cc_mask = ~a_is_prime & ~b_is_prime

    pc_b_vals = b_vals[pc_mask]
    cc_b_vals = b_vals[cc_mask]

    print(f"  PC composites: {len(pc_b_vals):,}")
    print(f"  CC composites: {len(cc_b_vals):,}")

    # Compute ω for each
    print(f"  Computing ω for PC composites...")
    t0 = time.time()
    pc_omega = np.array([compute_omega_from_spf(spf, int(b)) for b in pc_b_vals])
    print(f"    Completed in {time.time() - t0:.1f}s")

    print(f"  Computing ω for CC composites...")
    t0 = time.time()
    cc_omega = np.array([compute_omega_from_spf(spf, int(b)) for b in cc_b_vals])
    print(f"    Completed in {time.time() - t0:.1f}s")

    return pc_omega, cc_omega


def plot_distributions(pc_omega, cc_omega, K, output_dir: Path):
    """Create visualization plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up style
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['figure.figsize'] = (10, 6)

    # Count distributions
    pc_counts = Counter(pc_omega)
    cc_counts = Counter(cc_omega)

    max_omega = max(max(pc_counts.keys()), max(cc_counts.keys()))
    omega_vals = np.arange(1, max_omega + 1)

    pc_freq = np.array([pc_counts.get(w, 0) / len(pc_omega) for w in omega_vals])
    cc_freq = np.array([cc_counts.get(w, 0) / len(cc_omega) for w in omega_vals])

    # Stats
    pc_mean = np.mean(pc_omega)
    cc_mean = np.mean(cc_omega)
    pc_std = np.std(pc_omega)
    cc_std = np.std(cc_omega)

    # ========== Plot 1: Histogram overlay ==========
    fig, ax = plt.subplots(figsize=(10, 6))

    width = 0.35
    x = omega_vals

    bars1 = ax.bar(x - width/2, pc_freq, width, label=f'PC (a prime): μ={pc_mean:.3f}, σ={pc_std:.3f}',
                   color='#e74c3c', alpha=0.8, edgecolor='darkred')
    bars2 = ax.bar(x + width/2, cc_freq, width, label=f'CC (both comp): μ={cc_mean:.3f}, σ={cc_std:.3f}',
                   color='#3498db', alpha=0.8, edgecolor='darkblue')

    ax.set_xlabel('ω(b) = number of distinct prime factors')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of ω(b) conditioned on primality of a\n(K={K:,}, twin prime candidates (6k-1, 6k+1))')
    ax.legend(loc='upper right')
    ax.set_xlim(0.5, min(max_omega + 0.5, 10.5))
    ax.set_xticks(range(1, min(max_omega + 1, 11)))
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'omega_distribution_overlay.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'omega_distribution_overlay.pdf', bbox_inches='tight')
    print(f"  Saved: omega_distribution_overlay.png/pdf")
    plt.close()

    # ========== Plot 2: Difference plot ==========
    fig, ax = plt.subplots(figsize=(10, 5))

    diff = pc_freq - cc_freq
    colors = ['#e74c3c' if d > 0 else '#3498db' for d in diff]

    ax.bar(omega_vals, diff, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    ax.set_xlabel('ω(b)')
    ax.set_ylabel('P(ω|PC) - P(ω|CC)')
    ax.set_title('Distribution shift: PC composites have more prime factors\n(red = PC more frequent, blue = CC more frequent)')
    ax.set_xlim(0.5, min(max_omega + 0.5, 10.5))
    ax.set_xticks(range(1, min(max_omega + 1, 11)))
    ax.grid(axis='y', alpha=0.3)

    # Add annotation
    shift = pc_mean - cc_mean
    ax.annotate(f'Mean shift: +{shift:.4f}\n({100*shift/cc_mean:.2f}% increase)',
                xy=(0.98, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'omega_distribution_difference.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'omega_distribution_difference.pdf', bbox_inches='tight')
    print(f"  Saved: omega_distribution_difference.png/pdf")
    plt.close()

    # ========== Plot 3: Per-prime density comparison ==========
    fig, ax = plt.subplots(figsize=(10, 6))

    primes = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    baseline = [1/p for p in primes]
    conditioned = [1/(p-1) for p in primes]
    increment = [1/(p*(p-1)) for p in primes]

    x = np.arange(len(primes))
    width = 0.35

    ax.bar(x - width/2, baseline, width, label='Baseline: 1/p', color='#3498db', alpha=0.8)
    ax.bar(x + width/2, conditioned, width, label='Conditioned: 1/(p-1)', color='#e74c3c', alpha=0.8)

    ax.set_xlabel('Prime p')
    ax.set_ylabel('P(p | b)')
    ax.set_title('Per-prime divisibility probability\n(conditioning on "a is prime" boosts each probability)')
    ax.set_xticks(x)
    ax.set_xticklabels(primes)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add increment annotations
    for i, (b, c, inc) in enumerate(zip(baseline, conditioned, increment)):
        ax.annotate(f'+{inc:.4f}', xy=(i, c), xytext=(i, c + 0.01),
                   ha='center', va='bottom', fontsize=8, color='darkred')

    plt.tight_layout()
    plt.savefig(output_dir / 'per_prime_density_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'per_prime_density_comparison.pdf', bbox_inches='tight')
    print(f"  Saved: per_prime_density_comparison.png/pdf")
    plt.close()

    # ========== Plot 4: Cumulative sum of increments ==========
    fig, ax = plt.subplots(figsize=(10, 5))

    # Compute cumulative sum of 1/[p(p-1)] for primes up to various cutoffs
    from sympy import primerange
    all_primes = list(primerange(5, 1000))
    cumsum = np.cumsum([1/(p*(p-1)) for p in all_primes])

    ax.plot(all_primes, cumsum, 'b-', linewidth=2, label='Σ 1/[p(p-1)]')
    ax.axhline(y=cumsum[-1], color='red', linestyle='--', alpha=0.7,
               label=f'Limit ≈ {cumsum[-1]:.4f}')

    # Mark key points
    for target in [0.05, 0.08, 0.10]:
        idx = np.searchsorted(cumsum, target)
        if idx < len(all_primes):
            ax.plot(all_primes[idx], cumsum[idx], 'ko', markersize=6)
            ax.annotate(f'p={all_primes[idx]}: {cumsum[idx]:.3f}',
                       xy=(all_primes[idx], cumsum[idx]),
                       xytext=(all_primes[idx] + 50, cumsum[idx] - 0.005),
                       fontsize=9)

    ax.set_xlabel('Prime cutoff P')
    ax.set_ylabel('Cumulative increment in E[ω]')
    ax.set_title('Convergence of the selection bias constant\n(small primes dominate the effect)')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 500)

    plt.tight_layout()
    plt.savefig(output_dir / 'cumulative_increment.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'cumulative_increment.pdf', bbox_inches='tight')
    print(f"  Saved: cumulative_increment.png/pdf")
    plt.close()

    return {
        'pc_mean': pc_mean,
        'cc_mean': cc_mean,
        'pc_std': pc_std,
        'cc_std': cc_std,
        'shift': pc_mean - cc_mean,
        'relative_shift': (pc_mean - cc_mean) / cc_mean
    }


if __name__ == '__main__':
    import sys

    K = int(float(sys.argv[1])) if len(sys.argv) > 1 else 10_000_000

    pc_omega, cc_omega = compute_distributions(K)

    output_dir = Path('figures')
    stats = plot_distributions(pc_omega, cc_omega, K, output_dir)

    print(f"\nDistribution statistics:")
    print(f"  PC mean ω: {stats['pc_mean']:.4f}")
    print(f"  CC mean ω: {stats['cc_mean']:.4f}")
    print(f"  Shift: +{stats['shift']:.4f} ({100*stats['relative_shift']:.2f}%)")
    print(f"\nPlots saved to {output_dir}/")
