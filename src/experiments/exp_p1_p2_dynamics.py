#!/usr/bin/env python3
"""
Experiment: p1/p2 Dynamics and Log-Log Exponential Law (GPU-first)

Tests the "sieve as dynamical system" prediction:
- In log-log time, elimination events follow exponential distributions
- Rate = sieve dimension (2 for first hit, 1 for second hit)

For pairs (6k-1, 6k+1):
  p1 = min(spf(a), spf(b))  — first prime to hit either member
  p2 = max(spf(a), spf(b))  — "second shoe" prime (0 if censored)

Predictions (finite CRT model):
  V = log(log(p1)) - log(log(5)) ~ Exp(2)
  U = log(log(p2)) - log(log(p1)) ~ Exp(1)  [for uncensored observations]
  alpha = log(p2)/log(p1) ~ Pareto(1) tail

IMPORTANT NOTES:
1. PC/CP pairs have p2 = 0 (censored: survivor is prime).
2. CC pairs have finite p2 (uncensored).
3. CC-only U statistics are biased by censoring.

See docs/sieve_dynamics_framework.md for theoretical context.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
from scipy import stats

# Import project infrastructure
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.wheel_sieve import wheel_spf_sieve
from src.gpu_wheel_sieve import HAS_GPU
if HAS_GPU:
    from src.gpu_wheel_sieve import WheelGPUContext


# =============================================================================
# Statistical Analysis
# =============================================================================

def _safe_mean(arr: np.ndarray) -> float:
    return float(np.mean(arr)) if arr.size else float("nan")


def _safe_std(arr: np.ndarray) -> float:
    return float(np.std(arr)) if arr.size else float("nan")


def _safe_median(arr: np.ndarray) -> float:
    return float(np.median(arr)) if arr.size else float("nan")


def compute_statistics(p1: np.ndarray, p2: np.ndarray, states: np.ndarray, B: float = 5.0) -> Dict:
    """
    Compute U, V statistics with censoring-aware analysis.

    PC/CP pairs are censored (p2 = 0). CC pairs are uncensored.
    """
    state_counts = np.bincount(states, minlength=4)
    n_PP, n_PC, n_CP, n_CC = (int(x) for x in state_counts)
    n_non_pp = n_PC + n_CP + n_CC
    n_censored = n_PC + n_CP
    censoring_fraction = n_censored / n_non_pp if n_non_pp > 0 else 0.0

    # V uses all non-PP pairs (p1 > 0)
    p1_non_pp = p1[states != 0]
    log_log_B = np.log(np.log(B))
    V = np.log(np.log(p1_non_pp.astype(np.float64))) - log_log_B
    V_nonneg = V[V >= 0]

    # U and alpha use CC pairs only
    cc_mask = states == 3
    p1_cc = p1[cc_mask]
    p2_cc = p2[cc_mask]
    log_p1 = np.log(p1_cc.astype(np.float64))
    log_p2 = np.log(p2_cc.astype(np.float64))
    U = np.log(log_p2) - np.log(log_p1)
    U_nonneg = U[U >= 0]
    alpha = log_p2 / log_p1

    # KS tests
    ks_V, pval_V = (stats.kstest(V_nonneg, 'expon', args=(0, 0.5))
                    if V_nonneg.size else (float("nan"), float("nan")))
    ks_U, pval_U = (stats.kstest(U_nonneg, 'expon', args=(0, 1.0))
                    if U_nonneg.size else (float("nan"), float("nan")))

    results = {
        'state_counts': {
            'PP': n_PP,
            'PC': n_PC,
            'CP': n_CP,
            'CC': n_CC,
        },
        'n_non_pp': n_non_pp,
        'n_censored': n_censored,
        'censoring_fraction': censoring_fraction,

        # V statistics (first-hit law, all non-PP)
        'V_mean': _safe_mean(V_nonneg),
        'V_std': _safe_std(V_nonneg),
        'V_median': _safe_median(V_nonneg),
        'V_theoretical_mean': 0.5,
        'V_ks_statistic': float(ks_V),
        'V_ks_pvalue': float(pval_V),

        # U statistics (CC-only, biased by censoring)
        'U_mean': _safe_mean(U_nonneg),
        'U_std': _safe_std(U_nonneg),
        'U_median': _safe_median(U_nonneg),
        'U_theoretical_mean': 1.0,
        'U_ks_statistic': float(ks_U),
        'U_ks_pvalue': float(pval_U),
        'U_censoring_note': (
            f'CC-only; {100*censoring_fraction:.1f}% of observations are censored (p2=0)'
        ),

        # Alpha statistics
        'alpha_mean': _safe_mean(alpha),
        'alpha_median': _safe_median(alpha),
        'alpha_max': float(np.max(alpha)) if alpha.size else float("nan"),

        # Raw arrays for plotting
        'V': V,
        'U': U,
        'alpha': alpha,
        'p1_non_pp': p1_non_pp,
    }

    return results


# =============================================================================
# Plotting
# =============================================================================

def generate_plots(stats: Dict, output_dir: Path, K: int, seed: int) -> None:
    """Generate diagnostic plots with censoring notes."""
    try:
        import matplotlib.pyplot as plt
        from scipy.stats import expon
    except ImportError:
        print("  matplotlib/scipy not available, skipping plots")
        return

    rng = np.random.default_rng(seed)

    V = stats['V']
    U = stats['U']
    alpha = stats['alpha']
    p1_non_pp = stats['p1_non_pp']
    censoring_frac = stats['censoring_fraction']

    # Sample for plotting if too large
    max_points = 100000
    if V.size > max_points:
        V_plot = V[rng.choice(V.size, max_points, replace=False)]
    else:
        V_plot = V

    if U.size > max_points:
        U_plot = U[rng.choice(U.size, max_points, replace=False)]
    else:
        U_plot = U

    if alpha.size > max_points:
        alpha_plot = alpha[rng.choice(alpha.size, max_points, replace=False)]
    else:
        alpha_plot = alpha

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: V distribution (first-hit, Exp(2))
    ax = axes[0, 0]
    V_pos = V_plot[V_plot >= 0]
    if V_pos.size:
        ax.hist(V_pos, bins=50, density=True, alpha=0.7, color='forestgreen', label='Empirical')
        x = np.linspace(0, np.percentile(V_pos, 99), 100)
        ax.plot(x, expon.pdf(x, scale=0.5), 'r-', lw=2, label='Exp(2) theory')
    ax.set_xlabel('V = log(log(p1)) - log(log(5))')
    ax.set_ylabel('Density')
    ax.set_title(f'V Distribution (First Hit, Exp(2))\nMean={stats["V_mean"]:.3f}, Theory=0.5')
    ax.legend()

    ax = axes[0, 1]
    if V_pos.size:
        theoretical_quantiles = expon.ppf(np.linspace(0.01, 0.99, 100), scale=0.5)
        empirical_quantiles = np.percentile(V_pos, np.linspace(1, 99, 100))
        ax.scatter(theoretical_quantiles, empirical_quantiles, s=10, alpha=0.6, color='forestgreen')
        ax.plot([0, max(theoretical_quantiles)], [0, max(theoretical_quantiles)], 'r--', lw=2)
    ax.set_xlabel('Exp(2) Theoretical Quantiles')
    ax.set_ylabel('Empirical Quantiles')
    ax.set_title(f'Q-Q Plot: V vs Exp(2)\nKS={stats["V_ks_statistic"]:.4f}')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    primes_to_show = [5, 7, 11, 13, 17, 19, 23, 29, 31]
    counts = [(p1_non_pp == p).sum() for p in primes_to_show]
    ax.bar(range(len(primes_to_show)), counts,
           tick_label=[str(p) for p in primes_to_show], color='steelblue', alpha=0.7)
    ax.set_xlabel('p1 (first-hit prime)')
    ax.set_ylabel('Count')
    ax.set_title('p1 Distribution (first few primes)')

    # Row 2: U distribution (second-hit, CC-only)
    ax = axes[1, 0]
    U_pos = U_plot[U_plot >= 0]
    if U_pos.size:
        ax.hist(U_pos, bins=50, density=True, alpha=0.7, color='steelblue', label='Empirical (CC only)')
        x = np.linspace(0, np.percentile(U_pos, 99), 100)
        ax.plot(x, expon.pdf(x, scale=1), 'r-', lw=2, label='Exp(1) theory')
    ax.set_xlabel('U = log(log(p2)) - log(log(p1))')
    ax.set_ylabel('Density')
    ax.set_title(f'U Distribution (CC only, {100*censoring_frac:.0f}% censored)\n'
                 f'Mean={stats["U_mean"]:.3f}, Theory=1.0')
    ax.legend()

    ax = axes[1, 1]
    if U_pos.size:
        theoretical_quantiles = expon.ppf(np.linspace(0.01, 0.99, 100), scale=1)
        empirical_quantiles = np.percentile(U_pos, np.linspace(1, 99, 100))
        ax.scatter(theoretical_quantiles, empirical_quantiles, s=10, alpha=0.6)
        ax.plot([0, max(theoretical_quantiles)], [0, max(theoretical_quantiles)], 'r--', lw=2)
    ax.set_xlabel('Exp(1) Theoretical Quantiles')
    ax.set_ylabel('Empirical Quantiles')
    ax.set_title(f'Q-Q Plot: U vs Exp(1)\nKS={stats["U_ks_statistic"]:.4f}')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    if alpha_plot.size:
        alpha_sorted = np.sort(alpha_plot)
        survival = 1 - np.arange(1, len(alpha_sorted) + 1) / len(alpha_sorted)
        step = max(1, len(alpha_sorted) // 1000)
        ax.loglog(alpha_sorted[::step], survival[::step], 'b.', markersize=2, alpha=0.5, label='Empirical')
        t = np.linspace(1, min(alpha_sorted[-1], 20), 100)
        ax.loglog(t, 1/t, 'r-', lw=2, label='1/t (Pareto)')
    ax.set_xlabel('alpha = log(p2)/log(p1)')
    ax.set_ylabel('P(alpha > t)')
    ax.set_title(f'Run Exponent alpha Tail\nMax={stats["alpha_max"]:.1f}')
    ax.legend()
    ax.set_xlim(1, None)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f'Sieve Dynamics: Log-Log Exponential Law (K={K:.0e})\n'
        f'Note: U is biased by {100*censoring_frac:.0f}% censoring (PC/CP have p2=0)',
        fontsize=12, y=1.02
    )
    plt.tight_layout()

    fig_path = output_dir / "p1_p2_dynamics.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"    Saved {fig_path}")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Test p1/p2 dynamics and log-log exponential law")
    parser.add_argument("--K", type=float, default=1e8, help="Number of pairs (default: 1e8)")
    parser.add_argument("--output", type=str, default="data/reference/p1_p2_dynamics", help="Output directory")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for plot subsampling")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    args = parser.parse_args()

    if not HAS_GPU:
        raise RuntimeError("GPU not available. This experiment is GPU-only.")

    K = int(args.K)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("p1/p2 Dynamics Experiment")
    print("=" * 60)
    print(f"K = {K:,}")
    print(f"Output: {output_dir}")
    print()

    t0 = time.time()
    print("Building wheel SPF (GPU input)...")
    spf_wheel = wheel_spf_sieve(K)
    print("Setting up GPU context...")
    ctx = WheelGPUContext(K, spf_wheel)

    print("Computing p1/p2/states on GPU...")
    p1, p2, states = ctx.compute_p1_p2()
    compute_time = time.time() - t0

    print()
    stats = compute_statistics(p1, p2, states)

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    counts = stats['state_counts']
    print(f"State counts: PP={counts['PP']:,}, PC={counts['PC']:,}, CP={counts['CP']:,}, CC={counts['CC']:,}")
    print(f"Censoring: {stats['n_censored']:,} pairs have p2=0 ({100*stats['censoring_fraction']:.1f}%)")
    print()
    print("V = log(log(p1)) - log(log(5))  [First-hit law, Exp(2)]")
    print(f"  Mean:     {stats['V_mean']:.4f}  (theory: 0.5)")
    print(f"  Std:      {stats['V_std']:.4f}  (theory: 0.5)")
    print(f"  KS stat:  {stats['V_ks_statistic']:.4f}")
    print()
    print("U = log(log(p2)) - log(log(p1))  [Second-hit law, Exp(1)]")
    print(f"  NOTE: CC-only, biased by {100*stats['censoring_fraction']:.1f}% censoring")
    print(f"  Mean:     {stats['U_mean']:.4f}  (theory: 1.0, but biased low)")
    print(f"  Std:      {stats['U_std']:.4f}")
    print(f"  KS stat:  {stats['U_ks_statistic']:.4f}")
    print()
    print("alpha = log(p2)/log(p1)  [Pareto(1) tail]")
    print(f"  Mean:     {stats['alpha_mean']:.4f}")
    print(f"  Median:   {stats['alpha_median']:.4f}")
    print(f"  Max:      {stats['alpha_max']:.1f}")
    print()

    if not args.no_plot:
        generate_plots(stats, output_dir, K, seed=args.seed)

    stats_for_json = {k: v for k, v in stats.items() if not isinstance(v, np.ndarray)}

    meta = {
        'description': 'p1/p2 dynamics test of log-log exponential law (GPU wheel SPF)',
        'K': K,
        'computation_time_seconds': compute_time,
        'predictions': {
            'V': 'Exp(2) — mean=0.5, first-hit law (all non-PP pairs)',
            'U': 'Exp(1) — mean=1.0, second-hit law (CC-only, censoring-biased)',
            'alpha': 'Pareto(1) tail — P(alpha>t) ~ 1/t (CC-only)'
        },
        'caveats': [
            f"PC+CP pairs ({stats['n_censored']:,}) have p2=0, creating {100*stats['censoring_fraction']:.1f}% censoring",
            'U statistics are biased low because they condition on non-censoring',
            'Pareto tail truncated at p2 <= 6K+1'
        ],
        'statistics': stats_for_json,
    }

    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {output_dir / 'metadata.json'}")

    print()
    print(f"Total time: {compute_time:.1f}s")
    print("Done!")


if __name__ == "__main__":
    main()
