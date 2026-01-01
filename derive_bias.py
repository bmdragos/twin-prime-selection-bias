#!/usr/bin/env python3
"""
First-principles derivation of the selection bias.

The derivation has three layers:
1. EXACT: The local mutual exclusivity identity
2. HEURISTIC: The independent-prime sum
3. EMPIRICAL: The calibration factor

This script computes all three and compares to data.
"""

import math


def sieve_primes(n: int) -> list:
    """Simple sieve of Eratosthenes."""
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for p in range(2, int(n**0.5) + 1):
        if is_prime[p]:
            for i in range(p*p, n + 1, p):
                is_prime[i] = False
    return [p for p in range(n + 1) if is_prime[p]]


def main():
    print("=" * 70)
    print("FIRST-PRINCIPLES DERIVATION OF SELECTION BIAS")
    print("=" * 70)
    print()

    # ================================================================
    # LAYER 1: THE EXACT LOCAL MECHANISM
    # ================================================================
    print("-" * 70)
    print("LAYER 1: The Exact Local Identity (the 'engine')")
    print("-" * 70)
    print()
    print("Let a = 6k-1, b = 6k+1. For any prime p >= 5:")
    print()
    print("  - Exactly one residue class k (mod p) has p | a")
    print("  - Exactly one residue class k (mod p) has p | b")
    print("  - These CANNOT overlap: p | (b-a) = 2 requires p = 2")
    print()
    print("So 'p|a' and 'p|b' are mutually exclusive events.")
    print()
    print("The key conditional identity:")
    print()
    print("  P(p|b | p nmid a) = P(p|b) / P(p nmid a)")
    print("                    = (1/p) / (1 - 1/p)")
    print("                    = 1/(p-1)")
    print()
    print("This step is EXACT. Everything else is heuristic.")
    print()

    # ================================================================
    # LAYER 2: THE INDEPENDENT-PRIME SUM
    # ================================================================
    print("-" * 70)
    print("LAYER 2: The Independent-Prime Heuristic")
    print("-" * 70)
    print()
    print("Let X_p = 1 if p|b, 0 otherwise.")
    print()
    print("  Unconditionally:        E[X_p] ~ 1/p")
    print("  Under 'a is prime':     E[X_p | a prime] ~ 1/(p-1)")
    print()
    print("Per-prime increment:")
    print()
    print("  delta_p = 1/(p-1) - 1/p = 1/[p(p-1)]")
    print()

    # Compute the sum
    primes = [p for p in sieve_primes(100000) if p >= 5]
    sum_delta = sum(1 / (p * (p - 1)) for p in primes)

    print("Summing over all primes p >= 5:")
    print()
    print(f"  Sum of 1/[p(p-1)] = {sum_delta:.6f}")
    print()
    print("This sum CONVERGES, explaining why the bias stabilizes.")
    print()

    # Show first few terms
    print("First few terms:")
    cumsum = 0
    for p in primes[:10]:
        term = 1 / (p * (p - 1))
        cumsum += term
        print(f"  p={p:3d}: 1/[{p}*{p-1}] = {term:.6f}  (cumulative: {cumsum:.6f})")
    print(f"  ...")
    print(f"  Total (p <= {primes[-1]}): {sum_delta:.6f}")
    print()

    # ================================================================
    # LAYER 3: COMPARISON TO EMPIRICAL DATA
    # ================================================================
    print("-" * 70)
    print("LAYER 3: Comparison to Empirical Data")
    print("-" * 70)
    print()

    # Empirical values from K=10^9
    omega_pc = 2.9067      # E[omega(b) | PC]
    omega_cc = 2.8239      # E[omega | CC]
    omega_uncond = 2.8357  # E[omega | unconditional composite at 6k+1]

    delta_pc_cc = omega_pc - omega_cc
    delta_pc_uncond = omega_pc - omega_uncond

    print("Empirical values at K = 10^9:")
    print()
    print(f"  E[omega | PC composite]     = {omega_pc:.4f}")
    print(f"  E[omega | CC composite]     = {omega_cc:.4f}")
    print(f"  E[omega | unconditional]    = {omega_uncond:.4f}")
    print()
    print("Observed shifts:")
    print()
    print(f"  Delta_omega (PC vs CC):          {delta_pc_cc:.4f}")
    print(f"  Delta_omega (PC vs unconditional): {delta_pc_uncond:.4f}")
    print()

    # Calibration factors
    c_vs_cc = delta_pc_cc / sum_delta
    c_vs_uncond = delta_pc_uncond / sum_delta

    print("Calibration factors c = Delta_omega_emp / Sum[1/p(p-1)]:")
    print()
    print(f"  c (PC vs CC):           {delta_pc_cc:.4f} / {sum_delta:.4f} = {c_vs_cc:.3f}")
    print(f"  c (PC vs unconditional): {delta_pc_uncond:.4f} / {sum_delta:.4f} = {c_vs_uncond:.3f}")
    print()

    print("The naive independent-prime calculation overshoots by ~20-30%.")
    print()

    # ================================================================
    # WHY THE OVERSHOOT?
    # ================================================================
    print("-" * 70)
    print("WHY THE OVERSHOOT?")
    print("-" * 70)
    print()
    print("Two sources:")
    print()
    print("1. BASELINE CHOICE")
    print("   PC vs CC is smaller than PC vs unconditional.")
    print("   CC is itself a conditioned population (both sides composite),")
    print("   which already has slightly lower omega than unconditional.")
    print()
    print("2. INDEPENDENCE ASSUMPTION")
    print("   Divisibility events are not independent across primes.")
    print("   After conditioning on 'a is prime', there are residual")
    print("   correlations between divisibility by different primes.")
    print("   This is exactly what the transfer matrix model quantifies.")
    print()

    # ================================================================
    # THE CLEAN SUMMARY
    # ================================================================
    print("-" * 70)
    print("CLEAN SUMMARY (ready for paper)")
    print("-" * 70)
    print()
    print('''Heuristic derivation. For each prime p >= 5, the congruences
6k = +/-1 (mod p) have unique solutions, so among k (mod p) exactly
one class satisfies p | (6k-1) and exactly one class satisfies
p | (6k+1). These two classes are disjoint since p | 2 would be
required for overlap. Hence conditioning on p nmid (6k-1) boosts
the probability that p | (6k+1) from 1/p to 1/(p-1).

Summing the per-prime increment 1/(p-1) - 1/p = 1/[p(p-1)] over
p >= 5 yields a convergent absolute shift:''')
    print()
    print(f"  Sum_{{p>=5}} 1/[p(p-1)] = {sum_delta:.4f}")
    print()
    print('''predicting a few-percent relative uplift when divided by the
baseline mean E[omega] ~ log log n.

Empirically, the observed PC-CC shift at K=10^9 is:''')
    print()
    print(f"  Delta_omega = {delta_pc_cc:.4f}")
    print()
    print(f'''about {100*(1-c_vs_cc):.0f}% smaller than the naive independent-prime prediction,
consistent with residual arithmetic correlations and the fact that
CC is itself a conditioned composite population.''')
    print()

    # ================================================================
    # PERCENTAGE BIAS PREDICTION
    # ================================================================
    print("-" * 70)
    print("PERCENTAGE BIAS VS SCALE")
    print("-" * 70)
    print()
    print("The percentage bias = Delta_omega / E[omega|CC]")
    print()
    print("Since E[omega] ~ log log N, and Delta_omega is ~constant,")
    print("the percentage bias slowly decreases:")
    print()

    for K_exp in [7, 8, 9, 10, 12, 15]:
        K = 10**K_exp
        # Approximate E[omega|CC] ~ log log (6K)
        log_log_N = math.log(math.log(6 * K))
        # Use empirical calibration
        predicted_delta = c_vs_cc * sum_delta
        predicted_bias_pct = 100 * predicted_delta / log_log_N
        print(f"  K=10^{K_exp:2d}: log log(6K) ~ {log_log_N:.3f}, predicted bias ~ {predicted_bias_pct:.2f}%")

    print()
    print("The bias percentage decreases very slowly with scale.")
    print()


if __name__ == '__main__':
    main()
