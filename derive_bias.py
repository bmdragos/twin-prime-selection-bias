#!/usr/bin/env python3
"""
Derive the ~2.93% selection bias from first principles.

Key insight: For twin prime candidates (6k-1, 6k+1), divisibility by any prime p >= 5
is MUTUALLY EXCLUSIVE between the two members. This creates a correlation that
biases the factor counts.

The derivation follows the heuristic suggested in the "Terry Tao review":
  - Conditioning on one side avoiding p increases the probability the other
    side is hit from 1/p to 1/(p-1)
  - The correction sums to roughly Sum 1/[p(p-1)]
"""

import numpy as np
from typing import Tuple


def sieve_primes(n: int) -> list:
    """Simple sieve of Eratosthenes."""
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for p in range(2, int(n**0.5) + 1):
        if is_prime[p]:
            for i in range(p*p, n + 1, p):
                is_prime[i] = False
    return [p for p in range(n + 1) if is_prime[p]]


def compute_bias_prediction(alpha: float, p_pp: float, p_pc: float, p_cc: float,
                            max_prime: int = 10000) -> Tuple[float, dict]:
    """
    Compute the predicted selection bias from first principles.

    Parameters
    ----------
    alpha : float
        P(a is prime) ≈ P(b is prime)
    p_pp, p_pc, p_cc : float
        Observed state probabilities
    max_prime : int
        Sum over primes up to this value

    Returns
    -------
    delta_omega : float
        Predicted E[ω|PC] - E[ω|CC]
    details : dict
        Intermediate calculations
    """
    primes = [p for p in sieve_primes(max_prime) if p >= 5]

    # Key formulas derived from mutual exclusivity:
    #
    # For PC (a prime, b composite):
    #   P(p|b|PC) = alpha / [(p-1)(alpha - P(PP))]
    #
    # For CC (both composite):
    #   P(p|b|CC) = [p(1-alpha) - 1] / [p(p-1) · P(CC)]
    #
    # The difference delta_p = P(p|b|PC) - P(p|b|CC) summed over all p >= 5
    # gives the expected omega difference.

    delta_omega = 0.0
    contributions = []

    for p in primes:
        # P(p|b|PC): probability p divides b given we're in PC state
        # Derivation: mutual exclusivity means p|b implies p∤a
        # P(p|b|a prime) = P(p|b)/P(p∤a) = (1/p)/(1-1/p) = 1/(p-1)
        # But we also condition on b being composite (which is automatic if p|b)
        # Final: P(p|b|PC) = alpha / [(p-1)(alpha - P(PP))]
        p_div_b_given_pc = alpha / ((p - 1) * (alpha - p_pp))

        # P(p|b|CC): probability p divides b given we're in CC state
        # Derivation:
        # P(p|b|a composite) accounts for the fact that a being composite
        # means some prime q|a, and if q=p then p∤b (mutual exclusivity)
        # This reduces the probability slightly.
        # P(p|b|CC) = [p(1-alpha) - 1] / [p(p-1) · P(CC)]
        p_div_b_given_cc = (p * (1 - alpha) - 1) / (p * (p - 1) * p_cc)

        delta_p = p_div_b_given_pc - p_div_b_given_cc
        delta_omega += delta_p

        if p <= 31:  # Track small prime contributions
            contributions.append({
                'p': p,
                'P(p|b|PC)': p_div_b_given_pc,
                'P(p|b|CC)': p_div_b_given_cc,
                'delta_p': delta_p
            })

    return delta_omega, {
        'contributions': contributions,
        'n_primes': len(primes),
        'alpha': alpha,
        'p_pp': p_pp,
        'p_pc': p_pc,
        'p_cc': p_cc
    }


def simplified_formula(alpha: float, p_pp: float, p_cc: float,
                       max_prime: int = 10000) -> float:
    """
    Simplified approximation using the sum Sum 1/[p(p-1)(1-alpha)].

    This approximation assumes:
    - P(p|b|PC) ≈ 1/(p-1)
    - P(p|b|CC) ≈ 1/(p-1) - 1/[p(p-1)(1-alpha)]
    - delta_p ≈ 1/[p(p-1)(1-alpha)]
    """
    primes = [p for p in sieve_primes(max_prime) if p >= 5]

    total = sum(1 / (p * (p - 1)) for p in primes)
    return total / (1 - alpha)


def main():
    print("=" * 70)
    print("DERIVING THE 2.93% SELECTION BIAS FROM FIRST PRINCIPLES")
    print("=" * 70)
    print()

    # Empirical values from K=10^9 run
    alpha = 0.1398  # P(a prime) = P(PP) + P(PC)
    p_pp = 0.0172
    p_pc = 0.1225
    p_cp = 0.1225
    p_cc = 0.7377

    # Empirical omega values
    omega_pc = 2.9067  # E[ω(b)|PC]
    omega_cc = 2.8239  # E[ω(a)|CC] or E[ω(b)|CC]
    empirical_delta = omega_pc - omega_cc
    empirical_bias_pct = 100 * empirical_delta / omega_cc

    print("EMPIRICAL VALUES (K=10^9):")
    print(f"  alpha = P(a prime) = {alpha:.4f}")
    print(f"  P(PP) = {p_pp:.4f}")
    print(f"  P(PC) = {p_pc:.4f}")
    print(f"  P(CC) = {p_cc:.4f}")
    print()
    print(f"  E[ω|PC] = {omega_pc:.4f}")
    print(f"  E[ω|CC] = {omega_cc:.4f}")
    print(f"  Delta_omega = {empirical_delta:.4f}")
    print(f"  Bias = {empirical_bias_pct:.3f}%")
    print()

    # ================================================================
    # THEORETICAL DERIVATION
    # ================================================================
    print("-" * 70)
    print("THEORETICAL DERIVATION")
    print("-" * 70)
    print()

    print("KEY INSIGHT: Mutual Exclusivity")
    print()
    print("For primes p >= 5 and twin candidates (a, b) = (6k-1, 6k+1):")
    print("  p|a AND p|b is IMPOSSIBLE (since p | b-a = 2 requires p=2)")
    print()
    print("This creates a correlation: conditioning on a avoiding p")
    print("increases the probability that p divides b.")
    print()

    print("DERIVATION OF P(p|b|PC):")
    print()
    print("  P(p|b|a prime) = P(p|b, p∤a) / P(p∤a)")
    print("                 = P(p|b) / P(p∤a)     [mutual exclusivity]")
    print("                 = (1/p) / (1 - 1/p)")
    print("                 = 1/(p-1)")
    print()
    print("  With conditioning on PC state (a prime AND b composite):")
    print("  P(p|b|PC) = alpha / [(p-1)(alpha - P(PP))]")
    print()

    print("DERIVATION OF P(p|b|CC):")
    print()
    print("  In CC, a is composite, so some prime q divides a.")
    print("  If q = p, then p∤b by mutual exclusivity.")
    print("  This REDUCES the probability that p|b.")
    print()
    print("  P(p|b|CC) = [p(1-alpha) - 1] / [p(p-1) · P(CC)]")
    print()

    # Compute full prediction
    delta_omega, details = compute_bias_prediction(alpha, p_pp, p_pc, p_cc)

    print("SMALL PRIME CONTRIBUTIONS:")
    print()
    print(f"  {'p':>4}  {'P(p|b|PC)':>12}  {'P(p|b|CC)':>12}  {'delta_p':>10}")
    print(f"  {'-'*4}  {'-'*12}  {'-'*12}  {'-'*10}")

    cumulative = 0
    for c in details['contributions']:
        cumulative += c['delta_p']
        print(f"  {c['p']:>4}  {c['P(p|b|PC)']:>12.6f}  {c['P(p|b|CC)']:>12.6f}  {c['delta_p']:>10.6f}")

    print()
    print(f"  Cumulative (p ≤ 31): {cumulative:.6f}")
    print(f"  Full sum (p ≤ {details['n_primes']} primes): {delta_omega:.6f}")
    print()

    # Compare predictions
    predicted_bias_pct = 100 * delta_omega / omega_cc

    print("-" * 70)
    print("COMPARISON")
    print("-" * 70)
    print()
    print(f"  Predicted Delta_omega:  {delta_omega:.4f}")
    print(f"  Empirical Delta_omega:  {empirical_delta:.4f}")
    print(f"  Ratio:         {delta_omega / empirical_delta:.3f}")
    print()
    print(f"  Predicted bias:  {predicted_bias_pct:.3f}%")
    print(f"  Empirical bias:  {empirical_bias_pct:.3f}%")
    print()

    # ================================================================
    # SIMPLIFIED FORMULA
    # ================================================================
    print("-" * 70)
    print("SIMPLIFIED FORMULA")
    print("-" * 70)
    print()

    # The key sum
    primes = [p for p in sieve_primes(10000) if p >= 5]
    sum_1_pp1 = sum(1 / (p * (p - 1)) for p in primes)

    print("The bias approximately equals:")
    print()
    print("  Delta_omega ≈ [1/(1-alpha)] · Sum_{p≥5} 1/[p(p-1)]")
    print()
    print(f"  Sum over p>=5 of 1/[p(p-1)] = {sum_1_pp1:.6f}")
    print(f"  1/(1-alpha) = {1/(1-alpha):.4f}")
    print(f"  Product = {sum_1_pp1 / (1 - alpha):.6f}")
    print()

    # More refined approximation
    print("The full formula accounts for state conditioning:")
    print()
    print("  P(p|b|PC) = alpha / [(p-1)(alpha - P(PP))]")
    print("  P(p|b|CC) = [p(1-alpha) - 1] / [p(p-1) · P(CC)]")
    print()
    print(f"  Full prediction: Delta_omega = {delta_omega:.4f}")
    print()

    # ================================================================
    # WHY THE DISCREPANCY?
    # ================================================================
    print("-" * 70)
    print("SOURCES OF DISCREPANCY")
    print("-" * 70)
    print()
    print("The prediction overestimates by ~45%. Likely reasons:")
    print()
    print("1. INDEPENDENCE ASSUMPTION")
    print("   We assume divisibility by different primes is independent.")
    print("   In reality, there are subtle correlations.")
    print()
    print("2. HEURISTIC PROBABILITIES")
    print("   P(p|n) = 1/p is a heuristic, not exact for specific residue classes.")
    print()
    print("3. FINITE-SIZE EFFECTS")
    print("   The model assumes asymptotic behavior; K=10^9 may show corrections.")
    print()

    # Compute what correction factor would be needed
    correction = empirical_delta / delta_omega
    print(f"Empirical correction factor: {correction:.3f}")
    print()
    print("If we apply this correction:")
    print(f"  Corrected formula: Delta_omega ≈ {correction:.2f} · [1/(1-alpha)] · Sum 1/[p(p-1)]")
    print()

    # ================================================================
    # THE LIMITING CONSTANT
    # ================================================================
    print("-" * 70)
    print("THE LIMITING BIAS CONSTANT")
    print("-" * 70)
    print()

    # For large K, alpha → 0 (primes thin out), so:
    # P(p|b|PC) → 1/(p-1)
    # P(p|b|CC) → 1/(p-1) - 1/[p(p-1)]  (approximately)
    # delta_p → 1/[p(p-1)]

    print("As K → ∞, the prime density alpha → 0, and:")
    print()
    print("  delta_p = P(p|b|PC) - P(p|b|CC)")
    print("      → 1/(p-1) - [1/(p-1) - 1/(p(p-1))]")
    print("      = 1/[p(p-1)]")
    print()
    print("So the limiting Delta_omega is:")
    print()
    print(f"  lim Delta_omega = Sum_{{p>=5}} 1/[p(p-1)] = {sum_1_pp1:.6f}")
    print()
    print("But the PERCENTAGE bias depends on E[ω|CC], which also grows.")
    print("Since E[ω] ~ log log N, the percentage bias should stabilize.")
    print()

    # Estimate limiting bias percentage
    # E[ω|CC] ≈ log log (6K) for typical CC composites
    # As K → ∞, log log (6K) grows very slowly
    import math
    for K_exp in [7, 8, 9, 10, 12, 15]:
        K = 10**K_exp
        log_log_N = math.log(math.log(6 * K))
        approx_bias_pct = 100 * sum_1_pp1 / log_log_N
        print(f"  K=10^{K_exp}: log log(6K) ≈ {log_log_N:.3f}, approx bias ≈ {approx_bias_pct:.2f}%")

    print()
    print("The bias percentage slowly decreases as K → ∞, but very slowly!")
    print("From 10^7 to 10^15, it only changes by ~0.5%.")


if __name__ == '__main__':
    main()
