# Research Directions

This document tracks three potential extensions of the selection bias work, ordered by tractability.

## Status Summary

| Direction | Status | Effort | Value |
|-----------|--------|--------|-------|
| 1. Per-prime diagnostics | **Complete** ✓ | ~1hr each pattern | Strengthens empirical case |
| 2. Variance/shape analysis | **Complete** ✓ | ~half day | Found variance + shape shift |
| 3. Prove residue-class model | Not started | Days-weeks | Would be a theorem |

---

## 1. Per-Prime Diagnostics (Empirical)

**Goal:** Replicate Table 3.1 for Sophie Germain and cousin primes, showing P(p|b | a prime) = 1/(p-1) at each prime individually.

**Why it matters:** Table 3.1 is the most convincing evidence—it shows the mechanism works at *each* prime, not just that the sum happens to match. Extending this to other patterns turns "the mechanism generalizes" from a claim into demonstrated fact.

**Tasks:**
- [x] Twin primes: Table 3.1 in paper (K=10^9)
- [x] Sophie Germain (n, 2n+1): Experiment created
- [x] Cousin primes (n, n+4): Experiment created
- [x] Run Sophie Germain at N=10^9 on DGX Spark ✓
- [x] Run cousin primes at K=10^8 on DGX Spark ✓

**Results (2025-01-03):**

Sophie Germain at N=10^9 (50.8M primes, 449M composites):
| p | P(p\|b \| n prime) | Predicted 1/(p-1) |
|---|-------------------|-------------------|
| 3 | 0.4999**79** | 0.5000 |
| 5 | 0.2500**08** | 0.2500 |
| 7 | 0.1666**77** | 0.1667 |

Cousin primes at K=10^8 (31.3M primes, 168.7M composites):
| p | P(p\|b \| n prime) | Increment | Note |
|---|-------------------|-----------|------|
| 3 | 0.500016 | **+0.00002** | Residue-class cancellation confirmed! |
| 5 | 0.2499**84** | +0.0593 | Matches 1/(p-1) |
| 7 | 0.1666**69** | +0.0282 | Matches 1/(p-1) |

**Implementation:**
- `src/experiments/exp_per_prime_divisibility.py` — Twin primes (existing)
- `src/experiments/exp_sophie_germain_per_prime.py` — Sophie Germain (new)
- `src/experiments/exp_cousin_primes_per_prime.py` — Cousin primes (new)

**Run commands:**
```bash
# Sophie Germain (N=10^9, odd n)
python -m src.experiments.exp_sophie_germain_per_prime 1e9

# Cousin primes (K=10^8, 6k±1 candidates)
python -m src.experiments.exp_cousin_primes_per_prime 1e8
```

**Output:** Reference CSVs in `data/reference/` + tables in paper Section 5.

---

## 2. Variance/Shape Analysis (Empirical)

**Goal:** Determine whether conditioning on "a is prime" affects only the mean of ω(b), or also the variance and shape.

**Tasks:**
- [x] Compute full ω distributions for PC and CC at K=10^8
- [x] Compute moments (mean, variance, skewness, kurtosis)
- [x] KS test against normal
- [x] Derive predicted variance shift from per-prime Bernoulli model

**Predicted variance shift (per-prime Bernoulli model):**
```
Var(ω | a prime) - Var(ω | a composite)
  ≈ Σ [Var(Bernoulli(1/(p-1))) - Var(Bernoulli(1/p))]
  = Σ [(p-2)/(p-1)² - (p-1)/p²]
  ≈ 0.0724  (for p up to 10000)
```

**Results at K=10^8 (2025-01-03):**

| Metric | PC | CC | Difference |
|--------|-----|-----|------------|
| Mean ω | 2.817 | 2.737 | **+0.080** (97% of predicted 0.083) |
| Variance | 0.683 | 0.624 | **+0.059** (9.5% higher) |
| Skewness | 0.754 | 0.829 | **-0.075** |
| Excess kurtosis | -0.012 | 0.110 | **-0.12** |

**Key findings:**
1. **Variance is elevated** for PC (9.5% higher), confirming the per-prime effect extends to variance
2. **Variance shift is 82% of predicted** (0.059 vs 0.072) — some correlation reduces the effect
3. **Shape changes slightly** — PC is less right-skewed and less peaked than CC
4. **Neither is normal** — KS p-values ≈ 0 (expected since ω is discrete)

**Interpretation:**
The variance shift is real but smaller than naive per-prime prediction, likely because:
- The CC baseline is itself conditioned (both a and b composite)
- Correlations between divisibility events reduce variance

The shape differences (lower skewness and kurtosis for PC) suggest conditioning on "a prime" pulls the distribution slightly toward symmetric/Gaussian. This makes sense: PC composites have more small prime factors, making them more "typical" in their factorization structure.

**Implementation:** `src/experiments/exp_variance_analysis.py`

---

## 3. Prove Residue-Class Model (Rigorous)

**Goal:** A theorem with explicit error bounds, not just empirical verification.

**Target statement:**
> For ω_small(b) = #{p ≤ √N : p|b}, we have
> E[ω_small(b) | a prime] − E[ω_small(b)] = Σ_{5≤p≤√N} 1/[p(p-1)] + O(1/log N)

**Why this is hard:**
- Requires formalizing "primes are equidistributed among allowed residue classes"
- Need to invoke Siegel-Walfisz or similar equidistribution results
- Error terms from prime number theorem in arithmetic progressions
- May need to handle correlation between different primes carefully

**Relevant literature:**
- Elliott, *Probabilistic Number Theory* (1980) — additive functions on shifted primes
- Halberstam — ω(p-1) distribution
- Dixit et al. — ω_y(p+a) with explicit error terms
- Barban-Davenport-Halberstam inequality — equidistribution in residue classes

**What the residue-class model already provides:**
- For any *finite* set of primes P, the model is *exact* (CRT + mutual exclusivity)
- The heuristic step is: primes behave like random elements from allowed residue classes
- This is essentially true by PNT in arithmetic progressions, but making it rigorous requires care

**Possible approach:**
1. Fix threshold P (e.g., P = √N)
2. Show that for p ≤ P, the empirical frequency of "p | b among pairs where a is prime" converges to 1/(p-1)
3. Bound the error using equidistribution results
4. Sum over p to get the ω_small prediction with explicit error

**Status:** This direction is parked until directions 1-2 are complete. The empirical foundation makes it easier for future work (human or AI) to formalize.

---

## Notes

- The per-prime diagnostics (direction 1) provides the empirical foundation for direction 3
- Direction 2 is orthogonal—it explores a different aspect (distribution shape vs mean)
- All three directions use the same computational infrastructure
- Reference data in `data/reference/` serves as ground truth for any future formalization
