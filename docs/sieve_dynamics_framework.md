# Sieve as Dynamical System: Framework and Predictions

> **Core claim**: In log-log time, primes arrive as a Poisson process with rate = sieve dimension.

This document captures the "sieve as state machine" conceptual framework and its testable predictions, developed through exploration of twin prime selection bias.

---

## 1. The Conceptual Lens

### 1.1 State Machine View

For twin primes (6k-1, 6k+1), track pairs through states as primes "arrive":

```
PP (both prime candidates)
 ↓  first prime p₁ hits one member
PC/CP (one prime, one composite)
 ↓  second prime p₂ hits survivor
CC (both composite)
```

Each prime p acts as a constraint that can "eliminate" a surviving form.

### 1.2 Hazard Rates

For an odd prime p > 2 and the twin pattern:
- From **PP**: hazard ≈ 2/p (two targets, can hit either)
- From **PC**: hazard ≈ 1/p (one target remains)

The **dimension** = number of live linear forms = hazard rate multiplier.

### 1.3 Why This Works (CRT Independence)

If we model n as uniform over admissible residue classes mod ∏p≤z p, then:
- Residue classes mod distinct primes are **literally independent** (Chinese Remainder Theorem)
- The "hazard" at prime p is exactly ν(p)/p where ν(p) = forbidden residue classes
- The state machine evolution is a genuine Markov process

This is not metaphor—it's exact combinatorics in the finite model.

---

## 2. The Log-Log Exponential Law

### 2.1 Corridor Laws (Classical Form)

The survival probabilities follow power laws in log:

$$\prod_{B < p \leq y}\left(1 - \frac{2}{p}\right) \approx \left(\frac{\log B}{\log y}\right)^2$$

$$\prod_{B < p \leq y}\left(1 - \frac{1}{p}\right) \approx \frac{\log B}{\log y}$$

### 2.2 Exponential Form (The Key Insight)

Let t = log log y. Then:

$$\left(\frac{\log B}{\log y}\right)^k = \exp(-k \cdot (t - t_0)) = e^{-k(t - t_0)}$$

**Survival is exponential in log-log time, with rate = dimension.**

### 2.3 Testable Predictions

Define:
- p₁ = min(spf(a), spf(b)) — first prime to hit either member
- p₂ = max(spf(a), spf(b)) — "second shoe" prime
- B = 5 (smallest prime after the 6k±1 wheel excludes 2, 3)

**Prediction 1 (First Hit):**
$$V = \log\log p_1 - \log\log B \sim \text{Exp}(2)$$

Rate 2 because dimension = 2 (both forms alive).

**Prediction 2 (Second Hit):**
$$U = \log\log p_2 - \log\log p_1 \sim \text{Exp}(1)$$

Rate 1 because dimension = 1 (one form remains).

**Prediction 3 (Run Exponent):**
$$\alpha = \frac{\log p_2}{\log p_1} \implies P(\alpha > t) \approx \frac{1}{t}$$

The run exponent has a Pareto(1) heavy tail—"the second shoe can take forever."

---

## 3. Hierarchy of Exactness

| Statement | Status | Error Term |
|-----------|--------|------------|
| CRT independence mod distinct primes | **Exact** | None (algebra) |
| Hazard = ν(p)/p in finite model | **Exact** | None (combinatorics) |
| ∏(1-k/p) ≈ (log B / log y)^k | **Mertens** | O(1/log y) |
| U ~ Exp(1) for actual integers | **Heuristic** | Measure-dependent |

### 3.1 What's Exact

In the finite model (uniform on admissible residues mod primorial):
- Per-prime hazards are exact
- State transitions are exactly Markovian
- The log-log exponential law can be derived cleanly

### 3.2 What's Approximate

For actual integers in [N, 2N]:
- Equidistribution has error terms
- Conditioning on survival creates measure shifts
- Finite-size truncation affects tails

**Key insight**: Treat the finite model as null hypothesis, measure deviations in reality.

---

## 4. Connections to Existing Work

### 4.1 Per-Prime Verification (Already Done)

We verified: P(p|b | a prime) = 1/(p-1) to 5 significant figures.

This confirms the **marginal** hazard rate. The new predictions test **joint** structure.

### 4.2 Omega Decomposition

The ω(b) excess in PC vs CC pairs decomposes as:
$$\mathbb{E}[\omega(b) | \text{PC}] - \mathbb{E}[\omega(b) | \text{CC}] = \sum_p \frac{1}{p(p-1)}$$

This is the integrated effect of the per-prime hazard shifts.

### 4.3 Buchstab Function Connection

p₁ = min(spf(a), spf(b)) connects to Buchstab's function ω(u):
- Buchstab describes the proportion of integers with smallest prime factor > y
- The log-log exponential law claims Buchstab has exponential structure for k-tuples

### 4.4 Variance Analysis

The variance shift σ²(PC) vs σ²(CC) should also be predictable from the hazard framework—each prime contributes a Bernoulli variance term.

---

## 5. Proposed Experiments

### 5.1 Primary: p₁/p₂ Dynamics

```
For all CC pairs (both members composite):
  1. Compute p₁ = min(spf(6k-1), spf(6k+1))
  2. Compute p₂ = max(spf(6k-1), spf(6k+1))
  3. Compute U = log(log(p₂)) - log(log(p₁))
  4. Compute V = log(log(p₁)) - log(log(5))

Test:
  - U ~ Exp(1) via Q-Q plot, KS test
  - V ~ Exp(2) via Q-Q plot, KS test
  - α = log(p₂)/log(p₁) vs Pareto(1)
```

### 5.2 Conditional Survival Curves

Plot P(p₂ > y | p₁) binned by p₁, compare to:
- Exact: ∏_{p₁ < p ≤ y}(1 - 1/p)
- Asymptotic: log(p₁) / log(y)

### 5.3 Cross-Pattern Universality

Repeat for cousins (n, n+4) and Sophie Germain (n, 2n+1):
- Same exponential laws should hold
- Rate parameters should equal dimension
- Exceptional primes modify early steps only

### 5.4 Hazard Rate Measurement

Directly measure empirical hazard:
$$h(p) = P(\text{hit at } p \mid \text{survive to } p^-)$$

Compare to theoretical 2/p (from PP) and 1/p (from PC).

---

## 6. Caveats and Boundaries

### 6.1 Where the Lens is Sharp

- Local / small-prime regime where CRT gives exact control
- Conditioned statistics (PC vs CC, run lengths, variance)
- Finite-sample predictions with explicit error estimates

### 6.2 Where It Hits the Classical Wall

- Translating "survives small primes" into "is prime"
- Controlling large prime factors / smoothness
- Parity obstructions (sieve can't distinguish prime from semiprime)

These boundaries are the fault lines of sieve theory itself—not a failure of the lens.

### 6.3 The Singular Series Caveat

The corridor laws omit the singular series constant:
$$1 - \frac{2}{p} = \left(1 - \frac{1}{p}\right)^2 \cdot \left(1 - \frac{1}{(p-1)^2}\right)$$

The extra factor converges (it's part of the twin prime constant), but affects 1-5% level comparisons.

### 6.4 Exceptional Primes

For twins, p = 2 is exceptional (two forbidden classes collapse into one). The 6k±1 wheel handles this, but it must be tracked explicitly for other patterns.

---

## 7. Summary

**What we have**: A clean "interface" to sieve structure that makes dimension drop and conditioning effects viscerally visible.

**What it predicts**: Log-log exponential laws for elimination times, with rate = number of live forms.

**What it doesn't do**: Solve the twin prime conjecture or create information from nothing.

**Why it's useful**: It's a discovery machine—suggests new observables (p₁, p₂, α, run lengths) and makes their distributions predictable.

---

## References

- Mertens' theorems (asymptotic products over primes)
- Buchstab's function (smallest prime factor distribution)
- Kubilius model (probabilistic structure of prime factors)
- Hardy-Littlewood singular series (pattern-specific constants)

---

*Document created: 2026-01-04*
*Context: Following thread from twin prime selection bias project*
