# 1. Introduction

## 1.1 Motivation

Consider the integers of the form $6k \pm 1$ for positive integers $k$. Every prime greater than 3 belongs to one of these two residue classes modulo 6, and the pairs $(6k-1, 6k+1)$ exhaust all twin prime candidates beyond $(3, 5)$.

For each such pair, we may ask: if exactly one member is prime, what can we say about the arithmetic structure of the composite partner?

Naively, one might expect the composite to be "generic"---statistically indistinguishable from a random composite of similar size. This paper shows that this expectation is false. Composites adjacent to primes have systematically *more* distinct prime factors than composites in pairs where both members are composite.

## 1.2 Main Results

Let $\omega(n)$ denote the number of distinct prime factors of $n$. We classify pairs $(a, b) = (6k-1, 6k+1)$ into four states:

- **PP:** Both $a$ and $b$ are prime (twin primes)
- **PC:** $a$ is prime, $b$ is composite
- **CP:** $a$ is composite, $b$ is prime
- **CC:** Both are composite

**Observation 1.1** (Empirical). At $K = 10^9$ pairs:

$$
\mathbb{E}[\omega(b) \mid \text{PC}] - \mathbb{E}[\omega(b) \mid \text{CC}] = 0.0828 \pm 0.0001
$$

corresponding to a relative uplift of $2.93\%$.

**Observation 1.2** (Stability). The selection bias is stable:
- Across scale: $2.96\%, 2.94\%, 2.93\%$ at $K = 10^7, 10^8, 10^9$
- Within runs: $3.00\%$ in the tail $[0.9K, K]$ vs $2.93\%$ in the full sample

**Proposition 1.3** (Heuristic). The bias is explained by mutual exclusivity of divisibility. For each prime $p \geq 5$, the identity $\mathbb{P}(p \mid b \mid p \nmid a) = 1/(p-1)$ is exact. Under the heuristic that primes are equidistributed among allowed residue classes modulo $p$, this extends to $\mathbb{P}(p \mid b \mid a \text{ prime}) \approx 1/(p-1)$. Summing the per-prime increments:

$$
\sum_{p \geq 5} \frac{1}{p(p-1)} = 0.1065
$$

This predicts $\mathbb{E}[\omega(b) \mid a \text{ prime}] - \mathbb{E}[\omega(b) \mid a \text{ composite}]$. Empirically, this difference is $0.107$, matching the prediction to within $1\%$.

## 1.3 Context

The Hardy-Littlewood conjecture and its refinements predict the density of prime pairs, but the *internal structure* of near-prime pairs has received less attention. Our result complements density predictions with a structural observation: the "near-miss" composites adjacent to primes are arithmetically distinguished.

Related phenomena include:

- **Chebyshev's bias:** Among primes, there are slightly more congruent to $3 \pmod 4$ than to $1 \pmod 4$ in initial segments.
- **Prime races:** The relative densities of primes in different residue classes fluctuate in predictable ways.
- **Cramer's model:** Random models of primes fail to capture local correlations induced by small primes.
- **Chen-type results:** Sieve methods have long studied the factorization of $p + 2$ when $p$ is prime, showing that $p + 2$ is either prime or a product of two primes for infinitely many $p$ [6].

Our selection bias is of a different character: it concerns the *mean* number of prime factors of composites adjacent to primes, not the extremal factorization structure. Nevertheless, the underlying mechanism (mutual exclusivity of divisibility) is the same local constraint that drives sieve-theoretic results on shifted primes.

## 1.4 Methods

The computation required processing $10^9$ pairs, each requiring:
1. Primality classification (via sieving)
2. Complete factorization (via smallest-prime-factor lookup)
3. State-conditional aggregation

We achieved this in under 5 minutes on an NVIDIA DGX Spark (GB10, Blackwell architecture) using:
- A segmented sieve with $\sim 10^3$ workers
- GPU-accelerated $\omega$-computation with on-device aggregation
- Unified memory to eliminate CPU-GPU transfer bottlenecks

The implementation is available at [repository URL].

## 1.5 Organization

- **Section 2** presents the empirical observations: state distributions, selection bias, and stability across scale and within runs.
- **Section 3** derives the bias from first principles via the mutual exclusivity mechanism.
- **Section 4** formalizes the conditioning algebra using a residue-class model.
- **Section 5** discusses implications, open questions, and why the result is "not surprising, but still useful."
