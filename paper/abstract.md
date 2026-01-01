# A Selection Bias in the Prime Factors of Twin Prime Candidates

## Abstract

We document a selection bias in the arithmetic structure of composite numbers adjacent to primes. Among pairs $(6k-1, 6k+1)$ classified by the primality of each member, the composite element of a "prime-composite" pair has approximately 3% more distinct prime factors than composites in "composite-composite" pairs.

At $K = 10^9$ pairs, the mean $\omega$ (distinct prime factors) for PC composites is $2.907$ versus $2.824$ for CC composites, a relative uplift of $2.93\%$. This bias is stable across three orders of magnitude in $K$ and persists (slightly elevated) in tail windows, ruling out transient or drifting artifacts.

We derive the bias from first principles. For each prime $p \geq 5$, the congruence classes $k \pmod p$ where $p \mid (6k-1)$ and $p \mid (6k+1)$ are disjoint. This mutual exclusivity implies that conditioning on "$a$ is prime" boosts the probability that $p \mid b$ from $1/p$ to $1/(p-1)$. Summing the per-prime increments yields a convergent sum $\sum_{p \geq 5} 1/[p(p-1)] = 0.1065$, which predicts 78% of the observed effect.

The remaining discrepancy arises from residual correlations between primes, quantified via a transfer-matrix model. The calibration factor $c = 0.78$ encodes the net effect of these correlations.

**Keywords:** twin primes, prime factors, selection bias, Hardy-Littlewood, sieve methods

**MSC 2020:** 11N05, 11A41, 11Y11
