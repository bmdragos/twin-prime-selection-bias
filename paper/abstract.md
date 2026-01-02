# A Selection Bias in the Prime Factors of Twin Prime Candidates

## Abstract

We document a selection bias in the arithmetic structure of composite numbers adjacent to primes. Among pairs $(6k-1, 6k+1)$ classified by the primality of each member, the composite element of a "prime-composite" pair has approximately 3% more distinct prime factors than composites in "composite-composite" pairs.

At $K = 10^9$ pairs, the mean $\omega$ (distinct prime factors) for PC composites is $2.907$ versus $2.824$ for CC composites, a relative uplift of $2.93\%$. This bias is stable across three orders of magnitude in $K$ and persists (slightly elevated) in tail windows, ruling out transient or drifting artifacts.

We derive the bias from first principles. For each prime $p \geq 5$, the congruence classes $k \pmod p$ where $p \mid (6k-1)$ and $p \mid (6k+1)$ are disjoint. This mutual exclusivity implies $\mathbb{P}(p \mid b \mid p \nmid a) = 1/(p-1)$, an exact identity. Under the heuristic that primes are equidistributed among allowed residue classes, this extends to $\mathbb{P}(p \mid b \mid a \text{ prime}) \approx 1/(p-1)$. Summing the per-prime increments yields a convergent sum $\sum_{p \geq 5} 1/[p(p-1)] = 0.1065$.

When compared to the correctly-aligned conditional expectation $\mathbb{E}[\omega(b) \mid a \text{ prime}] - \mathbb{E}[\omega(b) \mid a \text{ composite}] = 0.107$, the heuristic matches empirical data to within $1\%$. The PC-vs-CC composite difference of $0.0828$ is smaller because it further conditions on "$b$ composite," which modifies expectations through elementary conditioning algebra.

**Keywords:** twin primes, prime factors, selection bias, Hardy-Littlewood, sieve methods

**MSC 2020:** 11N05, 11A41, 11Y11
