# 5. Discussion

## 5.1 Why This Is Not Surprising

The selection bias we observe is, in hindsight, predictable from first principles. The mutual exclusivity of divisibility (Proposition 3.1) is an elementary fact about modular arithmetic, and the resulting bias follows by linearity of expectation. A skeptical reader might reasonably ask: *what is new here?*

We offer three responses.

**The quantitative match is nontrivial.** While the qualitative direction of the bias (positive) follows from mutual exclusivity, the quantitative match is striking: the convergent sum $\sum 1/[p(p-1)] = 0.1065$ predicts $\mathbb{E}[\omega(b) \mid a \text{ prime}] - \mathbb{E}[\omega(b) \mid a \text{ composite}]$ to within $1\%$. This near-exact agreement was not guaranteed a priori.

**The stability requires explanation.** The bias persists unchanged across five orders of magnitude in $K$ and is stable (or slightly increasing) in tail windows. This rules out transient effects or slowly drifting artifacts. The convergence of the infinite sum explains this stability: almost all contribution comes from primes below 100.

**The baseline ambiguity is instructive.** The effect size changes from 2.93% (vs CC) to 2.50% (vs unconditional) depending on the baseline. This illustrates how conditioning can create statistical artifacts even when the underlying mechanism is simple. Similar ambiguities arise throughout computational number theory.

## 5.2 Implications for Twin Prime Heuristics

The Hardy-Littlewood conjecture predicts that the number of twin primes below $N$ is asymptotically:

$$
\pi_2(N) \sim 2C_2 \frac{N}{(\log N)^2}
$$

where $C_2 = \prod_{p \geq 3} (1 - 1/(p-1)^2) \approx 0.66$ is the twin prime constant.

Our work does not bear directly on this conjecture, but it illuminates the arithmetic structure that underlies it. The same mutual exclusivity that drives the selection bias also appears in the twin prime constant: the factor $(1 - 1/(p-1)^2)$ arises because a twin prime pair must avoid both residue classes $k \equiv \pm 1/6 \pmod{p}$.

**A complementary perspective.** Twin prime heuristics ask: "Given random $k$, what is the probability that both $6k-1$ and $6k+1$ are prime?" Our selection bias asks the inverse question: "Given that $6k-1$ is prime, how does the arithmetic structure of $6k+1$ change?" Both questions reduce to the same local mechanism.

## 5.3 Computational Lessons

Scaling to $K = 10^9$ required careful optimization:

- **GPU-side aggregation** reduced data transfer by $10^6\times$ (from 8GB per prime to 64 bytes).
- **Unified memory** on the NVIDIA GB10 eliminated traditional CPU-GPU copy bottlenecks.
- **Shared memory multiprocessing** avoided serialization overhead for large arrays.

The dominant cost at $K = 10^9$ is the sieve itself (100s of 280s total), not the bias computation. Further scaling to $K = 10^{10}$ is feasible with segmented sieves and distributed memory.

## 5.4 Open Questions

**1. Can the conditioning transformation be made explicit?**

The PC-vs-CC difference of $0.0828$ relates to the full conditional difference of $0.107$ through the conditioning on "$b$ composite." A closed-form expression for this transformation would clarify the relationship between the two statistics.

**2. Does the bias extend to other prime patterns?**

The mutual exclusivity mechanism applies to any pair $(a, b) = (f(k), g(k))$ where $f(k) - g(k)$ is small and coprime to large primes. Sophie Germain pairs $(p, 2p+1)$ and prime $k$-tuples are natural candidates.

**3. What is the distribution of $\omega$ conditional on primality?**

We have computed means. The full distribution of $\omega(b) \mid a$ prime could reveal higher-order structure. Is the variance also elevated? What about the maximum?

## 5.5 Why This Is Useful

Despite being "not surprising," the selection bias has practical value:

**For prime-testing heuristics.** If one member of a twin-candidate pair is known to be prime, the composite partner is slightly more likely to have many small factors. This could inform probabilistic primality tests that use trial division.

**For understanding conditioned populations.** The bias is a clean example of how conditioning on one variable (primality of $a$) affects the distribution of a related variable (prime factors of $b$). Similar selection effects appear throughout statistics and machine learning.

**As a pedagogical example.** The derivation---from exact local identity, through heuristic sum, to empirical verification---illustrates the interplay between rigor and heuristic in analytic number theory. The same pattern appears in the Hardy-Littlewood circle method, sieve theory, and random matrix conjectures.

## 5.6 Conclusion

We have documented a 3% selection bias in the number of distinct prime factors of composite numbers adjacent to primes. The bias:

- Is persistent across three orders of magnitude in sample size
- Is stable or slightly increasing in tail windows
- Admits a first-principles explanation via mutual exclusivity
- Is quantitatively predicted (to within $1\%$) by an elementary convergent sum, when compared to the correctly-aligned conditional expectation

The phenomenon is simple, the explanation is elementary, and the quantitative agreement is striking. Sometimes the most instructive results are those that confirm what "should" be true---and verify the heuristic to high precision.
