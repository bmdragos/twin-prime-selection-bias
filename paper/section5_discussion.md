# 5. Discussion

## 5.1 What Is (and Isn't) New Here

The $1/(p-1)$ factor in Proposition 3.1 is not new—it is the standard sifting density that appears when applying the linear sieve to shifted-prime sets like $\{p + 2 : p \leq x\}$. The formula has been understood since at least Brun's work on twin primes.

**Prior work on additive functions near primes.** The study of additive functions on shifted primes has a substantial literature. Elliott's *Probabilistic Number Theory* [7, 8] develops the general theory of additive functions on arithmetic progressions and shifted sequences, providing the probabilistic framework for understanding how primality conditions propagate through arithmetic structure. Erdős and Pomerance [9] studied $\omega(\varphi(n))$ and established foundational results on the distribution of prime factors of shifted primes $p-1$, showing that the mean of $\omega(p-1)$ over primes $p \leq x$ is asymptotically $(\log \log x)^2 / 2$. More recently, Dixit and collaborators have analyzed $\omega_y(p+a)$ (the count of prime factors up to $y$) with explicit error terms. These works establish that primes are equidistributed in residue classes and that conditioning on primality affects the local density of small factors.

A skeptical reader might reasonably ask: *if the mechanism is standard sieve theory, what is the contribution here?*

We offer three responses.

**The bridge to additive functions is explicit.** While the local density $1/(p-1)$ is standard, its consequence for additive functions like $\omega$ is typically left implicit. We make the connection explicit: summing the per-prime increments gives a constant $\sum 1/[p(p-1)] = 0.1065$ that predicts the mean shift in $\omega$ to within $1\%$. This is the simplest possible consequence of "one residue class is forbidden when the neighbor is prime," but stating it as a prediction with a constant (rather than vague heuristics) is what allows validation.

**The numerical validation is precise.** The prediction $\mathbb{P}(p \mid b \mid a \text{ prime}) = 1/(p-1)$ is confirmed to six decimal places at $K = 10^9$. The mean shift $0.1074$ matches the prediction $0.1065$ to within $1\%$. This precision rules out alternative explanations and confirms that the linearity argument captures the dominant effect.

**The decomposition separates small and large factors.** The transfer-matrix model naturally predicts $\omega_{\text{small}}$ (the count of small prime factors). Full $\omega$ includes a "large prime cofactor" contribution that requires separate analysis. The decomposition in Section 5.4 cleanly separates these effects: small primes contribute a 4.4% bias, large-prime cofactors cancel $\sim$13% of the absolute gap, yielding the observed 2.93% in full $\omega$.

## 5.2 Implications for Twin Prime Heuristics

The Hardy-Littlewood conjecture predicts that the number of twin primes below $N$ is asymptotically:

$$
\pi_2(N) \sim 2C_2 \frac{N}{(\log N)^2}
$$

where $C_2 = \prod_{p \geq 3} (1 - 1/(p-1)^2) \approx 0.66$ is the twin prime constant.

Our work does not bear directly on this conjecture, but it illuminates the arithmetic structure that underlies it. The same mutual exclusivity that drives the selection bias also appears in the twin prime constant: the factor $(1 - 1/(p-1)^2)$ arises because a twin prime pair must avoid both residue classes $k \equiv \pm 1/6 \pmod{p}$.

**A complementary perspective.** Twin prime heuristics ask: "Given random $k$, what is the probability that both $6k-1$ and $6k+1$ are prime?" Our selection bias asks the inverse question: "Given that $6k-1$ is prime, how does the arithmetic structure of $6k+1$ change?" Both questions reduce to the same local mechanism.

**The parity barrier.** Standard sieves cannot reliably separate integers by the parity of their number of prime factors—this is the classic parity obstruction. Results like Chen's theorem (every sufficiently large even number is the sum of a prime and a product of at most two primes) exist precisely because "prime or almost-prime" is what sieve methods can access without breaking the parity barrier. Our phenomenon lives comfortably within the sieve world: it concerns conditioning and additive functions, not the much harder problem of forcing exact primality in both components of a pair.

## 5.3 Computational Lessons

Scaling to $K = 10^9$ required careful optimization:

- **GPU-side aggregation** reduced data transfer by $10^6\times$ (from 8GB per prime to 64 bytes).
- **Unified memory** on the NVIDIA GB10 eliminated traditional CPU-GPU copy bottlenecks.
- **Shared memory multiprocessing** avoided serialization overhead for large arrays.

The dominant cost at $K = 10^9$ is the sieve itself (100s of 280s total), not the bias computation. Further scaling to $K = 10^{10}$ is feasible with segmented sieves and distributed memory.

## 5.4 Open Questions

**1. Is the PC-vs-CC bias derivable from Hardy-Littlewood asymptotics?**

The full conditional difference of $0.1074$ follows from the convergent sum plus equidistribution. The PC-vs-CC composite difference of $0.0828$ additionally depends on the PP/PC/CP/CC proportions, which we measured empirically. Hardy-Littlewood heuristics predict the asymptotic density of twin primes (PP pairs), and by extension should determine the other state proportions. Experts in sieve theory may already know this derivation; we pose it as a question for readers unfamiliar with the folklore.

**2. Does the bias extend to other admissible patterns?**

The mutual exclusivity mechanism is not specific to twin primes. For any pair of linear forms $(a, b) = (f(k), g(k))$, if $\gcd(f(k) - g(k), p) = 1$ for primes $p$ not dividing the leading coefficients, then the same disjointness holds: the residue classes where $p \mid a$ and $p \mid b$ are distinct.

This suggests a general principle: **any admissible prime tuple should exhibit the same "neighbor-survival inflates $\omega$" effect with predictable constants**. Demonstrating this across multiple patterns would elevate the result from a twin-prime curiosity to a general phenomenon.

**Sophie Germain pairs $(n, 2n+1)$.** If $q \mid n$ and $q \mid (2n+1)$, then $q \mid (2n+1 - 2n) = 1$, which is impossible. So mutual exclusivity holds, and we predict:
$$\mathbb{E}[\omega(2n+1) \mid n \text{ prime}] - \mathbb{E}[\omega(2n+1) \mid n \text{ composite}] \approx \sum_{q \geq 3} \frac{1}{q(q-1)} \approx 0.273$$
(The sum starts at $q = 3$ because $2 \mid (2n+1)$ is impossible for any integer $n$.)

At $N = 10^9$ among odd $n$ (GPU-accelerated, see `src/experiments/exp_sophie_germain_gpu.py`):

| Population | Mean $\omega(2n+1)$ | Sample size |
|------------|---------------------|-------------|
| $n$ prime | 3.083 | 50,847,533 |
| $n$ composite (odd) | 2.811 | 449,152,466 |
| **Difference** | **0.272** | — |

The empirical difference of $0.272$ matches the predicted $0.273$ to within **0.4%**. (Note: we restrict to odd $n$ because even $n$ are never prime except $n=2$, so including them contaminates the composite baseline.)

**Cousin primes $(n, n+4)$.** Same analysis: $q \mid n$ and $q \mid (n+4)$ implies $q \mid 4$, so only $q = 2$ fails mutual exclusivity. For $6k \pm 1$ candidates: if $n = 6k-1$, then $n+4 = 6k+3 \equiv 0 \pmod 3$ always; if $n = 6k+1$, then $n+4 = 6k+5 \not\equiv 0 \pmod 3$. Since $3 \mid (n+4)$ is determined by residue class (not primality of $n$)—and prime/composite $n$ have the same residue-class mix among $6k \pm 1$—the $p=3$ term contributes zero to the *difference*. The sum starts at $p = 5$: $\sum_{p \geq 5} 1/[p(p-1)] = 0.1065$.

At $K = 10^8$ among $6k \pm 1$ candidates (see `src/experiments/exp_cousin_primes_gpu.py`):

| Population | Mean $\omega(n+4)$ | Sample size |
|------------|-------------------|-------------|
| $n$ prime | 3.024 | 31,324,702 |
| $n$ composite | 2.918 | 168,675,298 |
| **Difference** | **0.106** | — |

The empirical difference of $0.1063$ matches the predicted $0.1064$ to within **0.1%**—even better than the twin prime result. This confirms the mechanism generalizes to other admissible patterns with constant gap.

**Future directions.** Natural extensions include prime 3-tuples like $(n, n+2, n+6)$ and longer admissible constellations. If the predicted constants match across all tested patterns, it establishes that "conditioning on rare arithmetic structure induces a quantifiable tilt in additive functions"—a principle with potential applications beyond prime tuples.

**3. What is the full distribution of $\omega$ under conditioning?**

The Erdős–Kac theorem states that $\omega(n)$, suitably centered and scaled, is asymptotically normal for random integers. A natural extension asks: does $\omega_{\text{small}}$ remain approximately normal under our conditioning, with a shifted mean? Does the variance shift as the Bernoulli-sum model predicts?

Preliminary analysis at $K = 10^6$ shows that the **variance is also elevated**:

| State | Mean $\omega(b)$ | Variance |
|-------|------------------|----------|
| PC (composite $b$) | 2.607 | 0.491 |
| CC (composite $b$) | 2.533 | 0.439 |

The variance for PC composites is 12% higher than for CC composites. This suggests the selection bias affects not just the central tendency but the entire distribution. A full characterization—showing that the conditioned populations satisfy an Erdős–Kac-type law with modified parameters—would turn "a mean bias" into "a full probabilistic law under conditioning."

**4. Decomposition: small primes vs. large prime cofactors**

The full bias decomposes into two competing effects. For composites $n \leq N$, at most one prime factor exceeds $\sqrt{N}$, so we have the exact identity (not a heuristic):

$$\omega(n) = \omega_{\text{small}}(n) + \mathbf{1}\{\exists\, p > \sqrt{N} : p \mid n\}$$

At $K = 10^9$ (where $\sqrt{N} = 77{,}459$):

| Component | PC | CC | Difference |
|-----------|-----|-----|------------|
| $\omega_{\text{small}}$ (factors $\leq \sqrt{N}$) | 2.231 | 2.136 | $+0.095$ |
| Has large prime factor $(> \sqrt{N})$ | 67.6% | 68.8% | $-0.012$ |
| **Full $\omega$** | 2.907 | 2.824 | $+0.083$ |

(Rows add exactly: $2.231 + 0.676 = 2.907$ and $2.136 + 0.688 = 2.824$.)

The small-prime component shows a **4.4% bias**—larger than the 2.93% bias in full $\omega$. CC composites are *more likely* to have a large prime cofactor (68.8% vs 67.6%), which partially offsets the small-prime advantage. The cancellation fraction is $0.012/0.095 \approx 12.6\%$, i.e., **large primes erase ~13% of the absolute small-prime gap**. (Computation time: 11 minutes on DGX Spark.)

**Smoothness interpretation.** "Has large prime" is equivalent to "not $\sqrt{N}$-smooth." The smoothness rates are:
- PC composites: $1 - 0.676 = 32.4\%$ are $\sqrt{N}$-smooth
- CC composites: $1 - 0.688 = 31.2\%$ are $\sqrt{N}$-smooth

PC composites are ~1.2 percentage points more likely to be smooth. This is exactly what the mechanism predicts: more small prime factors → more of the integer's "mass" is accounted for below $\sqrt{N}$ → less room for a large cofactor.

For comparison, the Dickman function gives $\rho(2) = 1 - \ln 2 \approx 0.307$ as the probability that a random integer $n$ is $\sqrt{n}$-smooth. The CC value $1 - 0.688 = 0.312$ is close to this "ambient" rate, while PC is slightly higher (more smooth), as the mechanism predicts.

**The transfer matrix and full $\omega$.** The transfer-matrix model tracks "hits" by primes $p \leq \sqrt{N}$, so it naturally predicts $\omega_{\text{small}}$. To predict full $\omega$, one needs a separate estimate of $\mathbb{P}(\text{large cofactor} \mid \text{state})$—a state-dependent smoothness probability. The decomposition above shows this probability is not independent of $\omega_{\text{small}}$: more small factors correlate with fewer large cofactors. A tighter theoretical treatment would model this coupling explicitly.

## 5.5 Why This Is Useful

Despite being "not surprising," the selection bias has practical value:

**As a niche observation for sieve practitioners.** If one member of a twin-candidate pair is known to be prime, the composite partner is slightly more likely to have many small factors. The effect size (~3%) is too small to be practically useful for primality testing, but it illustrates how conditioning propagates through arithmetic structure.

**For understanding conditioned populations.** The bias is a clean example of how conditioning on one variable (primality of $a$) affects the distribution of a related variable (prime factors of $b$). Similar selection effects appear throughout statistics and machine learning.

**As a pedagogical example.** The derivation---from exact local identity, through heuristic sum, to empirical verification---illustrates the interplay between rigor and heuristic in analytic number theory. The same pattern appears in the Hardy-Littlewood circle method, sieve theory, and random matrix conjectures.

## 5.6 Conclusion

We have documented a selection bias in the number of distinct prime factors of composite numbers adjacent to primes. The contribution is not the local density $1/(p-1)$—which is standard sieve theory—but the explicit bridge from local congruence facts to a measurable shift in the additive function $\omega$, validated numerically with high precision.

The bias:

- Is persistent across two orders of magnitude in sample size ($K = 10^7$ to $10^9$)
- Is stable or slightly increasing in tail windows
- Admits a first-principles explanation via mutual exclusivity
- Is quantitatively predicted (to within $1\%$) by the convergent sum $\sum_{p \geq 5} 1/[p(p-1)] = 0.1065$
- Decomposes cleanly: the transfer matrix predicts $\omega_{\text{small}}$, while a separate smoothness analysis explains the large-prime cofactor correction

The phenomenon generalizes beyond twin primes to Sophie Germain pairs and cousin primes, with predicted constants matching empirical values. This suggests a general principle: conditioning on rare arithmetic structure (a prime neighbor) induces a quantifiable tilt in additive functions, with the tilt predictable from elementary local densities.

The quantitative agreement is striking. Constants are where vague heuristics either become science or die—and this one survives.
