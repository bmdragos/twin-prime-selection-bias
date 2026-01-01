# 4. Residue-Class Model and Inclusion-Exclusion

## 4.1 Motivation

The independent-prime heuristic of Section 3 overshoots the empirical bias by $\approx 22\%$. This discrepancy arises because divisibility events are not independent: after conditioning on the primality of $a = 6k-1$, the divisibility of $b = 6k+1$ by different primes $p$ and $q$ becomes correlated.

To quantify these correlations exactly (for any finite set of primes), we formulate a residue-class model using inclusion-exclusion over the joint divisibility states. While this model can be written in matrix form, we do not invoke any spectral gap or decay-of-correlations results; it serves only as a bookkeeping device for exact computation of conditional expectations.

## 4.2 State Space

Fix a finite set $\mathcal{P} = \{p_1, \ldots, p_r\}$ of primes with $p_i \geq 5$. For each $k$, the residue of $k$ modulo $\prod_{p \in \mathcal{P}} p$ determines which primes divide $a = 6k-1$ and $b = 6k+1$.

Define the **state** of $k$ to be the pair $(\sigma_a, \sigma_b)$ where:
- $\sigma_a \subseteq \mathcal{P}$ is the set of primes dividing $a$
- $\sigma_b \subseteq \mathcal{P}$ is the set of primes dividing $b$

By the mutual exclusivity from Section 3.1, we have $\sigma_a \cap \sigma_b = \emptyset$: no prime can divide both $a$ and $b$.

The state space has size $3^r$ (each prime is in $\sigma_a$, in $\sigma_b$, or in neither), not $4^r$, due to this constraint.

## 4.3 Stationary Distribution

As $k$ ranges over $\{1, 2, \ldots, K\}$, the states are uniformly distributed over the $\prod_{p \in \mathcal{P}} p$ residue classes. The stationary probability of state $(\sigma_a, \sigma_b)$ is:

$$
\pi(\sigma_a, \sigma_b) = \prod_{p \in \sigma_a} \frac{1}{p} \cdot \prod_{p \in \sigma_b} \frac{1}{p} \cdot \prod_{p \notin \sigma_a \cup \sigma_b} \frac{p-2}{p}.
$$

The factor $(p-2)/p$ accounts for the $p - 2$ residue classes where $p$ divides neither $a$ nor $b$.

## 4.4 Conditional Expectations

The four macroscopic states (PP, PC, CP, CC) partition the state space based on whether $\sigma_a = \emptyset$ (i.e., $a$ is "prime" with respect to $\mathcal{P}$) and similarly for $b$.

For a random variable $f(\sigma_a, \sigma_b)$, the conditional expectation given macrostate $S$ is:

$$
\mathbb{E}[f \mid S] = \frac{\sum_{(\sigma_a, \sigma_b) \in S} f(\sigma_a, \sigma_b) \cdot \pi(\sigma_a, \sigma_b)}{\sum_{(\sigma_a, \sigma_b) \in S} \pi(\sigma_a, \sigma_b)}.
$$

Taking $f = |\sigma_b|$ (the number of primes in $\mathcal{P}$ dividing $b$) gives a finite-prime approximation to $\omega(b)$.

## 4.5 Explicit Computation for Small $\mathcal{P}$

**Example: $\mathcal{P} = \{5\}$.**

The three states are:
- $(5 \mid a, 5 \nmid b)$: probability $1/5$
- $(5 \nmid a, 5 \mid b)$: probability $1/5$
- $(5 \nmid a, 5 \nmid b)$: probability $3/5$

The macrostates:
- PP: $\sigma_a = \sigma_b = \emptyset$, probability $3/5$
- PC: $\sigma_a = \emptyset, \sigma_b = \{5\}$, probability $1/5$
- CP: $\sigma_a = \{5\}, \sigma_b = \emptyset$, probability $1/5$
- CC: impossible with only one prime

This recovers the mutual exclusivity: $\mathbb{P}(5 \mid b \mid 5 \nmid a) = (1/5) / (4/5) = 1/4 = 1/(5-1)$.

**Example: $\mathcal{P} = \{5, 7\}$.**

Nine states (since $\sigma_a \cap \sigma_b = \emptyset$). The CC macrostate requires $|\sigma_a| \geq 1$ and $|\sigma_b| \geq 1$, which has probability:

$$
\mathbb{P}(\text{CC}) = \frac{1}{5} \cdot \frac{1}{7} + \frac{1}{7} \cdot \frac{1}{5} = \frac{2}{35}.
$$

The conditional expectation $\mathbb{E}[|\sigma_b| \mid \text{CC}] = 1$ (exactly one prime divides $b$ in each CC state).

## 4.6 Convergence to the Singular-Series Correction

As $|\mathcal{P}| \to \infty$, the model expectations converge to true expectations over all primes. Numerical experiments show:

$$
\lim_{|\mathcal{P}| \to \infty} \frac{\mathbb{E}[|\sigma_b| \mid \text{PC}] - \mathbb{E}[|\sigma_b| \mid \text{CC}]}{\sum_{p \in \mathcal{P}} 1/[p(p-1)]} \approx 0.78.
$$

This ratio is the same correction factor observed empirically in Section 3.3. Its appearance in the residue-class model confirms that the discrepancy from the naive heuristic is a consequence of the local correlations (mutual exclusivity) rather than finite-size effects or large-prime contributions.

## 4.7 Connection to Sieve Theory

The residue-class model is essentially an application of inclusion-exclusion over the sieve dimension. In the language of sieve theory:

1. **The sieve weights.** The probability measure $\pi(\sigma_a, \sigma_b)$ corresponds to sieve weights on the residue classes modulo the primorial $P_\# = \prod_{p \in \mathcal{P}} p$.

2. **The fundamental lemma.** The mutual exclusivity constraint ($\sigma_a \cap \sigma_b = \emptyset$) reduces the state space from $4^r$ to $3^r$, encoding the local density corrections that produce the singular series.

3. **Product structure.** The factorization $\pi(\sigma_a, \sigma_b) = \prod_p (\text{local factor})$ mirrors the Euler product structure in the Hardy-Littlewood constant $C_2$.

The model provides exact computation of conditional expectations for any finite $\mathcal{P}$, giving rigorous (if non-asymptotic) bounds on the correction factor.

## 4.8 Limitations

The residue-class model treats all primes symmetrically and assumes uniform distribution of $k$ over residue classes. It does not account for:

- Large prime factors (which contribute to $\omega$ but are not in any finite $\mathcal{P}$)
- The density of primes themselves (which decreases like $1/\log n$)

These effects are subdominant for the selection bias, which is driven by small primes, but they prevent the model from predicting absolute values of $\mathbb{E}[\omega]$.
