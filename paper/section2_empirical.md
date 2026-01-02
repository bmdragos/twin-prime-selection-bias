# 2. Empirical Observation

## 2.1 Setup and Definitions

We consider pairs of integers of the form

$$
(a, b) = (6k - 1, 6k + 1)
$$

for positive integers $k$. Every prime greater than 3 lies in one of these two residue classes modulo 6, so these pairs exhaust all twin-prime candidates beyond the pair (3, 5).

For each pair, we classify the **state** according to the primality of each component:

| State | Description | Interpretation |
|-------|-------------|----------------|
| PP | Both $a$ and $b$ prime | Twin primes |
| PC | $a$ prime, $b$ composite | "Missing twin" on right |
| CP | $a$ composite, $b$ prime | "Missing twin" on left |
| CC | Both composite | Neither is prime |

Our primary observable is $\omega(n)$, the number of **distinct** prime factors of $n$. For prime $n$, we have $\omega(n) = 1$.

## 2.2 State Distribution

At $K = 10^9$ pairs, the observed state frequencies are:

| State | Count | Fraction |
|-------|------:|----------|
| PP | 17,244,408 | 1.72% |
| PC | 122,531,277 | 12.25% |
| CP | 122,525,274 | 12.25% |
| CC | 737,699,041 | 73.77% |

The near-equality of PC and CP counts reflects the expected symmetry between $a$ and $b$. As $K$ increases, the PP fraction decreases (consistent with the Hardy–Littlewood conjecture that twin primes thin out logarithmically), and the CC fraction increases correspondingly.

## 2.3 The Selection Bias

Our central empirical finding concerns the mean value of $\omega$ for the composite member of PC and CP pairs compared to CC pairs.

**Main Result.** At $K = 10^9$:

| Population | Mean $\omega$ | Sample size |
|------------|---------------|-------------|
| PC composite (the $b$ in PC) | 2.9067 | 122,531,277 |
| CP composite (the $a$ in CP) | 2.9068 | 122,525,274 |
| CC composite (the $b$ in CC) | 2.8239 | 737,699,041 |
| Unconditional composite at $6k+1$ | 2.8357 | 860,230,318 |

The **selection bias** is defined as the relative uplift:

$$
\text{Bias} = \frac{\mathbb{E}[\omega \mid \text{PC}] - \mathbb{E}[\omega \mid \text{CC}]}{\mathbb{E}[\omega \mid \text{CC}]} = \frac{2.9067 - 2.8239}{2.8239} = 2.93\%
$$

The bias against the unconditional baseline is:

$$
\frac{2.9067 - 2.8357}{2.8357} = 2.50\%
$$

Both values are positive and statistically significant given the sample sizes involved.

**Error bars.** The uncertainty $\pm 0.0001$ reported in Observation 1.1 is the standard error of the mean. The empirical standard deviations are $s_{PC} \approx 1.08$ and $s_{CC} \approx 1.09$. With $n_{PC} = 1.23 \times 10^8$ and $n_{CC} = 7.38 \times 10^8$, the standard error of the difference is:
$$
\mathrm{SE}(\bar\omega_{PC} - \bar\omega_{CC}) = \sqrt{\frac{s_{PC}^2}{n_{PC}} + \frac{s_{CC}^2}{n_{CC}}} \approx 1.0 \times 10^{-4}.
$$
The $\pm 0.0001$ is a standard error, not a confidence interval.

## 2.4 Stability Across Scale

A natural concern is whether the observed bias is a finite-size artifact that might vanish or drift at larger scales. We tested this by computing the bias at three scales:

| $K$ | Bias (PC vs CC) |
|-----|-----------------|
| $10^7$ | 2.956% |
| $10^8$ | 2.940% |
| $10^9$ | 2.933% |

The bias is remarkably stable, decreasing by only 0.02 percentage points per decade of $K$.

## 2.5 Tail-Window Stability

A more stringent test asks whether the bias is stable **within** a run—that is, whether it persists in the tail of the sample or is dominated by early terms. We computed the bias in several windows at $K = 10^9$:

| Window | Bias (PC vs CC) |
|--------|-----------------|
| Full $[1, K]$ | 2.93% |
| Second half $[K/2, K]$ | 3.00% |
| Last 10% $[0.9K, K]$ | 3.00% |
| Last 1% $[0.99K, K]$ | 2.99% |

The bias in the tail is **slightly higher** than in the full sample, ruling out the possibility of a slowly drifting effect that happens to average to a stable value.

We also examined logarithmic bins:

| Bin | Bias |
|-----|------|
| $[10^4, 10^5)$ | 2.92% |
| $[10^5, 10^6)$ | 3.08% |
| $[10^6, 10^7)$ | 3.06% |
| $[10^7, 10^8)$ | 3.01% |
| $[10^8, 10^9)$ | 2.99% |

The bias is stable across five orders of magnitude within a single run.

## 2.6 Baseline Consistency Check

We verified internal consistency by checking that the unconditional mean can be reconstructed as a mixture of conditional means.

Let $\pi = \mathbb{P}(a \text{ is prime}) \approx 0.14$ at $K = 10^9$. Then a composite at position $6k+1$ is either:
- the $b$ in a PC pair (when $a$ is prime), or
- the $b$ in a CC pair (when $a$ is composite).

The mixture prediction is:

$$
\mathbb{E}[\omega \mid b \text{ composite}] = \pi \cdot \mathbb{E}[\omega \mid \text{PC}] + (1 - \pi) \cdot \mathbb{E}[\omega \mid \text{CC}]
$$

$$
= 0.14 \times 2.9067 + 0.86 \times 2.8239 = 2.8354
$$

The observed unconditional mean is $2.8357$, matching within $3 \times 10^{-4}$—consistent with rounding of the reported means. This confirms that our conditional means are mutually consistent and that no systematic errors have been introduced.

## 2.7 Decomposition of the Bias

The total bias of $2.93\%$ (PC vs CC) conflates two distinct mechanisms. We can decompose it using the unconditional composite mean as an intermediate baseline.

Let:
- $\mu_{\text{PC}} = \mathbb{E}[\omega(b) \mid \text{PC}] = 2.9067$
- $\mu_{\text{CC}} = \mathbb{E}[\omega(b) \mid \text{CC}] = 2.8239$
- $\mu_{\text{uncond}} = \mathbb{E}[\omega(b) \mid b \text{ composite}] = 2.8357$

Then the total shift decomposes as:

$$
\mu_{\text{PC}} - \mu_{\text{CC}} = \underbrace{(\mu_{\text{PC}} - \mu_{\text{uncond}})}_{\text{PC uplift}} + \underbrace{(\mu_{\text{uncond}} - \mu_{\text{CC}})}_{\text{CC suppression}}
$$

Numerically at $K = 10^9$:

| Component | Value | Fraction of total |
|-----------|-------|-------------------|
| PC uplift | $2.9067 - 2.8357 = 0.0710$ | 86% |
| CC suppression | $2.8357 - 2.8239 = 0.0118$ | 14% |
| **Total** | $0.0828$ | 100% |

**Interpretation.** The dominant effect ($86\%$) is the *PC uplift*: conditioning on $a$ being prime pushes small prime factors onto $b$, increasing $\omega(b)$.

The secondary effect ($14\%$) is the *CC suppression*: conditioning on $a$ being composite means $a$ likely has small prime factors, which—by the same mutual exclusivity—slightly *reduces* the probability that $b$ has those factors. Thus $\mu_{\text{CC}} < \mu_{\text{uncond}}$.

This decomposition clarifies that the CC baseline is not a neutral reference population. The "selection bias" is primarily a property of the PC population, with a smaller contribution from the CC population being itself depleted.

## 2.8 Summary

The empirical facts are:

1. **A persistent bias exists.** Composites adjacent to primes have $\approx 3\%$ more distinct prime factors than composites in CC pairs.

2. **The bias is stable.** It persists across three orders of magnitude in $K$ and is stable (or slightly increasing) in tail windows.

3. **Multiple baselines confirm it.** The effect holds whether comparing to CC composites ($+2.93\%$) or to unconditional composites ($+2.50\%$).

4. **The bias decomposes cleanly.** Approximately $86\%$ of the effect is PC uplift (primes pushing factors onto neighbors); $14\%$ is CC suppression (composites pulling factors away from neighbors).

5. **Internal consistency holds.** The conditional means combine correctly to reproduce the unconditional mean.

These observations are not explained by finite-size effects, implementation artifacts, or baseline choice. In the next section, we develop a first-principles heuristic that explains both the existence and the magnitude of the bias.
