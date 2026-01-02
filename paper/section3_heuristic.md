# 3. First-Principles Heuristic

## 3.1 The Local Mechanism

The selection bias arises from a simple arithmetic identity. For a prime $p \geq 5$ and the pair $(a, b) = (6k-1, 6k+1)$:

- Exactly one residue class $k \pmod{p}$ satisfies $p \mid a$.
- Exactly one residue class $k \pmod{p}$ satisfies $p \mid b$.
- These two classes are **disjoint**: if both held, then $p \mid (b - a) = 2$, which is impossible for $p \geq 5$.

This mutual exclusivity is the engine of the bias. It implies that the events "$p \mid a$" and "$p \mid b$" are negatively correlated: knowing one fails makes the other slightly more likely.

**Figure 3.1: Mutual Exclusivity at $p = 5$.**

Consider the residue classes $k \bmod 5$:

```
k mod 5:    0       1       2       3       4
            │       │       │       │       │
  5|a?      ✗       ✓       ✗       ✗       ✗     (6k-1 ≡ 0 mod 5 when k ≡ 1)
  5|b?      ✗       ✗       ✗       ✗       ✓     (6k+1 ≡ 0 mod 5 when k ≡ 4)
            │       │       │       │       │
            └───────┴───────┴───────┴───────┘
                    ↑                   ↑
              "forbidden"         "forbidden"
               for b               for a
```

The key observation: the classes where $5 \mid a$ and where $5 \mid b$ are **disjoint**. When we condition on "$a$ is prime" (i.e., exclude $k \equiv 1$), the remaining 4 classes include the one where $5 \mid b$. Thus $\mathbb{P}(5 \mid b \mid 5 \nmid a) = 1/4$, not $1/5$.

**Proposition 3.1.** For any prime $p \geq 5$,

$$
\mathbb{P}(p \mid b \mid p \nmid a) = \frac{1}{p-1}.
$$

*Proof.* Among the $p$ residue classes modulo $p$, exactly one has $p \mid b$. Conditioning on $p \nmid a$ removes one class (the unique class with $p \mid a$), leaving $p - 1$ equally likely classes. Exactly one of these has $p \mid b$. $\square$

**Table 3.1: Per-Prime Verification.**

The local mechanism can be verified directly by measuring $\mathbb{P}(p \mid b \mid a \text{ prime})$ for small primes. Under the equidistribution heuristic, this should approximate $1/(p-1)$:

| $p$ | Predicted $\mathbb{P}(p \mid b \mid a \text{ prime})$ | Predicted $\mathbb{P}(p \mid b \mid a \text{ composite})$ | Increment |
|-----|------------------------------------------------------|----------------------------------------------------------|-----------|
| 5 | $1/4 = 0.2500$ | $\approx 1/5 = 0.2000$ | $0.0500$ |
| 7 | $1/6 = 0.1667$ | $\approx 1/7 = 0.1429$ | $0.0238$ |
| 11 | $1/10 = 0.1000$ | $\approx 1/11 = 0.0909$ | $0.0091$ |
| 13 | $1/12 = 0.0833$ | $\approx 1/13 = 0.0769$ | $0.0064$ |

Empirically, the measured values at $K = 10^9$ match these predictions to within statistical error, confirming that primes are approximately equidistributed among allowed residue classes.

The unconditional probability $\mathbb{P}(p \mid b) = 1/p$ is boosted to $1/(p-1)$ upon conditioning. The per-prime increment is:

$$
\delta_p = \frac{1}{p-1} - \frac{1}{p} = \frac{1}{p(p-1)}.
$$

## 3.2 The Independent-Prime Sum

We now make a heuristic leap: assume that the increments $\delta_p$ contribute additively to the expected number of distinct prime factors of $b$.

Let $\omega(n)$ denote the number of distinct primes dividing $n$. The exact identity $\mathbb{P}(p \mid b \mid p \nmid a) = 1/(p-1)$ leads, under the heuristic that "$a$ composite" is approximately unconditioned with respect to each small prime $p$, to:

$$
\mathbb{E}[\omega(b) \mid a \text{ prime}] - \mathbb{E}[\omega(b) \mid a \text{ composite}] \approx \sum_{p \geq 5} \delta_p = \sum_{p \geq 5} \frac{1}{p(p-1)}.
$$

(The key assumption: conditioning on "$a$ composite" does not significantly bias the residue class $k \bmod p$ for small $p$, since primes are sparse.)

This sum converges:

$$
\sum_{p \geq 5} \frac{1}{p(p-1)} = 0.1065\ldots
$$

The first few terms dominate:

| $p$ | $1/[p(p-1)]$ | Cumulative |
|-----|--------------|------------|
| 5 | 0.0500 | 0.0500 |
| 7 | 0.0238 | 0.0738 |
| 11 | 0.0091 | 0.0829 |
| 13 | 0.0064 | 0.0893 |
| 17 | 0.0037 | 0.0930 |

By $p = 97$, the cumulative sum reaches $0.1047$, about 98% of the limiting value.

**Remark.** The convergence of this sum explains why the bias stabilizes rather than growing or shrinking with $K$. Each prime contributes a fixed increment to the absolute shift $\Delta\omega$, and almost all of this contribution comes from small primes.

## 3.3 Empirical Verification

To test the prediction of Section 3.2, we compute the **full** conditional expectations—including contributions from cases where $b$ is prime.

**Computing $\mathbb{E}[\omega(b) \mid a \text{ prime}]$.** When $a$ is prime, $b$ is either prime (the PP case) or composite (the PC case). Using empirical frequencies at $K = 10^9$:

$$
\mathbb{P}(b \text{ prime} \mid a \text{ prime}) = \frac{N_{PP}}{N_{PP} + N_{PC}} = \frac{17{,}244{,}408}{139{,}775{,}685} \approx 0.1234
$$

For prime $b$, we have $\omega(b) = 1$. For composite $b$ in PC pairs, $\mathbb{E}[\omega(b)] = 2.9067$. Thus:

$$
\mathbb{E}[\omega(b) \mid a \text{ prime}] = 0.1234 \times 1 + 0.8766 \times 2.9067 = 2.671
$$

**Computing $\mathbb{E}[\omega(b) \mid a \text{ composite}]$.** When $a$ is composite, $b$ is either prime (the CP case) or composite (the CC case):

$$
\mathbb{P}(b \text{ prime} \mid a \text{ composite}) = \frac{N_{CP}}{N_{CP} + N_{CC}} = \frac{122{,}525{,}274}{860{,}224{,}315} \approx 0.1424
$$

For prime $b$, $\omega(b) = 1$. For composite $b$ in CC pairs, $\mathbb{E}[\omega(b)] = 2.8239$. Thus:

$$
\mathbb{E}[\omega(b) \mid a \text{ composite}] = 0.1424 \times 1 + 0.8576 \times 2.8239 = 2.564
$$

**The comparison.** The empirical difference is:

$$
\mathbb{E}[\omega(b) \mid a \text{ prime}] - \mathbb{E}[\omega(b) \mid a \text{ composite}] = 2.671 - 2.564 = 0.107
$$

This matches the heuristic prediction of $0.1065$ to within $1\%$:

$$
\frac{0.107}{0.1065} \approx 1.00
$$

The independent-prime heuristic accounts for essentially the entire observed effect when applied to the correctly-aligned conditional expectations.

## 3.4 The PC-vs-CC Comparison as a Derived Quantity

The empirical quantity that first motivated this investigation was the comparison of composite $\omega$-values:

$$
\mathbb{E}[\omega(b) \mid \text{PC}] - \mathbb{E}[\omega(b) \mid \text{CC}] = 2.9067 - 2.8239 = 0.0828
$$

This is smaller than the full conditional difference of $0.107$ because it excludes the prime cases. The relationship between the two is:

- When $a$ is prime, $b$ being prime ($\omega = 1$) pulls down the average relative to when $b$ is composite ($\omega \approx 2.9$).
- When $a$ is composite, $b$ being prime is more likely (by mutual exclusivity), so this pull-down effect is stronger.

The difference in these "pull-down" effects accounts for the gap between $0.107$ (full conditional) and $0.0828$ (composite-only).

## 3.5 The Percentage Bias

The percentage bias is:

$$
\text{Bias} = \frac{\Delta\omega}{\mathbb{E}[\omega \mid \text{CC}]}.
$$

Since $\Delta\omega$ is approximately constant while $\mathbb{E}[\omega]$ grows like $\log \log N$ (the Hardy-Ramanujan theorem), the percentage bias decreases slowly with scale. Empirically:

| $K$ | $\mathbb{E}[\omega \mid \text{CC}]$ | Bias |
|-----|--------------------------------------|------|
| $10^7$ | 2.74 | 2.96% |
| $10^8$ | 2.78 | 2.94% |
| $10^9$ | 2.82 | 2.93% |

The bias is remarkably stable, decreasing by only $\sim 0.01$ percentage points per order of magnitude in $K$.

## 3.6 Summary

The selection bias admits a clean first-principles explanation:

1. **Exact:** For each $p \geq 5$, conditioning on $p \nmid a$ boosts $\mathbb{P}(p \mid b)$ from $1/p$ to $1/(p-1)$.

2. **Heuristic:** Summing the per-prime increments gives $\sum_{p \geq 5} 1/[p(p-1)] = 0.1065$.

3. **Empirical verification:** The correctly-aligned conditional difference $\mathbb{E}[\omega(b) \mid a \text{ prime}] - \mathbb{E}[\omega(b) \mid a \text{ composite}] = 0.107$ matches the heuristic to within $1\%$.

4. **Derived quantity:** The PC-vs-CC composite comparison ($0.0828$) is smaller because it excludes the prime-$b$ cases, which have $\omega = 1$ and occur with different frequencies depending on whether $a$ is prime.

The independent-prime heuristic, despite its simplicity, accounts for essentially all of the observed effect.
