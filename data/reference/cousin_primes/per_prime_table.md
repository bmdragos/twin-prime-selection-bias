# Cousin Primes Per-Prime Verification

Pattern: $(n, n+4)$ among $6k \pm 1$ candidates

$K = 100,000,000$ (31,324,702 $n$ prime)

**Note:** For $p=3$, divisibility of $n+4$ depends on residue class:
- $n = 6k-1 \Rightarrow n+4 = 6k+3 \equiv 0 \pmod 3$ (always)
- $n = 6k+1 \Rightarrow n+4 = 6k+5 \equiv 2 \pmod 3$ (never)

Since prime/composite $n$ have the same residue-class mix, $p=3$ contributes zero to the difference.

Predicted: $\sum_{p \geq 5} 1/[p(p-1)] = 0.0893$

| $p$ | $\hat{P}(p \mid b \mid n \text{ prime})$ | $1/(p-1)$ | SE | $z$ | $\varepsilon_p$ | Note |
|-----|-----------------------------------------------|-----------|------------|-------|----------------|------|
| 3 | 0.500016 | — | 8.93e-05 | — | — | Residue-class determined (should cancel) |
| 5 | 0.249984 | 0.2500 | 7.74e-05 | -0.21 | -0.00007 |  |
| 7 | 0.166669 | 0.1667 | 6.66e-05 | +0.03 | +0.00001 |  |
| 11 | 0.099996 | 0.1000 | 5.36e-05 | -0.08 | -0.00004 |  |
| 13 | 0.083322 | 0.0833 | 4.94e-05 | -0.22 | -0.00013 |  |

$\varepsilon_p = (p-1) \cdot \hat{P} - 1$: scaled residual (0 if model exact)

$z$-scores within $\pm 2$ are expected sampling noise.
