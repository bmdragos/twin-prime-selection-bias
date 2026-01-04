# Sophie Germain Per-Prime Verification

Pattern: $(n, 2n+1)$ for odd $n$

$N = 1,000,000,000$ (50,847,533 odd $n$ prime)

Predicted: $\sum_{p \geq 3} 1/[p(p-1)] = 0.2560$

| $p$ | $\hat{P}(p \mid b \mid n \text{ prime})$ | $1/(p-1)$ | SE | $z$ | $\varepsilon_p$ |
|-----|-----------------------------------------------|-----------|------------|-------|----------------|
| 3 | 0.499979 | 0.5000 | 7.01e-05 | -0.30 | -0.00004 |
| 5 | 0.250008 | 0.2500 | 6.07e-05 | +0.14 | +0.00003 |
| 7 | 0.166677 | 0.1667 | 5.23e-05 | +0.20 | +0.00006 |
| 11 | 0.100000 | 0.1000 | 4.21e-05 | +0.00 | +0.00000 |
| 13 | 0.083337 | 0.0833 | 3.88e-05 | +0.10 | +0.00005 |

$\varepsilon_p = (p-1) \cdot \hat{P} - 1$: scaled residual (0 if model exact)

$z$-scores within $\pm 2$ are expected sampling noise.
