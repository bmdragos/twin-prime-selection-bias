# Variance and Shape Analysis

$K = 100,000,000$, $N = 6K+1 = 600,000,001$

$\log \log N = 3.0063$

## Means

| Population | Mean $\omega$ | Sample size |
|------------|---------------|-------------|
| PC composites | 2.8171 | 13,496,565 |
| CC composites | 2.7366 | 70,841,598 |
| **Difference** | **0.0805** | — |

## Variances

| Population | Variance | Std Dev |
|------------|----------|--------|
| PC composites | 0.6830 | 0.8264 |
| CC composites | 0.6236 | 0.7897 |
| **Ratio (PC/CC)** | **1.0953** | — |

PC variance is **9.5%** higher than CC.

Predicted variance shift from per-prime Bernoulli model: 0.0724

## Higher Moments

| Moment | PC | CC | Normal |
|--------|----|----|--------|
| Skewness | 0.7539 | 0.8287 | 0 |
| Excess kurtosis | -0.0121 | 0.1097 | 0 |

## Normality (KS test with empirical standardization)

- PC: KS = 0.2522, p = 0.00e+00
- CC: KS = 0.2771, p = 0.00e+00
