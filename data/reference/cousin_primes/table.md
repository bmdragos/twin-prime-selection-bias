# Cousin Primes (n, n+4) among 6k±1 candidates

$K = 100,000,000$

| Population | Mean ω(n+4) | Sample size |
|------------|-------------|-------------|
| n prime | 3.0238 | 31,324,702 |
| n composite | 2.9175 | 168,675,298 |
| **Difference** | **0.1063** | — |

Predicted: Σ 1/[p(p-1)] for p ≥ 5 = 0.1064

Relative error: 0.08%

Note on p=3: For n = 6k-1, n+4 = 6k+3 ≡ 0 (mod 3) always.
For n = 6k+1, n+4 = 6k+5 ≢ 0 (mod 3).
Since 3|n+4 is determined by residue class (not primality of n),
the p=3 term contributes zero to the DIFFERENCE. The sum starts at p=5.
