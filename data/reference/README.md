# Reference Results

Canonical results from DGX Spark (NVIDIA GB10) at K = 10^9.

These are the exact values cited in the paper. Use to verify your own runs.

## Files

### omega_decomposition_K1e9/
Omega decomposition into small (p ≤ √N) and large (p > √N) prime factors.

Key results:
- PC omega_small: 2.2312, CC omega_small: 2.1364 → diff +0.0948 (4.4% bias)
- PC has_big: 67.6%, CC has_big: 68.7% → diff −0.012
- PC omega_full: 2.9067, CC omega_full: 2.8239 → diff +0.0828 (2.93% bias)
- Large-prime reduction: 13%

### per_prime_K1e9/
Per-prime divisibility verification of P(p|b | a prime) = 1/(p−1).

Key results at p=5,7,11,13:
- Empirical P(p|b | a prime) matches 1/(p−1) to 6 decimal places
- Empirical P(p|b | a comp) is lower than naive 1/p (CC suppression)

### sophie_germain/
Sophie Germain pairs (n, 2n+1) at N = 10^9.

Key results:
- E[ω(2n+1) | n prime] = 3.083, E[ω(2n+1) | n composite] = 2.825
- Difference: 0.258 vs predicted 0.273 (5.7% error)
- Larger error than twin primes suggests heuristic less accurate for non-constant gaps

### cousin_primes/
Cousin primes (n, n+4) among 6k±1 candidates at K = 10^8.

Key results:
- E[ω(n+4) | n prime] = 3.024, E[ω(n+4) | n composite] = 2.918
- Difference: 0.1063 vs predicted 0.1064 (**0.1% error**)
- Confirms mechanism generalizes to other constant-gap patterns

## Reproducibility

```bash
# On DGX Spark or similar GPU system:
python -m src.experiments.exp_omega_decomposition_gpu --K 1e9 --save
python -m src.experiments.exp_per_prime_divisibility --K 1e9 --save
python -m src.experiments.exp_sophie_germain_gpu --N 1e9 --save
python -m src.experiments.exp_cousin_primes_gpu --K 1e8 --save
```

Runtime: ~11 min (omega decomposition), ~5 min (per-prime), ~1 min (Sophie Germain), ~15s (cousin primes).

## Verification

Expected file sizes (line counts including header):

| File | Lines |
|------|-------|
| `omega_decomposition_K1e9/results.csv` | 17 |
| `per_prime_K1e9/per_prime_divisibility.csv` | 8 |

Verify with: `wc -l data/reference/*/*.csv`
