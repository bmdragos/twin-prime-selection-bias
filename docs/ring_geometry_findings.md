# Twin Primes in Quadratic Integer Rings

## Summary

We investigated twin primes through the lens of Gaussian integers Z[i] and Eisenstein integers Z[ω], discovering that the bias patterns are **ring-dependent** rather than universal.

## The Three-Body View

Every twin prime pair (p, p+2) where p > 3 decomposes into a **triplet** in Z[i]:

```
Twin prime (p, q) = (π, π̄, q)
```

Where:
- One prime ("good", ≡1 mod 4) splits: p = ππ̄ (conjugate Gaussian primes)
- One prime ("bad", ≡3 mod 4) stays inert in Z[i]

This frames twin primes as a **bridge** between the split and inert worlds.

## Key Finding: Ring-Dependent Biases

### Z[i] (Gaussian Integers)
- **Split primes** (p ≡ 1 mod 4) create **7x larger biases** than inert primes
- p=5 dominates with the largest bias
- The "good" primes that factor in Z[i] show strong geometric constraints

### Z[ω] (Eisenstein Integers)
- **Inert primes** (p ≡ 2 mod 3) create **~1.2x larger biases** than split primes
- p=5 STILL dominates (41.8% bias) despite being inert in Z[ω]
- **Opposite pattern** to Z[i]!

### The p=5 Anomaly

p=5 dominates in BOTH rings despite opposite splitting behavior:

| Ring | p=5 status | p=5 bias |
|------|------------|----------|
| Z[i] | Split (5 ≡ 1 mod 4) | Largest |
| Z[ω] | Inert (5 ≡ 2 mod 3) | Largest (41.8%) |

**Interpretation**: p=5 is special because it's the **first prime interacting with the 6k±1 wheel structure**, not due to ring-theoretic properties.

## Four Classes of Primes (mod 12)

All primes p > 3 fall into exactly one class:

| Class | Condition | Z[i] | Z[ω] | Examples |
|-------|-----------|------|------|----------|
| Both split | p ≡ 1 (mod 12) | Split | Split | 13, 37, 61, 73, 97 |
| Gaussian only | p ≡ 5 (mod 12) | Split | Inert | 5, 17, 29, 41, 53 |
| Eisenstein only | p ≡ 7 (mod 12) | Inert | Split | 7, 19, 31, 43, 67 |
| Both inert | p ≡ 11 (mod 12) | Inert | Inert | 11, 23, 47, 59, 71 |

**Result**: "Doubly split" primes are **NOT** special. The ranking by combined bias:

| Class | n | Combined Bias |
|-------|---|---------------|
| Gaussian only (≡5 mod 12) | 6 | **9.65%** |
| Eisenstein only (≡7 mod 12) | 6 | 7.46% |
| Both split (≡1 mod 12) | 5 | 5.58% |
| Both inert (≡11 mod 12) | 6 | 4.72% |

**The p=5 dominance**: p=5 alone has 40.58% combined bias, dwarfing all other primes. Without p=5, all classes converge to ~3-5%.

## Connection to Hardy-Littlewood

The twin prime constant decomposes as:

```
C₂ = 2 × C_good × C_bad
```

where C_good/C_bad ≈ 1.29. The asymmetry comes almost entirely from p=3:
- p=3 contributes factor of 2 to C_good (since 3 doesn't divide 6k±1)
- Other primes contribute symmetrically

## Experimental Results

### Mod p Bias Comparison (K=10⁸)

```
Z[i] Results:
  Split primes (≡1 mod 4): Mean bias = 6.7x larger than inert

Z[ω] Results:
  Split primes (≡1 mod 3): Mean bias = 8.84%
  Inert primes (≡2 mod 3): Mean bias = 10.51%
  Ratio: 0.84x (inert slightly larger!)
```

## The p=5 Mystery

p=5 is overwhelmingly special across all analyses:

| Analysis | p=5 Bias | Next Largest |
|----------|----------|--------------|
| Z[i] Gaussian | Largest | p=13 (~12%) |
| Z[ω] Eisenstein | 41.8% | p=7 (~23%) |
| Combined | 40.58% | p=7 (~23%) |

**What makes p=5 unique?**

1. **First prime after 3**: p=5 is the smallest prime that interacts with the 6k±1 wheel structure
2. **Largest relative constraint**: 5 divides more residue classes relative to its size
3. **NOT about ring splitting**: p=5 dominates in Z[ω] despite being inert there

The bias appears to be about **arithmetic proximity** (5 is small) rather than algebraic structure.

## Open Questions

1. **Why p=5?** Is it simply the smallest interacting prime, or is there deeper structure?

2. **Doubly split primes**: Do primes ≡1 (mod 12) have unique properties for twin prime counts?

3. **Higher rings**: What happens in Z[ζ_n] for n > 6?

4. **Computational leverage**: Can ring factorizations provide faster twin prime sieves?

## Code

Experiments are in `src/experiments/`:
- `exp_modp_biases.py` - Z[i] analysis
- `exp_eisenstein_geometry.py` - Z[ω] analysis
- `exp_both_rings.py` - Combined mod 12 analysis
