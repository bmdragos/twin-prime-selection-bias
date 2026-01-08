# Twin Prime Selection Bias

A simple empirical observation: composites next to primes have ~3% more distinct prime factors than composites next to composites.

## The Observation

Among twin prime candidates $(6k-1, 6k+1)$, classify pairs by primality:
- **PC** — first prime, second composite
- **CC** — both composite

The composite in a PC pair has 2.93% more distinct prime factors (ω) than composites in CC pairs. That's it.

## Why It Happens

When $a = 6k-1$ is prime, it can't be divisible by any prime $p \geq 5$. This means $b = 6k+1$ avoids one residue class mod $p$ that composites normally occupy. Fewer "slots" taken → more room for other prime factors.

The bias is predicted by: $\sum_{p \geq 5} \frac{1}{p(p-1)} = 0.1065$

Empirical measurement: 0.1074 (within 1%).

## Generalizes to Other Patterns

| Pattern | Predicted | Empirical |
|---------|-----------|-----------|
| Twin primes | 0.1065 | 0.1074 |
| Sophie Germain | 0.273 | 0.272 |
| Cousin primes | 0.1065 | 0.1063 |

Same mechanism, predictable constants.

## Running It

```bash
python run_gpu.py --K 1e8    # GPU
uv run python run_all.py     # CPU
```

## What This Isn't

- Not a primality test
- Not evidence about twin prime density
- Not a breakthrough—just a conditioning effect that falls out of sieve theory

The math is standard. The contribution is making the prediction explicit and validating it numerically.

## License

MIT
