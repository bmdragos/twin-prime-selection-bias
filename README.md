# Twin Prime Selection Bias

Composites adjacent to primes have systematically higher factor counts than composites in purely composite pairs.

**[Interactive Results](https://bmdragos.github.io/twin-prime-selection-bias/)** | **Key Finding: +2.93% selection bias in ω**

## The Core Finding

Let **ω(n)** denote the number of distinct prime factors of n. Among twin prime candidates $(6k-1, 6k+1)$—numbers coprime to 2 and 3—we classify pairs by primality state:
- **PP** — Both prime (twin primes)
- **PC** — $a=6k-1$ prime, $b=6k+1$ composite
- **CP** — $a$ composite, $b$ prime
- **CC** — Both composite

**Two key comparisons** (often conflated):

| Comparison | Difference | Relative |
|------------|------------|----------|
| PC vs CC composites | +0.083 | **2.93%** |
| E[ω(b) \| a prime] vs E[ω(b) \| a composite] | +0.1074 | — |

The 2.93% headline compares composite ω-values only (PC vs CC). The 0.1074 difference is the full conditional (includes cases where b is prime, contributing ω=1). Both are predicted by the heuristic sum $\sum_{p≥5} 1/[p(p-1)] = 0.1065$ to within 1%.

**Note**: This is a conditioning/selection effect on composites adjacent to primes, not evidence about twin-prime density or a primality test.

## Generalization to Other Patterns

The mechanism generalizes beyond twin primes. Any admissible prime pair exhibits the same bias:

| Pattern | Predicted | Empirical | Error |
|---------|-----------|-----------|-------|
| Twin primes $(6k-1, 6k+1)$ | 0.1065 | 0.1074 | 0.8% |
| Sophie Germain $(n, 2n+1)$ | 0.273 | 0.272 | **0.4%** |
| Cousin primes $(n, n+4)$ | 0.1065 | 0.1063 | **0.1%** |

- **Sophie Germain**: Sum starts at $q=3$ (not 5) since $2 \nmid (2n+1)$ for odd $n$. Verified at $N=10^9$.
- **Cousin primes**: Same sum as twins. Verified at $K=10^8$ among $6k \pm 1$ candidates.

## What This Studies

We develop a transfer-matrix model that predicts this bias from first principles, treating the action of each prime p >= 5 as a Markov transition on pair states. The model produces quantitative predictions for:

- Expected number of distinct prime factors (omega) in each component
- The "tilt" in factor count distributions between twin prime candidates and random composites
- State transition probabilities as functions of prime cutoff P

We validate these predictions against empirical data and show strong agreement.

## Quick Start

```bash
# GPU-accelerated (recommended for K >= 10^8)
python run_gpu.py --K 1e8

# CPU-only
uv sync && uv run python run_all.py
```

Results saved to `data/results/`.

## Benchmarks

### DGX Spark (NVIDIA GB10, 128GB unified memory)

**Main selection bias run** (`run_gpu.py`):
| K | Runtime | Selection Bias |
|---|---------|----------------|
| 10^7 | 54s | 2.956% |
| 10^8 | ~60s | 2.940% |
| 10^9 | **190s** | 2.933% |

**Omega decomposition** (`exp_omega_decomposition_gpu.py`):
| K | Runtime | Notes |
|---|---------|-------|
| 10^8 | ~1 min | Small vs large prime factors |
| 10^9 | **11 min** | Dominated by wheel SPF sieve |

GPU-side aggregation provides **6.2x speedup** over CPU-parallel by avoiding 8GB array transfers per P value.

### Mac (M4 Pro, 48GB RAM)

| K | Runtime | Notes |
|---|---------|-------|
| 10^6 | ~10s | CPU only |
| 10^7 | ~2 min | CPU parallel |
| 10^8 | ~20 min | CPU parallel |

For K >= 10^9, use DGX Spark or similar high-memory GPU system.

## Project Structure

```
twin-prime-selection-bias/
├── run_gpu.py                # GPU-accelerated runner (recommended)
├── run_all.py                # CPU-only runner
├── benchmark_gpu.py          # GPU optimization benchmarks
├── src/
│   ├── gpu_factorization.py  # CUDA kernels, unified memory support
│   ├── parallel_sieve.py     # CPU-parallel SPF sieve
│   ├── primes.py             # Prime generation
│   ├── sieve_pairs.py        # (6k-1, 6k+1) pair logic
│   ├── factorization.py      # SPF sieve, omega functions
│   ├── transfer_matrix.py    # Mean-field Markov model
│   ├── coefficient_extraction.py  # Model predictions
│   └── experiments/
│       ├── exp_omega_decomposition_gpu.py  # Small vs large prime decomposition
│       ├── exp_sophie_germain_gpu.py       # Sophie Germain verification
│       ├── exp_cousin_primes_gpu.py        # Cousin primes verification
│       └── exp_per_prime_divisibility.py   # Per-prime 1/(p-1) verification
├── docs/                     # GitHub Pages site
├── dgx-spark/                # Container setup for DGX
├── config/                   # Experiment parameters
└── notebooks/                # Interactive exploration
```

## Reference Results

Canonical results in [`data/reference/`](data/reference/README.md) for reproducibility verification:

| Experiment | Scale | Key Result |
|------------|-------|------------|
| Twin primes (omega decomposition) | K=10^9 | PC vs CC: +2.93% bias |
| Sophie Germain $(n, 2n+1)$ | N=10^9 | 0.272 vs predicted 0.273 |
| Cousin primes $(n, n+4)$ | K=10^8 | 0.1063 vs predicted 0.1065 |
| Per-prime divisibility | K=10^9 | P(p\|b \| a prime) = 1/(p-1) to 6 decimal places |

Reproduce with:
```bash
python -m src.experiments.exp_omega_decomposition_gpu --K 1e9
python -m src.experiments.exp_sophie_germain_gpu --N 1e9
python -m src.experiments.exp_cousin_primes_gpu --K 1e8
python -m src.experiments.exp_per_prime_divisibility --K 1e9
```

## Technical Details

- **Population**: All analysis is conditioned on 6k±1 candidates (coprime to 2 and 3). Results do not generalize directly to "all composites."
- **SPF Sieve**: uint32 with 0=prime sentinel (24GB for K=10^9 vs 48GB with int64)
- **GPU Kernels**: Numba CUDA with unified memory support for Grace Hopper/Blackwell
- **Aggregation**: GPU-side reduction avoids transferring 8GB omega arrays per P value
- **CPU Fallback**: Shared memory multiprocessing for systems without GPU

## License

MIT License. See LICENSE file for details.
