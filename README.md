# Twin Prime Selection Bias

Composites adjacent to primes have systematically higher factor counts than composites in purely composite pairs.

**[Interactive Results](https://bmdragos.github.io/twin-prime-selection-bias/)** | **Key Finding: +2.93% selection bias in omega**

## The Core Finding

Among twin prime candidates (6k-1, 6k+1), we classify pairs by primality state:
- **PP** — Both prime (twin primes)
- **PC** — a prime, b composite
- **CP** — a composite, b prime
- **CC** — Both composite

The composite member of a PC/CP pair has **2.93% more distinct prime factors** than composites in CC pairs. This bias is stable across K=10^7 to K=10^9 pairs.

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

| K | Runtime | Selection Bias |
|---|---------|----------------|
| 10^7 | 54s | 2.956% |
| 10^8 | ~60s | 2.940% |
| 10^9 | **190s** | 2.933% |

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
│   └── experiments/          # Individual experiments
├── docs/                     # GitHub Pages site
├── dgx-spark/                # Container setup for DGX
├── config/                   # Experiment parameters
└── notebooks/                # Interactive exploration
```

## Technical Details

- **SPF Sieve**: uint32 with 0=prime sentinel (24GB for K=10^9 vs 48GB with int64)
- **GPU Kernels**: Numba CUDA with unified memory support for Grace Hopper/Blackwell
- **Aggregation**: GPU-side reduction avoids transferring 8GB omega arrays per P value
- **CPU Fallback**: Shared memory multiprocessing for systems without GPU

## License

MIT License. See LICENSE file for details.
