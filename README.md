# Twin Prime Selection Bias

This repository contains reproducible code for studying selection bias in the factor counts of twin prime candidates.

## What This Paper Studies

Among integers of the form (6k-1, 6k+1), we study how conditioning on primality affects the distribution of small prime factors in the composite component. When one member of the pair is prime and the other is composite, the composite inherits a non-trivial bias in its factorization structure.

We develop a transfer-matrix model that predicts this bias from first principles, treating the action of each prime p >= 5 as a Markov transition on pair states (PP, PC, CP, CC). The model produces quantitative predictions for:

- Expected number of distinct prime factors (omega) in each component
- The "tilt" in factor count distributions between twin prime candidates and random composites
- State transition probabilities as functions of prime cutoff P

We validate these predictions against empirical data from the first million pairs and show strong agreement, with residual discrepancies that suggest avenues for refinement.

## Reproducing Results

```bash
uv sync
uv run python run_all.py
```

Or with pip (slower):
```bash
pip install -r requirements.txt
python run_all.py
```

All tables and figures will be saved to `data/results/`.

## Expected Runtime

- **Mac (M1/M2)**: ~5-10 minutes for K=1,000,000
- **DGX / High-memory server**: Similar or faster; main bottleneck is single-threaded sieve

For larger K, the sieve backend can be swapped for a parallel implementation with minimal changes to `src/primes.py` and `src/factorization.py`.

## Project Structure

```
twin-prime-selection-bias/
├── config/default.yaml       # Experiment parameters
├── src/
│   ├── primes.py             # Prime generation
│   ├── sieve_pairs.py        # (6k-1, 6k+1) pair logic
│   ├── factorization.py      # SPF sieve, omega functions
│   ├── metrics.py            # Paper statistics definitions
│   ├── null_models.py        # Skeptic-proofing permutations
│   ├── transfer_matrix.py    # Mean-field Markov model
│   ├── coefficient_extraction.py  # Model predictions
│   ├── plotting.py           # Visualization
│   └── experiments/
│       ├── exp_selection_bias.py      # Section 3
│       ├── exp_model_vs_empirical.py  # Sections 5-6
│       └── exp_run_lengths.py         # Appendix
├── run_all.py                # Full reproducibility script
├── data/
│   ├── intermediate/         # Cached computations
│   └── results/              # Output tables and figures
└── notebooks/
    └── sanity_checks.ipynb   # Interactive exploration
```

## License

MIT License. See LICENSE file for details.
