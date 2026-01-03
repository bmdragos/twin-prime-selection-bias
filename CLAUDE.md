# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository studies the selection bias in twin prime factor counts: composites adjacent to primes have systematically higher distinct prime factor counts (omega) than composites in purely composite pairs. The project provides both empirical verification and a transfer-matrix model that predicts this bias from first principles.

**Key finding**: The composite member of a PC/CP pair (prime-composite) has ~2.93% more distinct prime factors than composites in CC pairs (both composite).

## Commands

### GPU-accelerated (recommended for K >= 10^8)
```bash
python run_gpu.py              # K=10^8 default
python run_gpu.py --K 1e9      # K=10^9 (requires 24GB+ RAM)
python run_gpu.py --K 1e7      # Quick test
```

### CPU-only
```bash
uv sync && uv run python run_all.py                    # Default config (K=10^6)
uv run python run_all.py --config config/large.yaml   # Larger K
```

### DGX Spark deployment
```bash
# From Mac: copy to DGX, run container, execute
scp -r . spark-dcf7.local:~/twin-prime-selection-bias
ssh spark-dcf7.local
cd ~/twin-prime-selection-bias/dgx-spark && ./run-container.sh
# Inside container:
cd /workspace/project && python run_gpu.py --K 1e9
```

## Architecture

### Core Mathematical Model (`src/transfer_matrix.py`)
- `T_p(p)`: 4x4 ungraded transfer matrix for prime p (PP, PC, CP, CC states)
- `T_p_graded(p, A, B)`: Graded transfer matrix tracking factor counts
- Key insight: Each prime p >= 5 acts as a Markov transition on pair states (6k-1, 6k+1)

### Data Flow
1. **Sieve Phase**: Generate SPF (smallest prime factor) array for all integers up to 6K+1
   - CPU: `src/factorization.py:spf_sieve()` or `src/parallel_sieve.py:spf_sieve_parallel()`
   - Uses uint32 with 0=prime sentinel to halve memory (24GB vs 48GB for K=10^9)

2. **State Classification**: Classify each pair (6k-1, 6k+1) as PP/PC/CP/CC
   - GPU: `src/gpu_factorization.py:compute_states_gpu()`

3. **Omega Computation**: Count distinct prime factors per state
   - GPU: `UnifiedMemoryGPUContext.compute_omega_leq_P_aggregated()` - aggregates on GPU, returns only 8 numbers instead of 8GB arrays
   - CPU: `src/parallel_sieve.py:omega_leq_P_parallel()` - shared memory multiprocessing

4. **Model Comparison**: Compare empirical omega means to transfer-matrix predictions
   - `src/coefficient_extraction.py:model_mean_omega()`, `state_probabilities()`

### GPU Memory Strategy
- **Unified memory (Grace Hopper/Blackwell)**: GPU accesses full system RAM directly; use GPU-side aggregation to avoid 8GB transfers per P value
- **Discrete GPU**: Limited by VRAM; falls back to CPU-parallel for K >= 5×10^8

### State Encoding
- PP=0 (both prime), PC=1 (a prime, b composite), CP=2, CC=3
- Twin primes are the PP pairs

## Configuration
Config files in `config/` specify:
- `K`: Number of pairs to analyze (10^6 to 10^9)
- `P_grid`: Prime cutoff values for model comparison
- `block_size`: 2310 = 2×3×5×7×11 (primorial for sieve optimization)

## Output
Results saved to `data/results/`:
- `selection_bias_summary.csv`: State counts and mean omega per state
- `model_vs_empirical.csv`: Model predictions vs empirical measurements
