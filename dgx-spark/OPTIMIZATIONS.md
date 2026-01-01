# DGX Spark Optimization Guide

Lessons learned optimizing for the NVIDIA GB10 (Grace Hopper architecture) with 128GB unified memory.

## Key Insight: Unified Memory Changes Everything

The GB10's unified memory means CPU and GPU share the same 128GB physical RAM. This eliminates the traditional "copy to GPU, compute, copy back" pattern.

**What this enables:**
- GPU can access arrays larger than "GPU memory" (because it's all system memory)
- Data transfers are cheap (just pointer passing, not DMA copies)
- The bottleneck shifts from transfer bandwidth to kernel compute

## Optimizations That Worked

### 1. GPU-Side Aggregation (6x speedup on per-P computation)

**Problem:** Each P value required transferring 8GB of omega arrays back to CPU for aggregation.

**Solution:** Aggregate on GPU, transfer only 8 numbers (sums per state).

```python
# Before: transfer 8GB per P
omega_a, omega_b = gpu_ctx.compute_omega_leq_P(P)  # returns 2x 4GB arrays
mean_a = {i: np.mean(omega_a[state_codes == i]) for i in range(4)}

# After: transfer 64 bytes per P
sums_a, sums_b = gpu_ctx.compute_omega_leq_P_aggregated(P, d_state_codes)
mean_a = {i: sums_a[i] / counts[i] for i in range(4)}
```

**Implementation:** Two-phase reduction with shared memory:
1. Each block computes partial sums per state using `cuda.atomic.add` to shared memory
2. Final kernel sums block-level results

**Result:** 5.81x speedup at K=10^8, similar at K=10^9

### 2. uint32 SPF with 0=Prime Sentinel (50% memory reduction)

**Problem:** SPF array for K=10^9 was 48GB with int64.

**Solution:** Use uint32 (max SPF value is sqrt(6K) ≈ 77k, fits easily). Use 0 as sentinel for "n is prime" (SPF equals n itself).

```python
# spf[n] == 0 means n is prime
# This halves memory: 48GB → 24GB
spf = np.zeros(N, dtype=np.uint32)
```

**Kernel update:**
```python
p = spf[n]
if p == 0:  # n is prime
    p = n
```

### 3. Transfer Arrays Once, Reuse (eliminated memory leak)

**Problem:** Initial implementation called `cuda.to_device()` on every kernel invocation, leaking GPU memory.

**Solution:** Transfer in `__init__`, reuse device arrays for all calls.

```python
class UnifiedMemoryGPUContext:
    def __init__(self, a_vals, b_vals, spf):
        # Transfer once
        self.d_a = cuda.to_device(a_vals)
        self.d_b = cuda.to_device(b_vals)
        self.d_spf = cuda.to_device(spf)
        self.d_results_a = cuda.device_array(self.n, dtype=np.int32)

    def compute_omega_leq_P(self, P):
        # Reuse device arrays
        _kernel[...](self.d_a, self.d_spf, P, self.d_results_a)
```

### 4. Shared Memory for CPU Multiprocessing (eliminated pickle overhead)

**Problem:** Python multiprocessing pickles function arguments. For 24GB SPF array, this was catastrophic.

**Solution:** Use `multiprocessing.shared_memory` to share SPF across workers.

```python
shm = shared_memory.SharedMemory(create=True, size=spf.nbytes)
shm_spf = np.ndarray(spf.shape, dtype=spf.dtype, buffer=shm.buf)
shm_spf[:] = spf[:]

# Workers access via shared memory name
with Pool(initializer=_init_worker_shm, initargs=(shm.name, shape, dtype)):
    ...
```

## Optimizations That Failed

### Fused Kernels (2x slower!)

**Hypothesis:** Fusing a/b processing into one kernel would reduce launch overhead.

**Reality:** For memory-bound workloads, fusion hurts:
- Increases register pressure
- Reduces occupancy
- Cache thrashing between a and b SPF lookups

**Benchmark:**
- Separate kernels: P=97 in 22.8s
- Fused kernel: P=97 in 47.9s (2x slower!)

**Lesson:** Memory-bound kernels benefit from smaller, focused kernels.

## Opportunities Not Yet Implemented

### Warp-Level Reduction (optional ~1.3x on aggregation)

Current aggregation uses block-level atomics. Could reduce by 32x with warp shuffle:

```python
# Warp-level reduction before atomic
val = cuda.warp_reduce_sum(omega_val)
if lane_id == 0:
    cuda.atomic.add(shared_sums, state, val)
```

**Skip unless:** Aggregation becomes >20% of total time.

### Batched Factor-Once (potential 8x fewer SPF lookups)

Instead of factoring each number 8 times (once per P), factor once and count for all P values:

```python
# Factor once, collect primes
factors = [p1, p2, p3, ...]

# Count for each P threshold
for P in P_grid:
    omega_leq_P[i] = sum(1 for p in factors if p <= P)
```

**Trade-off:** Uses local memory (20 factors max), may increase register pressure. Needs microbenchmark.

## Performance Summary

| Configuration | K=10^9 Runtime |
|---------------|----------------|
| CPU-parallel only | 1168s |
| GPU (no aggregation) | 292s |
| GPU + aggregation | **190s** |

**Bottleneck breakdown (K=10^9):**
- SPF sieve (CPU): ~100s (52%)
- GPU context setup: ~5s (3%)
- Per-P omega computation: ~70s (37%)
- Full omega for bias: ~3s (2%)
- Other: ~12s (6%)

The SPF sieve is now the dominant cost. Further optimization would require:
- Parallel segmented sieve (already implemented, ~100s is good)
- GPU-accelerated sieve (complex, diminishing returns)
