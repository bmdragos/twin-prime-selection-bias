"""
GPU-accelerated factorization using Numba CUDA.

For DGX Spark or any CUDA-capable GPU.
Supports unified memory on Grace Hopper/Blackwell architectures.
"""

import numpy as np

try:
    from numba import cuda
    HAS_GPU = cuda.is_available()
except ImportError:
    HAS_GPU = False


def has_unified_memory() -> bool:
    """
    Detect if GPU has unified memory (Grace Hopper/Blackwell).

    Compute capability 9.0+ (Hopper) or 12.0+ (Blackwell) have unified memory.
    """
    if not HAS_GPU:
        return False
    try:
        device = cuda.get_current_device()
        cc = device.compute_capability
        # Grace Hopper (sm_90) and Blackwell (sm_120+) have unified memory
        return cc[0] >= 9
    except Exception:
        return False


HAS_UNIFIED_MEMORY = has_unified_memory() if HAS_GPU else False


if HAS_GPU:
    import numba

    @cuda.jit
    def _omega_leq_P_kernel(numbers, spf, P, results):
        """
        CUDA kernel: compute omega_leq_P for each number in parallel.
        SPF uses 0=prime sentinel (spf[n]==0 means n is prime).
        """
        idx = cuda.grid(1)
        if idx >= numbers.shape[0]:
            return

        n = numbers[idx]
        if n <= 1:
            results[idx] = 0
            return

        count = 0
        prev = 0

        while n > 1:
            p = spf[n]
            if p == 0:  # n is prime, SPF = n itself
                p = n
            if p != prev and p <= P:
                count += 1
            prev = p
            n //= p

        results[idx] = count

    @cuda.jit
    def _omega_kernel(numbers, spf, results):
        """
        CUDA kernel: compute full omega (distinct prime factors).
        SPF uses 0=prime sentinel (spf[n]==0 means n is prime).
        """
        idx = cuda.grid(1)
        if idx >= numbers.shape[0]:
            return

        n = numbers[idx]
        if n <= 1:
            results[idx] = 0
            return

        count = 0
        prev = 0

        while n > 1:
            p = spf[n]
            if p == 0:  # n is prime, SPF = n itself
                p = n
            if p != prev:
                count += 1
            prev = p
            n //= p

        results[idx] = count

    @cuda.jit
    def _omega_fused_kernel(a_vals, b_vals, spf, P, results_a, results_b):
        """
        FUSED kernel: process both a and b arrays in single launch.
        Reduces kernel launch overhead by 50%.
        """
        idx = cuda.grid(1)
        if idx >= a_vals.shape[0]:
            return

        # Process a
        n = a_vals[idx]
        if n <= 1:
            results_a[idx] = 0
        else:
            count = 0
            prev = 0
            while n > 1:
                p = spf[n]
                if p == 0:
                    p = n
                if p != prev and p <= P:
                    count += 1
                prev = p
                n //= p
            results_a[idx] = count

        # Process b
        n = b_vals[idx]
        if n <= 1:
            results_b[idx] = 0
        else:
            count = 0
            prev = 0
            while n > 1:
                p = spf[n]
                if p == 0:
                    p = n
                if p != prev and p <= P:
                    count += 1
                prev = p
                n //= p
            results_b[idx] = count

    @cuda.jit
    def _omega_fused_full_kernel(a_vals, b_vals, spf, results_a, results_b):
        """
        FUSED kernel for full omega (no P limit).
        """
        idx = cuda.grid(1)
        if idx >= a_vals.shape[0]:
            return

        # Process a
        n = a_vals[idx]
        if n <= 1:
            results_a[idx] = 0
        else:
            count = 0
            prev = 0
            while n > 1:
                p = spf[n]
                if p == 0:
                    p = n
                if p != prev:
                    count += 1
                prev = p
                n //= p
            results_a[idx] = count

        # Process b
        n = b_vals[idx]
        if n <= 1:
            results_b[idx] = 0
        else:
            count = 0
            prev = 0
            while n > 1:
                p = spf[n]
                if p == 0:
                    p = n
                if p != prev:
                    count += 1
                prev = p
                n //= p
            results_b[idx] = count

    @cuda.jit
    def _omega_batched_kernel(numbers, spf, P_values, n_P, results_matrix):
        """
        BATCHED kernel: factor once, count for ALL P values.

        Reduces factorization work by 8x!
        results_matrix shape: (n_P + 1, n_numbers)
        Row i = omega_leq_P[i], last row = full omega
        """
        idx = cuda.grid(1)
        if idx >= numbers.shape[0]:
            return

        n = numbers[idx]
        if n <= 1:
            for p_idx in range(n_P + 1):
                results_matrix[p_idx, idx] = 0
            return

        # Collect distinct prime factors in local memory (max 20 factors)
        factors = cuda.local.array(20, dtype=numba.int64)
        n_factors = 0
        prev = 0

        temp_n = n
        while temp_n > 1:
            p = spf[temp_n]
            if p == 0:
                p = temp_n
            if p != prev and n_factors < 20:
                factors[n_factors] = p
                n_factors += 1
            prev = p
            temp_n //= p

        # Count for each P threshold
        for p_idx in range(n_P):
            P = P_values[p_idx]
            count = 0
            for i in range(n_factors):
                if factors[i] <= P:
                    count += 1
            results_matrix[p_idx, idx] = count

        # Full omega (last row)
        results_matrix[n_P, idx] = n_factors

    @cuda.jit
    def _pair_states_kernel(prime_flags, K, results):
        """
        CUDA kernel: compute pair states in parallel.
        States: PP=0, PC=1, CP=2, CC=3
        """
        idx = cuda.grid(1)
        if idx >= K:
            return

        k = idx + 1
        a = 6 * k - 1
        b = 6 * k + 1

        a_prime = prime_flags[a]
        b_prime = prime_flags[b]

        if a_prime and b_prime:
            results[idx] = 0
        elif a_prime and not b_prime:
            results[idx] = 1
        elif not a_prime and b_prime:
            results[idx] = 2
        else:
            results[idx] = 3

    @cuda.jit
    def _aggregate_by_state_kernel(omega, state_codes, n, block_sums):
        """
        Two-phase reduction: each block computes partial sums per state.

        block_sums shape: (n_blocks, 4) - sum of omega for each state per block
        Uses shared memory to reduce atomic contention.
        """
        # Shared memory for this block's partial sums
        shared_sums = cuda.shared.array(4, dtype=numba.float64)

        tid = cuda.threadIdx.x
        bid = cuda.blockIdx.x
        idx = cuda.grid(1)

        # Initialize shared memory
        if tid < 4:
            shared_sums[tid] = 0.0
        cuda.syncthreads()

        # Each thread adds its value to appropriate state bucket
        if idx < n:
            state = state_codes[idx]
            cuda.atomic.add(shared_sums, state, float(omega[idx]))

        cuda.syncthreads()

        # First 4 threads write block results
        if tid < 4:
            block_sums[bid, tid] = shared_sums[tid]

    @cuda.jit
    def _reduce_block_sums_kernel(block_sums, n_blocks, final_sums):
        """
        Final reduction: sum all block partial sums.
        final_sums shape: (4,)
        """
        state = cuda.grid(1)
        if state >= 4:
            return

        total = 0.0
        for i in range(n_blocks):
            total += block_sums[i, state]
        final_sums[state] = total

    class GPUContext:
        """
        Manages GPU memory for efficient batch processing.
        Transfer data once, run multiple kernels.

        For discrete GPUs with separate VRAM.
        """
        def __init__(self, a_vals: np.ndarray, b_vals: np.ndarray, spf: np.ndarray):
            """
            Initialize GPU context with pre-transferred arrays.

            Parameters
            ----------
            a_vals : np.ndarray
                Array of 6k-1 values
            b_vals : np.ndarray
                Array of 6k+1 values
            spf : np.ndarray
                Smallest prime factor array (uint32, 0=prime sentinel)
            """
            print("    Transferring data to GPU...", end=" ", flush=True)
            self.n = len(a_vals)
            self.d_a = cuda.to_device(a_vals.astype(np.int64))
            self.d_b = cuda.to_device(b_vals.astype(np.int64))
            # Keep SPF as uint32 to save memory
            self.d_spf = cuda.to_device(spf.astype(np.uint32))
            self.d_results_a = cuda.device_array(self.n, dtype=np.int32)
            self.d_results_b = cuda.device_array(self.n, dtype=np.int32)

            # Compute grid dimensions once
            self.threads_per_block = 256
            self.blocks = (self.n + self.threads_per_block - 1) // self.threads_per_block

            cuda.synchronize()
            print("done")

        def compute_omega_leq_P(self, P: int):
            """Compute omega_leq_P for both a and b values, return results."""
            _omega_leq_P_kernel[self.blocks, self.threads_per_block](
                self.d_a, self.d_spf, P, self.d_results_a
            )
            _omega_leq_P_kernel[self.blocks, self.threads_per_block](
                self.d_b, self.d_spf, P, self.d_results_b
            )
            cuda.synchronize()

            return self.d_results_a.copy_to_host(), self.d_results_b.copy_to_host()

        def compute_omega(self):
            """Compute full omega for both a and b values."""
            _omega_kernel[self.blocks, self.threads_per_block](
                self.d_a, self.d_spf, self.d_results_a
            )
            _omega_kernel[self.blocks, self.threads_per_block](
                self.d_b, self.d_spf, self.d_results_b
            )
            cuda.synchronize()

            return self.d_results_a.copy_to_host(), self.d_results_b.copy_to_host()


    class UnifiedMemoryGPUContext:
        """
        GPU context optimized for unified memory systems (Grace Hopper/Blackwell).

        On unified memory, CPU and GPU share the same physical memory.
        Transfer data once, reuse for all kernel calls.

        Note: Separate kernels for a/b are FASTER than fused for memory-bound workloads.
        Fusion increases register pressure and reduces parallelism.
        """
        def __init__(self, a_vals: np.ndarray, b_vals: np.ndarray, spf: np.ndarray):
            """
            Initialize unified memory GPU context.

            Transfers arrays once to device memory (cheap on unified memory).
            """
            print("    Setting up unified memory GPU context...", end=" ", flush=True)
            self.n = len(a_vals)

            # Transfer arrays once - on unified memory this is just pinning
            self.d_a = cuda.to_device(np.ascontiguousarray(a_vals, dtype=np.int64))
            self.d_b = cuda.to_device(np.ascontiguousarray(b_vals, dtype=np.int64))
            self.d_spf = cuda.to_device(np.ascontiguousarray(spf, dtype=np.uint32))

            # Results arrays (reused across calls)
            self.d_results_a = cuda.device_array(self.n, dtype=np.int32)
            self.d_results_b = cuda.device_array(self.n, dtype=np.int32)

            # 256 threads works well for this memory-bound workload
            self.threads_per_block = 256
            self.blocks = (self.n + self.threads_per_block - 1) // self.threads_per_block

            cuda.synchronize()
            print("done")
            print(f"    Arrays: a/b={self.n:,} elements, SPF={len(spf):,} elements")

        def compute_omega_leq_P(self, P: int):
            """Compute omega_leq_P using separate kernels (faster for memory-bound)."""
            _omega_leq_P_kernel[self.blocks, self.threads_per_block](
                self.d_a, self.d_spf, P, self.d_results_a
            )
            _omega_leq_P_kernel[self.blocks, self.threads_per_block](
                self.d_b, self.d_spf, P, self.d_results_b
            )
            cuda.synchronize()

            return self.d_results_a.copy_to_host(), self.d_results_b.copy_to_host()

        def compute_omega(self):
            """Compute full omega using separate kernels."""
            _omega_kernel[self.blocks, self.threads_per_block](
                self.d_a, self.d_spf, self.d_results_a
            )
            _omega_kernel[self.blocks, self.threads_per_block](
                self.d_b, self.d_spf, self.d_results_b
            )
            cuda.synchronize()

            return self.d_results_a.copy_to_host(), self.d_results_b.copy_to_host()

        def compute_omega_leq_P_aggregated(self, P: int, d_state_codes):
            """
            Compute omega_leq_P and aggregate by state ON GPU.

            Returns only 8 numbers: (sums_a[4], sums_b[4]) instead of 8GB arrays.
            This eliminates the expensive host transfer.
            """
            # Compute omega
            _omega_leq_P_kernel[self.blocks, self.threads_per_block](
                self.d_a, self.d_spf, P, self.d_results_a
            )
            _omega_leq_P_kernel[self.blocks, self.threads_per_block](
                self.d_b, self.d_spf, P, self.d_results_b
            )

            # Allocate block-level partial sums (reuse if already allocated)
            if not hasattr(self, 'd_block_sums_a'):
                self.d_block_sums_a = cuda.device_array((self.blocks, 4), dtype=np.float64)
                self.d_block_sums_b = cuda.device_array((self.blocks, 4), dtype=np.float64)
                self.d_final_sums_a = cuda.device_array(4, dtype=np.float64)
                self.d_final_sums_b = cuda.device_array(4, dtype=np.float64)

            # Aggregate omega_a by state
            _aggregate_by_state_kernel[self.blocks, self.threads_per_block](
                self.d_results_a, d_state_codes, self.n, self.d_block_sums_a
            )
            _reduce_block_sums_kernel[1, 4](
                self.d_block_sums_a, self.blocks, self.d_final_sums_a
            )

            # Aggregate omega_b by state
            _aggregate_by_state_kernel[self.blocks, self.threads_per_block](
                self.d_results_b, d_state_codes, self.n, self.d_block_sums_b
            )
            _reduce_block_sums_kernel[1, 4](
                self.d_block_sums_b, self.blocks, self.d_final_sums_b
            )

            cuda.synchronize()

            # Transfer only 8 numbers instead of 8GB!
            return self.d_final_sums_a.copy_to_host(), self.d_final_sums_b.copy_to_host()

        def compute_omega_aggregated(self, d_state_codes):
            """
            Compute full omega and aggregate by state ON GPU.

            Returns only 8 numbers: (sums_a[4], sums_b[4]).
            """
            # Compute omega
            _omega_kernel[self.blocks, self.threads_per_block](
                self.d_a, self.d_spf, self.d_results_a
            )
            _omega_kernel[self.blocks, self.threads_per_block](
                self.d_b, self.d_spf, self.d_results_b
            )

            # Allocate if needed
            if not hasattr(self, 'd_block_sums_a'):
                self.d_block_sums_a = cuda.device_array((self.blocks, 4), dtype=np.float64)
                self.d_block_sums_b = cuda.device_array((self.blocks, 4), dtype=np.float64)
                self.d_final_sums_a = cuda.device_array(4, dtype=np.float64)
                self.d_final_sums_b = cuda.device_array(4, dtype=np.float64)

            # Aggregate
            _aggregate_by_state_kernel[self.blocks, self.threads_per_block](
                self.d_results_a, d_state_codes, self.n, self.d_block_sums_a
            )
            _reduce_block_sums_kernel[1, 4](
                self.d_block_sums_a, self.blocks, self.d_final_sums_a
            )

            _aggregate_by_state_kernel[self.blocks, self.threads_per_block](
                self.d_results_b, d_state_codes, self.n, self.d_block_sums_b
            )
            _reduce_block_sums_kernel[1, 4](
                self.d_block_sums_b, self.blocks, self.d_final_sums_b
            )

            cuda.synchronize()

            return self.d_final_sums_a.copy_to_host(), self.d_final_sums_b.copy_to_host()

    def omega_leq_P_gpu(numbers: np.ndarray, spf: np.ndarray, P: int) -> np.ndarray:
        """
        Compute omega_leq_P for array of numbers on GPU.
        (Simple version - transfers data each call. Use GPUContext for batch.)
        """
        n_numbers = len(numbers)
        d_numbers = cuda.to_device(numbers.astype(np.int64))
        d_spf = cuda.to_device(spf.astype(np.int64))
        d_results = cuda.device_array(n_numbers, dtype=np.int64)

        threads_per_block = 256
        blocks = (n_numbers + threads_per_block - 1) // threads_per_block

        _omega_leq_P_kernel[blocks, threads_per_block](d_numbers, d_spf, P, d_results)
        cuda.synchronize()

        return d_results.copy_to_host()

    def omega_gpu(numbers: np.ndarray, spf: np.ndarray) -> np.ndarray:
        """Compute omega for array of numbers on GPU."""
        n_numbers = len(numbers)
        d_numbers = cuda.to_device(numbers.astype(np.int64))
        d_spf = cuda.to_device(spf.astype(np.int64))
        d_results = cuda.device_array(n_numbers, dtype=np.int64)

        threads_per_block = 256
        blocks = (n_numbers + threads_per_block - 1) // threads_per_block

        _omega_kernel[blocks, threads_per_block](d_numbers, d_spf, d_results)
        cuda.synchronize()

        return d_results.copy_to_host()

    def compute_states_gpu(K: int, prime_flags: np.ndarray) -> np.ndarray:
        """Compute pair states on GPU."""
        d_flags = cuda.to_device(prime_flags.astype(np.uint8))
        d_results = cuda.device_array(K, dtype=np.int32)

        threads_per_block = 256
        blocks = (K + threads_per_block - 1) // threads_per_block

        _pair_states_kernel[blocks, threads_per_block](d_flags, K, d_results)
        cuda.synchronize()

        return d_results.copy_to_host()


def check_gpu():
    """Check GPU availability and print info."""
    if not HAS_GPU:
        print("GPU not available (numba not installed or no CUDA device)")
        return False

    device = cuda.get_current_device()
    print(f"GPU available: {device.name}")
    print(f"  Compute capability: {device.compute_capability}")
    try:
        print(f"  Total memory: {device.total_memory / 1e9:.1f} GB")
    except AttributeError:
        pass

    if HAS_UNIFIED_MEMORY:
        print(f"  Unified memory: YES (GPU can access system RAM directly)")
    else:
        print(f"  Unified memory: NO (discrete GPU with separate VRAM)")

    return True
