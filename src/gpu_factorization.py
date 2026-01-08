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

    @cuda.jit
    def _aggregate_batched_row_kernel(results_row, state_codes, n, block_sums):
        """
        Aggregate a single row from batched results by state.
        results_row: 1D array of omega values for one P threshold
        """
        shared_sums = cuda.shared.array(4, dtype=numba.float64)

        tid = cuda.threadIdx.x
        bid = cuda.blockIdx.x
        idx = cuda.grid(1)

        if tid < 4:
            shared_sums[tid] = 0.0
        cuda.syncthreads()

        if idx < n:
            state = state_codes[idx]
            cuda.atomic.add(shared_sums, state, float(results_row[idx]))

        cuda.syncthreads()

        if tid < 4:
            block_sums[bid, tid] = shared_sums[tid]

    @cuda.jit
    def _omega_fused_batched_kernel(a_vals, b_vals, spf, P_values, n_P,
                                     state_codes, n, block_sums):
        """
        FUSED kernel: factor once, count for all P, aggregate by state.

        Eliminates ALL intermediate storage by aggregating directly.

        block_sums shape: (n_blocks, n_P+1, 4, 2)
        - n_P+1: one row per P threshold + one for full omega
        - 4: states (PP=0, PC=1, CP=2, CC=3)
        - 2: a values (0) and b values (1)
        """
        # Shared memory for this block's partial sums
        # Layout: [p_idx][state][ab] where ab=0 for a, ab=1 for b
        # Size: (n_P+1) * 4 * 2 = up to 64 doubles for 7 P values
        shared_sums = cuda.shared.array((8, 4, 2), dtype=numba.float64)

        tid = cuda.threadIdx.x
        bid = cuda.blockIdx.x
        idx = cuda.grid(1)

        # Initialize shared memory (first 64 threads)
        init_idx = tid
        while init_idx < (n_P + 1) * 4 * 2:
            p_idx = init_idx // 8
            state = (init_idx // 2) % 4
            ab = init_idx % 2
            if p_idx <= n_P:
                shared_sums[p_idx, state, ab] = 0.0
            init_idx += cuda.blockDim.x
        cuda.syncthreads()

        if idx < n:
            state = state_codes[idx]

            # ===== Process a value =====
            val_a = a_vals[idx]
            if val_a <= 1:
                # All counts are 0, nothing to add
                pass
            else:
                # Factor and collect distinct primes
                factors = cuda.local.array(20, dtype=numba.int64)
                n_factors = 0
                prev = 0
                temp_n = val_a

                while temp_n > 1:
                    p = spf[temp_n]
                    if p == 0:
                        p = temp_n
                    if p != prev and n_factors < 20:
                        factors[n_factors] = p
                        n_factors += 1
                    prev = p
                    temp_n //= p

                # Count for each P threshold and aggregate
                for p_idx in range(n_P):
                    P = P_values[p_idx]
                    count = 0
                    for i in range(n_factors):
                        if factors[i] <= P:
                            count += 1
                    cuda.atomic.add(shared_sums, (p_idx, state, 0), float(count))

                # Full omega (last row)
                cuda.atomic.add(shared_sums, (n_P, state, 0), float(n_factors))

            # ===== Process b value =====
            val_b = b_vals[idx]
            if val_b <= 1:
                pass
            else:
                factors = cuda.local.array(20, dtype=numba.int64)
                n_factors = 0
                prev = 0
                temp_n = val_b

                while temp_n > 1:
                    p = spf[temp_n]
                    if p == 0:
                        p = temp_n
                    if p != prev and n_factors < 20:
                        factors[n_factors] = p
                        n_factors += 1
                    prev = p
                    temp_n //= p

                for p_idx in range(n_P):
                    P = P_values[p_idx]
                    count = 0
                    for i in range(n_factors):
                        if factors[i] <= P:
                            count += 1
                    cuda.atomic.add(shared_sums, (p_idx, state, 1), float(count))

                cuda.atomic.add(shared_sums, (n_P, state, 1), float(n_factors))

        cuda.syncthreads()

        # Write block results to global memory
        write_idx = tid
        while write_idx < (n_P + 1) * 4 * 2:
            p_idx = write_idx // 8
            state = (write_idx // 2) % 4
            ab = write_idx % 2
            if p_idx <= n_P:
                block_sums[bid, p_idx, state, ab] = shared_sums[p_idx, state, ab]
            write_idx += cuda.blockDim.x

    @cuda.jit
    def _reduce_fused_block_sums_kernel(block_sums, n_blocks, n_P, final_sums):
        """
        Reduce block sums to final sums.

        block_sums shape: (n_blocks, n_P+1, 4, 2)
        final_sums shape: (n_P+1, 4, 2)
        """
        # Each thread handles one (p_idx, state, ab) combination
        flat_idx = cuda.grid(1)
        max_idx = (n_P + 1) * 4 * 2

        if flat_idx >= max_idx:
            return

        p_idx = flat_idx // 8
        state = (flat_idx // 2) % 4
        ab = flat_idx % 2

        total = 0.0
        for i in range(n_blocks):
            total += block_sums[i, p_idx, state, ab]

        final_sums[p_idx, state, ab] = total

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

        def compute_all_omega_batched(self, P_grid: list, d_state_codes):
            """
            BATCHED computation: factor ONCE, count for ALL P values.

            This is ~7x faster than calling compute_omega_leq_P_aggregated in a loop
            because we only traverse the factorization tree once per number.

            Returns:
                dict mapping P -> (sums_a, sums_b) where each is array of 4 state sums
                Also includes 'full' key for full omega (no P limit)
            """
            import time
            n_P = len(P_grid)

            # Allocate results matrices: (n_P + 1, n) - last row is full omega
            # For K=1e8: 8 * 100M * 4 bytes = 3.2GB per matrix
            print(f"    Allocating batched results ({(n_P + 1) * self.n * 4 / 1e9:.1f}GB per array)...", end=" ", flush=True)
            d_results_a = cuda.device_array((n_P + 1, self.n), dtype=np.int32)
            d_results_b = cuda.device_array((n_P + 1, self.n), dtype=np.int32)

            # P values array on device
            d_P_values = cuda.to_device(np.array(P_grid, dtype=np.int64))
            cuda.synchronize()
            print("done")

            # Run batched kernels - factor once, count for all P
            print(f"    Running batched factorization (1 pass instead of {n_P})...", end=" ", flush=True)
            t0 = time.time()
            _omega_batched_kernel[self.blocks, self.threads_per_block](
                self.d_a, self.d_spf, d_P_values, n_P, d_results_a
            )
            _omega_batched_kernel[self.blocks, self.threads_per_block](
                self.d_b, self.d_spf, d_P_values, n_P, d_results_b
            )
            cuda.synchronize()
            print(f"{time.time() - t0:.1f}s")

            # Allocate aggregation buffers (reuse for all P)
            if not hasattr(self, 'd_block_sums_a'):
                self.d_block_sums_a = cuda.device_array((self.blocks, 4), dtype=np.float64)
                self.d_block_sums_b = cuda.device_array((self.blocks, 4), dtype=np.float64)
                self.d_final_sums_a = cuda.device_array(4, dtype=np.float64)
                self.d_final_sums_b = cuda.device_array(4, dtype=np.float64)

            # Aggregate each P's results by state
            results = {}
            print(f"    Aggregating by state for {n_P} P values...", end=" ", flush=True)
            t0 = time.time()

            for p_idx, P in enumerate(P_grid):
                # Aggregate row p_idx for a values
                _aggregate_batched_row_kernel[self.blocks, self.threads_per_block](
                    d_results_a[p_idx], d_state_codes, self.n, self.d_block_sums_a
                )
                _reduce_block_sums_kernel[1, 4](
                    self.d_block_sums_a, self.blocks, self.d_final_sums_a
                )

                # Aggregate row p_idx for b values
                _aggregate_batched_row_kernel[self.blocks, self.threads_per_block](
                    d_results_b[p_idx], d_state_codes, self.n, self.d_block_sums_b
                )
                _reduce_block_sums_kernel[1, 4](
                    self.d_block_sums_b, self.blocks, self.d_final_sums_b
                )

                cuda.synchronize()
                results[P] = (
                    self.d_final_sums_a.copy_to_host().copy(),
                    self.d_final_sums_b.copy_to_host().copy()
                )

            # Also get full omega (last row)
            _aggregate_batched_row_kernel[self.blocks, self.threads_per_block](
                d_results_a[n_P], d_state_codes, self.n, self.d_block_sums_a
            )
            _reduce_block_sums_kernel[1, 4](
                self.d_block_sums_a, self.blocks, self.d_final_sums_a
            )

            _aggregate_batched_row_kernel[self.blocks, self.threads_per_block](
                d_results_b[n_P], d_state_codes, self.n, self.d_block_sums_b
            )
            _reduce_block_sums_kernel[1, 4](
                self.d_block_sums_b, self.blocks, self.d_final_sums_b
            )

            cuda.synchronize()
            results['full'] = (
                self.d_final_sums_a.copy_to_host().copy(),
                self.d_final_sums_b.copy_to_host().copy()
            )

            print(f"{time.time() - t0:.1f}s")

            return results

        def compute_all_omega_fused(self, P_grid: list, d_state_codes):
            """
            FUSED computation: factor, count, AND aggregate all in one kernel.

            This eliminates ALL intermediate storage - no 6.4GB arrays!
            Only allocates ~100MB for block-level partial sums.

            Returns:
                dict mapping P -> (sums_a, sums_b) where each is array of 4 state sums
                Also includes 'full' key for full omega (no P limit)
            """
            import time
            n_P = len(P_grid)

            # Only allocate block-level sums: (n_blocks, n_P+1, 4, 2)
            # For K=1e8: 400K blocks * 8 * 4 * 2 * 8 bytes = ~200MB (vs 6.4GB before!)
            block_sums_size = self.blocks * (n_P + 1) * 4 * 2 * 8 / 1e6
            print(f"    Allocating fused block sums ({block_sums_size:.1f}MB)...", end=" ", flush=True)

            d_block_sums = cuda.device_array((self.blocks, n_P + 1, 4, 2), dtype=np.float64)
            d_final_sums = cuda.device_array((n_P + 1, 4, 2), dtype=np.float64)
            d_P_values = cuda.to_device(np.array(P_grid, dtype=np.int64))
            cuda.synchronize()
            print("done")

            # Run single fused kernel
            print(f"    Running FUSED kernel (factor + count + aggregate)...", end=" ", flush=True)
            t0 = time.time()
            _omega_fused_batched_kernel[self.blocks, self.threads_per_block](
                self.d_a, self.d_b, self.d_spf, d_P_values, n_P,
                d_state_codes, self.n, d_block_sums
            )
            cuda.synchronize()
            kernel_time = time.time() - t0
            print(f"{kernel_time:.2f}s")

            # Reduce block sums
            print(f"    Reducing block sums...", end=" ", flush=True)
            t0 = time.time()
            n_threads = (n_P + 1) * 4 * 2  # One thread per output element
            _reduce_fused_block_sums_kernel[1, max(n_threads, 32)](
                d_block_sums, self.blocks, n_P, d_final_sums
            )
            cuda.synchronize()
            print(f"{time.time() - t0:.2f}s")

            # Extract results
            final_sums = d_final_sums.copy_to_host()

            results = {}
            for p_idx, P in enumerate(P_grid):
                sums_a = final_sums[p_idx, :, 0].copy()
                sums_b = final_sums[p_idx, :, 1].copy()
                results[P] = (sums_a, sums_b)

            # Full omega (last row)
            results['full'] = (
                final_sums[n_P, :, 0].copy(),
                final_sums[n_P, :, 1].copy()
            )

            return results

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
