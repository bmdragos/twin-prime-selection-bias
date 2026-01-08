"""
GPU-accelerated wheel sieve for 6k±1 numbers.

Uses wheel-indexed SPF array (3x smaller than full SPF).
Enables K=10^10 on 128GB unified memory systems.

Memory at K=10^10:
- Full SPF: 240GB (doesn't fit)
- Wheel SPF: 80GB (fits!)
"""

import numpy as np

try:
    from numba import cuda
    import numba
    HAS_GPU = cuda.is_available()
except ImportError:
    HAS_GPU = False


if HAS_GPU:
    # ========== Device Functions ==========

    @cuda.jit(device=True)
    def n_to_index_device(n: int) -> int:
        """Convert number (≡ 1 or 5 mod 6) to wheel index. Device function."""
        r = n % 6
        if r == 5:
            return (n - 5) // 3
        else:  # r == 1
            return (n - 4) // 3

    @cuda.jit(device=True)
    def wheel_spf_lookup_device(n: int, spf_wheel) -> int:
        """
        Get smallest prime factor using wheel-indexed SPF. Device function.

        Handles:
        - n divisible by 2 → 2
        - n divisible by 3 → 3
        - n ≡ 1 or 5 (mod 6) → lookup in wheel array
        """
        if n <= 1:
            return n
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3

        # n is ≡ 1 or 5 (mod 6)
        idx = n_to_index_device(n)
        p = spf_wheel[idx]
        if p == 0:  # n is prime
            return n
        return p

    # ========== GPU Kernels ==========

    @cuda.jit
    def _omega_wheel_kernel(numbers, spf_wheel, results):
        """
        CUDA kernel: compute omega (distinct prime factors) using wheel SPF.
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
            p = wheel_spf_lookup_device(n, spf_wheel)
            if p != prev:
                count += 1
                prev = p
            n //= p

        results[idx] = count

    @cuda.jit
    def _omega_leq_P_wheel_kernel(numbers, spf_wheel, P, results):
        """
        CUDA kernel: compute omega_leq_P using wheel SPF.
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
            p = wheel_spf_lookup_device(n, spf_wheel)
            if p != prev and p <= P:
                count += 1
            if p != prev:
                prev = p
            n //= p

        results[idx] = count

    @cuda.jit
    def _omega_wheel_onthefly_kernel(K, spf_wheel, P, results_a, results_b):
        """
        CUDA kernel: compute omega_leq_P for pair k WITHOUT storing a/b arrays.

        Generates 6k-1 and 6k+1 on-the-fly, saving 160GB at K=10^10.
        """
        k = cuda.grid(1) + 1  # k from 1 to K
        if k > K:
            return

        # Generate pair values on-the-fly
        a = 6 * k - 1
        b = 6 * k + 1
        idx = k - 1  # 0-indexed results

        # Compute omega_leq_P for a
        n = a
        count = 0
        prev = 0
        while n > 1:
            p = wheel_spf_lookup_device(n, spf_wheel)
            if p != prev and p <= P:
                count += 1
            if p != prev:
                prev = p
            n //= p
        results_a[idx] = count

        # Compute omega_leq_P for b
        n = b
        count = 0
        prev = 0
        while n > 1:
            p = wheel_spf_lookup_device(n, spf_wheel)
            if p != prev and p <= P:
                count += 1
            if p != prev:
                prev = p
            n //= p
        results_b[idx] = count

    @cuda.jit
    def _omega_wheel_full_onthefly_kernel(K, spf_wheel, results_a, results_b):
        """
        CUDA kernel: compute full omega for pair k on-the-fly.
        """
        k = cuda.grid(1) + 1
        if k > K:
            return

        a = 6 * k - 1
        b = 6 * k + 1
        idx = k - 1

        # omega for a
        n = a
        count = 0
        prev = 0
        while n > 1:
            p = wheel_spf_lookup_device(n, spf_wheel)
            if p != prev:
                count += 1
                prev = p
            n //= p
        results_a[idx] = count

        # omega for b
        n = b
        count = 0
        prev = 0
        while n > 1:
            p = wheel_spf_lookup_device(n, spf_wheel)
            if p != prev:
                count += 1
                prev = p
            n //= p
        results_b[idx] = count

    @cuda.jit
    def _pair_states_wheel_kernel(K, spf_wheel, results):
        """
        CUDA kernel: compute pair states on-the-fly using wheel SPF.

        A number n is prime iff wheel_spf_lookup returns n itself.
        States: PP=0, PC=1, CP=2, CC=3
        """
        k = cuda.grid(1) + 1
        if k > K:
            return

        a = 6 * k - 1
        b = 6 * k + 1
        idx = k - 1

        # Check if prime: SPF(n) == n means prime
        a_prime = (wheel_spf_lookup_device(a, spf_wheel) == a)
        b_prime = (wheel_spf_lookup_device(b, spf_wheel) == b)

        if a_prime and b_prime:
            results[idx] = 0
        elif a_prime and not b_prime:
            results[idx] = 1
        elif not a_prime and b_prime:
            results[idx] = 2
        else:
            results[idx] = 3

    @cuda.jit
    def _p1_p2_wheel_kernel(K, spf_wheel, p1_out, p2_out, state_out):
        """
        CUDA kernel: compute p1, p2, and state codes on-the-fly.

        p1 = min(spf(a), spf(b)) for CC pairs
        p2 = max(spf(a), spf(b)) for CC pairs
        For PC/CP: p1 = SPF(composite), p2 = 0 (censored)
        For PP: p1 = 0, p2 = 0
        """
        k = cuda.grid(1) + 1
        if k > K:
            return

        a = 6 * k - 1
        b = 6 * k + 1
        idx = k - 1

        spf_a = wheel_spf_lookup_device(a, spf_wheel)
        spf_b = wheel_spf_lookup_device(b, spf_wheel)

        a_prime = (spf_a == a)
        b_prime = (spf_b == b)

        if a_prime and b_prime:
            state_out[idx] = 0
            p1_out[idx] = 0
            p2_out[idx] = 0
        elif a_prime:
            state_out[idx] = 1
            p1_out[idx] = spf_b
            p2_out[idx] = 0
        elif b_prime:
            state_out[idx] = 2
            p1_out[idx] = spf_a
            p2_out[idx] = 0
        else:
            state_out[idx] = 3
            if spf_a <= spf_b:
                p1_out[idx] = spf_a
                p2_out[idx] = spf_b
            else:
                p1_out[idx] = spf_b
                p2_out[idx] = spf_a

    @cuda.jit
    def _aggregate_by_state_kernel(omega, state_codes, n, block_sums):
        """
        Two-phase reduction: each block computes partial sums per state.
        (Same as in gpu_factorization.py)
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
            cuda.atomic.add(shared_sums, state, float(omega[idx]))

        cuda.syncthreads()

        if tid < 4:
            block_sums[bid, tid] = shared_sums[tid]

    @cuda.jit
    def _reduce_block_sums_kernel(block_sums, n_blocks, final_sums):
        """Final reduction: sum all block partial sums."""
        state = cuda.grid(1)
        if state >= 4:
            return

        total = 0.0
        for i in range(n_blocks):
            total += block_sums[i, state]
        final_sums[state] = total

    # ========== GPU Context Class ==========

    class WheelGPUContext:
        """
        GPU context for wheel-indexed SPF on unified memory systems.

        Memory usage at K=10^10:
        - spf_wheel: 80GB (vs 240GB for full SPF)
        - results: 40GB (2 × 10^10 × 4 bytes)
        - state_codes: 10GB
        - Total: ~130GB (fits in 128GB with on-the-fly generation)

        With on-the-fly a/b generation, no a_vals/b_vals arrays needed.
        """

        def __init__(self, K: int, spf_wheel: np.ndarray):
            """
            Initialize wheel GPU context.

            Parameters
            ----------
            K : int
                Number of pairs (will compute for k=1..K)
            spf_wheel : np.ndarray
                Wheel-indexed SPF array from wheel_spf_sieve(K)
            """
            print(f"    Setting up wheel GPU context for K={K:,}...", end=" ", flush=True)
            self.K = K

            # Transfer wheel SPF (this is the big one: 80GB at K=10^10)
            self.d_spf_wheel = cuda.to_device(np.ascontiguousarray(spf_wheel, dtype=np.uint32))

            # Results arrays
            self.d_results_a = cuda.device_array(K, dtype=np.int32)
            self.d_results_b = cuda.device_array(K, dtype=np.int32)

            # Grid dimensions
            self.threads_per_block = 256
            self.blocks = (K + self.threads_per_block - 1) // self.threads_per_block

            cuda.synchronize()
            print("done")

            spf_gb = spf_wheel.nbytes / 1e9
            print(f"    Wheel SPF: {spf_gb:.1f}GB ({len(spf_wheel):,} elements)")

        def compute_states(self) -> np.ndarray:
            """Compute pair states on-the-fly."""
            d_states = cuda.device_array(self.K, dtype=np.int32)

            _pair_states_wheel_kernel[self.blocks, self.threads_per_block](
                self.K, self.d_spf_wheel, d_states
            )
            cuda.synchronize()

            return d_states.copy_to_host()

        def compute_p1_p2(self):
            """Compute p1, p2, and state codes for all pairs."""
            d_p1 = cuda.device_array(self.K, dtype=np.uint32)
            d_p2 = cuda.device_array(self.K, dtype=np.uint32)
            d_states = cuda.device_array(self.K, dtype=np.uint8)

            _p1_p2_wheel_kernel[self.blocks, self.threads_per_block](
                self.K, self.d_spf_wheel, d_p1, d_p2, d_states
            )
            cuda.synchronize()

            return d_p1.copy_to_host(), d_p2.copy_to_host(), d_states.copy_to_host()

        def compute_states_gpu(self):
            """Compute pair states, keep on GPU."""
            if not hasattr(self, 'd_state_codes'):
                self.d_state_codes = cuda.device_array(self.K, dtype=np.int32)

            _pair_states_wheel_kernel[self.blocks, self.threads_per_block](
                self.K, self.d_spf_wheel, self.d_state_codes
            )
            cuda.synchronize()

            return self.d_state_codes

        def compute_omega_leq_P(self, P: int):
            """Compute omega_leq_P for all pairs on-the-fly."""
            _omega_wheel_onthefly_kernel[self.blocks, self.threads_per_block](
                self.K, self.d_spf_wheel, P, self.d_results_a, self.d_results_b
            )
            cuda.synchronize()

            return self.d_results_a.copy_to_host(), self.d_results_b.copy_to_host()

        def compute_omega(self):
            """Compute full omega for all pairs on-the-fly."""
            _omega_wheel_full_onthefly_kernel[self.blocks, self.threads_per_block](
                self.K, self.d_spf_wheel, self.d_results_a, self.d_results_b
            )
            cuda.synchronize()

            return self.d_results_a.copy_to_host(), self.d_results_b.copy_to_host()

        def compute_omega_leq_P_aggregated(self, P: int, d_state_codes=None):
            """
            Compute omega_leq_P and aggregate by state ON GPU.

            Returns only 8 numbers instead of 80GB arrays.
            """
            if d_state_codes is None:
                d_state_codes = self.compute_states_gpu()

            # Compute omega
            _omega_wheel_onthefly_kernel[self.blocks, self.threads_per_block](
                self.K, self.d_spf_wheel, P, self.d_results_a, self.d_results_b
            )

            # Allocate block sums if needed
            if not hasattr(self, 'd_block_sums_a'):
                self.d_block_sums_a = cuda.device_array((self.blocks, 4), dtype=np.float64)
                self.d_block_sums_b = cuda.device_array((self.blocks, 4), dtype=np.float64)
                self.d_final_sums_a = cuda.device_array(4, dtype=np.float64)
                self.d_final_sums_b = cuda.device_array(4, dtype=np.float64)

            # Aggregate omega_a by state
            _aggregate_by_state_kernel[self.blocks, self.threads_per_block](
                self.d_results_a, d_state_codes, self.K, self.d_block_sums_a
            )
            _reduce_block_sums_kernel[1, 4](
                self.d_block_sums_a, self.blocks, self.d_final_sums_a
            )

            # Aggregate omega_b by state
            _aggregate_by_state_kernel[self.blocks, self.threads_per_block](
                self.d_results_b, d_state_codes, self.K, self.d_block_sums_b
            )
            _reduce_block_sums_kernel[1, 4](
                self.d_block_sums_b, self.blocks, self.d_final_sums_b
            )

            cuda.synchronize()

            return self.d_final_sums_a.copy_to_host(), self.d_final_sums_b.copy_to_host()

        def compute_omega_aggregated(self, d_state_codes=None):
            """Compute full omega and aggregate by state ON GPU."""
            if d_state_codes is None:
                d_state_codes = self.compute_states_gpu()

            _omega_wheel_full_onthefly_kernel[self.blocks, self.threads_per_block](
                self.K, self.d_spf_wheel, self.d_results_a, self.d_results_b
            )

            if not hasattr(self, 'd_block_sums_a'):
                self.d_block_sums_a = cuda.device_array((self.blocks, 4), dtype=np.float64)
                self.d_block_sums_b = cuda.device_array((self.blocks, 4), dtype=np.float64)
                self.d_final_sums_a = cuda.device_array(4, dtype=np.float64)
                self.d_final_sums_b = cuda.device_array(4, dtype=np.float64)

            _aggregate_by_state_kernel[self.blocks, self.threads_per_block](
                self.d_results_a, d_state_codes, self.K, self.d_block_sums_a
            )
            _reduce_block_sums_kernel[1, 4](
                self.d_block_sums_a, self.blocks, self.d_final_sums_a
            )

            _aggregate_by_state_kernel[self.blocks, self.threads_per_block](
                self.d_results_b, d_state_codes, self.K, self.d_block_sums_b
            )
            _reduce_block_sums_kernel[1, 4](
                self.d_block_sums_b, self.blocks, self.d_final_sums_b
            )

            cuda.synchronize()

            return self.d_final_sums_a.copy_to_host(), self.d_final_sums_b.copy_to_host()


def check_wheel_gpu():
    """Check GPU availability for wheel sieve."""
    if not HAS_GPU:
        print("GPU not available")
        return False

    from numba import cuda
    device = cuda.get_current_device()
    print(f"GPU: {device.name}")
    print(f"Compute capability: {device.compute_capability}")
    return True
