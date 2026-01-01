"""
Parallel segmented sieve for SPF computation.

Uses multiprocessing to parallelize across CPU cores.
For large arrays, uses shared memory to avoid pickling overhead.
"""

import numpy as np
from multiprocessing import Pool, cpu_count, shared_memory
from typing import Tuple
import math

# Global variables for worker processes (set via initializer)
_worker_spf = None
_worker_shm = None


def _sieve_small_primes(limit: int) -> np.ndarray:
    """Sieve of Eratosthenes for small primes."""
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False

    for p in range(2, int(limit**0.5) + 1):
        if is_prime[p]:
            is_prime[p*p::p] = False

    return np.nonzero(is_prime)[0]


def _process_segment(args: Tuple[int, int, np.ndarray]) -> np.ndarray:
    """
    Process a single segment: compute SPF for range [start, end).

    Returns SPF array for this segment.
    Uses uint32 with 0 = prime sentinel (SPF values are at most sqrt(N) < 77k).
    """
    start, end, small_primes = args
    size = end - start

    # Initialize to 0 (meaning "prime" - SPF equals the number itself)
    # Use uint32 to halve memory usage (48GB -> 24GB for K=10^9)
    spf = np.zeros(size, dtype=np.uint32)

    # Special case for 0 and 1: mark as non-prime with SPF=1
    if start == 0:
        spf[0] = 1  # 0 has no SPF, use 1 as marker
        if size > 1:
            spf[1] = 1  # 1 has no SPF, use 1 as marker

    # Sieve with small primes
    for p in small_primes:
        if p * p >= end:
            break

        # Find first multiple of p in range [start, end)
        first = ((start + p - 1) // p) * p
        if first < p * p:
            first = p * p

        # Mark multiples (only if not already marked)
        for j in range(first - start, size, p):
            if spf[j] == 0:  # Not yet marked (still prime candidate)
                spf[j] = p

    return spf


def spf_sieve_parallel(N: int, num_workers: int = None) -> np.ndarray:
    """
    Compute SPF for all integers up to N using parallel segmented sieve.

    Parameters
    ----------
    N : int
        Upper bound (inclusive).
    num_workers : int, optional
        Number of parallel workers. Defaults to CPU count.

    Returns
    -------
    np.ndarray
        Array where spf[i] is the smallest prime factor of i.
    """
    if num_workers is None:
        num_workers = cpu_count()

    # Step 1: Sieve small primes up to sqrt(N)
    sqrt_n = int(math.sqrt(N)) + 1
    small_primes = _sieve_small_primes(sqrt_n)

    print(f"    Found {len(small_primes)} small primes up to {sqrt_n}")

    # Step 2: Determine segment size and create tasks
    # Aim for ~100 segments per worker for good load balancing
    segment_size = max(10**6, (N + 1) // (num_workers * 100))

    segments = []
    for start in range(0, N + 1, segment_size):
        end = min(start + segment_size, N + 1)
        segments.append((start, end, small_primes))

    print(f"    Processing {len(segments)} segments with {num_workers} workers...")

    # Step 3: Process segments in parallel
    with Pool(num_workers) as pool:
        results = pool.map(_process_segment, segments)

    # Step 4: Concatenate results
    spf = np.concatenate(results)

    return spf


def prime_flags_parallel(N: int, num_workers: int = None) -> np.ndarray:
    """
    Compute prime flags using parallel sieve.

    A number n > 1 is prime iff spf[n] == n.
    """
    spf = spf_sieve_parallel(N, num_workers)

    flags = np.zeros(N + 1, dtype=bool)
    flags[2:] = (spf[2:] == np.arange(2, N + 1))

    return flags


def _init_worker_shm(shm_name: str, shape: tuple, dtype):
    """Initialize worker with shared memory SPF array."""
    global _worker_spf, _worker_shm
    _worker_shm = shared_memory.SharedMemory(name=shm_name)
    _worker_spf = np.ndarray(shape, dtype=dtype, buffer=_worker_shm.buf)


def _omega_leq_P_chunk_shm(args):
    """Compute omega_leq_P for a chunk using shared memory SPF.

    SPF uses 0 = prime sentinel (spf[n] == 0 means n is prime, SPF = n).
    """
    numbers, P = args
    global _worker_spf
    spf = _worker_spf  # Read-only access
    results = np.zeros(len(numbers), dtype=np.int32)

    for i, n in enumerate(numbers):
        if n <= 1:
            continue
        count = 0
        prev = 0
        while n > 1:
            p = spf[n]
            if p == 0:  # n is prime
                p = n
            if p != prev and p <= P:
                count += 1
            prev = p
            n //= p
        results[i] = count

    return results


def _omega_chunk_shm(args):
    """Compute full omega for a chunk using shared memory SPF.

    SPF uses 0 = prime sentinel (spf[n] == 0 means n is prime, SPF = n).
    """
    numbers = args
    global _worker_spf
    spf = _worker_spf  # Read-only access
    results = np.zeros(len(numbers), dtype=np.int32)

    for i, n in enumerate(numbers):
        if n <= 1:
            continue
        count = 0
        prev = 0
        while n > 1:
            p = spf[n]
            if p == 0:  # n is prime
                p = n
            if p != prev:
                count += 1
            prev = p
            n //= p
        results[i] = count

    return results


def _omega_leq_P_chunk(args):
    """Compute omega_leq_P for a chunk of numbers.

    SPF uses 0 = prime sentinel (spf[n] == 0 means n is prime, SPF = n).
    """
    numbers, spf, P = args
    results = np.zeros(len(numbers), dtype=np.int32)

    for i, n in enumerate(numbers):
        if n <= 1:
            continue
        count = 0
        prev = 0
        while n > 1:
            p = spf[n]
            if p == 0:  # n is prime
                p = n
            if p != prev and p <= P:
                count += 1
            prev = p
            n //= p
        results[i] = count

    return results


def _omega_chunk(args):
    """Compute full omega for a chunk of numbers.

    SPF uses 0 = prime sentinel (spf[n] == 0 means n is prime, SPF = n).
    """
    numbers, spf = args
    results = np.zeros(len(numbers), dtype=np.int32)

    for i, n in enumerate(numbers):
        if n <= 1:
            continue
        count = 0
        prev = 0
        while n > 1:
            p = spf[n]
            if p == 0:  # n is prime
                p = n
            if p != prev:
                count += 1
            prev = p
            n //= p
        results[i] = count

    return results


def omega_leq_P_parallel(numbers: np.ndarray, spf: np.ndarray, P: int,
                         num_workers: int = None, chunk_size: int = 100000) -> np.ndarray:
    """
    Compute omega_leq_P for array of numbers using parallel CPU.

    Uses shared memory for SPF array to avoid pickling large arrays.
    """
    if num_workers is None:
        num_workers = cpu_count()

    # For large SPF arrays (>1GB), use shared memory
    use_shm = spf.nbytes > 10**9

    if use_shm:
        # Create shared memory block
        shm = shared_memory.SharedMemory(create=True, size=spf.nbytes)
        try:
            # Copy SPF to shared memory
            shm_spf = np.ndarray(spf.shape, dtype=spf.dtype, buffer=shm.buf)
            shm_spf[:] = spf[:]

            # Create chunks (only numbers and P, not SPF)
            n = len(numbers)
            chunks = []
            for i in range(0, n, chunk_size):
                end = min(i + chunk_size, n)
                chunks.append((numbers[i:end].copy(), P))

            # Process with shared memory initializer
            with Pool(num_workers, initializer=_init_worker_shm,
                      initargs=(shm.name, spf.shape, spf.dtype)) as pool:
                results = pool.map(_omega_leq_P_chunk_shm, chunks)

            return np.concatenate(results)
        finally:
            shm.close()
            shm.unlink()
    else:
        # Small SPF: use original approach
        n = len(numbers)
        chunks = []
        for i in range(0, n, chunk_size):
            end = min(i + chunk_size, n)
            chunks.append((numbers[i:end], spf, P))

        with Pool(num_workers) as pool:
            results = pool.map(_omega_leq_P_chunk, chunks)

        return np.concatenate(results)


def omega_parallel(numbers: np.ndarray, spf: np.ndarray,
                   num_workers: int = None, chunk_size: int = 100000) -> np.ndarray:
    """
    Compute full omega for array of numbers using parallel CPU.

    Uses shared memory for SPF array to avoid pickling large arrays.
    """
    if num_workers is None:
        num_workers = cpu_count()

    # For large SPF arrays (>1GB), use shared memory
    use_shm = spf.nbytes > 10**9

    if use_shm:
        # Create shared memory block
        shm = shared_memory.SharedMemory(create=True, size=spf.nbytes)
        try:
            # Copy SPF to shared memory
            shm_spf = np.ndarray(spf.shape, dtype=spf.dtype, buffer=shm.buf)
            shm_spf[:] = spf[:]

            # Create chunks (only numbers, not SPF)
            n = len(numbers)
            chunks = []
            for i in range(0, n, chunk_size):
                end = min(i + chunk_size, n)
                chunks.append(numbers[i:end].copy())

            # Process with shared memory initializer
            with Pool(num_workers, initializer=_init_worker_shm,
                      initargs=(shm.name, spf.shape, spf.dtype)) as pool:
                results = pool.map(_omega_chunk_shm, chunks)

            return np.concatenate(results)
        finally:
            shm.close()
            shm.unlink()
    else:
        # Small SPF: use original approach
        n = len(numbers)
        chunks = []
        for i in range(0, n, chunk_size):
            end = min(i + chunk_size, n)
            chunks.append((numbers[i:end], spf))

        with Pool(num_workers) as pool:
            results = pool.map(_omega_chunk, chunks)

        return np.concatenate(results)


if __name__ == '__main__':
    import time

    # Benchmark
    for N in [10**7, 10**8, 6 * 10**8]:
        print(f"\nN = {N:,}")

        # Parallel
        t0 = time.time()
        spf_par = spf_sieve_parallel(N)
        t_par = time.time() - t0
        print(f"  Parallel: {t_par:.1f}s")

        # Sequential (for comparison, only for small N)
        if N <= 10**8:
            from factorization import spf_sieve
            t0 = time.time()
            spf_seq = spf_sieve(N)
            t_seq = time.time() - t0
            print(f"  Sequential: {t_seq:.1f}s")
            print(f"  Speedup: {t_seq/t_par:.1f}x")

            # Verify
            assert np.array_equal(spf_par, spf_seq), "Results don't match!"
            print("  ✓ Verified")
