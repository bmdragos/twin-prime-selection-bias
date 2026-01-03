"""
Wheel sieve for 6k±1 numbers only.

Instead of storing SPF for all integers (6K elements for K pairs),
only store SPF for numbers ≡ 1 or 5 (mod 6). This is 1/3 the memory.

Memory savings:
- K=10^9: 24GB → 8GB
- K=10^10: 240GB → 80GB

Index mapping:
- Index 2i   → 6(i+1) - 1 = 6i + 5  (≡ 5 mod 6)
- Index 2i+1 → 6(i+1) + 1 = 6i + 7  (≡ 1 mod 6)

Reverse (n → index):
- n ≡ 5 (mod 6): index = (n - 5) // 3 * 2     = 2 * ((n-5)//6)
- n ≡ 1 (mod 6): index = (n - 7) // 3 * 2 + 1 = 2 * ((n-7)//6) + 1

For n=5: (5-5)//6 = 0, index = 0 ✓
For n=7: (7-7)//6 = 0, index = 1 ✓
For n=11: (11-5)//6 = 1, index = 2 ✓
For n=13: (13-7)//6 = 1, index = 3 ✓
"""

import numpy as np
from typing import Tuple


def index_to_n(i: int) -> int:
    """Convert wheel index to actual number."""
    # i=0 → 5, i=1 → 7, i=2 → 11, i=3 → 13, ...
    if i % 2 == 0:
        return 3 * i + 5  # 6k-1 numbers
    else:
        return 3 * i + 4  # 6k+1 numbers


def n_to_index(n: int) -> int:
    """Convert number (must be ≡ 1 or 5 mod 6) to wheel index."""
    r = n % 6
    if r == 5:
        return (n - 5) // 3
    elif r == 1:
        return (n - 4) // 3
    else:
        raise ValueError(f"{n} is not ≡ 1 or 5 (mod 6)")


def wheel_spf_sieve(K: int) -> np.ndarray:
    """
    Compute SPF for all 6k±1 numbers up to 6K+1.

    Parameters
    ----------
    K : int
        Number of pairs. Will sieve up to 6K+1.

    Returns
    -------
    np.ndarray
        Wheel-indexed SPF array of size 2K.
        spf_wheel[i] = smallest prime factor of index_to_n(i)
        Value 0 means the number is prime.
    """
    N = 6 * K + 1
    size = 2 * K  # Only 2K entries needed

    # Initialize to 0 (prime sentinel)
    spf = np.zeros(size, dtype=np.uint32)

    # Sieve with primes starting from 5
    # (2 and 3 don't divide 6k±1 numbers)
    sqrt_N = int(N ** 0.5) + 1

    # Generate small primes for sieving (only need 5, 7, 11, 13, ...)
    # These are exactly the 6k±1 numbers up to sqrt(N)
    sqrt_K = (sqrt_N + 5) // 6 + 1

    for idx in range(min(size, 2 * sqrt_K)):
        p = index_to_n(idx)
        if p > sqrt_N:
            break
        if spf[idx] != 0:  # p is not prime
            continue

        # p is prime - mark its multiples
        # We need to mark p * m where m ≡ 1 or 5 (mod 6) and p*m >= p^2

        # Starting multiplier: smallest m ≡ 1 or 5 (mod 6) such that p*m >= p^2
        # That's m >= p, and m must be ≡ 1 or 5 (mod 6)

        # p itself is ≡ 1 or 5 (mod 6)
        # p*p is ≡ 1 (mod 6) if p ≡ 1 or 5 (since 1*1=1, 5*5=25≡1)
        # So p^2 ≡ 1 (mod 6) always for p >= 5

        # For p ≡ 1 (mod 6):
        #   p * 1 = p ≡ 1 (mod 6) ✓
        #   p * 5 ≡ 5 (mod 6) ✓
        #   p * 7 ≡ 1 (mod 6) ✓
        #   ...
        # For p ≡ 5 (mod 6):
        #   p * 1 = p ≡ 5 (mod 6) ✓
        #   p * 5 ≡ 25 ≡ 1 (mod 6) ✓
        #   p * 7 ≡ 35 ≡ 5 (mod 6) ✓
        #   ...

        # Multipliers that give 6k±1: 1, 5, 7, 11, 13, 17, 19, ...
        # These are exactly the 6k±1 numbers themselves!

        # Start from p (which gives p^2 when multiplied by p... wait no)
        # We want p * m >= p^2, so m >= p
        # Smallest m >= p that is ≡ 1 or 5 (mod 6):

        # Find first multiplier m >= p where m ≡ 1 or 5 (mod 6)
        m_start = p
        r = m_start % 6
        if r == 0:
            m_start += 1  # → 1 mod 6
        elif r == 2:
            m_start += 3  # → 5 mod 6
        elif r == 3:
            m_start += 2  # → 5 mod 6
        elif r == 4:
            m_start += 1  # → 5 mod 6
        # r == 1 or 5 already good

        # Now iterate through multipliers: m, then next 6k±1 after m
        m = m_start
        while p * m <= N:
            prod = p * m
            prod_idx = n_to_index(prod)
            if prod_idx < size and spf[prod_idx] == 0:
                spf[prod_idx] = p

            # Next multiplier: if m ≡ 5 (mod 6), next is m+2 (≡ 1)
            #                  if m ≡ 1 (mod 6), next is m+4 (≡ 5)
            if m % 6 == 5:
                m += 2
            else:
                m += 4

    return spf


def wheel_spf_lookup(n: int, spf_wheel: np.ndarray) -> int:
    """
    Get smallest prime factor of n using wheel-indexed SPF array.

    Handles all cases:
    - n divisible by 2 → returns 2
    - n divisible by 3 → returns 3
    - n ≡ 1 or 5 (mod 6) → looks up in wheel array
    """
    if n <= 1:
        return n
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3

    # n is ≡ 1 or 5 (mod 6)
    idx = n_to_index(n)
    p = spf_wheel[idx]
    if p == 0:  # n is prime
        return n
    return p


def omega_wheel(n: int, spf_wheel: np.ndarray) -> int:
    """
    Count distinct prime factors using wheel-indexed SPF.
    """
    if n <= 1:
        return 0

    count = 0
    prev = 0

    while n > 1:
        p = wheel_spf_lookup(n, spf_wheel)
        if p != prev:
            count += 1
            prev = p
        n //= p

    return count


def omega_leq_P_wheel(n: int, spf_wheel: np.ndarray, P: int) -> int:
    """
    Count distinct prime factors <= P using wheel-indexed SPF.
    """
    if n <= 1:
        return 0

    count = 0
    prev = 0

    while n > 1:
        p = wheel_spf_lookup(n, spf_wheel)
        if p != prev and p <= P:
            count += 1
        if p != prev:
            prev = p
        n //= p

    return count


# Vectorized versions for arrays
def omega_wheel_array(numbers: np.ndarray, spf_wheel: np.ndarray) -> np.ndarray:
    """Compute omega for array of numbers using wheel SPF."""
    results = np.zeros(len(numbers), dtype=np.int32)
    for i, n in enumerate(numbers):
        results[i] = omega_wheel(n, spf_wheel)
    return results


def omega_leq_P_wheel_array(numbers: np.ndarray, spf_wheel: np.ndarray, P: int) -> np.ndarray:
    """Compute omega_leq_P for array of numbers using wheel SPF."""
    results = np.zeros(len(numbers), dtype=np.int32)
    for i, n in enumerate(numbers):
        results[i] = omega_leq_P_wheel(n, spf_wheel, P)
    return results


if __name__ == '__main__':
    # Quick sanity check
    print("Testing wheel sieve...")

    # Test index mapping
    for i in range(10):
        n = index_to_n(i)
        i_back = n_to_index(n)
        print(f"  i={i} → n={n} → i={i_back}")
        assert i == i_back

    # Test small sieve
    K = 100
    spf = wheel_spf_sieve(K)
    print(f"\nWheel SPF for K={K} (size={len(spf)}):")

    # Check some known primes and composites
    test_cases = [
        (5, 5),   # prime
        (7, 7),   # prime
        (11, 11), # prime
        (25, 5),  # 5^2
        (35, 5),  # 5*7
        (49, 7),  # 7^2
        (77, 7),  # 7*11
        (121, 11), # 11^2
    ]

    print("\nSPF lookups:")
    for n, expected_spf in test_cases:
        got = wheel_spf_lookup(n, spf)
        status = "✓" if got == expected_spf else f"✗ (expected {expected_spf})"
        print(f"  SPF({n}) = {got} {status}")

    # Test omega
    print("\nOmega tests:")
    omega_cases = [
        (30, 3),   # 2*3*5
        (60, 3),   # 2^2*3*5
        (210, 4),  # 2*3*5*7
        (35, 2),   # 5*7
    ]
    for n, expected in omega_cases:
        got = omega_wheel(n, spf)
        status = "✓" if got == expected else f"✗ (expected {expected})"
        print(f"  ω({n}) = {got} {status}")

    print("\nMemory comparison:")
    for test_K in [10**6, 10**9]:
        full_size = (6 * test_K + 1) * 4 / 1e9
        wheel_size = (2 * test_K) * 4 / 1e9
        print(f"  K={test_K:.0e}: full={full_size:.1f}GB, wheel={wheel_size:.1f}GB, savings={full_size/wheel_size:.1f}x")
