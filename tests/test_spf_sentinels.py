"""
Tests for SPF (smallest prime factor) sentinel conventions.

These tests prevent regression of the sentinel mismatch bugs found by Codex CLI:
- spf_sieve uses spf[p] = p for primes
- spf_sieve_parallel uses spf[p] = 0 for primes
- Functions must use the correct convention for their SPF source
"""

import numpy as np
import pytest

from src.factorization import spf_sieve, spf_sieve_with_flags, omega, omega_leq_P
from src.primes import prime_flags_upto
from src.parallel_sieve import spf_sieve_parallel, prime_flags_parallel


# Known small primes for testing
SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
SMALL_COMPOSITES = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25]


class TestSpfSieveSentinel:
    """Test that spf_sieve uses spf[p] = p for primes."""

    def test_primes_have_spf_equal_to_self(self):
        """For spf_sieve, primes should have spf[p] = p."""
        N = 100
        spf = spf_sieve(N)

        for p in SMALL_PRIMES:
            if p <= N:
                assert spf[p] == p, f"spf_sieve: spf[{p}] should be {p}, got {spf[p]}"

    def test_composites_have_spf_less_than_self(self):
        """For composites, spf[n] < n."""
        N = 100
        spf = spf_sieve(N)

        for n in SMALL_COMPOSITES:
            if n <= N:
                assert spf[n] < n, f"spf_sieve: spf[{n}] should be < {n}, got {spf[n]}"
                assert spf[n] > 0, f"spf_sieve: spf[{n}] should be > 0, got {spf[n]}"

    def test_spf_zero_and_one(self):
        """spf[0] = 0 and spf[1] = 1."""
        spf = spf_sieve(10)
        assert spf[0] == 0
        assert spf[1] == 1


class TestSpfSieveParallelSentinel:
    """Test that spf_sieve_parallel uses spf[p] = 0 for primes."""

    def test_primes_have_spf_zero(self):
        """For spf_sieve_parallel, primes should have spf[p] = 0."""
        N = 50
        spf = spf_sieve_parallel(N, num_workers=1)

        for p in SMALL_PRIMES:
            if p <= N:
                assert spf[p] == 0, f"spf_sieve_parallel: spf[{p}] should be 0, got {spf[p]}"

    def test_composites_have_nonzero_spf(self):
        """For composites, spf[n] > 0 and spf[n] < n."""
        N = 50
        spf = spf_sieve_parallel(N, num_workers=1)

        for n in SMALL_COMPOSITES:
            if n <= N:
                assert spf[n] > 0, f"spf_sieve_parallel: spf[{n}] should be > 0, got {spf[n]}"
                assert spf[n] < n, f"spf_sieve_parallel: spf[{n}] should be < {n}, got {spf[n]}"

    def test_spf_zero_and_one_marked(self):
        """spf[0] = 1 and spf[1] = 1 (non-prime markers)."""
        spf = spf_sieve_parallel(10, num_workers=1)
        assert spf[0] == 1, f"spf[0] should be 1 (non-prime), got {spf[0]}"
        assert spf[1] == 1, f"spf[1] should be 1 (non-prime), got {spf[1]}"


class TestPrimeFlagsConsistency:
    """Test that prime flag functions are consistent with their SPF sources."""

    def test_prime_flags_upto_matches_known_primes(self):
        """prime_flags_upto should correctly identify primes."""
        N = 100
        flags = prime_flags_upto(N)

        for p in SMALL_PRIMES:
            if p <= N:
                assert flags[p], f"prime_flags_upto: {p} should be prime"

        for n in SMALL_COMPOSITES:
            if n <= N:
                assert not flags[n], f"prime_flags_upto: {n} should not be prime"

        assert not flags[0]
        assert not flags[1]

    def test_prime_flags_parallel_matches_known_primes(self):
        """prime_flags_parallel should correctly identify primes using 0 sentinel."""
        N = 50
        flags = prime_flags_parallel(N, num_workers=1)

        for p in SMALL_PRIMES:
            if p <= N:
                assert flags[p], f"prime_flags_parallel: {p} should be prime"

        for n in SMALL_COMPOSITES:
            if n <= N:
                assert not flags[n], f"prime_flags_parallel: {n} should not be prime"

        assert not flags[0]
        assert not flags[1]

    def test_prime_flags_consistency(self):
        """Both prime flag methods should give identical results."""
        N = 200
        flags_upto = prime_flags_upto(N)
        flags_parallel = prime_flags_parallel(N, num_workers=1)

        assert np.array_equal(flags_upto, flags_parallel), \
            "prime_flags_upto and prime_flags_parallel should give identical results"


class TestOmegaWithCorrectSpf:
    """Test omega functions work correctly with spf_sieve (spf[p]=p convention)."""

    def test_omega_of_primes(self):
        """omega(p) = 1 for primes."""
        spf = spf_sieve(100)

        for p in SMALL_PRIMES:
            if p <= 100:
                assert omega(p, spf) == 1, f"omega({p}) should be 1"

    def test_omega_of_prime_powers(self):
        """omega(p^k) = 1 for prime powers."""
        spf = spf_sieve(1000)

        prime_powers = [(4, 1), (8, 1), (9, 1), (25, 1), (27, 1), (32, 1), (49, 1), (125, 1)]
        for n, expected in prime_powers:
            assert omega(n, spf) == expected, f"omega({n}) should be {expected}"

    def test_omega_of_products(self):
        """omega(p*q) = 2 for distinct primes."""
        spf = spf_sieve(1000)

        products = [(6, 2), (10, 2), (14, 2), (15, 2), (21, 2), (35, 2)]
        for n, expected in products:
            assert omega(n, spf) == expected, f"omega({n}) should be {expected}"

    def test_omega_leq_P(self):
        """omega_leq_P counts only factors <= P."""
        spf = spf_sieve(1000)

        # 30 = 2 * 3 * 5
        assert omega_leq_P(30, spf, 2) == 1  # only 2
        assert omega_leq_P(30, spf, 3) == 2  # 2 and 3
        assert omega_leq_P(30, spf, 5) == 3  # 2, 3, and 5
        assert omega_leq_P(30, spf, 100) == 3  # all factors


class TestSentinelMismatchPrevention:
    """
    Regression tests for the specific bugs found by Codex CLI.

    These tests would have caught the original bugs:
    - run_stability_analysis.py used (spf == 0) with spf_sieve
    - prime_flags_parallel used (spf == n) with spf_sieve_parallel
    """

    def test_spf_sieve_zero_check_gives_no_primes(self):
        """
        BUG REGRESSION: Using (spf == 0) with spf_sieve gives no primes.
        This was the bug in run_stability_analysis.py:76.
        """
        spf = spf_sieve(100)

        # The WRONG way (what the bug did):
        wrong_flags = (spf == 0)
        wrong_prime_count = np.sum(wrong_flags[2:])

        # The RIGHT way:
        right_flags = (spf == np.arange(len(spf)))
        right_prime_count = np.sum(right_flags[2:])

        assert wrong_prime_count == 0, "BUG: (spf == 0) with spf_sieve should find 0 primes"
        assert right_prime_count == 25, "There should be 25 primes <= 100"

    def test_spf_parallel_self_check_gives_no_primes(self):
        """
        BUG REGRESSION: Using (spf == n) with spf_sieve_parallel gives no primes.
        This was the bug in prime_flags_parallel before fix.
        """
        N = 50
        spf = spf_sieve_parallel(N, num_workers=1)

        # The WRONG way (what the bug did):
        wrong_flags = np.zeros(N + 1, dtype=bool)
        wrong_flags[2:] = (spf[2:] == np.arange(2, N + 1))
        wrong_prime_count = np.sum(wrong_flags)

        # The RIGHT way:
        right_flags = np.zeros(N + 1, dtype=bool)
        right_flags[2:] = (spf[2:] == 0)
        right_prime_count = np.sum(right_flags)

        assert wrong_prime_count == 0, "BUG: (spf == n) with spf_sieve_parallel should find 0 primes"
        assert right_prime_count == 15, "There should be 15 primes <= 50"

    def test_omega_with_parallel_spf_gives_wrong_result(self):
        """
        BUG REGRESSION: omega() with parallel SPF (0 sentinel) gives wrong results.

        The omega function does: n //= p where p = spf[n].
        If spf[n] = 0 for primes, n //= 0 produces 0, causing the loop to
        exit early with wrong count.
        """
        import warnings
        spf_parallel = spf_sieve_parallel(50, num_workers=1)  # Small N, single worker

        # Using omega with parallel SPF gives wrong result
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore NumPy's divide-by-zero warning
            result = omega(5, spf_parallel)  # 5 is prime, spf[5] = 0

        # Result is wrong (should be 1, but we get 0 due to n //= 0 → 0)
        assert result != 1, "omega(5) with parallel SPF should give wrong result"


class TestSpfSieveWithFlags:
    """Test the combined SPF + flags helper."""

    def test_returns_correct_spf(self):
        """SPF from spf_sieve_with_flags matches spf_sieve."""
        N = 1000
        spf_combined, _ = spf_sieve_with_flags(N)
        spf_separate = spf_sieve(N)

        assert np.array_equal(spf_combined, spf_separate)

    def test_returns_correct_flags(self):
        """Flags from spf_sieve_with_flags match prime_flags_upto."""
        N = 1000
        _, flags_combined = spf_sieve_with_flags(N)
        flags_separate = prime_flags_upto(N)

        assert np.array_equal(flags_combined, flags_separate)

    def test_single_pass_efficiency(self):
        """Verify we get both outputs from one call."""
        N = 100
        spf, flags = spf_sieve_with_flags(N)

        # Check types
        assert spf.dtype == np.int64
        assert flags.dtype == bool

        # Check consistency: primes should have spf[p] == p and flags[p] == True
        for p in SMALL_PRIMES:
            if p <= N:
                assert spf[p] == p
                assert flags[p]


class TestCrossValidation:
    """Cross-validate that both sieves produce consistent factorizations."""

    def test_factorizations_match(self):
        """
        Both SPF arrays should produce the same complete factorization,
        just with different prime detection.
        """
        N = 200
        spf_seq = spf_sieve(N)
        spf_par = spf_sieve_parallel(N, num_workers=1)

        # For composites, both should have the same SPF
        for n in range(4, N + 1):
            if spf_seq[n] != n:  # n is composite in sequential
                assert spf_seq[n] == spf_par[n], \
                    f"SPF mismatch at composite {n}: seq={spf_seq[n]}, par={spf_par[n]}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
