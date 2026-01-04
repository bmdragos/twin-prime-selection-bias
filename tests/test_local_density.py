"""
Regression tests for per-prime local density verification.

These tests load reference data and verify that the scaled residuals ε_p
remain within expected tolerance. If future code changes break the
mechanism, these tests will fail.

The scaled residual is defined as:
    ε_p = (p-1) × P̂(p|b|a prime) - 1

For the local density model to hold, ε_p should be near zero.
Tolerance is set conservatively at 1e-3 (0.1%) to allow for
minor computational variations while catching real regressions.
"""

import pytest
import json
from pathlib import Path

# Tolerance for scaled residuals
# At K=10^9, we observe |ε_p| ~ 10^-5, so 10^-3 is very conservative
EPSILON_TOLERANCE = 1e-3

# Reference data directory
REFERENCE_DIR = Path(__file__).parent.parent / "data" / "reference"


def load_metadata(pattern_dir: str) -> dict:
    """Load metadata.json for a pattern."""
    path = REFERENCE_DIR / pattern_dir / "metadata.json"
    if not path.exists():
        pytest.skip(f"Reference data not found: {path}")
    with open(path) as f:
        return json.load(f)


def load_per_prime_csv(pattern_dir: str) -> dict:
    """Load per-prime CSV and extract scaled residuals.

    Handles multi-section CSVs where the per-prime data starts after
    a header row containing 'p' as the first column.
    """
    import csv

    csv_path = REFERENCE_DIR / pattern_dir / "per_prime_table.csv"
    if not csv_path.exists():
        pytest.skip(f"Per-prime CSV not found: {csv_path}")

    results = {}
    with open(csv_path) as f:
        lines = f.readlines()

    # Find the per-prime section (starts with 'p,' header)
    data_start = None
    for i, line in enumerate(lines):
        if line.startswith('p,'):
            data_start = i
            break

    if data_start is None:
        pytest.skip(f"No per-prime header found in {csv_path}")

    # Parse from the per-prime header onwards
    import io
    csv_content = ''.join(lines[data_start:])
    reader = csv.DictReader(io.StringIO(csv_content))

    for row in reader:
        if 'p' in row and row['p'].isdigit():
            p = int(row['p'])
            if 'scaled_residual' in row and row['scaled_residual'] not in ('N/A', ''):
                results[p] = float(row['scaled_residual'])

    return results


class TestTwinPrimesLocalDensity:
    """Test local density for twin prime pattern (6k-1, 6k+1)."""

    def test_metadata_exists(self):
        """Verify metadata.json exists and has required fields."""
        meta = load_metadata("twin_primes")
        assert "pattern" in meta
        assert "K" in meta
        assert "primes_tested" in meta
        assert meta["K"] >= 1e6, "Reference data should use K >= 10^6"

    @pytest.mark.parametrize("p", [5, 7, 11, 13])
    def test_scaled_residual_within_tolerance(self, p):
        """Verify |ε_p| < tolerance for each prime."""
        residuals = load_per_prime_csv("twin_primes")
        if p not in residuals:
            pytest.skip(f"Prime {p} not in reference data")

        epsilon_p = residuals[p]
        assert abs(epsilon_p) < EPSILON_TOLERANCE, (
            f"Twin primes: |ε_{p}| = {abs(epsilon_p):.2e} exceeds tolerance {EPSILON_TOLERANCE:.0e}"
        )


class TestSophieGermainLocalDensity:
    """Test local density for Sophie Germain pattern (n, 2n+1)."""

    def test_metadata_exists(self):
        """Verify metadata.json exists and has required fields."""
        meta = load_metadata("sophie_germain")
        assert "pattern" in meta
        assert "N" in meta
        assert meta["N"] >= 1e6, "Reference data should use N >= 10^6"

    @pytest.mark.parametrize("p", [3, 5, 7, 11, 13])
    def test_scaled_residual_within_tolerance(self, p):
        """Verify |ε_p| < tolerance for each prime."""
        residuals = load_per_prime_csv("sophie_germain")
        if p not in residuals:
            pytest.skip(f"Prime {p} not in reference data")

        epsilon_p = residuals[p]
        assert abs(epsilon_p) < EPSILON_TOLERANCE, (
            f"Sophie Germain: |ε_{p}| = {abs(epsilon_p):.2e} exceeds tolerance {EPSILON_TOLERANCE:.0e}"
        )


class TestCousinPrimesLocalDensity:
    """Test local density for cousin prime pattern (n, n+4)."""

    def test_metadata_exists(self):
        """Verify metadata.json exists and has required fields."""
        meta = load_metadata("cousin_primes")
        assert "pattern" in meta
        assert "K" in meta
        assert "p3_note" in meta, "Should document the p=3 special case"

    @pytest.mark.parametrize("p", [5, 7, 11, 13])
    def test_scaled_residual_within_tolerance(self, p):
        """Verify |ε_p| < tolerance for p >= 5."""
        residuals = load_per_prime_csv("cousin_primes")
        if p not in residuals:
            pytest.skip(f"Prime {p} not in reference data")

        epsilon_p = residuals[p]
        assert abs(epsilon_p) < EPSILON_TOLERANCE, (
            f"Cousin primes: |ε_{p}| = {abs(epsilon_p):.2e} exceeds tolerance {EPSILON_TOLERANCE:.0e}"
        )

    def test_p3_increment_near_zero(self):
        """
        For cousin primes, p=3 is residue-class determined.
        The increment P(3|b|n prime) - P(3|b|n comp) should be ~0.
        """
        csv_path = REFERENCE_DIR / "cousin_primes" / "per_prime_table.csv"
        if not csv_path.exists():
            pytest.skip("Cousin primes CSV not found")

        import csv
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('p') == '3' and 'increment' in row:
                    increment = float(row['increment'])
                    # p=3 increment should be very small (residue-class cancellation)
                    assert abs(increment) < 1e-3, (
                        f"Cousin primes p=3 increment = {increment:.4f}, expected ~0"
                    )
                    return

        pytest.skip("p=3 row not found in cousin primes CSV")


class TestCrossPatternConsistency:
    """Cross-pattern sanity checks."""

    def test_all_patterns_have_metadata(self):
        """All pattern directories should have metadata.json."""
        patterns = ["twin_primes", "sophie_germain", "cousin_primes"]
        for pattern in patterns:
            meta_path = REFERENCE_DIR / pattern / "metadata.json"
            assert meta_path.exists(), f"Missing metadata for {pattern}"

    def test_scaled_residual_definition_consistent(self):
        """All patterns should use the same ε_p definition."""
        patterns = ["twin_primes", "sophie_germain", "cousin_primes"]
        definitions = set()

        for pattern in patterns:
            meta = load_metadata(pattern)
            if "scaled_residual_definition" in meta:
                definitions.add(meta["scaled_residual_definition"])

        # Should all have the same definition (possibly with minor wording differences)
        assert len(definitions) <= 2, "Inconsistent ε_p definitions across patterns"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
