# Validation against C flagmatic reference output.
#
# Strategy: parse the C binary's flags.py and flags.rat for two known problems,
# then check that our generation counts and pair densities agree exactly.
# Any discrepancy in pair densities would produce wrong SDP matrices.
#
# Two reference cases:
#   k2n4 — K=2, n=4 (ordinary graphs, 4 vertices)
#   k3n5 — K=3, n=5 (3-uniform hypergraphs, 5 vertices)

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from zaszlo.densities import compute_pair_densities
from zaszlo.generation import generate_admissible, generate_flags, generate_types
from zaszlo.types import Hypergraph

from .reference import parse_flags_py, parse_flags_rat

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# K=2, n=4 reference
# ---------------------------------------------------------------------------

class TestK2N4:
    @pytest.fixture(scope="class")
    def ref(self):
        return parse_flags_py(FIXTURES / "k2n4" / "flags.py", k=2)

    @pytest.fixture(scope="class")
    def rat(self):
        return parse_flags_rat(FIXTURES / "k2n4" / "flags.rat")

    def test_admissible_count(self, ref):
        our = generate_admissible(ref["n"], 2)
        assert len(our) == len(ref["H"])  # 11

    def test_type_count(self, ref):
        # Valid type orders for K=2, n=4: 0 and 2 (same parity as n, ≤ n-2)
        our_types = generate_types(0, 2) + generate_types(2, 2)
        assert len(our_types) == ref["num_types"]  # 3

    def test_flag_counts_per_type(self, ref):
        # Use C's own types as input so counts are directly comparable.
        for sigma, t in enumerate(ref["types"]):
            s = t.type_size
            n = ref["n"]
            m = (n + s) // 2 if (n - s) % 2 == 0 else (n + s - 1) // 2
            our_flags = generate_flags(m, t, [], [])
            assert len(our_flags) == ref["num_flags"][sigma], (
                f"Type sigma={sigma}: expected {ref['num_flags'][sigma]} flags, "
                f"got {len(our_flags)}"
            )

    def test_pair_densities_match_rat(self, ref, rat):
        """Pair densities must match C's flags.rat exactly (rational arithmetic)."""
        for Hi, H in enumerate(ref["H"], start=1):  # H_idx is 1-based in flags.rat
            pair_dens = compute_pair_densities(H, ref["types"], ref["flags"])

            for sigma in range(ref["num_types"]):
                nf = ref["num_flags"][sigma]
                for i in range(1, nf + 1):
                    for j in range(i, nf + 1):
                        our_val = pair_dens[sigma][i - 1][j - 1]
                        ref_val = rat.get((Hi, sigma, i, j), Fraction(0))
                        assert our_val == ref_val, (
                            f"H={Hi}, sigma={sigma}, i={i}, j={j}: "
                            f"got {our_val}, expected {ref_val}"
                        )


# ---------------------------------------------------------------------------
# K=3, n=5 reference
# ---------------------------------------------------------------------------

class TestK3N5:
    @pytest.fixture(scope="class")
    def ref(self):
        return parse_flags_py(FIXTURES / "k3n5" / "flags.py", k=3)

    @pytest.fixture(scope="class")
    def rat(self):
        return parse_flags_rat(FIXTURES / "k3n5" / "flags.rat")

    def test_admissible_count(self, ref):
        our = generate_admissible(ref["n"], 3)
        assert len(our) == len(ref["H"])  # 34

    def test_type_count(self, ref):
        # Valid type orders for K=3, n=5: 1 and 3 (same parity as n=5, ≤ n-2=3)
        our_types = generate_types(1, 3) + generate_types(3, 3)
        assert len(our_types) == ref["num_types"]  # 3

    def test_flag_counts_per_type(self, ref):
        for sigma, t in enumerate(ref["types"]):
            s = t.type_size
            n = ref["n"]
            m = (n + s) // 2 if (n - s) % 2 == 0 else (n + s - 1) // 2
            our_flags = generate_flags(m, t, [], [])
            assert len(our_flags) == ref["num_flags"][sigma], (
                f"Type sigma={sigma}: expected {ref['num_flags'][sigma]} flags, "
                f"got {len(our_flags)}"
            )

    def test_pair_densities_match_rat(self, ref, rat):
        """Pair densities must match C's flags.rat exactly (rational arithmetic)."""
        for Hi, H in enumerate(ref["H"], start=1):
            pair_dens = compute_pair_densities(H, ref["types"], ref["flags"])

            for sigma in range(ref["num_types"]):
                nf = ref["num_flags"][sigma]
                for i in range(1, nf + 1):
                    for j in range(i, nf + 1):
                        our_val = pair_dens[sigma][i - 1][j - 1]
                        ref_val = rat.get((Hi, sigma, i, j), Fraction(0))
                        assert our_val == ref_val, (
                            f"H={Hi}, sigma={sigma}, i={i}, j={j}: "
                            f"got {our_val}, expected {ref_val}"
                        )
