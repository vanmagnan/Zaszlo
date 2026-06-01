# Tests for FlagProblem validation and the complete() constructor.

from __future__ import annotations

import pytest

from zaszlo import FlagProblem, Hypergraph, complete


# ---------------------------------------------------------------------------
# complete()
# ---------------------------------------------------------------------------

class TestComplete:
    def test_k2_default(self):
        g = complete(3)
        assert g.n == 3
        assert g.k == 2
        assert len(g.edges) == 3  # C(3,2)

    def test_k2_explicit(self):
        g = complete(4, 2)
        assert g.n == 4
        assert g.k == 2
        assert len(g.edges) == 6  # C(4,2)

    def test_k3(self):
        g = complete(4, 3)
        assert g.n == 4
        assert g.k == 3
        assert len(g.edges) == 4  # C(4,3)

    def test_k4(self):
        g = complete(5, 4)
        assert len(g.edges) == 5  # C(5,4)

    def test_returns_hypergraph(self):
        assert isinstance(complete(3), Hypergraph)

    def test_all_edges_present(self):
        g = complete(4, 2)
        expected = {(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)}
        assert set(g.edges) == expected

    def test_complete_3_is_k3(self):
        k3_explicit = Hypergraph(3, 2, [(1, 2), (1, 3), (2, 3)])
        assert complete(3) == k3_explicit

    def test_complete_4_3_matches_k4_minus_parent(self):
        # complete(4, 3) has all 4 edges; k4_minus() has 3 of them
        from zaszlo import k4_minus
        full = complete(4, 3)
        minus = k4_minus()
        assert len(full.edges) == 4
        assert len(minus.edges) == 3
        assert all(e in full.edges for e in minus.edges)


# ---------------------------------------------------------------------------
# FlagProblem validation
# ---------------------------------------------------------------------------

class TestFlagProblemValidation:
    # --- parity ---

    def test_parity_mismatch_even_n_odd_order(self):
        with pytest.raises(ValueError, match="parity"):
            FlagProblem(4, 1, 2)  # n=4 even, type_order=1 odd

    def test_parity_mismatch_odd_n_even_order(self):
        with pytest.raises(ValueError, match="parity"):
            FlagProblem(5, 2, 2)  # n=5 odd, type_order=2 even

    def test_parity_ok_even_n_even_order(self):
        FlagProblem(4, 2, 2)  # should not raise

    def test_parity_ok_odd_n_odd_order(self):
        FlagProblem(5, 3, 2)  # should not raise

    def test_parity_ok_zero_order_even_n(self):
        FlagProblem(4, 0, 2)  # type_order=0 is valid for even n

    # --- type_order too large ---

    def test_order_equals_n_minus_1_even_n(self):
        # n=4, type_order=3: different parity → parity error fires first
        with pytest.raises(ValueError, match="parity"):
            FlagProblem(4, 3, 2)

    def test_order_equals_n_even_n(self):
        # n=4, type_order=4: same parity but too large
        with pytest.raises(ValueError, match="n-2"):
            FlagProblem(4, 4, 2)

    def test_order_equals_n_minus_2_is_valid(self):
        FlagProblem(4, 2, 2)  # n-2 = 2, exactly at the limit — valid

    def test_order_exceeds_n_minus_2(self):
        # n=6, type_order=6: same parity (both even) but too large
        with pytest.raises(ValueError, match="n-2"):
            FlagProblem(6, 6, 2)

    # --- error message quality ---

    def test_parity_error_lists_valid_values(self):
        with pytest.raises(ValueError, match="Valid type_order values"):
            FlagProblem(4, 1, 2)

    def test_size_error_names_n_minus_2(self):
        with pytest.raises(ValueError, match="n-2="):
            FlagProblem(6, 6, 2)
