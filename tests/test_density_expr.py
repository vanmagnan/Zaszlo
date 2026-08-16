# Tests for DensityExpr: construction, canonicalization, evaluation, rendering,
# and integration with FlagProblem.

from __future__ import annotations

from fractions import Fraction

import pytest

from zaszlo import DensityExpr, FlagProblem, Hypergraph, complete
from zaszlo.densities import induced_density


K3 = complete(3)
K4 = complete(4)
K2 = complete(2)


# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_single_term_ok(self):
        e = DensityExpr([(1, K3)])
        assert len(e.terms) == 1
        assert e.terms[0] == (Fraction(1), K3)
        assert e.k == 2

    def test_int_coerced_to_fraction(self):
        e = DensityExpr([(2, K3)])
        assert isinstance(e.terms[0][0], Fraction)
        assert e.terms[0][0] == Fraction(2)

    def test_negative_int_coefficient(self):
        e = DensityExpr([(-3, K3)])
        assert e.terms[0][0] == Fraction(-3)

    def test_fraction_coefficient(self):
        e = DensityExpr([(Fraction(1, 2), K3)])
        assert e.terms[0][0] == Fraction(1, 2)

    def test_multi_term_preserves_order(self):
        e = DensityExpr([(1, K3), (Fraction(1, 2), K4)])
        graphs_in_order = [g for _, g in e.terms]
        assert graphs_in_order == [K3, K4]

    # --- rejections ---

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="at least one"):
            DensityExpr([])

    def test_rejects_float_coefficient(self):
        with pytest.raises(ValueError, match="int or Fraction"):
            DensityExpr([(0.5, K3)])

    def test_rejects_bool_coefficient(self):
        # bool is a subclass of int; DensityExpr must still reject it.
        with pytest.raises(ValueError, match="int or Fraction"):
            DensityExpr([(True, K3)])

    def test_rejects_non_hypergraph_graph(self):
        with pytest.raises(ValueError, match="Hypergraph"):
            DensityExpr([(1, "not a graph")])

    def test_rejects_bad_tuple_shape(self):
        with pytest.raises(ValueError, match="tuple"):
            DensityExpr([(1, K3, "extra")])

    def test_rejects_mixed_uniformity(self):
        k4_3unif = complete(4, 3)
        with pytest.raises(ValueError, match="uniformity"):
            DensityExpr([(1, K3), (1, k4_3unif)])


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------

class TestCanonicalization:
    def test_merges_duplicate_graphs(self):
        e = DensityExpr([(1, K3), (2, K3)])
        assert len(e.terms) == 1
        assert e.terms[0] == (Fraction(3), K3)

    def test_drops_zero_coefficient_terms(self):
        e = DensityExpr([(1, K3), (0, K4)])
        assert len(e.terms) == 1
        assert e.terms[0][1] == K3

    def test_drops_terms_that_cancel_to_zero(self):
        e = DensityExpr([(1, K3), (-1, K3), (2, K4)])
        assert len(e.terms) == 1
        assert e.terms[0] == (Fraction(2), K4)

    def test_rejects_when_all_terms_cancel(self):
        with pytest.raises(ValueError, match="reduced to the zero"):
            DensityExpr([(1, K3), (-1, K3)])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_single_term_matches_induced_density(self):
        H = complete(5)
        e = DensityExpr([(1, K3)])
        assert e.evaluate(H) == induced_density(H, K3)

    def test_scalar_multiple_of_induced_density(self):
        H = complete(5)
        e = DensityExpr([(3, K3)])
        assert e.evaluate(H) == 3 * induced_density(H, K3)

    def test_linear_combination_matches_manual_sum(self):
        H = complete(5)
        c1, c2 = Fraction(2), Fraction(-1, 2)
        e = DensityExpr([(c1, K3), (c2, K4)])
        expected = c1 * induced_density(H, K3) + c2 * induced_density(H, K4)
        assert e.evaluate(H) == expected

    def test_evaluate_returns_fraction(self):
        H = complete(4)
        e = DensityExpr([(1, K3)])
        assert isinstance(e.evaluate(H), Fraction)

    def test_evaluate_on_empty_graph(self):
        empty = Hypergraph(5, 2, [])
        e = DensityExpr([(1, K3), (5, K4)])
        # No induced K3 or K4 exists in the empty graph.
        assert e.evaluate(empty) == Fraction(0)


# ---------------------------------------------------------------------------
# Equality / hashing / repr
# ---------------------------------------------------------------------------

class TestIdentity:
    def test_equal_when_terms_match(self):
        a = DensityExpr([(1, K3), (Fraction(1, 2), K4)])
        b = DensityExpr([(1, K3), (Fraction(1, 2), K4)])
        assert a == b
        assert hash(a) == hash(b)

    def test_equal_after_merging_duplicates(self):
        a = DensityExpr([(3, K3)])
        b = DensityExpr([(1, K3), (2, K3)])
        assert a == b

    def test_not_equal_to_other_types(self):
        e = DensityExpr([(1, K3)])
        assert (e == "K3") is False

    def test_repr_contains_type_name(self):
        e = DensityExpr([(1, K3)])
        assert "DensityExpr" in repr(e)


# ---------------------------------------------------------------------------
# Rendering (smoke tests — content, not exact layout)
# ---------------------------------------------------------------------------

class TestRendering:
    def test_explain_returns_string(self):
        e = DensityExpr([(1, K3), (-1, K4)])
        text = e.explain()
        assert isinstance(text, str)
        assert "DensityExpr" in text
        assert "d(" in text  # formula uses "d(...)"

    def test_html_returns_string(self):
        e = DensityExpr([(1, K3)])
        html = e._repr_html_()
        assert isinstance(html, str)
        assert "DensityExpr" in html

    def test_mimebundle_has_both_types(self):
        e = DensityExpr([(1, K3)])
        bundle = e._repr_mimebundle_()
        assert "text/html" in bundle
        assert "text/plain" in bundle


# ---------------------------------------------------------------------------
# Integration with FlagProblem
# ---------------------------------------------------------------------------

class TestFlagProblemIntegration:
    def test_flag_problem_accepts_density_expr(self):
        expr = DensityExpr([(1, K3), (-Fraction(1, 2), K4)])
        prob = FlagProblem(5, 3, 2, target=expr)
        assert prob.target is expr

    def test_flag_problem_rejects_mismatched_k(self):
        # DensityExpr on k=2 graphs used in a k=3 problem.
        expr = DensityExpr([(1, K3)])
        with pytest.raises(ValueError, match="uniformity"):
            FlagProblem(5, 3, 3, target=expr)

    def test_flag_problem_rejects_invalid_target_type(self):
        with pytest.raises(ValueError, match="target must be"):
            FlagProblem(4, 2, 2, target="not a target")

    def test_explain_problem_mentions_functional(self):
        expr = DensityExpr([(1, K3), (-1, K4)])
        prob = FlagProblem(5, 3, 2, target=expr)
        text = prob.explain()
        assert "functional" in text.lower()
