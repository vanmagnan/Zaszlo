# Tests for the flag-algebra kernel: FlagAlgebraElement, UnlabeledExpr,
# AuxiliaryConstraint, Flag arithmetic, flag_product, unlabel, lift_to.
#
# The correctness anchor for flag_product + unlabel is the library's own
# compute_pair_densities: unlabel(f_i · f_j).evaluate(H) must equal
# pair_dens[H][σ][min(i,j)][max(i,j)] for every admissible H.

from __future__ import annotations

from fractions import Fraction

import pytest

from zaszlo import (
    AuxiliaryConstraint,
    Flag,
    FlagAlgebraElement,
    FlagProblem,
    Hypergraph,
    UnlabeledExpr,
    complete,
    flag_product,
    lift_to,
    unlabel,
)
from zaszlo.densities import compute_pair_densities
from zaszlo.generation import generate_admissible, generate_flags
from zaszlo.graphs import canonical, induce


# --- Common fixtures ------------------------------------------------------

SIGMA_1V_K2 = Flag(Hypergraph(1, 2, []), 1)           # 1-vertex type, k=2
FLAGS_2V_1V = generate_flags(2, SIGMA_1V_K2, [], [])   # [f0=no edges, f1=1 edge]
SIGMA_2V_K2 = Flag(Hypergraph(2, 2, [(1, 2)]), 2)      # 2-vertex type with edge


# ---------------------------------------------------------------------------
# FlagAlgebraElement — construction & validation
# ---------------------------------------------------------------------------

class TestFlagAlgebraElement:
    def test_single_term(self):
        e = FlagAlgebraElement(SIGMA_1V_K2, 2, [(1, FLAGS_2V_1V[0])])
        assert len(e.terms) == 1
        assert e.terms[0][0] == Fraction(1)
        assert e.n == 2
        assert e.s == 1
        assert e.k == 2

    def test_zero_element_from_empty_terms(self):
        e = FlagAlgebraElement(SIGMA_1V_K2, 2, [])
        assert e.is_zero
        assert e.n == 2

    def test_zero_element_from_cancelling_terms(self):
        e = FlagAlgebraElement(SIGMA_1V_K2, 2, [(1, FLAGS_2V_1V[0]), (-1, FLAGS_2V_1V[0])])
        assert e.is_zero

    def test_merges_duplicate_flags(self):
        e = FlagAlgebraElement(SIGMA_1V_K2, 2, [(2, FLAGS_2V_1V[0]), (3, FLAGS_2V_1V[0])])
        assert len(e.terms) == 1
        assert e.terms[0][0] == Fraction(5)

    def test_rejects_non_type_flag(self):
        # A flag with type_size < graph.n is not a type.
        not_a_type = FLAGS_2V_1V[0]  # type_size=1, graph.n=2
        with pytest.raises(ValueError, match="must be a type"):
            FlagAlgebraElement(not_a_type, 2, [])

    def test_rejects_float_coefficient(self):
        with pytest.raises(ValueError, match="int or Fraction"):
            FlagAlgebraElement(SIGMA_1V_K2, 2, [(0.5, FLAGS_2V_1V[0])])

    def test_rejects_bool_coefficient(self):
        with pytest.raises(ValueError, match="int or Fraction"):
            FlagAlgebraElement(SIGMA_1V_K2, 2, [(True, FLAGS_2V_1V[0])])

    def test_rejects_wrong_type_size(self):
        # Flag with type_size 2 in an element declared for type_size 1.
        bad_flag = Flag(Hypergraph(2, 2, [(1, 2)]), 2)
        with pytest.raises(ValueError, match="type_size"):
            FlagAlgebraElement(SIGMA_1V_K2, 2, [(1, bad_flag)])

    def test_rejects_wrong_grade(self):
        # Flag on 3 vertices in an element declared for grade 2.
        f3 = generate_flags(3, SIGMA_1V_K2, [], [])[0]
        with pytest.raises(ValueError, match="graph.n"):
            FlagAlgebraElement(SIGMA_1V_K2, 2, [(1, f3)])

    def test_rejects_grade_below_type_size(self):
        with pytest.raises(ValueError, match="at least the type size"):
            FlagAlgebraElement(SIGMA_1V_K2, 0, [])


# ---------------------------------------------------------------------------
# FlagAlgebraElement — arithmetic
# ---------------------------------------------------------------------------

class TestFlagAlgebraArithmetic:
    def test_addition(self):
        a = FlagAlgebraElement(SIGMA_1V_K2, 2, [(1, FLAGS_2V_1V[0])])
        b = FlagAlgebraElement(SIGMA_1V_K2, 2, [(1, FLAGS_2V_1V[1])])
        c = a + b
        assert len(c.terms) == 2

    def test_addition_merges(self):
        a = FlagAlgebraElement(SIGMA_1V_K2, 2, [(1, FLAGS_2V_1V[0])])
        b = FlagAlgebraElement(SIGMA_1V_K2, 2, [(2, FLAGS_2V_1V[0])])
        c = a + b
        assert len(c.terms) == 1
        assert c.terms[0][0] == Fraction(3)

    def test_subtraction_cancels(self):
        a = FlagAlgebraElement(SIGMA_1V_K2, 2, [(1, FLAGS_2V_1V[0])])
        assert (a - a).is_zero

    def test_scalar_multiplication(self):
        a = FlagAlgebraElement(SIGMA_1V_K2, 2, [(1, FLAGS_2V_1V[0])])
        b = 3 * a
        assert b.terms[0][0] == Fraction(3)

    def test_fraction_scalar_multiplication(self):
        a = FlagAlgebraElement(SIGMA_1V_K2, 2, [(1, FLAGS_2V_1V[0])])
        b = Fraction(1, 2) * a
        assert b.terms[0][0] == Fraction(1, 2)

    def test_product_grade(self):
        # deg(f · g) = n_f + n_g − s
        f, g = FLAGS_2V_1V
        prod = FlagAlgebraElement(SIGMA_1V_K2, 2, [(1, f)]) * FlagAlgebraElement(SIGMA_1V_K2, 2, [(1, g)])
        assert prod.n == 2 + 2 - 1  # 3

    def test_product_commutes(self):
        f, g = FLAGS_2V_1V
        ef = FlagAlgebraElement(SIGMA_1V_K2, 2, [(1, f)])
        eg = FlagAlgebraElement(SIGMA_1V_K2, 2, [(1, g)])
        assert ef * eg == eg * ef

    def test_type_mismatch_addition(self):
        a_1v = FlagAlgebraElement(SIGMA_1V_K2, 2, [(1, FLAGS_2V_1V[0])])
        f_over_2v = Flag(Hypergraph(3, 2, [(1, 2)]), 2)
        b_2v = FlagAlgebraElement(SIGMA_2V_K2, 3, [(1, f_over_2v)])
        with pytest.raises(ValueError, match="matching type"):
            _ = a_1v + b_2v

    def test_type_mismatch_product(self):
        a_1v = FlagAlgebraElement(SIGMA_1V_K2, 2, [(1, FLAGS_2V_1V[0])])
        f_over_2v = Flag(Hypergraph(3, 2, [(1, 2)]), 2)
        b_2v = FlagAlgebraElement(SIGMA_2V_K2, 3, [(1, f_over_2v)])
        with pytest.raises(ValueError, match="matching type"):
            _ = a_1v * b_2v

    def test_grade_mismatch_addition(self):
        a2 = FlagAlgebraElement(SIGMA_1V_K2, 2, [(1, FLAGS_2V_1V[0])])
        f3 = generate_flags(3, SIGMA_1V_K2, [], [])[0]
        a3 = FlagAlgebraElement(SIGMA_1V_K2, 3, [(1, f3)])
        with pytest.raises(ValueError, match="different grades"):
            _ = a2 + a3

    def test_pow_squares(self):
        a = FlagAlgebraElement(SIGMA_1V_K2, 2, [(1, FLAGS_2V_1V[0])])
        assert a ** 2 == a * a

    def test_pow_cubes(self):
        a = FlagAlgebraElement(SIGMA_1V_K2, 2, [(1, FLAGS_2V_1V[0])])
        assert a ** 3 == a * a * a

    def test_pow_rejects_zero(self):
        a = FlagAlgebraElement(SIGMA_1V_K2, 2, [(1, FLAGS_2V_1V[0])])
        with pytest.raises(ValueError, match="positive integer"):
            a ** 0


# ---------------------------------------------------------------------------
# Flag operators — lift into FlagAlgebraElement
# ---------------------------------------------------------------------------

class TestFlagOperators:
    def test_flag_plus_flag_returns_element(self):
        f, g = FLAGS_2V_1V
        r = f + g
        assert isinstance(r, FlagAlgebraElement)
        assert r.n == 2 and r.s == 1

    def test_flag_minus_flag(self):
        f, g = FLAGS_2V_1V
        r = f - g
        assert isinstance(r, FlagAlgebraElement)
        assert r.terms[0] == (Fraction(1), f)
        assert r.terms[1] == (Fraction(-1), g)

    def test_flag_neg(self):
        f = FLAGS_2V_1V[0]
        r = -f
        assert r.terms[0] == (Fraction(-1), f)

    def test_scalar_times_flag(self):
        f = FLAGS_2V_1V[0]
        r = 2 * f
        assert r.terms[0] == (Fraction(2), f)

    def test_flag_times_scalar(self):
        f = FLAGS_2V_1V[0]
        r = f * Fraction(1, 3)
        assert r.terms[0] == (Fraction(1, 3), f)

    def test_flag_times_flag_returns_element_at_correct_grade(self):
        f, g = FLAGS_2V_1V
        r = f * g
        assert isinstance(r, FlagAlgebraElement)
        assert r.n == 3

    def test_flag_pow(self):
        f = FLAGS_2V_1V[0]
        r = f ** 2
        assert r == f * f

    def test_type_mismatch_between_flags(self):
        f_over_1v = FLAGS_2V_1V[0]
        f_over_2v = Flag(Hypergraph(3, 2, [(1, 2)]), 2)
        with pytest.raises(ValueError):
            _ = f_over_1v + f_over_2v


# ---------------------------------------------------------------------------
# flag_product — semantics against library pair_dens
# ---------------------------------------------------------------------------

class TestFlagProductAgainstPairDens:
    """The strongest correctness check: unlabel(f_i · f_j) evaluated at each
    admissible H matches pair_dens[H][σ][min,max]."""

    @pytest.mark.parametrize("n_grade", [3, 4])
    def test_matches_pair_dens_for_1v_type_k2(self, n_grade):
        # Compare flag_product+unlabel against compute_pair_densities on all
        # (H, i, j) triples for the 1-vertex-type k=2 setup at various grades.
        sigma = SIGMA_1V_K2
        # Flags at grade n_grade//2+1 (roughly matches what pair_dens uses in a real problem).
        m = 2  # 2v flags, matches n=3 admissibles below when grade=3
        flags = generate_flags(m, sigma, [], [])
        admissible = generate_admissible(n_grade, 2)
        pair_dens = [compute_pair_densities(H, [sigma], [flags]) for H in admissible]
        # Only compare when 2m - s equals n_grade (unlabel grade = n_grade)
        if 2 * m - 1 != n_grade:
            pytest.skip("skip when grade doesn't line up with 2m - s")

        for h_idx, H in enumerate(admissible):
            H_canon = canonical(Flag(H, 0)).graph.edges
            for i, fi in enumerate(flags):
                for j, fj in enumerate(flags):
                    if i > j:
                        continue
                    prod = flag_product(fi, fj)
                    elem = FlagAlgebraElement(sigma, 2 * m - 1, prod)
                    u = unlabel(elem)
                    # Look up coefficient of H_canon
                    coef = Fraction(0)
                    for c, gg in u.terms:
                        if gg.edges == H_canon:
                            coef = c
                            break
                    expected = pair_dens[h_idx][0][i][j]
                    assert coef == expected, (
                        f"mismatch at H={h_idx}, i={i}, j={j}: "
                        f"got {coef}, expected {expected}"
                    )


class TestFlagProductBasic:
    def test_product_is_commutative(self):
        f, g = FLAGS_2V_1V
        left = flag_product(f, g)
        right = flag_product(g, f)
        # Same set of (coef, flag) pairs.
        assert sorted(left, key=lambda t: t[1].graph.edges) == sorted(right, key=lambda t: t[1].graph.edges)

    def test_bilinearity_scalar(self):
        # (2·f) · g == 2 · (f · g) as FlagAlgebraElements.
        f, g = FLAGS_2V_1V
        e = FlagAlgebraElement(SIGMA_1V_K2, 2, [(2, f)])
        ge = FlagAlgebraElement(SIGMA_1V_K2, 2, [(1, g)])
        base = FlagAlgebraElement(SIGMA_1V_K2, 2, [(1, f)])
        assert e * ge == 2 * (base * ge)

    def test_bilinearity_sum(self):
        # (f + f') · g == f · g + f' · g
        f, fp = FLAGS_2V_1V
        g = FLAGS_2V_1V[0]
        left = (FlagAlgebraElement(SIGMA_1V_K2, 2, [(1, f)]) + FlagAlgebraElement(SIGMA_1V_K2, 2, [(1, fp)])) * FlagAlgebraElement(SIGMA_1V_K2, 2, [(1, g)])
        right = FlagAlgebraElement(SIGMA_1V_K2, 2, [(1, f)]) * FlagAlgebraElement(SIGMA_1V_K2, 2, [(1, g)]) + FlagAlgebraElement(SIGMA_1V_K2, 2, [(1, fp)]) * FlagAlgebraElement(SIGMA_1V_K2, 2, [(1, g)])
        assert left == right

    def test_type_flag_is_left_identity(self):
        # For σ (type flag itself as a 0-extra-vertex σ-flag), σ · f = f.
        sigma_as_flag = SIGMA_1V_K2  # sits at grade s=1
        f = FLAGS_2V_1V[0]
        prod = flag_product(sigma_as_flag, f)
        # Should be [(1, f_canonical)]
        assert len(prod) == 1
        c, resulting_flag = prod[0]
        assert c == Fraction(1)
        # Result grade = 1 + 2 − 1 = 2.
        assert resulting_flag.graph.n == 2


# ---------------------------------------------------------------------------
# unlabel
# ---------------------------------------------------------------------------

class TestUnlabel:
    def test_unlabel_of_zero_is_zero(self):
        e = FlagAlgebraElement(SIGMA_1V_K2, 2, [])
        u = unlabel(e)
        assert u.is_zero
        assert u.n == 2

    def test_unlabel_preserves_grade(self):
        f = FLAGS_2V_1V[0]
        e = FlagAlgebraElement(SIGMA_1V_K2, 2, [(1, f)])
        assert unlabel(e).n == 2

    def test_unlabel_linear(self):
        f, g = FLAGS_2V_1V
        ef = FlagAlgebraElement(SIGMA_1V_K2, 2, [(1, f)])
        eg = FlagAlgebraElement(SIGMA_1V_K2, 2, [(1, g)])
        assert unlabel(2 * ef - eg) == 2 * unlabel(ef) - unlabel(eg)


# ---------------------------------------------------------------------------
# UnlabeledExpr
# ---------------------------------------------------------------------------

class TestUnlabeledExpr:
    def test_empty_terms_is_zero(self):
        u = UnlabeledExpr(2, 3)
        assert u.is_zero

    def test_evaluate_matches_induced_density(self):
        # For a single-term expression u = 1 · H, evaluate(G) = induced_density(G, H).
        from zaszlo.densities import induced_density
        K3 = complete(3)
        u = UnlabeledExpr(2, 3, [(1, K3)])
        # Evaluate on a bigger graph.
        K4 = complete(4)
        assert u.evaluate(K4) == induced_density(K4, K3)

    def test_evaluate_grade_mismatch(self):
        u = UnlabeledExpr(2, 3, [(1, complete(3))])
        # H at grade 2 < u.n=3 should error.
        H_small = Hypergraph(2, 2, [(1, 2)])
        with pytest.raises(ValueError, match="undefined"):
            u.evaluate(H_small)

    def test_evaluate_uniformity_mismatch(self):
        u = UnlabeledExpr(2, 3, [(1, complete(3))])
        with pytest.raises(ValueError, match="uniformity"):
            u.evaluate(complete(3, 3))

    def test_addition_grade_mismatch(self):
        u2 = UnlabeledExpr(2, 2, [(1, Hypergraph(2, 2, [(1, 2)]))])
        u3 = UnlabeledExpr(2, 3, [(1, complete(3))])
        with pytest.raises(ValueError, match="different grades"):
            _ = u2 + u3


# ---------------------------------------------------------------------------
# lift_to
# ---------------------------------------------------------------------------

class TestLiftTo:
    def test_lift_identity_when_same_grade(self):
        u = UnlabeledExpr(2, 3, [(1, complete(3))])
        assert lift_to(u, 3) is u

    def test_lift_zero(self):
        u = UnlabeledExpr(2, 3, [])
        assert lift_to(u, 5).is_zero

    def test_lift_rejects_smaller_target(self):
        u = UnlabeledExpr(2, 4, [(1, complete(4))])
        with pytest.raises(ValueError, match="cannot lower grade"):
            lift_to(u, 3)

    def test_lift_matches_induced_density(self):
        # lift_to(1·H_small) evaluated on H_big equals induced_density(H_big, H_small).
        from zaszlo.densities import induced_density
        K3 = complete(3)
        u = UnlabeledExpr(2, 3, [(1, K3)])
        u_lifted = lift_to(u, 4)
        for H in generate_admissible(4, 2):
            expected = induced_density(H, K3)
            actual = Fraction(0)
            for c, gg in u_lifted.terms:
                if gg == H:
                    actual = c
                    break
            assert actual == expected

    def test_lifted_evaluate_equals_direct_evaluate(self):
        # For any admissible H at grade ≥ u.n, evaluate should agree with lifted evaluate.
        K3 = complete(3)
        u = UnlabeledExpr(2, 3, [(2, K3)])
        u_lifted = lift_to(u, 4)
        for H in generate_admissible(4, 2):
            assert u_lifted.evaluate(H) == u.evaluate(H)


# ---------------------------------------------------------------------------
# AuxiliaryConstraint
# ---------------------------------------------------------------------------

class TestAuxiliaryConstraint:
    def test_wraps_unlabeled_expr(self):
        u = UnlabeledExpr(2, 3, [(1, complete(3))])
        c = AuxiliaryConstraint(u)
        assert c.expr is u
        assert c.n == 3 and c.k == 2

    def test_rejects_non_unlabeled(self):
        with pytest.raises(ValueError, match="UnlabeledExpr"):
            AuxiliaryConstraint(complete(3))

    def test_rejects_unsupported_sense(self):
        u = UnlabeledExpr(2, 3, [(1, complete(3))])
        with pytest.raises(ValueError, match=">=0"):
            AuxiliaryConstraint(u, sense="<=0")


# ---------------------------------------------------------------------------
# FlagProblem.add_constraint accepts multiple input types
# ---------------------------------------------------------------------------

class TestFlagProblemAddConstraint:
    def test_add_auxiliary_constraint(self):
        prob = FlagProblem(4, 2, 2, forbidden=[complete(3)])
        u = UnlabeledExpr(2, 3, [(1, complete(3))])
        prob.add_constraint(AuxiliaryConstraint(u))
        assert len(prob.aux_constraints) == 1

    def test_add_unlabeled_expr_directly(self):
        prob = FlagProblem(4, 2, 2, forbidden=[complete(3)])
        u = UnlabeledExpr(2, 3, [(1, complete(3))])
        prob.add_constraint(u)
        assert len(prob.aux_constraints) == 1
        assert isinstance(prob.aux_constraints[0], AuxiliaryConstraint)

    def test_add_flag_algebra_element_auto_unlabels(self):
        prob = FlagProblem(4, 2, 2, forbidden=[complete(3)])
        f = FLAGS_2V_1V[0]
        e = FlagAlgebraElement(SIGMA_1V_K2, 2, [(1, f)])
        prob.add_constraint(e)
        assert len(prob.aux_constraints) == 1

    def test_reject_grade_larger_than_problem_n(self):
        prob = FlagProblem(4, 2, 2, forbidden=[complete(3)])
        # 5v UnlabeledExpr in a 4v problem.
        K5 = Hypergraph(5, 2, [])
        u = UnlabeledExpr(2, 5, [(1, K5)])
        with pytest.raises(ValueError, match="exceeds problem"):
            prob.add_constraint(u)

    def test_reject_k_mismatch(self):
        prob = FlagProblem(4, 2, 2, forbidden=[complete(3)])
        u = UnlabeledExpr(3, 3, [(1, Hypergraph(3, 3, []))])
        with pytest.raises(ValueError, match="k="):
            prob.add_constraint(u)
