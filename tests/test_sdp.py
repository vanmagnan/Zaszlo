# End-to-end SDP tests.
#
# Mirrors the Julia runtests.jl end-to-end tests exactly.
# Solver: Clarabel (analogous to COSMO used in Julia).

from __future__ import annotations

from fractions import Fraction

import pytest

from unittest.mock import patch

from zaszlo import (
    Certificate,
    CertificationError,
    DensityExpr,
    FlagAlgebraResult,
    FlagProblem,
    Hypergraph,
    build_flag_algebra_data,
    certify,
    certify_at_bound,
    complete,
    identify_sharps,
    round_certificate,
    solve,
    solve_sdp,
    verify_certificate,
)


# ---------------------------------------------------------------------------
# Mantel's theorem: max edge density of triangle-free graphs = 1/2
# K=2, n=4, forbidden=[K3]
# ---------------------------------------------------------------------------

class TestMantel:
    @pytest.fixture(scope="class")
    def data(self):
        k3 = Hypergraph(3, 2, [(1, 2), (1, 3), (2, 3)])
        prob = FlagProblem(4, 2, 2, forbidden=[k3], minimize=False)
        return build_flag_algebra_data(prob)

    @pytest.fixture(scope="class")
    def result(self, data):
        return solve_sdp(data, extract_Q=True)

    def test_admissible_count(self, data):
        assert len(data.admissible) == 7   # K3-free graphs on 4 vertices

    def test_type_count(self, data):
        assert len(data.types) == 3

    def test_flag_counts(self, data):
        assert [len(fs) for fs in data.flags] == [2, 4, 3]

    def test_status(self, result):
        assert result.status in ("optimal", "optimal_inaccurate")

    def test_bound(self, result):
        assert abs(result.bound - 0.5) < 1e-3

    def test_certificate(self, data, result):
        cert = verify_certificate(data, result)
        assert cert["min_psd_eigval"] >= -1e-5
        assert cert["min_residual"] >= -1e-4   # allow small solver slop

    def test_sharps(self, data, result):
        sharps = identify_sharps(data, result)
        # Empty graph has near-zero slack at optimality (complementary slackness in the SDP)
        assert 0 in sharps.indices
        # At least one sharp graph has density = 1/2 (the extremal K_{n/2,n/2}-like graph)
        assert any(d == Fraction(1, 2) for d in sharps.densities)


# ---------------------------------------------------------------------------
# K4-minus-free 3-graphs: max edge density ≈ 1/3
# K=3, n=5, forbidden=[K4-]
# Flagmatic Users Guide: "Approximate floating point bound is 0.33333334"
# ---------------------------------------------------------------------------

class TestK4MinusFree:
    @pytest.fixture(scope="class")
    def data(self):
        k4minus = Hypergraph(4, 3, [(1, 2, 3), (1, 2, 4), (1, 3, 4)])
        prob = FlagProblem(5, 3, 3, forbidden=[k4minus], minimize=False)
        return build_flag_algebra_data(prob)

    @pytest.fixture(scope="class")
    def result(self, data):
        return solve_sdp(data, extract_Q=True)

    def test_admissible_count(self, data):
        assert len(data.admissible) == 11

    def test_type_count(self, data):
        assert len(data.types) == 3

    def test_flag_counts(self, data):
        assert [len(fs) for fs in data.flags] == [2, 7, 4]

    def test_status(self, result):
        assert result.status in ("optimal", "optimal_inaccurate")

    def test_bound(self, result):
        assert abs(result.bound - 1 / 3) < 1e-3

    def test_certificate_psd(self, data, result):
        cert = verify_certificate(data, result)
        assert cert["min_psd_eigval"] >= -1e-5


# ---------------------------------------------------------------------------
# Rational rounding: Mantel's theorem certificate
# ---------------------------------------------------------------------------

class TestRoundCertificate:
    @pytest.fixture(scope="class")
    def data(self):
        k3 = Hypergraph(3, 2, [(1, 2), (1, 3), (2, 3)])
        prob = FlagProblem(4, 2, 2, forbidden=[k3], minimize=False)
        return build_flag_algebra_data(prob)

    @pytest.fixture(scope="class")
    def rounded(self, data):
        result = solve_sdp(data, extract_Q=True)
        return round_certificate(data, result, denom_limit=1000)

    def test_q_matrices_present(self, rounded):
        assert rounded.Q is not None
        assert len(rounded.Q) == 3

    def test_bound_rounded(self, rounded):
        # Certified bound should be close to 1/2 (may be slightly above due to rounding).
        assert abs(rounded.bound - 0.5) < 1e-2

    def test_q_psd_by_construction(self, rounded):
        # PSD is guaranteed by construction (Q = L_rat @ L_rat.T).
        import numpy as np
        for Q in rounded.Q:
            eigvals = np.linalg.eigvalsh(Q)
            assert eigvals.min() >= -1e-12

    def test_cholesky_factors_stored(self, rounded):
        assert rounded.cholesky_factors is not None
        assert len(rounded.cholesky_factors) == len(rounded.Q)

    def test_cholesky_reconstruction(self, rounded):
        # Q == L @ L.T exactly (up to float arithmetic).
        import numpy as np
        for Q, L in zip(rounded.Q, rounded.cholesky_factors):
            assert Q.shape == L.shape
            assert np.allclose(Q, L @ L.T, atol=1e-12)

    def test_verify_passes(self, data, rounded):
        # Rounded certificate must be exactly valid: PSD by construction, all residuals ≥ 0.
        cert = verify_certificate(data, rounded)
        assert cert["valid"] is True
        assert cert["min_psd_eigval"] >= -1e-10
        assert cert["min_residual"] >= 0


# ---------------------------------------------------------------------------
# Max induced C5-density in triangle-free graphs = 24/625
# K=2, n=5, forbidden=[K3], target=C5
# Flagmatic Users Guide: "Approximate floating point bound is 0.03840000"
# 24/625 = 0.0384
# ---------------------------------------------------------------------------

class TestC5Density:
    @pytest.fixture(scope="class")
    def data(self):
        c5 = Hypergraph(5, 2, [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)])
        k3 = Hypergraph(3, 2, [(1, 2), (1, 3), (2, 3)])
        prob = FlagProblem(5, 3, 2, forbidden=[k3], target=c5, minimize=False)
        return build_flag_algebra_data(prob)

    @pytest.fixture(scope="class")
    def result(self, data):
        return solve_sdp(data, extract_Q=True)

    def test_admissible_count(self, data):
        assert len(data.admissible) == 14

    def test_type_count(self, data):
        assert len(data.types) == 4

    def test_flag_counts(self, data):
        assert [len(fs) for fs in data.flags] == [5, 8, 6, 5]

    def test_status(self, result):
        assert result.status in ("optimal", "optimal_inaccurate")

    def test_bound(self, result):
        assert abs(result.bound - 24 / 625) < 1e-3

    def test_certificate_psd(self, data, result):
        cert = verify_certificate(data, result)
        assert cert["min_psd_eigval"] >= -1e-5


# ---------------------------------------------------------------------------
# certify(): Certificate object
# ---------------------------------------------------------------------------

class TestCertify:
    @pytest.fixture(scope="class")
    def data(self):
        k3 = Hypergraph(3, 2, [(1, 2), (1, 3), (2, 3)])
        prob = FlagProblem(4, 2, 2, forbidden=[k3], minimize=False)
        return build_flag_algebra_data(prob)

    @pytest.fixture(scope="class")
    def result(self, data):
        return solve_sdp(data, extract_Q=True)

    @pytest.fixture(scope="class")
    def proof(self, result):
        return certify(result)

    def test_returns_certificate(self, proof):
        assert isinstance(proof, Certificate)

    def test_bound_is_fraction(self, proof):
        assert isinstance(proof.bound, Fraction)

    def test_bound_value(self, proof):
        assert abs(float(proof.bound) - 0.5) < 1e-2

    def test_valid(self, proof):
        assert proof.valid is True

    def test_q_matrices_count(self, proof):
        assert proof.Q is not None
        assert len(proof.Q) == 3  # Mantel has 3 types

    def test_residuals_all_nonneg(self, proof):
        assert all(r >= 0 for r in proof.residuals)

    def test_residuals_count(self, proof, data):
        assert len(proof.residuals) == len(data.admissible)

    def test_active_constraints_have_zero_residual(self, proof):
        ac = proof.active_constraints
        assert len(ac) > 0
        for i in ac:
            assert proof.residuals[i] == Fraction(0)

    def test_explicit_data_argument(self, data, result):
        proof = certify(result, data)
        assert proof.valid is True

    def test_raises_without_q_matrices(self, data):
        result_no_q = solve_sdp(data)
        with pytest.raises(ValueError):
            certify(result_no_q)


# ---------------------------------------------------------------------------
# solve(): high-level entry point
# ---------------------------------------------------------------------------

class TestSolve:
    @pytest.fixture(scope="class")
    def prob(self):
        return FlagProblem(4, 2, 2, forbidden=[complete(3)])

    @pytest.fixture(scope="class")
    def result(self, prob):
        return solve(prob)

    @pytest.fixture(scope="class")
    def result_certified(self, prob):
        return solve(prob, certify=True)

    # --- certify=False (default) ---

    def test_returns_flag_algebra_result(self, result):
        assert isinstance(result, FlagAlgebraResult)

    def test_bound_no_certify(self, result):
        assert abs(result.bound - 0.5) < 1e-3

    def test_certificate_is_none_without_certify(self, result):
        assert result.certificate is None

    def test_data_attached(self, result):
        assert result.data is not None
        assert len(result.data.admissible) == 7

    def test_q_not_extracted_without_certify(self, result):
        # solve() without certify does not extract Q (no need for rounding).
        assert result.Q is None

    # --- certify=True ---

    def test_returns_flag_algebra_result_certified(self, result_certified):
        assert isinstance(result_certified, FlagAlgebraResult)

    def test_bound_with_certify(self, result_certified):
        assert abs(result_certified.bound - 0.5) < 1e-3

    def test_certificate_populated(self, result_certified):
        assert result_certified.certificate is not None
        assert isinstance(result_certified.certificate, Certificate)

    def test_certificate_valid(self, result_certified):
        assert result_certified.certificate.valid is True

    def test_certificate_bound_is_fraction(self, result_certified):
        assert isinstance(result_certified.certificate.bound, Fraction)

    def test_certificate_bound_value(self, result_certified):
        assert abs(float(result_certified.certificate.bound) - 0.5) < 1e-2

    def test_data_accessible_via_result(self, result_certified):
        # The full pipeline is reachable from a single returned object.
        assert result_certified.data is not None
        assert result_certified.data.problem is not None

    # --- verbose output ---

    def test_verbose_prints_data_step(self, prob, capsys):
        solve(prob, verbose=True)
        assert "Building flag algebra data" in capsys.readouterr().out

    def test_verbose_prints_sdp_step(self, prob, capsys):
        solve(prob, verbose=True)
        assert "Solving SDP" in capsys.readouterr().out

    def test_verbose_certify_prints_certificate_step(self, prob, capsys):
        solve(prob, certify=True, verbose=True)
        out = capsys.readouterr().out
        assert "Certifying" in out
        assert "valid" in out

    def test_silent_by_default(self, prob, capsys):
        solve(prob)
        assert capsys.readouterr().out == ""

    # --- CertificationError raised on invalid certificate ---

    def test_raises_certification_error_on_invalid_certificate(self, prob):
        k3 = Hypergraph(3, 2, [(1, 2), (1, 3), (2, 3)])
        fp = FlagProblem(4, 2, 2, forbidden=[k3])
        bad_cert = Certificate(
            problem=fp,
            bound=Fraction(1, 2),
            valid=False,
            Q=[],
            residuals=[Fraction(-1, 847), Fraction(1, 10)],
            data=None,
        )
        with patch("zaszlo.api._certify", return_value=bad_cert):
            with pytest.raises(CertificationError):
                solve(prob, certify=True)

    def test_certification_error_carries_result(self, prob):
        k3 = Hypergraph(3, 2, [(1, 2), (1, 3), (2, 3)])
        fp = FlagProblem(4, 2, 2, forbidden=[k3])
        bad_cert = Certificate(
            problem=fp,
            bound=Fraction(1, 2),
            valid=False,
            Q=[],
            residuals=[Fraction(-1, 847), Fraction(1, 10)],
            data=None,
        )
        with patch("zaszlo.api._certify", return_value=bad_cert):
            with pytest.raises(CertificationError) as exc_info:
                solve(prob, certify=True)
        assert exc_info.value.result.certificate is bad_cert


# ---------------------------------------------------------------------------
# CertificationError: exception type and message content
# ---------------------------------------------------------------------------

class TestCertificationError:
    @pytest.fixture
    def invalid_result(self):
        """FlagAlgebraResult with a Certificate containing one negative residual."""
        k3 = Hypergraph(3, 2, [(1, 2), (1, 3), (2, 3)])
        prob = FlagProblem(4, 2, 2, forbidden=[k3])
        cert = Certificate(
            problem=prob,
            bound=Fraction(1, 2),
            valid=False,
            Q=[],
            residuals=[Fraction(-3, 100), Fraction(1, 10)],
            data=None,
        )
        result = FlagAlgebraResult(
            problem=prob,
            status="optimal",
            bound=0.5,
            Q=None,
            slacks=[],
            certificate=cert,
        )
        return result

    def test_importable_from_zaszlo(self):
        from zaszlo import CertificationError as CE
        assert CE is CertificationError

    def test_is_subclass_of_runtime_error(self):
        assert issubclass(CertificationError, RuntimeError)

    def test_result_attribute(self, invalid_result):
        err = CertificationError(invalid_result, denom_limit=1000)
        assert err.result is invalid_result

    def test_certificate_accessible_via_result(self, invalid_result):
        err = CertificationError(invalid_result, denom_limit=1000)
        assert err.result.certificate is invalid_result.certificate

    def test_message_contains_failing_count(self, invalid_result):
        err = CertificationError(invalid_result, denom_limit=1000)
        # 1 negative residual out of 2 total
        assert "1 of 2" in str(err)

    def test_message_contains_worst_offender_index(self, invalid_result):
        err = CertificationError(invalid_result, denom_limit=1000)
        # index 0 has the only negative residual
        assert "graph index 0" in str(err)

    def test_message_contains_exact_residual(self, invalid_result):
        err = CertificationError(invalid_result, denom_limit=1000)
        assert "-3/100" in str(err)

    def test_message_contains_denom_limit(self, invalid_result):
        err = CertificationError(invalid_result, denom_limit=500)
        assert "500" in str(err)

    def test_worst_offender_is_most_negative(self):
        """When multiple residuals are negative, the worst (most negative) is reported."""
        k3 = Hypergraph(3, 2, [(1, 2), (1, 3), (2, 3)])
        prob = FlagProblem(4, 2, 2, forbidden=[k3])
        cert = Certificate(
            problem=prob,
            bound=Fraction(1, 2),
            valid=False,
            Q=[],
            residuals=[Fraction(-1, 100), Fraction(-7, 100), Fraction(1, 10)],
            data=None,
        )
        result = FlagAlgebraResult(
            problem=prob, status="optimal", bound=0.5,
            Q=None, slacks=[], certificate=cert,
        )
        err = CertificationError(result, denom_limit=1000)
        assert "2 of 3" in str(err)
        assert "graph index 1" in str(err)   # index 1 has residual -7/100
        assert "-7/100" in str(err)


# ---------------------------------------------------------------------------
# Linear-combination target: DensityExpr
# ---------------------------------------------------------------------------

class TestLinearComboMatchesScalar:
    """A one-term DensityExpr must produce the same bound as a bare Hypergraph target."""

    @pytest.fixture(scope="class")
    def bound_scalar(self):
        c5 = Hypergraph(5, 2, [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)])
        k3 = Hypergraph(3, 2, [(1, 2), (1, 3), (2, 3)])
        prob = FlagProblem(5, 3, 2, forbidden=[k3], target=c5, minimize=False)
        return solve_sdp(build_flag_algebra_data(prob)).bound

    @pytest.fixture(scope="class")
    def bound_expr(self):
        c5 = Hypergraph(5, 2, [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)])
        k3 = Hypergraph(3, 2, [(1, 2), (1, 3), (2, 3)])
        prob = FlagProblem(5, 3, 2, forbidden=[k3], target=DensityExpr([(1, c5)]), minimize=False)
        return solve_sdp(build_flag_algebra_data(prob)).bound

    def test_bounds_agree(self, bound_scalar, bound_expr):
        assert abs(bound_scalar - bound_expr) < 1e-6

    def test_bound_matches_known_value(self, bound_expr):
        assert abs(bound_expr - 24 / 625) < 1e-3


class TestLinearComboAffineMantel:
    """Affine transform of Mantel: maximize  (3/2)·edge − (1/2)·(1 − edge)  = (3/2)·d − 1/2
    on triangle-free 4-vertex graphs.  Written as a linear combination of induced densities
    over the edge (K2) and the empty 2-vertex graph.  At the Mantel optimum edge = 1/2, so
    the value is (3/2)(1/2) − 1/2 = 1/4.
    """

    @pytest.fixture(scope="class")
    def result(self):
        k3 = Hypergraph(3, 2, [(1, 2), (1, 3), (2, 3)])
        edge = complete(2)                       # K2 = single edge
        empty_2v = Hypergraph(2, 2, [])          # 2-vertex empty graph
        expr = DensityExpr([
            (1, edge),
            (-Fraction(1, 2), empty_2v),
        ])
        prob = FlagProblem(4, 2, 2, forbidden=[k3], target=expr, minimize=False)
        return solve_sdp(build_flag_algebra_data(prob), extract_Q=True)

    def test_status(self, result):
        assert result.status in ("optimal", "optimal_inaccurate")

    def test_bound_value(self, result):
        # (3/2)(1/2) − 1/2 = 1/4
        assert abs(result.bound - 0.25) < 1e-3

    def test_certificate_psd(self, result):
        cert = verify_certificate(result.data, result)
        assert cert["min_psd_eigval"] >= -1e-5

    def test_certify_valid(self, result):
        proof = certify(result)
        assert proof.valid
        assert abs(float(proof.bound) - 0.25) < 1e-2


# ---------------------------------------------------------------------------
# Multiple forbidden subgraphs
# ---------------------------------------------------------------------------

class TestMultipleForbidden:
    """Multiple entries in `forbidden` must all be enforced.

    Since K4 contains K3, the constraint set {K3, K4}-free coincides exactly with
    K3-free, so both problems must produce the same admissible graphs, the same
    number of types/flags, and the same Mantel bound of 1/2.
    """

    @pytest.fixture(scope="class")
    def data_single(self):
        k3 = Hypergraph(3, 2, [(1, 2), (1, 3), (2, 3)])
        return build_flag_algebra_data(FlagProblem(4, 2, 2, forbidden=[k3]))

    @pytest.fixture(scope="class")
    def data_multi(self):
        k3 = Hypergraph(3, 2, [(1, 2), (1, 3), (2, 3)])
        k4 = complete(4)
        return build_flag_algebra_data(FlagProblem(4, 2, 2, forbidden=[k3, k4]))

    def test_admissible_counts_match(self, data_single, data_multi):
        assert len(data_single.admissible) == len(data_multi.admissible)

    def test_flag_counts_match(self, data_single, data_multi):
        assert [len(fs) for fs in data_single.flags] == [len(fs) for fs in data_multi.flags]

    def test_bound_matches_mantel(self, data_multi):
        r = solve_sdp(data_multi)
        assert abs(r.bound - 0.5) < 1e-3

    def test_two_incomparable_forbidden(self):
        """Two forbidden graphs that do not contain one another (K3 and K4-minus, mixed
        uniformity would be invalid, so we use two 3-uniform patterns).  Verify the SDP
        still produces a finite bound.  This locks in that the pipeline handles distinct
        forbidden patterns without silently dropping any."""
        k4minus = Hypergraph(4, 3, [(1, 2, 3), (1, 2, 4), (1, 3, 4)])
        c5_tight = Hypergraph(5, 3, [(1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 1), (5, 1, 2)])
        prob = FlagProblem(5, 3, 3, forbidden=[k4minus, c5_tight])
        r = solve_sdp(build_flag_algebra_data(prob))
        assert r.status in ("optimal", "optimal_inaccurate")
        # K4-minus alone gives ~1/3; adding c5 as a forbidden pattern can only tighten
        # the bound, so the result must be no larger than the K4-minus-only bound plus
        # solver slop.
        assert r.bound <= 1 / 3 + 1e-3


# ---------------------------------------------------------------------------
# Auxiliary flag-algebra constraints
# ---------------------------------------------------------------------------

class TestAuxConstraintRedundant:
    """Adding a redundant SoS constraint <<f²>> >= 0 to Mantel must preserve
    validity (the bound may change slightly with rounding, but a valid
    certificate must exist and match 1/2 numerically)."""

    @pytest.fixture(scope="class")
    def solved(self):
        from zaszlo import Flag, DensityExpr
        from zaszlo.algebra import unlabel
        from zaszlo.generation import generate_flags

        K3 = complete(3)
        # A trivially-valid SoS constraint: <<f²>> >= 0 where f is any admissible flag.
        sigma_empty = Flag(Hypergraph(0, 2, []), 0)
        f = generate_flags(2, sigma_empty, [], [])[0]  # 2v admissible flag
        aux_expr = unlabel(f * f)  # UnlabeledExpr at grade 4

        prob = FlagProblem(4, 2, 2, forbidden=[K3])
        prob.add_constraint(aux_expr)
        return solve(prob, certify=True)

    def test_bound_unchanged(self, solved):
        assert abs(solved.bound - 0.5) < 1e-3

    def test_certificate_still_valid(self, solved):
        assert solved.certificate.valid

    def test_certified_bound_close_to_half(self, solved):
        assert abs(float(solved.certificate.bound) - 0.5) < 1e-2

    def test_mu_extracted(self, solved):
        assert len(solved.mu) == 1

    def test_mu_nonneg(self, solved):
        assert all(m >= -1e-9 for m in solved.mu)

    def test_certificate_carries_mu(self, solved):
        # Certificate.mu is the exact rational μ (may be exactly 0 for redundant aux).
        assert len(solved.certificate.mu) == 1
        assert all(m >= 0 for m in solved.certificate.mu)


class TestAuxConstraintSquareOfDifference:
    """<<(f0 - f1)²>> >= 0 for f0, f1 the two 2v flags over the 1-vertex type.
    This constraint has mixed-sign lifted coefficients (see the algebra tests)
    but is a valid SoS inequality on graphons. Adding it to Mantel must keep
    the certificate valid and the bound close to 1/2."""

    @pytest.fixture(scope="class")
    def solved(self):
        from zaszlo import Flag
        from zaszlo.algebra import unlabel
        from zaszlo.generation import generate_flags

        K3 = complete(3)
        sigma_1v = Flag(Hypergraph(1, 2, []), 1)
        f0, f1 = generate_flags(2, sigma_1v, [], [])
        aux_element = (f0 - f1) * (f0 - f1)   # grade 3
        aux_expr = unlabel(aux_element)

        prob = FlagProblem(4, 2, 2, forbidden=[K3])
        # aux_expr is at grade 3, prob.n is 4 — lifting handled by pipeline
        prob.add_constraint(aux_expr)
        return solve(prob, certify=True)

    def test_bound_close_to_half(self, solved):
        assert abs(solved.bound - 0.5) < 1e-3

    def test_certificate_valid(self, solved):
        assert solved.certificate.valid

    def test_residuals_all_nonneg(self, solved):
        assert all(r >= 0 for r in solved.certificate.residuals)

    def test_mu_nonneg(self, solved):
        assert all(m >= 0 for m in solved.certificate.mu)


# ---------------------------------------------------------------------------
# certify_at_bound(): Cohn–de Laat–Leijenhorst pipeline
# ---------------------------------------------------------------------------

class TestCertifyAtBound:
    """certify_at_bound must return an exact Certificate at the user-supplied
    Fraction bound, valid in exact rational arithmetic.

    The primary test is Mantel's theorem: max edge density in triangle-free
    4-vertex graphs = exactly 1/2.  Standard certify() cannot reach Fraction(1,2)
    because Cholesky rounding always overshoots; this pipeline must hit it exactly.
    """

    @pytest.fixture(scope="class")
    def mantel_data(self):
        k3 = Hypergraph(3, 2, [(1, 2), (1, 3), (2, 3)])
        prob = FlagProblem(4, 2, 2, forbidden=[k3], minimize=False)
        return build_flag_algebra_data(prob)

    @pytest.fixture(scope="class")
    def mantel_cert(self, mantel_data):
        return certify_at_bound(mantel_data, Fraction(1, 2))

    # --- core correctness ---

    def test_returns_certificate(self, mantel_cert):
        assert isinstance(mantel_cert, Certificate)

    def test_bound_is_exactly_half(self, mantel_cert):
        assert mantel_cert.bound == Fraction(1, 2)

    def test_bound_is_fraction(self, mantel_cert):
        assert isinstance(mantel_cert.bound, Fraction)

    def test_valid(self, mantel_cert):
        assert mantel_cert.valid is True

    def test_residuals_all_nonneg(self, mantel_cert):
        assert all(r >= 0 for r in mantel_cert.residuals)

    def test_residuals_are_fractions(self, mantel_cert):
        assert all(isinstance(r, Fraction) for r in mantel_cert.residuals)

    def test_residuals_count(self, mantel_cert, mantel_data):
        assert len(mantel_cert.residuals) == len(mantel_data.admissible)

    def test_q_matrices_present(self, mantel_cert):
        assert mantel_cert.Q is not None
        assert len(mantel_cert.Q) == 3  # Mantel has 3 types

    def test_active_constraints_have_zero_residual(self, mantel_cert):
        ac = mantel_cert.active_constraints
        assert len(ac) > 0
        for i in ac:
            assert mantel_cert.residuals[i] == Fraction(0)

    # --- exact bound vs certify() ---

    def test_exact_bound_unreachable_by_certify(self, mantel_data):
        # Confirm the motivation: certify() cannot reach Fraction(1, 2) exactly.
        from zaszlo import solve_sdp
        result = solve_sdp(mantel_data, extract_Q=True)
        proof = certify(result)
        assert proof.bound != Fraction(1, 2)

    # --- Turán's theorem: max edge density in K4-free graphs = 2/3 ---

    @pytest.fixture(scope="class")
    def turan_data(self):
        prob = FlagProblem(5, 3, 2, forbidden=[complete(4)], minimize=False)
        return build_flag_algebra_data(prob)

    @pytest.fixture(scope="class")
    def turan_cert(self, turan_data):
        return certify_at_bound(turan_data, Fraction(2, 3))

    def test_turan_bound_exact(self, turan_cert):
        assert turan_cert.bound == Fraction(2, 3)

    def test_turan_valid(self, turan_cert):
        assert turan_cert.valid is True

    def test_turan_residuals_nonneg(self, turan_cert):
        assert all(r >= 0 for r in turan_cert.residuals)

    # --- type validation ---

    def test_raises_on_non_fraction_bound(self, mantel_data):
        with pytest.raises(TypeError):
            certify_at_bound(mantel_data, 0.5)

    def test_raises_on_int_bound(self, mantel_data):
        with pytest.raises(TypeError):
            certify_at_bound(mantel_data, 1)

    # --- infeasible bound returns invalid certificate ---

    def test_infeasible_bound_returns_invalid(self, mantel_data):
        # Fraction(1, 3) is below the Mantel optimum; the feasibility SDP is
        # infeasible (no Q certifies a bound that tight for triangle-free graphs).
        cert = certify_at_bound(mantel_data, Fraction(1, 3))
        assert isinstance(cert, Certificate)
        assert cert.valid is False

    # --- aux constraints: μ carried through correctly ---

    def test_aux_constraint_problem(self):
        from zaszlo import Flag
        from zaszlo.algebra import unlabel
        from zaszlo.generation import generate_flags

        K3 = complete(3)
        sigma_empty = Flag(Hypergraph(0, 2, []), 0)
        f = generate_flags(2, sigma_empty, [], [])[0]
        aux_expr = unlabel(f * f)

        prob = FlagProblem(4, 2, 2, forbidden=[K3])
        prob.add_constraint(aux_expr)
        data = build_flag_algebra_data(prob)
        cert = certify_at_bound(data, Fraction(1, 2))
        assert cert.valid is True

    # --- provenance: pipeline records how the certificate was produced ---

    def test_provenance_pipeline_certify_at_bound(self, mantel_cert):
        assert mantel_cert.provenance.pipeline == "certify_at_bound"

    def test_provenance_denominator_limit(self, mantel_cert):
        assert mantel_cert.provenance.denom_limit == 1000

    def test_provenance_feasibility_status_optimal(self, mantel_cert):
        assert mantel_cert.provenance.feasibility_status == "optimal"

    def test_provenance_kernel_dims_recorded(self, mantel_cert):
        # Mantel has three types, each with a 1-dimensional kernel at the optimum.
        assert mantel_cert.provenance.kernel_dims == [1, 1, 1]

    def test_provenance_correction_reports_rank(self, mantel_cert):
        corr = mantel_cert.provenance.correction
        assert corr is not None
        assert corr["num_sharp"] == 3
        assert corr["rank"] == 2  # Mantel: 3 sharp constraints, only 2 independent
        assert corr["float_rank"] == 2
        assert corr["applied"] is True

    def test_provenance_psd_safeguard_passed(self, mantel_cert):
        assert mantel_cert.provenance.psd_safeguard == "passed"

    def test_provenance_no_reason_when_valid(self, mantel_cert):
        assert mantel_cert.provenance.reason is None

    def test_provenance_infeasible_bound_surfaces_reason(self, mantel_data):
        # Bound tighter than what the SDP can achieve → feasibility solve is
        # infeasible; provenance must surface the Clarabel status and a reason.
        cert = certify_at_bound(mantel_data, Fraction(1, 3))
        assert cert.valid is False
        assert cert.provenance.pipeline == "certify_at_bound"
        assert cert.provenance.feasibility_status == "infeasible"
        assert cert.provenance.reason is not None
        assert "infeasible" in cert.provenance.reason.lower()

    def test_provenance_certify_pipeline_tag(self, mantel_data):
        # certify() (Cholesky pipeline) must also record its provenance.
        from zaszlo import solve_sdp

        result = solve_sdp(mantel_data, extract_Q=True)
        cert = certify(result)
        assert cert.provenance.pipeline == "certify"
        assert cert.provenance.denom_limit == 1000
        assert cert.provenance.chol_reg == 1e-10

    def test_provenance_survives_to_dict(self, mantel_cert):
        d = mantel_cert.to_dict()
        assert "provenance" in d
        assert d["provenance"]["pipeline"] == "certify_at_bound"
        assert d["provenance"]["feasibility_status"] == "optimal"
        assert d["provenance"]["kernel_dims"] == [1, 1, 1]

    def test_diagnostic_report_includes_provenance(self, mantel_cert):
        report = mantel_cert.diagnose()
        assert report.provenance["pipeline"] == "certify_at_bound"
        assert report.provenance["feasibility_status"] == "optimal"

    def test_explain_mentions_provenance(self, mantel_cert):
        text = mantel_cert.explain()
        assert "How this was produced" in text
        assert "certify_at_bound" in text

    def test_explain_mentions_reason_when_invalid(self, mantel_data):
        cert = certify_at_bound(mantel_data, Fraction(1, 3))
        text = cert.explain()
        assert "Why invalid" in text
        assert "infeasible" in text.lower()
