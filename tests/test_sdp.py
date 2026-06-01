# End-to-end SDP tests.
#
# Mirrors the Julia runtests.jl end-to-end tests exactly.
# Solver: Clarabel (analogous to COSMO used in Julia).

from __future__ import annotations

from fractions import Fraction

import pytest

from zaszlo import (
    FlagProblem,
    Hypergraph,
    build_flag_algebra_data,
    identify_sharps,
    round_certificate,
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
        # Empty graph is always sharp (density 0 ≤ bound)
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
