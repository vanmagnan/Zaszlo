# Tests for the L1 diagnostic advisor (DiagnosticReport).
#
# The report surfaces observations, not interpretations.  Tests check field
# presence, types, and specific values on problems whose ground truth we know
# (Mantel, an aux-constrained variant, a lower-bound problem).

from __future__ import annotations

import json
from fractions import Fraction

import pytest

from zaszlo import (
    Flag,
    FlagProblem,
    Hypergraph,
    complete,
    solve,
    solve_sdp,
    build_flag_algebra_data,
)
from zaszlo.algebra import unlabel
from zaszlo.diagnose import DiagnosticReport
from zaszlo.generation import generate_flags


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mantel_result():
    prob = FlagProblem(4, 2, 2, forbidden=[complete(3)])
    return solve(prob, certify=True)


@pytest.fixture(scope="module")
def mantel_result_no_cert():
    # solve without certify still produces a FlagAlgebraResult; diagnose must
    # cope with the certificate being None.
    prob = FlagProblem(4, 2, 2, forbidden=[complete(3)])
    return solve(prob)


@pytest.fixture(scope="module")
def aux_result():
    prob = FlagProblem(4, 2, 2, forbidden=[complete(3)])
    sigma_empty = Flag(Hypergraph(0, 2, []), 0)
    u = unlabel(generate_flags(2, sigma_empty, [], [])[0] ** 2)
    prob.add_constraint(u)
    return solve(prob, certify=True)


# ---------------------------------------------------------------------------
# Field presence and types
# ---------------------------------------------------------------------------

class TestFieldPresence:
    def test_all_fields_present(self, mantel_result):
        d = mantel_result.diagnose()
        expected = {
            "kind", "status", "is_optimal", "is_valid",
            "bound_float", "bound_exact",
            "num_admissible", "num_types", "flag_counts",
            "num_active_constraints", "active_constraint_indices",
            "sharp_indices", "sharp_densities",
            "max_sharp_density_matches_bound",
            "aux_summary", "Q_summary",
            "min_residual", "worst_residual_index",
            "provenance",
        }
        assert set(d.to_dict().keys()) == expected

    def test_kind_is_flag_algebra_result(self, mantel_result):
        assert mantel_result.diagnose().kind == "FlagAlgebraResult"

    def test_kind_from_certificate(self, mantel_result):
        assert mantel_result.certificate.diagnose().kind == "Certificate"

    def test_field_types(self, mantel_result):
        d = mantel_result.diagnose()
        assert isinstance(d.status, str)
        assert isinstance(d.is_optimal, bool)
        assert isinstance(d.is_valid, bool)
        assert isinstance(d.bound_float, float)
        assert isinstance(d.bound_exact, str)
        assert isinstance(d.num_admissible, int)
        assert isinstance(d.flag_counts, list)
        assert isinstance(d.active_constraint_indices, list)
        assert isinstance(d.aux_summary, list)
        assert isinstance(d.Q_summary, list)


# ---------------------------------------------------------------------------
# Mantel-specific observations
# ---------------------------------------------------------------------------

class TestMantelObservations:
    def test_status_optimal(self, mantel_result):
        d = mantel_result.diagnose()
        assert d.is_optimal is True

    def test_valid_certificate(self, mantel_result):
        d = mantel_result.diagnose()
        assert d.is_valid is True

    def test_bound_float_close_to_half(self, mantel_result):
        d = mantel_result.diagnose()
        assert abs(d.bound_float - 0.5) < 1e-2

    def test_num_admissible_correct(self, mantel_result):
        # Mantel = 7 admissible triangle-free 4v graphs.
        assert mantel_result.diagnose().num_admissible == 7

    def test_num_types_correct(self, mantel_result):
        assert mantel_result.diagnose().num_types == 3

    def test_flag_counts(self, mantel_result):
        assert mantel_result.diagnose().flag_counts == [2, 4, 3]

    def test_at_least_one_active_constraint(self, mantel_result):
        d = mantel_result.diagnose()
        assert d.num_active_constraints >= 1

    def test_sharps_contain_half_density(self, mantel_result):
        d = mantel_result.diagnose()
        # 1/2 density is the extremal one.
        assert "1/2" in d.sharp_densities

    def test_q_summary_lists_all_blocks(self, mantel_result):
        d = mantel_result.diagnose()
        assert len(d.Q_summary) == 3
        assert [q["size"] for q in d.Q_summary] == [2, 4, 3]

    def test_min_residual_is_zero_at_active_index(self, mantel_result):
        d = mantel_result.diagnose()
        # min residual should be exactly 0 at one of the active constraint indices
        assert d.min_residual == "0"
        assert d.worst_residual_index in d.active_constraint_indices


# ---------------------------------------------------------------------------
# No-certificate case
# ---------------------------------------------------------------------------

class TestNoCertificate:
    def test_is_valid_is_none(self, mantel_result_no_cert):
        d = mantel_result_no_cert.diagnose()
        assert d.is_valid is None

    def test_bound_exact_is_none(self, mantel_result_no_cert):
        d = mantel_result_no_cert.diagnose()
        assert d.bound_exact is None

    def test_active_constraints_empty(self, mantel_result_no_cert):
        d = mantel_result_no_cert.diagnose()
        assert d.num_active_constraints == 0

    def test_min_residual_none(self, mantel_result_no_cert):
        d = mantel_result_no_cert.diagnose()
        assert d.min_residual is None


# ---------------------------------------------------------------------------
# Aux constraint case
# ---------------------------------------------------------------------------

class TestAuxConstraintReport:
    def test_aux_summary_length_matches_constraint_count(self, aux_result):
        d = aux_result.diagnose()
        assert len(d.aux_summary) == 1

    def test_aux_entry_fields(self, aux_result):
        d = aux_result.diagnose()
        entry = d.aux_summary[0]
        assert set(entry.keys()) == {"grade", "num_terms", "mu_str", "mu_float", "mu_active"}

    def test_aux_mu_str_is_rational_string(self, aux_result):
        entry = aux_result.diagnose().aux_summary[0]
        # Either "0" or "num/den" or "num" — all valid rational strings.
        assert isinstance(entry["mu_str"], str)


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_to_dict_returns_dict(self, mantel_result):
        assert isinstance(mantel_result.diagnose().to_dict(), dict)

    def test_to_json_valid(self, mantel_result):
        s = mantel_result.diagnose().to_json()
        parsed = json.loads(s)
        assert parsed["kind"] == "FlagAlgebraResult"

    def test_to_json_round_trip_agrees_with_to_dict(self, mantel_result):
        d = mantel_result.diagnose()
        parsed = json.loads(d.to_json())
        # Same keys and values (all fields are JSON-primitive after asdict).
        assert parsed["num_admissible"] == d.num_admissible
        assert parsed["sharp_densities"] == d.sharp_densities


# ---------------------------------------------------------------------------
# Explain smoke tests
# ---------------------------------------------------------------------------

class TestExplain:
    def test_explain_returns_string(self, mantel_result):
        assert isinstance(mantel_result.diagnose().explain(), str)

    def test_explain_mentions_bound(self, mantel_result):
        assert "Bound" in mantel_result.diagnose().explain()

    def test_repr_summary(self, mantel_result):
        r = repr(mantel_result.diagnose())
        assert "DiagnosticReport" in r
        assert "bound" in r.lower()
