# Tests for structured JSON / dict serialization.
#
# The contract:
#   - Every top-level payload has "kind".
#   - Exact rationals encode as {"num", "den"} losslessly.
#   - to_json() round-trips through json.loads back to the same dict.
#   - Field shapes are stable enough to be consumed by external tools / the agent.

from __future__ import annotations

import json
from fractions import Fraction

import pytest

from zaszlo import (
    AuxiliaryConstraint,
    DensityExpr,
    Flag,
    FlagAlgebraElement,
    FlagProblem,
    Hypergraph,
    UnlabeledExpr,
    complete,
    identify_sharps,
    solve,
)
from zaszlo.algebra import unlabel
from zaszlo.generation import generate_flags
from zaszlo.serialize import (
    encode_aux_constraint,
    encode_density_expr,
    encode_fraction,
    encode_hypergraph,
    encode_problem,
    encode_unlabeled_expr,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mantel_result():
    prob = FlagProblem(4, 2, 2, forbidden=[complete(3)])
    return solve(prob, certify=True)


@pytest.fixture(scope="module")
def mantel_sharps(mantel_result):
    return identify_sharps(mantel_result.data, mantel_result)


# ---------------------------------------------------------------------------
# Primitive encoders
# ---------------------------------------------------------------------------

class TestPrimitiveEncoders:
    def test_fraction(self):
        d = encode_fraction(Fraction(3, 7))
        assert d == {"num": 3, "den": 7}

    def test_fraction_negative(self):
        d = encode_fraction(Fraction(-2, 5))
        assert d == {"num": -2, "den": 5}

    def test_fraction_zero(self):
        d = encode_fraction(Fraction(0))
        assert d == {"num": 0, "den": 1}

    def test_hypergraph(self):
        g = complete(3)
        d = encode_hypergraph(g)
        assert d["n"] == 3 and d["k"] == 2
        assert d["edges"] == [[1, 2], [1, 3], [2, 3]]

    def test_density_expr(self):
        e = DensityExpr([(1, complete(2)), (-Fraction(1, 2), Hypergraph(2, 2, []))])
        d = encode_density_expr(e)
        assert d["kind"] == "DensityExpr"
        assert d["k"] == 2
        assert len(d["terms"]) == 2
        assert d["terms"][0]["coef"] == {"num": 1, "den": 1}
        assert d["terms"][1]["coef"] == {"num": -1, "den": 2}

    def test_unlabeled_expr(self):
        u = UnlabeledExpr(2, 3, [(2, complete(3))])
        d = encode_unlabeled_expr(u)
        assert d["kind"] == "UnlabeledExpr" and d["n"] == 3
        assert d["terms"] == [{"coef": {"num": 2, "den": 1}, "graph": encode_hypergraph(complete(3))}]


class TestProblemEncoding:
    def test_edge_density_target(self):
        p = FlagProblem(4, 2, 2, forbidden=[complete(3)])
        d = encode_problem(p)
        assert d["n"] == 4 and d["type_order"] == 2 and d["k"] == 2
        assert d["target"] == {"kind": "edge_density"}
        assert d["num_aux_constraints"] == 0
        assert len(d["forbidden"]) == 1

    def test_hypergraph_target(self):
        p = FlagProblem(5, 3, 2, forbidden=[complete(3)], target=complete(5))
        d = encode_problem(p)
        assert d["target"]["kind"] == "induced_density"
        assert d["target"]["graph"]["n"] == 5

    def test_density_expr_target(self):
        expr = DensityExpr([(1, complete(2)), (-Fraction(1, 2), Hypergraph(2, 2, []))])
        p = FlagProblem(4, 2, 2, target=expr)
        d = encode_problem(p)
        assert d["target"]["kind"] == "DensityExpr"

    def test_aux_constraints_metadata(self):
        prob = FlagProblem(4, 2, 2, forbidden=[complete(3)])
        u = unlabel(generate_flags(2, Flag(Hypergraph(0, 2, []), 0), [], [])[0] ** 2)
        prob.add_constraint(u)
        d = encode_problem(prob)
        assert d["num_aux_constraints"] == 1
        assert d["aux_constraints"][0]["kind"] == "AuxiliaryConstraint"
        assert d["aux_constraints"][0]["sense"] == ">=0"


# ---------------------------------------------------------------------------
# Certificate serialization
# ---------------------------------------------------------------------------

class TestCertificateToDict:
    def test_top_level_kind(self, mantel_result):
        d = mantel_result.certificate.to_dict()
        assert d["kind"] == "Certificate"

    def test_required_fields_present(self, mantel_result):
        d = mantel_result.certificate.to_dict()
        for key in [
            "problem", "bound", "bound_float", "valid",
            "active_constraints", "num_admissible", "residuals",
            "Q_dimensions", "Q", "mu",
        ]:
            assert key in d, f"missing key: {key}"

    def test_bound_is_num_den(self, mantel_result):
        d = mantel_result.certificate.to_dict()
        assert set(d["bound"].keys()) == {"num", "den"}
        assert d["bound"]["den"] > 0

    def test_residuals_length_matches_admissible_count(self, mantel_result):
        d = mantel_result.certificate.to_dict()
        assert len(d["residuals"]) == d["num_admissible"]

    def test_q_dimensions_match_num_types(self, mantel_result):
        d = mantel_result.certificate.to_dict()
        assert d["Q_dimensions"] == [2, 4, 3]  # Mantel: 3 types with these flag counts

    def test_q_matrices_are_lists_of_lists_of_rationals(self, mantel_result):
        d = mantel_result.certificate.to_dict()
        for Q, dim in zip(d["Q"], d["Q_dimensions"]):
            assert len(Q) == dim
            for row in Q:
                assert len(row) == dim
                for entry in row:
                    assert set(entry.keys()) == {"num", "den"}

    def test_valid_is_bool(self, mantel_result):
        assert isinstance(mantel_result.certificate.to_dict()["valid"], bool)

    def test_bound_float_matches_bound(self, mantel_result):
        d = mantel_result.certificate.to_dict()
        reconstructed = d["bound"]["num"] / d["bound"]["den"]
        assert abs(reconstructed - d["bound_float"]) < 1e-9


class TestCertificateToJson:
    def test_returns_string(self, mantel_result):
        s = mantel_result.certificate.to_json()
        assert isinstance(s, str)

    def test_valid_json(self, mantel_result):
        s = mantel_result.certificate.to_json()
        parsed = json.loads(s)
        assert parsed["kind"] == "Certificate"

    def test_round_trip_agrees_with_to_dict(self, mantel_result):
        d = mantel_result.certificate.to_dict()
        parsed = json.loads(mantel_result.certificate.to_json())
        # Keys should be identical (sort_keys=True); values should match.
        assert set(parsed.keys()) == set(d.keys())


# ---------------------------------------------------------------------------
# FlagAlgebraResult serialization
# ---------------------------------------------------------------------------

class TestResultToDict:
    def test_kind(self, mantel_result):
        assert mantel_result.to_dict()["kind"] == "FlagAlgebraResult"

    def test_certificate_nested_when_present(self, mantel_result):
        d = mantel_result.to_dict()
        assert d["certificate"] is not None
        assert d["certificate"]["kind"] == "Certificate"

    def test_certificate_null_when_absent(self):
        prob = FlagProblem(4, 2, 2, forbidden=[complete(3)])
        r = solve(prob)  # no certify
        assert r.to_dict()["certificate"] is None

    def test_mu_lists(self, mantel_result):
        # No aux constraints -> mu is []
        d = mantel_result.to_dict()
        assert d["mu"] == []


# ---------------------------------------------------------------------------
# SharpsResult serialization
# ---------------------------------------------------------------------------

class TestSharpsToDict:
    def test_kind(self, mantel_sharps):
        assert mantel_sharps.to_dict()["kind"] == "SharpsResult"

    def test_graphs_are_hypergraphs(self, mantel_sharps):
        d = mantel_sharps.to_dict()
        for g in d["graphs"]:
            assert set(g.keys()) == {"n", "k", "edges"}

    def test_densities_are_rationals(self, mantel_sharps):
        d = mantel_sharps.to_dict()
        for r in d["densities"]:
            assert set(r.keys()) == {"num", "den"}


# ---------------------------------------------------------------------------
# Aux constraint serialization end-to-end
# ---------------------------------------------------------------------------

class TestAuxConstraintSerialization:
    @pytest.fixture(scope="class")
    def solved_with_aux(self):
        prob = FlagProblem(4, 2, 2, forbidden=[complete(3)])
        u = unlabel(generate_flags(2, Flag(Hypergraph(0, 2, []), 0), [], [])[0] ** 2)
        prob.add_constraint(u)
        return solve(prob, certify=True)

    def test_certificate_mu_serialized(self, solved_with_aux):
        d = solved_with_aux.certificate.to_dict()
        assert len(d["mu"]) == 1
        assert set(d["mu"][0].keys()) == {"num", "den"}

    def test_result_mu_serialized(self, solved_with_aux):
        d = solved_with_aux.to_dict()
        assert len(d["mu"]) == 1  # one aux → one mu

    def test_problem_carries_aux_constraint_metadata(self, solved_with_aux):
        d = solved_with_aux.to_dict()
        assert d["problem"]["num_aux_constraints"] == 1
        assert d["problem"]["aux_constraints"][0]["sense"] == ">=0"
