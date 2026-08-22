# Pipeline integrity tests.
#
# Verify that the full solve → certify → identify_sharps → diagnose → to_json
# chain runs without error and produces structurally valid output.
# Mathematical correctness (bound values, exact Fractions) is the
# responsibility of test_sdp.py and test_corpus.py.
#
# Run:           pytest tests/test_e2e.py -v
# Skip in CI:    pytest tests/ -m "not slow"

from __future__ import annotations

import json

import pytest

from zaszlo import corpus, identify_sharps, solve
from zaszlo.diagnose import DiagnosticReport

pytestmark = pytest.mark.slow

# Four entries covering distinct code paths:
#   mantel               — baseline k=2 graph, edge-density target
#   k4_minus_free_3graphs — k=3 hypergraph
#   affine_mantel        — DensityExpr target
#   mantel_with_aux_sos  — AuxiliaryConstraint path
_E2E_SLUGS = [
    "mantel",
    "k4_minus_free_3graphs",
    "affine_mantel",
    "mantel_with_aux_sos",
]


@pytest.fixture(scope="module", params=_E2E_SLUGS)
def pipeline(request):
    entry = corpus.get(request.param)
    result = solve(entry.problem_factory(), certify=True)
    sharps = identify_sharps(result.data, result)
    return result, sharps


class TestCertificate:
    def test_valid(self, pipeline):
        result, _ = pipeline
        assert result.certificate.valid

    def test_residuals_all_nonneg(self, pipeline):
        result, _ = pipeline
        assert all(r >= 0 for r in result.certificate.residuals)

    def test_active_constraints_nonempty(self, pipeline):
        result, _ = pipeline
        assert len(result.certificate.active_constraints) > 0

    def test_json_parses(self, pipeline):
        result, _ = pipeline
        parsed = json.loads(result.certificate.to_json())
        assert parsed["kind"] == "Certificate"

    def test_json_has_required_keys(self, pipeline):
        result, _ = pipeline
        parsed = json.loads(result.certificate.to_json())
        for key in ("bound", "valid", "residuals", "Q", "active_constraints"):
            assert key in parsed


class TestSharps:
    def test_nonempty(self, pipeline):
        _, sharps = pipeline
        assert len(sharps.indices) > 0

    def test_json_parses(self, pipeline):
        _, sharps = pipeline
        parsed = json.loads(sharps.to_json())
        assert parsed["kind"] == "SharpsResult"

    def test_json_has_required_keys(self, pipeline):
        _, sharps = pipeline
        parsed = json.loads(sharps.to_json())
        for key in ("graphs", "densities", "indices"):
            assert key in parsed


class TestDiagnose:
    def test_returns_diagnostic_report(self, pipeline):
        result, _ = pipeline
        assert isinstance(result.diagnose(), DiagnosticReport)

    def test_reports_optimal(self, pipeline):
        result, _ = pipeline
        assert result.diagnose().is_optimal

    def test_reports_valid(self, pipeline):
        result, _ = pipeline
        assert result.diagnose().is_valid

    def test_json_parses(self, pipeline):
        result, _ = pipeline
        parsed = json.loads(result.diagnose().to_json())
        assert parsed["kind"] == "FlagAlgebraResult"

    def test_json_has_required_keys(self, pipeline):
        result, _ = pipeline
        parsed = json.loads(result.diagnose().to_json())
        for key in ("status", "bound_float", "num_admissible", "sharp_indices", "Q_summary"):
            assert key in parsed
