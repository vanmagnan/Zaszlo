"""Tests for the Lean/Flagmatic-schema exporter.

The Lean tactic in taeyool/lean-flag-algebras-release consumes a specific
JSON format. This file has three tiers of tests:

1. Unit tests on the string / rational encoders.
2. Envelope tests — every out-of-scope situation raises LeanExportError.
3. End-to-end tests that build a real Zászló Certificate (from the corpus)
   and put its exported JSON through a local re-implementation of the
   taeyool loader's parsing (see ``_taeyool_parse_*`` helpers). This is
   the strongest check we can do without Lean installed: if the schema
   drifts, or if our per-type Q blocks fail to be PSD in exact ℚ, one
   of these tests catches it.

The taeyool committed Mantel certificate is checked in at
``LeanFlagAlgebras/Flagmatic/Certificates/Mantel_cert.json`` in their
repo; here we hard-code the parts of it we need to compare against so
this file is self-contained and doesn't require the taeyool repo be
present locally.
"""

from __future__ import annotations

import json
from fractions import Fraction

import pytest

from zaszlo import corpus
from zaszlo.export.lean import (
    LeanExportError,
    flagmatic_bound,
    flagmatic_flag_string,
    flagmatic_graph_string,
    to_flagmatic_certificate,
    write_flagmatic_certificate,
)
from zaszlo.pipeline import build_flag_algebra_data
from zaszlo.sdp import certify_at_bound
from zaszlo.types import (
    Flag,
    Hypergraph,
    complete,
    k4_minus,
)


# ---------------------------------------------------------------------------
# Local re-implementation of the taeyool JSON loader's parsing steps.
# Kept minimal and independent of the Zászló import so this file catches
# accidental schema drift. Sourced from ``flagmatic_to_lean.py`` in
# taeyool/lean-flag-algebras-release (Apache-2.0).
# ---------------------------------------------------------------------------

def _taeyool_parse_rat(x) -> Fraction:
    if isinstance(x, int):
        return Fraction(x)
    if isinstance(x, str):
        if "/" in x:
            num, den = x.split("/")
            return Fraction(int(num), int(den))
        return Fraction(int(x))
    raise TypeError(f"cannot parse {x!r} as rational")


def _taeyool_parse_qdash(qdash: list[list]) -> list[list[Fraction]]:
    n = len(qdash)
    for i, row in enumerate(qdash):
        if len(row) != n - i:
            raise ValueError(
                f"row {i} has {len(row)} entries; upper-triangular row-major "
                f"expects {n - i}"
            )
    M = [[Fraction(0)] * n for _ in range(n)]
    for i, row in enumerate(qdash):
        for k, val in enumerate(row):
            j = i + k
            v = _taeyool_parse_rat(val)
            M[i][j] = v
            M[j][i] = v
    return M


def _taeyool_matmul(A, B):
    n = len(A)
    p = len(A[0]) if A else 0
    q = len(B[0]) if B else 0
    if any(len(row) != p for row in A) or any(len(row) != q for row in B) or len(B) != p:
        raise ValueError("matmul dimension mismatch")
    return [
        [sum(A[i][k] * B[k][j] for k in range(p)) for j in range(q)]
        for i in range(n)
    ]


def _taeyool_transpose(A):
    if not A:
        return []
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]


def _taeyool_assemble_block(qdash: list[list], r: list[list]) -> list[list[Fraction]]:
    """M = R · Q' · Rᵀ (with the upper-triangular Q' parsed to symmetric)."""
    Q = _taeyool_parse_qdash(qdash)
    R = [[_taeyool_parse_rat(x) for x in row] for row in r]
    return _taeyool_matmul(_taeyool_matmul(R, Q), _taeyool_transpose(R))


def _taeyool_ldl(M: list[list[Fraction]]) -> tuple[list[list[Fraction]], list[Fraction]]:
    """Exact ℚ LDLᵀ. Returns (L, D). Raises if a diagonal is negative."""
    n = len(M)
    for row in M:
        if len(row) != n:
            raise ValueError("M is not square")
    L = [[Fraction(0)] * n for _ in range(n)]
    D = [Fraction(0)] * n
    for i in range(n):
        L[i][i] = Fraction(1)
    for j in range(n):
        s = M[j][j] - sum(L[j][k] * L[j][k] * D[k] for k in range(j))
        if s < 0:
            raise ValueError(f"LDL failed: D[{j}] = {s} < 0 (M not PSD)")
        D[j] = s
        for i in range(j + 1, n):
            if D[j] == 0:
                # M PSD with zero diagonal ⇒ off-diagonal must also be zero.
                offdiag = M[i][j] - sum(L[i][k] * L[j][k] * D[k] for k in range(j))
                if offdiag != 0:
                    raise ValueError(
                        f"LDL failed at zero pivot D[{j}]: L[{i},{j}]·D[{j}] "
                        f"residual = {offdiag} ≠ 0 (M not PSD in this direction)"
                    )
                L[i][j] = Fraction(0)
            else:
                L[i][j] = (
                    M[i][j] - sum(L[i][k] * L[j][k] * D[k] for k in range(j))
                ) / D[j]
    return L, D


# ---------------------------------------------------------------------------
# 1. Encoder unit tests
# ---------------------------------------------------------------------------

class TestFlagmaticBound:
    def test_integer(self):
        assert flagmatic_bound(Fraction(2)) == 2

    def test_negative_integer(self):
        assert flagmatic_bound(Fraction(-3)) == -3

    def test_rational(self):
        assert flagmatic_bound(Fraction(1, 2)) == "1/2"
        assert flagmatic_bound(Fraction(24, 625)) == "24/625"

    def test_negative_rational_keeps_sign_in_numerator(self):
        assert flagmatic_bound(Fraction(-1, 2)) == "-1/2"

    def test_rejects_float(self):
        with pytest.raises(LeanExportError):
            flagmatic_bound(0.5)  # type: ignore[arg-type]


class TestFlagmaticGraphString:
    def test_empty_on_three_vertices(self):
        assert flagmatic_graph_string(Hypergraph(3, 2, [])) == "3:"

    def test_k3(self):
        # K3 vertices 1,2,3 with edges 12, 13, 23 → "3:121323".
        assert flagmatic_graph_string(complete(3)) == "3:121323"

    def test_p3_matches_taeyool_admissible(self):
        # P3 (path on 3 vertices, center = 1): edges 12, 13 → "3:1213".
        p3 = Hypergraph(3, 2, [(1, 2), (1, 3)])
        assert flagmatic_graph_string(p3) == "3:1213"

    def test_single_edge_on_2(self):
        # Edge = induced K2 → "2:12".
        assert flagmatic_graph_string(complete(2)) == "2:12"

    def test_rejects_hypergraph(self):
        with pytest.raises(LeanExportError, match="k=2"):
            flagmatic_graph_string(k4_minus())

    def test_rejects_n_over_9(self):
        with pytest.raises(LeanExportError, match="n ≤ 9"):
            flagmatic_graph_string(Hypergraph(10, 2, []))


class TestFlagmaticFlagString:
    def test_type_1_flag(self):
        # 2-vertex graph, edge 12, first 1 vertex is type: "2:12(1)".
        f = Flag(Hypergraph(2, 2, [(1, 2)]), type_size=1)
        assert flagmatic_flag_string(f) == "2:12(1)"

    def test_type_1_empty_flag(self):
        f = Flag(Hypergraph(2, 2, []), type_size=1)
        assert flagmatic_flag_string(f) == "2:(1)"


# ---------------------------------------------------------------------------
# 2. Envelope tests
# ---------------------------------------------------------------------------

def _lean_cert(slug: str):
    """Build an exact-at-bound certificate using Lean-compatible params.

    Uses the entry's ``lean_problem_factory`` when available (to avoid
    empty-type blocks that the taeyool tactic does not support), falling
    back to ``problem_factory`` otherwise.
    """
    entry = corpus.get(slug)
    factory = entry.lean_problem_factory or entry.problem_factory
    prob = factory()
    data = build_flag_algebra_data(prob)
    return certify_at_bound(data, entry.expected_bound)


def _corpus_cert(slug: str):
    """Build a cert using the entry's corpus (non-Lean) params."""
    entry = corpus.get(slug)
    prob = entry.problem_factory()
    data = build_flag_algebra_data(prob)
    return certify_at_bound(data, entry.expected_bound)


@pytest.fixture(scope="module")
def mantel_cert():
    return _lean_cert("mantel")


class TestEnvelope:
    def test_valid_mantel_passes(self, mantel_cert):
        # Should not raise.
        to_flagmatic_certificate(mantel_cert)

    def test_rejects_minimize(self, mantel_cert):
        mantel_cert.problem.minimize = True
        with pytest.raises(LeanExportError, match="maximize"):
            to_flagmatic_certificate(mantel_cert)
        mantel_cert.problem.minimize = False  # restore

    def test_rejects_empty_type_block(self):
        # Mantel at corpus params (n=4, type_order=2) generates a '0:' block.
        cert = _corpus_cert("mantel")
        with pytest.raises(LeanExportError, match="empty type"):
            to_flagmatic_certificate(cert)

    def test_rejects_multi_forbid(self):
        cert = _corpus_cert("mantel_multi_forbidden")
        with pytest.raises(LeanExportError, match="single forbidden pattern"):
            to_flagmatic_certificate(cert)

    def test_rejects_aux_constraints(self):
        cert = _corpus_cert("mantel_with_aux_sos")
        with pytest.raises(LeanExportError, match="auxiliary flag-algebra"):
            to_flagmatic_certificate(cert)

    def test_rejects_k3_hypergraph(self):
        cert = _corpus_cert("k4_minus_free_3graphs")
        with pytest.raises(LeanExportError, match="k=2"):
            to_flagmatic_certificate(cert)

    def test_rejects_density_expr_target(self):
        cert = _corpus_cert("affine_mantel")
        with pytest.raises(LeanExportError, match="DensityExpr"):
            to_flagmatic_certificate(cert)


# ---------------------------------------------------------------------------
# 3. End-to-end schema tests on Mantel
# ---------------------------------------------------------------------------

class TestMantelExport:
    def test_top_level_keys(self, mantel_cert):
        payload = to_flagmatic_certificate(mantel_cert)
        assert set(payload.keys()) == {
            "description",
            "bound",
            "order_of_admissible_graphs",
            "number_of_admissible_graphs",
            "admissible_graphs",
            "number_of_types",
            "types",
            "numbers_of_flags",
            "flags",
            "qdash_matrices",
            "r_matrices",
            "admissible_graph_densities",
        }

    def test_description_matches_taeyool_fixture(self, mantel_cert):
        # Taeyool Mantel_cert.json committed value.
        payload = to_flagmatic_certificate(mantel_cert)
        assert payload["description"] == "2-graph; maximize 2:12 density; forbid 3:121323"

    def test_bound_is_one_half(self, mantel_cert):
        payload = to_flagmatic_certificate(mantel_cert)
        assert payload["bound"] == "1/2"

    def test_json_roundtrip(self, mantel_cert):
        payload = to_flagmatic_certificate(mantel_cert)
        # No non-serializable types.
        assert json.loads(json.dumps(payload)) == payload

    def test_numbers_of_flags_matches_qdash_dims(self, mantel_cert):
        payload = to_flagmatic_certificate(mantel_cert)
        for i, block in enumerate(payload["qdash_matrices"]):
            n = payload["numbers_of_flags"][i]
            assert len(block) == n
            for row_idx, row in enumerate(block):
                # Upper-triangular row-major: row i has n-i entries.
                assert len(row) == n - row_idx

    def test_r_matrices_are_identity(self, mantel_cert):
        payload = to_flagmatic_certificate(mantel_cert)
        for i, R in enumerate(payload["r_matrices"]):
            n = payload["numbers_of_flags"][i]
            expected = [[1 if a == b else 0 for b in range(n)] for a in range(n)]
            assert R == expected

    def test_admissible_and_types_have_correct_length(self, mantel_cert):
        payload = to_flagmatic_certificate(mantel_cert)
        assert payload["number_of_admissible_graphs"] == len(payload["admissible_graphs"])
        assert payload["number_of_types"] == len(payload["types"])
        assert len(payload["numbers_of_flags"]) == payload["number_of_types"]
        assert len(payload["flags"]) == payload["number_of_types"]
        assert len(payload["qdash_matrices"]) == payload["number_of_types"]
        assert len(payload["r_matrices"]) == payload["number_of_types"]
        assert len(payload["admissible_graph_densities"]) == payload["number_of_admissible_graphs"]

    def test_write_file(self, mantel_cert, tmp_path):
        out = tmp_path / "Mantel_cert.json"
        write_flagmatic_certificate(mantel_cert, str(out))
        reloaded = json.loads(out.read_text())
        assert reloaded["description"].startswith("2-graph;")


# ---------------------------------------------------------------------------
# 3b. Round-trip through the taeyool loader parsing (schema + PSD check)
# ---------------------------------------------------------------------------

class TestTaeyoolLoaderRoundTrip:
    """Exercise the same parsing path the taeyool tactic uses at load time.

    If any of these fails, the taeyool tactic would also fail elaboration.
    """

    def test_bound_parses(self, mantel_cert):
        payload = to_flagmatic_certificate(mantel_cert)
        assert _taeyool_parse_rat(payload["bound"]) == Fraction(1, 2)

    def test_densities_parse(self, mantel_cert):
        payload = to_flagmatic_certificate(mantel_cert)
        for d in payload["admissible_graph_densities"]:
            _ = _taeyool_parse_rat(d)  # should not raise

    def test_blocks_ldlt_decompose_in_exact_Q(self, mantel_cert):
        payload = to_flagmatic_certificate(mantel_cert)
        for i, (qdash, R) in enumerate(zip(
            payload["qdash_matrices"], payload["r_matrices"]
        )):
            M = _taeyool_assemble_block(qdash, R)
            # LDLᵀ succeeds iff M is PSD in exact ℚ; this is what the taeyool
            # tactic relies on to synthesize the PSD witness on the Lean side.
            _, D = _taeyool_ldl(M)
            assert all(d >= 0 for d in D), f"block {i} diagonal has negative entry: {D}"

    def test_M_equals_cert_Q_when_R_is_identity(self, mantel_cert):
        # When R = I, assembled M_t should equal cert.Q[i] exactly.
        payload = to_flagmatic_certificate(mantel_cert)
        for i, (qdash, R) in enumerate(zip(
            payload["qdash_matrices"], payload["r_matrices"]
        )):
            M = _taeyool_assemble_block(qdash, R)
            assert M == mantel_cert.Q[i], f"block {i} mismatch: M != cert.Q[{i}]"
