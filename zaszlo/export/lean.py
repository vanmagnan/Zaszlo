"""Export a :class:`~zaszlo.types.Certificate` as a Flagmatic-format JSON
consumable by the ``flag_certificate`` Lean tactic of taeyool's
``lean-flag-algebras-release``.

Reference implementation and target schema:

- Jeong, Park, Hyun, Oum, Yang. *Formalizing Flag Algebras in Lean.*
  arXiv:2607.23500.
- https://github.com/taeyool/lean-flag-algebras-release (Apache-2.0),
  specifically ``LeanFlagAlgebras/Flagmatic/flagmatic_to_lean.py`` and the
  ``Certificates/*.json`` fixtures.

The taeyool tactic elaborates a schema-conformant JSON into an axiom-free
Lean proof of the density bound. Its envelope: 2-graphs, maximize
direction, a single non-induced forbidden pattern, and no auxiliary
constraints. Certificates outside that envelope raise :class:`LeanExportError`.

The taeyool schema decomposes each per-type PSD block as ``M_t = R_t Q'_t R_tᵀ``
and re-derives ``M_t`` at load time. Since :attr:`Certificate.Q` already stores
the assembled ``M_t`` in exact rationals, we emit ``R_t = I`` and place the
upper triangle of ``M_t`` into ``qdash_matrices``. The tactic then LDLᵀ-factors
``M_t`` on the Lean side to produce the PSD witness.
"""

from __future__ import annotations

import json
from fractions import Fraction
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..types import Certificate, Flag, FlagProblem, Hypergraph


class LeanExportError(ValueError):
    """Raised when a Certificate cannot be expressed in the taeyool schema."""


# ---------------------------------------------------------------------------
# Primitive encoders
# ---------------------------------------------------------------------------

def flagmatic_bound(f: Fraction) -> int | str:
    """Encode a rational as an ``int`` (when integral) or ``"num/den"`` string.

    Matches the ``_parse_rat`` reader in ``flagmatic_to_lean.py``.
    """
    if not isinstance(f, Fraction):
        raise LeanExportError(
            f"flagmatic_bound expects a Fraction; got {type(f).__name__}"
        )
    if f.denominator == 1:
        return f.numerator
    return f"{f.numerator}/{f.denominator}"


def flagmatic_graph_string(g: "Hypergraph") -> str:
    """Encode a k=2 hypergraph as ``"n:edges"`` (Flagmatic convention).

    Vertices are 1-indexed and rendered as single digits, so ``n ≤ 9`` is
    required — matching the taeyool corpus (largest case is n=5). Edges
    are the sorted pairs concatenated: e.g. K3 becomes ``"3:121323"``.
    """
    if g.k != 2:
        raise LeanExportError(
            f"Flagmatic string encoding requires k=2 (ordinary graphs); "
            f"got k={g.k}. The taeyool Lean library is written on SimpleGraph "
            f"and does not consume hypergraph certificates."
        )
    if g.n > 9:
        raise LeanExportError(
            f"Flagmatic single-digit encoding requires n ≤ 9; got n={g.n}. "
            f"Multi-digit vertex labels would produce ambiguous edge strings."
        )
    body = "".join(f"{u}{v}" for u, v in g.edges)
    return f"{g.n}:{body}"


def flagmatic_flag_string(f: "Flag") -> str:
    """Encode a σ-flag as ``"n:edges(s)"`` (Flagmatic convention).

    The parenthesized tail is the type size; the first ``s`` vertices are the
    labeled/type vertices.
    """
    return f"{flagmatic_graph_string(f.graph)}({f.type_size})"


# ---------------------------------------------------------------------------
# Description string
# ---------------------------------------------------------------------------

def _target_description(prob: "FlagProblem") -> str:
    """Return the ``maximize <string> density`` fragment of the description."""
    from ..types import DensityExpr, Hypergraph
    if prob.target is None:
        # Edge density = induced density of K2.
        return "2:12"
    if isinstance(prob.target, Hypergraph):
        return flagmatic_graph_string(prob.target)
    if isinstance(prob.target, DensityExpr):
        raise LeanExportError(
            "The taeyool schema does not encode DensityExpr (linear-combination) "
            "targets. Use a single Hypergraph or None (edge density) as target."
        )
    raise LeanExportError(
        f"unrecognized target type: {type(prob.target).__name__}"
    )


def _description(prob: "FlagProblem") -> str:
    """Assemble the top-level ``description`` field.

    Format: ``"<k>-graph; maximize <target> density; forbid <pattern>"``.
    """
    target_str = _target_description(prob)
    forbid_str = flagmatic_graph_string(prob.forbidden[0])
    return f"{prob.k}-graph; maximize {target_str} density; forbid {forbid_str}"


# ---------------------------------------------------------------------------
# Envelope check
# ---------------------------------------------------------------------------

def _check_envelope(cert: "Certificate") -> None:
    """Raise :class:`LeanExportError` if the certificate is outside the
    taeyool tactic's supported envelope."""
    from ..types import DensityExpr

    prob = cert.problem

    if prob.k != 2:
        raise LeanExportError(
            f"Lean export requires k=2 (the taeyool Lean library targets "
            f"SimpleGraph); this problem has k={prob.k}."
        )
    if prob.minimize:
        raise LeanExportError(
            "Lean export requires the maximize direction; this problem sets "
            "minimize=True."
        )
    if isinstance(prob.target, DensityExpr):
        raise LeanExportError(
            "Lean export cannot encode DensityExpr targets — the taeyool "
            "description grammar is 'maximize <graph> density; forbid ...'."
        )
    if not prob.forbidden:
        raise LeanExportError(
            "Lean export requires at least one non-induced forbidden pattern "
            "(prob.forbidden is empty)."
        )
    if len(prob.forbidden) > 1:
        raise LeanExportError(
            f"Lean export supports a single forbidden pattern; got "
            f"{len(prob.forbidden)} in prob.forbidden. Remove redundant "
            "patterns (e.g. K3 already implies K4-freeness) before exporting."
        )
    if prob.forbidden_induced:
        raise LeanExportError(
            "Lean export does not support induced-forbidden patterns; move "
            "the pattern to prob.forbidden if it forbids all non-induced "
            "copies, or drop it."
        )
    if prob.aux_constraints:
        raise LeanExportError(
            f"Lean export does not encode auxiliary flag-algebra constraints "
            f"(the taeyool tactic has no schema slot for μ · ⟦e⟧). This "
            f"certificate carries {len(prob.aux_constraints)} aux constraint(s)."
        )
    if not cert.valid:
        raise LeanExportError(
            "Refusing to export an invalid certificate (cert.valid is False). "
            "The taeyool tactic would fail elaboration; fix rounding first "
            "(raise denom_limit or inspect cert.residuals)."
        )
    if cert.data is None:
        raise LeanExportError(
            "Certificate.data is None — the exporter needs the pre-SDP data "
            "(types, flags, admissible list, densities) to build the schema. "
            "Certify via solve(prob, certify=True) so data is attached."
        )
    empty_type_blocks = [t for t in cert.data.types if t.type_size == 0]
    if empty_type_blocks:
        raise LeanExportError(
            f"Lean export requires all SDP blocks to have a non-empty type "
            f"(type_size ≥ 1). This certificate has {len(empty_type_blocks)} "
            f"block(s) for the empty type (σ = ∅, encoded as '0:' in Flagmatic), "
            f"which the flag_certificate tactic does not support. "
            f"Re-solve with an odd type_order — e.g. type_order=1 for Mantel "
            f"(K₃-free), type_order=3 for K₅-free. Use the corpus entry's "
            f"lean_problem_factory if one is provided."
        )


# ---------------------------------------------------------------------------
# Q -> upper-triangular row-major
# ---------------------------------------------------------------------------

def _upper_triangle_rows(Q: list[list[Fraction]]) -> list[list[int | str]]:
    """Return ``[[Q[i,i], Q[i,i+1], …, Q[i,n-1]] for i in range(n)]``.

    Rationals are encoded via :func:`flagmatic_bound`.
    """
    n = len(Q)
    for i, row in enumerate(Q):
        if len(row) != n:
            raise LeanExportError(
                f"Q matrix is not square: row {i} has {len(row)} entries, "
                f"expected {n}."
            )
    return [[flagmatic_bound(Q[i][j]) for j in range(i, n)] for i in range(n)]


def _identity_r_matrix(n: int) -> list[list[int]]:
    """The ``n × n`` identity, encoded as a full list-of-lists.

    Emitted as ``R_t`` so that the taeyool loader reconstructs ``M_t = Q'_t``.
    """
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def to_flagmatic_certificate(cert: "Certificate") -> dict[str, Any]:
    """Return a taeyool-schema JSON-ready dict for this certificate.

    Raises
    ------
    LeanExportError
        If the certificate is outside the taeyool tactic's envelope (see
        :func:`_check_envelope`) or contains data the encoders reject
        (e.g. n > 9 vertices).

    The emitted dict has the following top-level keys, in the order the
    taeyool README documents them::

        description, bound,
        order_of_admissible_graphs, number_of_admissible_graphs,
        admissible_graphs,
        number_of_types, types,
        numbers_of_flags, flags,
        qdash_matrices, r_matrices,
        admissible_graph_densities
    """
    _check_envelope(cert)
    data = cert.data
    assert data is not None  # guaranteed by _check_envelope
    prob = cert.problem

    admissible_strs = [flagmatic_graph_string(g) for g in data.admissible]
    type_strs = [flagmatic_graph_string(t.graph) for t in data.types]
    flag_strs = [
        [flagmatic_flag_string(f) for f in flags_for_type]
        for flags_for_type in data.flags
    ]

    qdash = [_upper_triangle_rows(Q) for Q in cert.Q]
    r_mats = [_identity_r_matrix(len(Q)) for Q in cert.Q]

    return {
        "description": _description(prob),
        "bound": flagmatic_bound(cert.bound),
        "order_of_admissible_graphs": prob.n,
        "number_of_admissible_graphs": len(data.admissible),
        "admissible_graphs": admissible_strs,
        "number_of_types": len(data.types),
        "types": type_strs,
        "numbers_of_flags": [len(fs) for fs in data.flags],
        "flags": flag_strs,
        "qdash_matrices": qdash,
        "r_matrices": r_mats,
        "admissible_graph_densities": [flagmatic_bound(d) for d in data.densities],
    }


def write_flagmatic_certificate(
    cert: "Certificate", path: str, *, indent: int | None = 4
) -> None:
    """Write ``cert`` as a taeyool-schema JSON file at ``path``.

    ``indent=4`` matches the formatting of the committed fixtures in
    ``LeanFlagAlgebras/Flagmatic/Certificates/`` for easier diffing.
    """
    payload = to_flagmatic_certificate(cert)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, ensure_ascii=False)
        if indent is not None:
            f.write("\n")
