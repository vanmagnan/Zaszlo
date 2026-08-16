"""Structured JSON / dict serialization for user-facing result objects.

Exact rationals are encoded as ``{"num": int, "den": int}`` to preserve
round-trip fidelity across languages without depending on Python's
:class:`fractions.Fraction`.  Hypergraphs, flags, and problem specifications
carry enough metadata that a downstream consumer can reconstruct the
mathematical claim without needing to load Zászló itself.

Convention
----------
Every top-level export includes a ``"kind"`` field identifying the class
(``"Certificate"``, ``"FlagAlgebraResult"``, ``"SharpsResult"``,
``"DiagnosticReport"``) so consumers can dispatch generically.

Scope
-----
Export is **one-way**.  There is no ``from_dict`` in this pass — the exact-
rational Q matrices are large, and no consumer of the agent workflow needs
to reload a certificate into Python objects.  Downstream verification tools
consume the dict/JSON directly.
"""

from __future__ import annotations

import json
from fractions import Fraction
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .types import (
        AuxiliaryConstraint,
        Certificate,
        DensityExpr,
        Flag,
        FlagAlgebraResult,
        FlagProblem,
        Hypergraph,
        SharpsResult,
        UnlabeledExpr,
    )


# ---------------------------------------------------------------------------
# Primitive encoders
# ---------------------------------------------------------------------------

def encode_fraction(f: Fraction) -> dict[str, int]:
    """Encode an exact Fraction as {"num", "den"}."""
    return {"num": f.numerator, "den": f.denominator}


def encode_hypergraph(g: "Hypergraph") -> dict[str, Any]:
    """Encode a Hypergraph as {"n", "k", "edges"} with edges as lists of ints."""
    return {
        "n": g.n,
        "k": g.k,
        "edges": [list(e) for e in g.edges],
    }


def encode_flag(f: "Flag") -> dict[str, Any]:
    """Encode a Flag as {"graph", "type_size"}."""
    return {
        "graph": encode_hypergraph(f.graph),
        "type_size": f.type_size,
    }


def encode_density_expr(expr: "DensityExpr") -> dict[str, Any]:
    """Encode a DensityExpr as {"kind":"DensityExpr", "k", "terms":[{"coef","graph"}]}."""
    return {
        "kind": "DensityExpr",
        "k": expr.k,
        "terms": [
            {"coef": encode_fraction(c), "graph": encode_hypergraph(g)}
            for c, g in expr.terms
        ],
    }


def encode_unlabeled_expr(u: "UnlabeledExpr") -> dict[str, Any]:
    """Encode an UnlabeledExpr as {"kind","k","n","terms":[{"coef","graph"}]}."""
    return {
        "kind": "UnlabeledExpr",
        "k": u.k,
        "n": u.n,
        "terms": [
            {"coef": encode_fraction(c), "graph": encode_hypergraph(g)}
            for c, g in u.terms
        ],
    }


def encode_aux_constraint(c: "AuxiliaryConstraint") -> dict[str, Any]:
    return {
        "kind": "AuxiliaryConstraint",
        "sense": c.sense,
        "expr": encode_unlabeled_expr(c.expr),
    }


def encode_problem(p: "FlagProblem") -> dict[str, Any]:
    """Encode a FlagProblem's structural spec (not the SDP data itself)."""
    from .types import DensityExpr, Hypergraph
    if p.target is None:
        target_repr: Any = {"kind": "edge_density"}
    elif isinstance(p.target, Hypergraph):
        target_repr = {"kind": "induced_density", "graph": encode_hypergraph(p.target)}
    elif isinstance(p.target, DensityExpr):
        target_repr = encode_density_expr(p.target)
    else:  # defensive; FlagProblem.__init__ already rejects other types
        target_repr = {"kind": "unknown", "repr": repr(p.target)}
    return {
        "n": p.n,
        "type_order": p.type_order,
        "k": p.k,
        "minimize": p.minimize,
        "forbidden": [encode_hypergraph(g) for g in p.forbidden],
        "forbidden_induced": [encode_hypergraph(g) for g in p.forbidden_induced],
        "target": target_repr,
        "num_aux_constraints": len(p.aux_constraints),
        "aux_constraints": [encode_aux_constraint(c) for c in p.aux_constraints],
    }


def _encode_matrix_exact(mat: list[list[Fraction]]) -> list[list[dict[str, int]]]:
    return [[encode_fraction(x) for x in row] for row in mat]


# ---------------------------------------------------------------------------
# Result-object encoders
# ---------------------------------------------------------------------------

def certificate_to_dict(c: "Certificate") -> dict[str, Any]:
    """Encode a Certificate as a JSON-ready dict.

    Includes the full rational Q matrices and per-graph residuals.  For very
    large certificates the payload can be substantial — filter downstream if
    only the summary fields (bound, valid, active_constraints) are needed.
    """
    problem = encode_problem(c.problem)
    admissible = c.data.admissible if c.data is not None else None
    return {
        "kind": "Certificate",
        "problem": problem,
        "bound": encode_fraction(c.bound),
        "bound_float": float(c.bound),
        "valid": c.valid,
        "active_constraints": list(c.active_constraints),
        "num_admissible": len(c.residuals),
        "residuals": [encode_fraction(r) for r in c.residuals],
        "Q_dimensions": [len(Q) for Q in c.Q],
        "Q": [_encode_matrix_exact(Q) for Q in c.Q],
        "mu": [encode_fraction(m) for m in c.mu],
        "admissible_graphs": (
            [encode_hypergraph(H) for H in admissible]
            if admissible is not None else None
        ),
    }


def result_to_dict(r: "FlagAlgebraResult") -> dict[str, Any]:
    """Encode a FlagAlgebraResult as a JSON-ready dict.

    The float Q matrices are included when extracted; exact Q (post-rounding)
    lives on the attached Certificate if any and can be retrieved separately.
    """
    Q_float = None if r.Q is None else [[list(row) for row in Q.tolist()] for Q in r.Q]
    Q_dims = None if r.Q is None else [Q.shape[0] for Q in r.Q]
    return {
        "kind": "FlagAlgebraResult",
        "problem": encode_problem(r.problem),
        "status": r.status,
        "bound_float": r.bound,
        "bound_exact": encode_fraction(r.bound_exact) if r.bound_exact is not None else None,
        "slacks": list(r.slacks),
        "mu": list(r.mu),
        "mu_exact": (
            [encode_fraction(m) for m in r.mu_exact]
            if r.mu_exact is not None else None
        ),
        "Q_dimensions": Q_dims,
        "Q_float": Q_float,
        "certificate": (
            certificate_to_dict(r.certificate) if r.certificate is not None else None
        ),
    }


def sharps_to_dict(s: "SharpsResult") -> dict[str, Any]:
    """Encode a SharpsResult as a JSON-ready dict."""
    return {
        "kind": "SharpsResult",
        "problem": encode_problem(s.problem),
        "num_sharps": len(s.indices),
        "indices": list(s.indices),
        "graphs": [encode_hypergraph(g) for g in s.graphs],
        "densities": [encode_fraction(d) for d in s.densities],
        "residuals": [encode_fraction(r) for r in s.residuals],
    }


# ---------------------------------------------------------------------------
# JSON wrappers
# ---------------------------------------------------------------------------

def to_json(payload: dict[str, Any], indent: int | None = 2) -> str:
    """Serialize a dict payload to a JSON string (UTF-8, sorted keys)."""
    return json.dumps(payload, indent=indent, sort_keys=True, ensure_ascii=False)
