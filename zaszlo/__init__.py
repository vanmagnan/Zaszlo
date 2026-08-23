"""Zaszlo: flag algebra computations for extremal combinatorics.

The main entry point is :func:`solve`:

- ``solve(prob)``               — build flag algebra data and solve the SDP.
- ``solve(prob, certify=True)`` — also produce an exact rational certificate.

For step-by-step control the full pipeline is also public:

- :class:`FlagProblem` — specify the hypergraph uniformity, vertex count,
  forbidden subgraphs, and optimization direction.
- :func:`build_flag_algebra_data` — generate types, flags, admissible graphs,
  and pair-density matrices.
- :func:`solve_sdp` — build and solve the SDP.
- :func:`certify` — round and verify in exact rational arithmetic; returns a
  :class:`Certificate`.
- :func:`identify_sharps` — find extremal graphs from a certificate.

Quick start::

    from zaszlo import FlagProblem, complete, solve

    # Mantel's theorem: max edge density in triangle-free graphs = 1/2
    prob   = FlagProblem(4, 2, 2, forbidden=[complete(3)])
    result = solve(prob)
    print(result.bound)          # ≈ 0.5

    # Same problem with an exact rational certificate
    result = solve(prob, certify=True)
    print(result.certificate.bound)   # Fraction close to 1/2
    print(result.certificate.valid)   # True
    print(result.certificate.explain())

    # Turán's theorem: max edge density in K₄-free graphs = 2/3
    prob   = FlagProblem(5, 3, 2, forbidden=[complete(4)])
    result = solve(prob, certify=True)
    print(result.certificate.bound)   # Fraction close to 2/3
"""

from .types import (
    Hypergraph,
    Flag,
    FlagProblem,
    FlagAlgebraData,
    FlagAlgebraResult,
    SharpsResult,
    Certificate,
    CertificateProvenance,
    CertificationError,
    DensityExpr,
    FlagAlgebraElement,
    UnlabeledExpr,
    AuxiliaryConstraint,
    complete,
    k4_minus,
    c5_3uniform,
    f32,
)
from .pipeline import build_flag_algebra_data
from .sdp import build_sdp, solve_sdp, verify_certificate, identify_sharps, round_certificate, certify, certify_at_bound
from .algebra import flag_product, unlabel, lift_to
from .api import solve
from .diagnose import DiagnosticReport
from . import corpus

__all__ = [
    # One-call entry point
    "solve",
    # Types
    "Hypergraph",
    "Flag",
    "FlagProblem",
    "FlagAlgebraData",
    "FlagAlgebraResult",
    "SharpsResult",
    "Certificate",
    "CertificateProvenance",
    "DensityExpr",
    "FlagAlgebraElement",
    "UnlabeledExpr",
    "AuxiliaryConstraint",
    # Exceptions
    "CertificationError",
    # Named graph constructors
    "complete",
    "k4_minus",
    "c5_3uniform",
    "f32",
    # Step-by-step pipeline
    "build_flag_algebra_data",
    "build_sdp",
    "solve_sdp",
    "verify_certificate",
    "identify_sharps",
    "round_certificate",
    "certify",
    "certify_at_bound",
    # Flag algebra kernel
    "flag_product",
    "unlabel",
    "lift_to",
    # Diagnostics
    "DiagnosticReport",
    # Reference problem corpus (sub-module)
    "corpus",
]
