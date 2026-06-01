"""Zaszlo: flag algebra computations for extremal combinatorics.

The main entry points are:

- :class:`FlagProblem` — specify the hypergraph uniformity, vertex count,
  forbidden subgraphs, and optimization direction.
- :func:`build_flag_algebra_data` — generate types, flags, admissible graphs,
  and pair-density matrices from a ``FlagProblem``.
- :func:`solve_sdp` — build and solve the SDP; returns a
  :class:`FlagAlgebraResult`.
- :func:`verify_certificate` / :func:`identify_sharps` — check the certificate
  in exact rational arithmetic and find extremal graphs.

Quick start::

    from zaszlo import complete, FlagProblem, build_flag_algebra_data, solve_sdp

    # Mantel's theorem: max edge density in triangle-free graphs = 1/2
    # type_order = n - 2 is the recommended choice for the tightest bound at a given n
    prob = FlagProblem(4, 2, 2, forbidden=[complete(3)])
    data = build_flag_algebra_data(prob)
    result = solve_sdp(data, extract_Q=True)
    print(result.bound)          # ≈ 0.5
    print(result.explain())      # plain-text summary of the result

    # Turán's theorem: max edge density in K₄-free graphs = 2/3
    prob = FlagProblem(5, 3, 2, forbidden=[complete(4)])
    data = build_flag_algebra_data(prob)
    result = solve_sdp(data, extract_Q=True)
    print(result.bound)          # ≈ 0.6667
"""

from .types import (
    Hypergraph,
    Flag,
    FlagProblem,
    FlagAlgebraData,
    FlagAlgebraResult,
    SharpsResult,
    complete,
    k4_minus,
    c5_3uniform,
    f32,
)
from .pipeline import build_flag_algebra_data
from .sdp import build_sdp, solve_sdp, verify_certificate, identify_sharps, round_certificate

__all__ = [
    "Hypergraph",
    "Flag",
    "FlagProblem",
    "FlagAlgebraData",
    "FlagAlgebraResult",
    "SharpsResult",
    "complete",
    "k4_minus",
    "c5_3uniform",
    "f32",
    "build_flag_algebra_data",
    "build_sdp",
    "solve_sdp",
    "verify_certificate",
    "identify_sharps",
    "round_certificate",
]
