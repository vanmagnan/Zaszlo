"""Seed entries for the reference corpus.

Every entry has a `problem_factory` that returns a fresh :class:`FlagProblem`.
The registered bound is either exact (a closed-form Fraction from the
literature) or a rational approximation good enough for the SDP tolerance
noted in the entry's ``notes`` field.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Callable

from ..algebra import unlabel
from ..generation import generate_flags
from ..types import (
    AuxiliaryConstraint,
    DensityExpr,
    Flag,
    FlagProblem,
    Hypergraph,
    c5_3uniform,
    complete,
    f32,
    k4_minus,
)


@dataclass(frozen=True)
class CorpusEntry:
    """A curated reference problem with a known flag-algebra bound.

    Attributes
    ----------
    slug:
        Short unique identifier used with :func:`zaszlo.corpus.get`.
    name:
        Human-readable title (e.g., "Mantel's theorem").
    citation:
        Bibliographic pointer to the original result.
    description:
        One-paragraph mathematical description of the problem.
    tags:
        Freeform categorization: e.g., ``('turan', 'k=2', 'edge-density')``.
    problem_factory:
        Zero-argument callable that returns a fresh :class:`FlagProblem`.
        Rebuilding the problem on each call avoids shared mutable state.
    expected_bound:
        The known bound as a :class:`~fractions.Fraction`.  For entries where
        only an approximate SDP bound is currently attainable, this is a close
        rational and the entry's ``notes`` explains the tolerance.
    notes:
        Free-text notes: what the extremal construction is, tolerance caveats,
        common variations, references to related entries.
    lean_slug:
        For entries whose bound is proved in the taeyool
        ``lean-flag-algebras-release`` Lean 4 library, the base filename of
        the corresponding Flagmatic certificate JSON under
        ``LeanFlagAlgebras/Flagmatic/Certificates/`` (e.g. ``"Mantel_cert"``).
        ``None`` when no matching Lean case study exists.
    lean_problem_factory:
        If set, use this factory instead of ``problem_factory`` when generating
        a certificate for Lean export. The taeyool ``flag_certificate`` tactic
        requires all SDP blocks to have a non-empty type (type_size ≥ 1); the
        Flagmatic empty type (σ = ∅, "0:") is not supported. This means k=2
        problems with an even type_order need a different parameterization for
        Lean export (even type_order always introduces an empty-type block;
        odd type_order does not). ``None`` when ``problem_factory`` already
        produces a Lean-compatible certificate.
    """

    slug: str
    name: str
    citation: str
    description: str
    tags: tuple[str, ...]
    problem_factory: Callable[[], FlagProblem]
    expected_bound: Fraction
    notes: str
    lean_slug: str | None = None
    lean_problem_factory: Callable[[], FlagProblem] | None = None


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _mantel() -> FlagProblem:
    return FlagProblem(4, 2, 2, forbidden=[complete(3)])


def _mantel_lean() -> FlagProblem:
    # type_order=2 (even) introduces an empty-type block; use type_order=1.
    return FlagProblem(3, 1, 2, forbidden=[complete(3)])


def _turan_k4() -> FlagProblem:
    return FlagProblem(5, 3, 2, forbidden=[complete(4)])


def _turan_k5() -> FlagProblem:
    return FlagProblem(6, 4, 2, forbidden=[complete(5)])


def _turan_k5_lean() -> FlagProblem:
    # type_order=4 (even) introduces an empty-type block; use type_order=3.
    return FlagProblem(5, 3, 2, forbidden=[complete(5)])


def _pentagon_c5() -> FlagProblem:
    C5 = Hypergraph(5, 2, [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)])
    return FlagProblem(5, 3, 2, forbidden=[complete(3)], target=C5)


def _k4_minus_free_3g() -> FlagProblem:
    return FlagProblem(5, 3, 3, forbidden=[k4_minus()])


def _f32_free_3g() -> FlagProblem:
    return FlagProblem(5, 3, 3, forbidden=[f32()])


def _c5_tight_free_3g() -> FlagProblem:
    return FlagProblem(5, 3, 3, forbidden=[c5_3uniform()])


def _affine_mantel() -> FlagProblem:
    """Max of (3/2)·d(K2) − 1/2 in triangle-free 4-vertex graphs.

    Equivalent to `d(K2) − (1/2)(1 − d(K2))` = `(3/2)·d(K2) − 1/2`.
    At the Mantel optimum d(K2) = 1/2 the value is 1/4.
    """
    edge = Hypergraph(2, 2, [(1, 2)])
    empty_2v = Hypergraph(2, 2, [])
    expr = DensityExpr([(1, edge), (-Fraction(1, 2), empty_2v)])
    return FlagProblem(4, 2, 2, forbidden=[complete(3)], target=expr)


def _mantel_multi_forbidden() -> FlagProblem:
    """K3 and K4 both forbidden; equivalent to K3-free (Mantel)."""
    return FlagProblem(4, 2, 2, forbidden=[complete(3), complete(4)])


def _mantel_with_aux_sos() -> FlagProblem:
    """Mantel with a redundant SoS auxiliary inequality ⟦f²⟧ ≥ 0.

    Demonstrates the aux-constraint API without changing the bound.  The
    solver assigns μ ≈ 0 to the redundant constraint.
    """
    prob = FlagProblem(4, 2, 2, forbidden=[complete(3)])
    sigma_empty = Flag(Hypergraph(0, 2, []), 0)
    admissible_2v_flag = generate_flags(2, sigma_empty, [], [])[0]
    aux_expr = unlabel(admissible_2v_flag * admissible_2v_flag)
    prob.add_constraint(AuxiliaryConstraint(aux_expr))
    return prob


# ---------------------------------------------------------------------------
# Entries
# ---------------------------------------------------------------------------

ALL_ENTRIES: tuple[CorpusEntry, ...] = (
    CorpusEntry(
        slug="mantel",
        name="Mantel's theorem",
        citation="Mantel (1907)",
        description=(
            "A triangle-free graph on n vertices has at most ⌊n²/4⌋ edges. "
            "Equivalently, the edge density satisfies d(K₂) ≤ 1/2, with "
            "equality for the balanced complete bipartite graph."
        ),
        tags=("turan", "k=2", "edge-density"),
        problem_factory=_mantel,
        expected_bound=Fraction(1, 2),
        notes=(
            "Bound is tight at n=4 already. The extremal graphon is the "
            "balanced bipartite graphon; sharp graphs at n=4 include the "
            "empty graph, the 3-edge path/matching, and K₂,₂."
        ),
        lean_slug="Mantel_cert",
        lean_problem_factory=_mantel_lean,
    ),
    CorpusEntry(
        slug="turan_k4",
        name="Turán's theorem for K₄",
        citation="Turán (1941)",
        description=(
            "A K₄-free graph has edge density at most 2/3, attained by the "
            "balanced complete 3-partite graph (the Turán graph T(n,3))."
        ),
        tags=("turan", "k=2", "edge-density"),
        problem_factory=_turan_k4,
        expected_bound=Fraction(2, 3),
        notes="The Turán density of Kₖ is (k−2)/(k−1); this is k=4 giving 2/3.",
        lean_slug="K4freeEdge_cert",
    ),
    CorpusEntry(
        slug="turan_k5",
        name="Turán's theorem for K₅",
        citation="Turán (1941)",
        description=(
            "A K₅-free graph has edge density at most 3/4, attained by the "
            "balanced complete 4-partite graph T(n,4)."
        ),
        tags=("turan", "k=2", "edge-density"),
        problem_factory=_turan_k5,
        expected_bound=Fraction(3, 4),
        notes=(
            "n=6, type_order=4 for a tight bound. Numerical solve returns 0.75 "
            "cleanly."
        ),
        lean_slug="K5freeEdge_cert",
        lean_problem_factory=_turan_k5_lean,
    ),
    CorpusEntry(
        slug="pentagon_c5_density",
        name="Pentagon problem — maximum C₅ density in triangle-free graphs",
        citation="Grzesik (2012); Hatami–Hladký–Král'–Norine–Razborov (2013)",
        description=(
            "Among triangle-free graphs, the maximum induced density of the "
            "5-cycle C₅ is 24/625 ≈ 0.0384, attained by the balanced blow-up "
            "of C₅."
        ),
        tags=("target-density", "k=2", "induced"),
        problem_factory=_pentagon_c5,
        expected_bound=Fraction(24, 625),
        notes=(
            "Requires n=5, type_order=3 for the tight bound. In a triangle-"
            "free graph every C₅ is automatically induced."
        ),
        lean_slug="ErdosPentagon_cert",
    ),
    CorpusEntry(
        slug="k4_minus_free_3graphs",
        name="K₄⁻-free 3-uniform hypergraphs",
        citation="Frankl–Füredi (1984 conj.); Keevash–Sudakov (2005)",
        description=(
            "The maximum edge density of a K₄⁻-free 3-uniform hypergraph is "
            "1/3. The extremal construction partitions the vertex set into "
            "three equal parts and includes all triples with at least two "
            "vertices in a single part."
        ),
        tags=("turan", "k=3", "hypergraph", "edge-density"),
        problem_factory=_k4_minus_free_3g,
        expected_bound=Fraction(1, 3),
        notes=(
            "K₄⁻ is the 3-uniform hypergraph on 4 vertices with 3 edges "
            "(one edge missing from the complete K₄⁽³⁾)."
        ),
    ),
    CorpusEntry(
        slug="f32_free_3graphs",
        name="F₃₂-free 3-uniform hypergraphs",
        citation="Frankl–Füredi (1983)",
        description=(
            "F₃₂ is the 5-vertex 3-graph with 4 edges sharing a common pair. "
            "F₃₂-free 3-graphs have edge density at most 1/2."
        ),
        tags=("turan", "k=3", "hypergraph", "edge-density"),
        problem_factory=_f32_free_3g,
        expected_bound=Fraction(1, 2),
        notes="Bound is exact at n=5, type_order=3.",
    ),
    CorpusEntry(
        slug="c5_tight_free_3graphs",
        name="Tight-C₅-free 3-uniform hypergraphs",
        citation="Mubayi–Rödl (2011)",
        description=(
            "The tight 5-cycle C₅ in 3-uniform hypergraphs consists of the "
            "edges {i, i+1, i+2} on ℤ/5. The maximum edge density in tight-"
            "C₅-free 3-graphs is conjectured to converge to 1/2."
        ),
        tags=("turan", "k=3", "hypergraph", "edge-density"),
        problem_factory=_c5_tight_free_3g,
        expected_bound=Fraction(1, 2),
        notes=(
            "At n=5 the SDP gives ≈ 0.4996, so use tolerance ~1e-2. "
            "Larger n tightens further; exact bound remains open."
        ),
    ),
    CorpusEntry(
        slug="affine_mantel",
        name="Affine transform of Mantel — a DensityExpr worked example",
        citation="Derived from Mantel (1907)",
        description=(
            "Maximize the linear functional f(H) = (3/2)·d(H, K₂) − 1/2 "
            "over triangle-free 4-vertex graphs. Written as a DensityExpr "
            "over the K₂ and empty-2v basis: 1·d(H, K₂) − (1/2)·d(H, empty)."
        ),
        tags=("linear-combination", "k=2", "density-expr"),
        problem_factory=_affine_mantel,
        expected_bound=Fraction(1, 4),
        notes=(
            "Exact bound: at the Mantel optimum d(K₂) = 1/2 the value is "
            "(3/2)(1/2) − 1/2 = 1/4. Demonstrates the DensityExpr target API."
        ),
    ),
    CorpusEntry(
        slug="mantel_multi_forbidden",
        name="Mantel with redundant multiple forbidden — {K₃, K₄}-free",
        citation="Corollary of Mantel (1907)",
        description=(
            "Forbid both K₃ and K₄; K₃-freeness already implies K₄-freeness, "
            "so the admissible class and bound coincide with Mantel's theorem. "
            "Exercises the multi-forbidden API path."
        ),
        tags=("turan", "k=2", "multi-forbidden"),
        problem_factory=_mantel_multi_forbidden,
        expected_bound=Fraction(1, 2),
        notes=(
            "Included as a sanity check on the multi-forbidden pipeline; "
            "admissible-set / flag counts must match single-K₃-forbidden."
        ),
    ),
    CorpusEntry(
        slug="mantel_with_aux_sos",
        name="Mantel with an auxiliary SoS constraint",
        citation="Trivial extension of Mantel (1907)",
        description=(
            "Adds a redundant flag-algebra inequality ⟦f²⟧ ≥ 0 (for f an "
            "arbitrary admissible flag) as an auxiliary SDP constraint on "
            "Mantel. Bound unchanged; μ ≈ 0 in the solved SDP. Demonstrates "
            "the AuxiliaryConstraint API path."
        ),
        tags=("turan", "k=2", "aux-constraint"),
        problem_factory=_mantel_with_aux_sos,
        expected_bound=Fraction(1, 2),
        notes=(
            "This entry exists to test the aux-constraint infrastructure "
            "without requiring a genuinely tightening auxiliary."
        ),
    ),
)
