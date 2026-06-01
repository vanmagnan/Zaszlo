# Core data structures for the flag algebra method.
#
# A K-uniform hypergraph has every edge of size K.
# K=2 gives ordinary graphs; K=3 gives 3-uniform hypergraphs.
#
# Edges are stored as sorted K-tuples of 1-based vertex indices.

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations
from typing import Optional
import numpy as np


# ---------------------------------------------------------------------------
# Hypergraph
# ---------------------------------------------------------------------------

class Hypergraph:
    """K-uniform hypergraph on n labeled vertices.

    Parameters
    ----------
    n:
        Number of vertices (1-based labels 1..n).
    k:
        Edge uniformity.
    edges:
        Iterable of k-tuples of vertex indices. Each edge is sorted on
        construction; the edge list itself is also sorted and deduplicated.
    """

    def __init__(self, n: int, k: int, edges: list[tuple[int, ...]]):
        self.n = n
        self.k = k
        # Normalize: sort each edge internally, then sort the edge list.
        normalized = sorted({tuple(sorted(e)) for e in edges})
        self.edges: list[tuple[int, ...]] = normalized

    # Equality and hashing so Hypergraph can be used in sets/dicts.
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Hypergraph):
            return NotImplemented
        return self.n == other.n and self.k == other.k and self.edges == other.edges

    def __hash__(self) -> int:
        return hash((self.n, self.k, tuple(self.edges)))

    def __repr__(self) -> str:
        return f"Hypergraph(n={self.n}, k={self.k}, edges={self.edges})"

    def __str__(self) -> str:
        return f"{self.n}v {self.k}-uniform hypergraph, {len(self.edges)} edges"

    def explain(self) -> str:
        """Return a plain-text explanation of this hypergraph for learning."""
        from .explain import explain_hypergraph
        return explain_hypergraph(self)

    def _repr_html_(self) -> str:
        from .explain import html_hypergraph
        return html_hypergraph(self)

    def _repr_mimebundle_(self, include=None, exclude=None, **kwargs) -> dict:
        bundle = {"text/html": self._repr_html_(), "text/plain": repr(self)}
        if include:
            bundle = {k: v for k, v in bundle.items() if k in include}
        if exclude:
            bundle = {k: v for k, v in bundle.items() if k not in exclude}
        return bundle


# ---------------------------------------------------------------------------
# Flag
# ---------------------------------------------------------------------------

class Flag:
    """Hypergraph with type_size labeled vertices (1..type_size fixed).

    Parameters
    ----------
    graph:
        The underlying hypergraph.
    type_size:
        Number of labeled (fixed) vertices. Must satisfy 0 ≤ type_size ≤ graph.n.

    Special cases:
      type_size == 0          → admissible graph (no labeled vertices)
      type_size == graph.n    → type (all vertices labeled)
    """

    def __init__(self, graph: Hypergraph, type_size: int):
        self.graph = graph
        self.type_size = type_size

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Flag):
            return NotImplemented
        return self.type_size == other.type_size and self.graph == other.graph

    def __hash__(self) -> int:
        return hash((self.type_size, self.graph))

    def __repr__(self) -> str:
        if self.type_size == 0:
            return f"Admissible: {self.graph}"
        if self.type_size == self.graph.n:
            return f"Type: {self.graph}"
        return f"Flag over {self.type_size}-vertex type: {self.graph}"

    def explain(self) -> str:
        """Return a plain-text explanation of this flag for learning."""
        from .explain import explain_flag
        return explain_flag(self)

    def _repr_html_(self) -> str:
        from .explain import html_flag
        return html_flag(self)

    def _repr_mimebundle_(self, include=None, exclude=None, **kwargs) -> dict:
        bundle = {"text/html": self._repr_html_(), "text/plain": repr(self)}
        if include:
            bundle = {k: v for k, v in bundle.items() if k in include}
        if exclude:
            bundle = {k: v for k, v in bundle.items() if k not in exclude}
        return bundle


# ---------------------------------------------------------------------------
# FlagProblem
# ---------------------------------------------------------------------------

class FlagProblem:
    """Specification of a flag algebra SDP.

    Parameters
    ----------
    n:
        Vertex count for admissible graphs.
    type_order:
        Maximum type vertex count.  Must satisfy ``type_order <= n - 2`` and
        have the same parity as ``n`` (so that flag sizes ``(n + s) / 2`` are
        integers).  ``type_order = n - 2`` is the standard choice — it uses the
        largest types whose flags still fit strictly inside the admissible-graph
        vertex count, giving the tightest bound for a given ``n``.
    k:
        Edge uniformity (k=2 for ordinary graphs, k=3 for 3-uniform
        hypergraphs, etc.).
    forbidden:
        Hypergraphs whose non-induced copies are forbidden.
    forbidden_induced:
        Hypergraphs whose induced copies are forbidden.
    target:
        Hypergraph whose density is optimized. None means edge density.
    minimize:
        If True, minimize the target density (lower bound); otherwise maximize
        (upper bound, the default).

    Raises
    ------
    ValueError
        If ``type_order`` violates the parity or size constraints.

    Examples
    --------
    Mantel's theorem — maximum edge density in triangle-free graphs is 1/2.
    Use ``type_order = n - 2`` (here 2), the standard choice::

        from zaszlo import complete, FlagProblem, build_flag_algebra_data, solve_sdp
        prob = FlagProblem(4, 2, 2, forbidden=[complete(3)])
        data = build_flag_algebra_data(prob)
        result = solve_sdp(data)
        # result.bound ≈ 0.5

    Turán's theorem — maximum edge density in K₄-free graphs is 2/3.
    Raise n to 5 and type_order to 3 for the tight bound::

        prob = FlagProblem(5, 3, 2, forbidden=[complete(4)])
        data = build_flag_algebra_data(prob)
        result = solve_sdp(data)
        # result.bound ≈ 0.6667
    """

    def __init__(
        self,
        n: int,
        type_order: int,
        k: int,
        *,
        forbidden: list[Hypergraph] | None = None,
        forbidden_induced: list[Hypergraph] | None = None,
        target: Optional[Hypergraph] = None,
        minimize: bool = False,
    ):
        if (n - type_order) % 2 != 0:
            valid = list(range(n % 2, n - 1, 2))
            raise ValueError(
                f"type_order={type_order} and n={n} have different parities. "
                f"Flag sizes (n+s)/2 must be integers, so type_order must match "
                f"the parity of n. Valid type_order values for n={n}: {valid}."
            )
        if type_order > n - 2:
            raise ValueError(
                f"type_order={type_order} must be <= n-2={n - 2}. "
                f"Types that large would produce flags as large as the admissible "
                f"graphs (n={n} vertices), making the SDP trivial."
            )
        self.n = n
        self.type_order = type_order
        self.k = k
        self.forbidden: list[Hypergraph] = forbidden or []
        self.forbidden_induced: list[Hypergraph] = forbidden_induced or []
        self.target = target
        self.minimize = minimize

    def __repr__(self) -> str:
        return (
            f"FlagProblem(n={self.n}, type_order={self.type_order}, k={self.k}, "
            f"minimize={self.minimize})"
        )

    def explain(self) -> str:
        """Return a plain-text explanation of this problem for learning."""
        from .explain import explain_problem
        return explain_problem(self)

    def _repr_html_(self) -> str:
        from .explain import html_problem
        return html_problem(self)

    def _repr_mimebundle_(self, include=None, exclude=None, **kwargs) -> dict:
        bundle = {"text/html": self._repr_html_(), "text/plain": repr(self)}
        if include:
            bundle = {k: v for k, v in bundle.items() if k in include}
        if exclude:
            bundle = {k: v for k, v in bundle.items() if k not in exclude}
        return bundle


# ---------------------------------------------------------------------------
# FlagAlgebraData
# ---------------------------------------------------------------------------

class FlagAlgebraData:
    """Fully computed data ready for SDP construction.

    Attributes
    ----------
    problem:
        The originating FlagProblem.
    types:
        All non-isomorphic types (Flags with type_size == graph.n).
    flags:
        flags[i] = list of flags over types[i].
    admissible:
        Non-isomorphic admissible graphs on n vertices.
    densities:
        densities[i] = Fraction density of admissible[i].
    pair_dens:
        pair_dens[i][j] = upper-triangular Fraction matrix for
        admissible graph i and type j.
    """

    def __init__(
        self,
        problem: FlagProblem,
        types: list[Flag],
        flags: list[list[Flag]],
        admissible: list[Hypergraph],
        densities: list[Fraction],
        pair_dens: list[list[list[list[Fraction]]]],
    ):
        """Store precomputed flag algebra data produced by build_flag_algebra_data."""
        self.problem = problem
        self.types = types
        self.flags = flags
        self.admissible = admissible
        self.densities = densities
        self.pair_dens = pair_dens  # pair_dens[H_idx][sigma] = 2D list of Fraction

    def explain(self) -> str:
        """Return a plain-text explanation of this pre-SDP data for learning."""
        from .explain import explain_data
        return explain_data(self)

    def _repr_html_(self) -> str:
        from .explain import html_data
        return html_data(self)

    def _repr_mimebundle_(self, include=None, exclude=None, **kwargs) -> dict:
        bundle = {"text/html": self._repr_html_(), "text/plain": repr(self)}
        if include:
            bundle = {k: v for k, v in bundle.items() if k in include}
        if exclude:
            bundle = {k: v for k, v in bundle.items() if k not in exclude}
        return bundle


# ---------------------------------------------------------------------------
# FlagAlgebraResult
# ---------------------------------------------------------------------------

class FlagAlgebraResult:
    """Output of solve_sdp.

    Attributes
    ----------
    problem:
        The originating FlagProblem.
    status:
        Solver status string (e.g. 'optimal').
    bound:
        Optimal objective value (float).
    Q:
        PSD certificate matrices (one numpy array per type), or None if not
        extracted.
    slacks:
        Primal slack per admissible graph (0.0 = sharp/extremal).

    Notes
    -----
    In Jupyter, results render automatically as HTML summary cards.  Pass the
    originating ``FlagAlgebraData`` to ``explain()`` or ``_repr_html_()`` for
    richer output that names the sharp (extremal) graphs::

        print(result.explain(data=data))   # names sharp graphs in plain text
        result._repr_html_(data=data)      # richer HTML card in a notebook
    """

    def __init__(
        self,
        problem: FlagProblem,
        status: str,
        bound: float,
        Q: Optional[list[np.ndarray]],
        slacks: list[float],
        cholesky_factors: Optional[list[np.ndarray]] = None,
        Q_exact: Optional[list[list[list[Fraction]]]] = None,
        bound_exact: Optional[Fraction] = None,
    ):
        """Store the output of solve_sdp. Constructed internally; not called directly."""
        self.problem = problem
        self.status = status
        self.bound = bound
        self.Q = Q
        self.slacks = slacks
        self.cholesky_factors = cholesky_factors
        self.Q_exact = Q_exact
        self.bound_exact = bound_exact

    def __repr__(self) -> str:
        q_desc = "not extracted" if self.Q is None else f"{len(self.Q)} matrices"
        l_desc = "" if self.cholesky_factors is None else "  (Cholesky factors stored)"
        return (
            f"FlagAlgebraResult\n"
            f"  status : {self.status}\n"
            f"  bound  : {self.bound}\n"
            f"  Q      : {q_desc}{l_desc}"
        )

    def explain(self, data=None) -> str:
        """Return a plain-text explanation of this result for learning.

        Parameters
        ----------
        data:
            Optional FlagAlgebraData from the same problem. When provided,
            sharp graphs are shown with their actual structure and densities.
        """
        from .explain import explain_result
        return explain_result(self, data=data)

    def explain_certificate(self) -> str:
        """Explain the SDP sum-of-squares certificate and Q matrix properties.

        Raises ValueError if Q matrices were not extracted (extract_Q=True).
        """
        from .explain import explain_certificate
        return explain_certificate(self)

    def _repr_html_(self, data=None) -> str:
        from .explain import html_result
        return html_result(self, data=data)

    def _repr_mimebundle_(self, include=None, exclude=None, **kwargs) -> dict:
        bundle = {"text/html": self._repr_html_(), "text/plain": repr(self)}
        if include:
            bundle = {k: v for k, v in bundle.items() if k in include}
        if exclude:
            bundle = {k: v for k, v in bundle.items() if k not in exclude}
        return bundle


# ---------------------------------------------------------------------------
# SharpsResult
# ---------------------------------------------------------------------------

class SharpsResult:
    """Output of identify_sharps — the extremal (sharp) admissible graphs.

    Attributes
    ----------
    problem:
        The originating FlagProblem.
    indices:
        0-based positions of sharp graphs in the admissible list.
    graphs:
        The sharp Hypergraph objects.
    densities:
        Exact Fraction densities of the sharp graphs.
    residuals:
        Exact rational residuals for *all* admissible graphs (not just sharps).
        Index i corresponds to the i-th admissible graph; sharps have residual ≈ 0.
    """

    def __init__(
        self,
        problem: FlagProblem,
        indices: list[int],
        graphs: list,
        densities: list[Fraction],
        residuals: list[Fraction],
    ):
        self.problem = problem
        self.indices = indices
        self.graphs = graphs
        self.densities = densities
        self.residuals = residuals

    def __repr__(self) -> str:
        return (
            f"SharpsResult  {len(self.indices)} sharp graph(s)"
            f"  (n={self.problem.n}, k={self.problem.k})"
        )

    def explain(self) -> str:
        """Return a plain-text explanation of the sharp graphs."""
        from .explain import explain_sharps
        return explain_sharps(self)

    def _repr_html_(self) -> str:
        from .explain import html_sharps
        return html_sharps(self)

    def _repr_mimebundle_(self, include=None, exclude=None, **kwargs) -> dict:
        bundle = {"text/html": self._repr_html_(), "text/plain": repr(self)}
        if include:
            bundle = {k: v for k, v in bundle.items() if k in include}
        if exclude:
            bundle = {k: v for k, v in bundle.items() if k not in exclude}
        return bundle


# ---------------------------------------------------------------------------
# Named graph shortcuts
# ---------------------------------------------------------------------------

def complete(n: int, k: int = 2) -> Hypergraph:
    """Complete k-uniform hypergraph on n vertices (all C(n,k) edges).

    The most common forbidden patterns in Turán-style problems are complete
    graphs, so ``k`` defaults to 2.  For hypergraph problems pass the
    uniformity explicitly, e.g. ``complete(4, 3)`` for K₄⁽³⁾.

    Examples
    --------
    ::

        complete(3)       # K₃ (triangle), k=2
        complete(4)       # K₄, k=2
        complete(4, 3)    # complete 3-uniform hypergraph on 4 vertices
    """
    return Hypergraph(n, k, list(combinations(range(1, n + 1), k)))


def k4_minus() -> Hypergraph:
    """K4-minus: 3-uniform hypergraph on 4 vertices with 3 edges.

    Corresponds to C flagmatic's --forbid-k4- ("4.3").
    """
    return Hypergraph(4, 3, [(1, 2, 3), (1, 2, 4), (1, 3, 4)])


def c5_3uniform() -> Hypergraph:
    """Tight 5-cycle in 3-uniform hypergraphs.

    Edges: {1,2,3},{2,3,4},{3,4,5},{4,5,1},{5,1,2}.
    Corresponds to C flagmatic's --forbid-c5.
    """
    return Hypergraph(5, 3, [(1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 1), (5, 1, 2)])


def f32() -> Hypergraph:
    """F32: 5-vertex 3-graph with 4 edges sharing common pair {4,5}.

    Edges: {1,2,3},{1,4,5},{2,4,5},{3,4,5}.
    Corresponds to C flagmatic's --forbid-f32.
    """
    return Hypergraph(5, 3, [(1, 2, 3), (1, 4, 5), (2, 4, 5), (3, 4, 5)])
