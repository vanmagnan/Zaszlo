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
from typing import Iterable, Optional, Union
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
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        for edge in edges:
            if len(edge) != k:
                raise ValueError(
                    f"edge {edge!r} has size {len(edge)}, expected k={k}"
                )
            if len(set(edge)) != k:
                raise ValueError(f"edge {edge!r} contains repeated vertices")
            if any(v < 1 or v > n for v in edge):
                raise ValueError(
                    f"edge {edge!r} contains a vertex outside 1..{n}"
                )
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
        if not (0 <= type_size <= graph.n):
            raise ValueError(
                f"type_size={type_size} must satisfy 0 <= type_size <= graph.n={graph.n}"
            )
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
# DensityExpr
# ---------------------------------------------------------------------------

_CoefLike = Union[int, Fraction]


class DensityExpr:
    """Linear combination of induced densities:  Σ_i  c_i · d(·, G_i).

    Used as ``FlagProblem.target`` when the quantity to optimize is not the
    density of a single induced subgraph but a rational linear combination of
    several such densities.  Evaluating the expression on an admissible graph
    ``H`` returns the exact rational value

        Σ_i  c_i · induced_density(H, G_i).

    Coefficients must be ``int`` or :class:`fractions.Fraction` — floats are
    rejected to keep the certified-bound story exact.  Constituent graphs must
    all share the same uniformity ``k``.

    The constructor is intentionally explicit; arithmetic operators are not
    defined on :class:`Hypergraph` or :class:`Flag` so that a future
    flag-algebra element type can claim that surface.

    Parameters
    ----------
    terms:
        Iterable of ``(coefficient, hypergraph)`` pairs.  Duplicate graphs are
        merged (coefficients summed) and zero-coefficient terms are dropped, so
        the stored ``.terms`` list is a canonical form.

    Raises
    ------
    ValueError
        If ``terms`` is empty, coefficients are floats, graphs disagree in
        uniformity, or the resulting canonical form has no nonzero terms.

    Examples
    --------
    A single-term expression is equivalent to a bare hypergraph target::

        expr = DensityExpr([(1, complete(3))])

    A genuine linear combination::

        from fractions import Fraction
        expr = DensityExpr([
            (2, complete(3)),
            (-Fraction(1, 2), c5_3uniform()),
        ])
        prob = FlagProblem(6, 4, 3, target=expr)
    """

    def __init__(self, terms: Iterable[tuple[_CoefLike, "Hypergraph"]]):
        raw = list(terms)
        if not raw:
            raise ValueError("DensityExpr requires at least one (coefficient, graph) term")

        # Coefficient validation + coercion to Fraction.
        coerced: list[tuple[Fraction, Hypergraph]] = []
        for pair in raw:
            if not (isinstance(pair, tuple) and len(pair) == 2):
                raise ValueError(
                    f"each term must be a (coefficient, Hypergraph) tuple; got {pair!r}"
                )
            coef, graph = pair
            if isinstance(coef, bool) or not isinstance(coef, (int, Fraction)):
                raise ValueError(
                    f"coefficient must be int or Fraction (got {type(coef).__name__}: {coef!r}). "
                    "Floats are rejected to keep certified bounds exact; wrap the value in "
                    "Fraction(...) or Fraction(numerator, denominator) explicitly."
                )
            if not isinstance(graph, Hypergraph):
                raise ValueError(
                    f"each term's second element must be a Hypergraph; got {type(graph).__name__}"
                )
            coerced.append((Fraction(coef), graph))

        # Uniformity check across all terms.
        ks = {g.k for _, g in coerced}
        if len(ks) > 1:
            raise ValueError(
                f"all graphs in a DensityExpr must share uniformity k; got {sorted(ks)}"
            )

        # Merge duplicates (by graph equality) and drop zero coefficients.
        merged: dict[Hypergraph, Fraction] = {}
        order: list[Hypergraph] = []
        for coef, graph in coerced:
            if graph in merged:
                merged[graph] += coef
            else:
                merged[graph] = coef
                order.append(graph)

        canonical_terms = [(merged[g], g) for g in order if merged[g] != 0]
        if not canonical_terms:
            raise ValueError(
                "DensityExpr reduced to the zero functional after merging duplicate "
                "graphs; provide at least one nonzero term."
            )

        self.terms: list[tuple[Fraction, Hypergraph]] = canonical_terms
        self.k: int = next(iter(ks))

    def evaluate(self, H: "Hypergraph") -> Fraction:
        """Evaluate  Σ_i c_i · induced_density(H, G_i)  in exact rational arithmetic."""
        from .densities import induced_density
        total = Fraction(0)
        for coef, graph in self.terms:
            total += coef * induced_density(H, graph)
        return total

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DensityExpr):
            return NotImplemented
        return self.terms == other.terms

    def __hash__(self) -> int:
        return hash(tuple(self.terms))

    def __repr__(self) -> str:
        summary = ", ".join(
            f"({coef}, {g.n}v/{g.k}-unif, {len(g.edges)}e)" for coef, g in self.terms
        )
        return f"DensityExpr[{summary}]"

    def explain(self) -> str:
        """Return a plain-text description of this density functional."""
        from .explain import explain_density_expr
        return explain_density_expr(self)

    def _repr_html_(self) -> str:
        from .explain import html_density_expr
        return html_density_expr(self)

    def _repr_mimebundle_(self, include=None, exclude=None, **kwargs) -> dict:
        bundle = {"text/html": self._repr_html_(), "text/plain": repr(self)}
        if include:
            bundle = {k: v for k, v in bundle.items() if k in include}
        if exclude:
            bundle = {k: v for k, v in bundle.items() if k not in exclude}
        return bundle


# ---------------------------------------------------------------------------
# FlagAlgebraElement  —  homogeneous elements of A^σ
# ---------------------------------------------------------------------------
#
# The flag algebra A^σ (Razborov, 2007) is the vector space spanned by
# isomorphism classes of σ-flags (hypergraphs with a distinguished labeling of
# σ embedded in the first s vertices), equipped with the "half-average" product
#
#     (f · g)_h  =  P[ random 2-partition of the m − s unlabeled vertices of h
#                       yields f on one side and g on the other side ],
#
# where h ranges over σ-flags on m = n_f + n_g − s vertices.
#
# The algebra is graded by vertex count.  This class stores a single
# homogeneous grade: a fixed type σ, a fixed vertex count n_e, and a rational
# linear combination of canonical σ-flags on n_e vertices.
# ---------------------------------------------------------------------------

class FlagAlgebraElement:
    """Element of the flag algebra A^σ at a fixed vertex count.

    Represents a rational linear combination

        e  =  Σᵢ cᵢ · fᵢ

    where every basis flag fᵢ shares the same type σ and total vertex count nₑ.
    Elements are homogeneous (fixed grade).  Addition requires matching
    ``(σ, nₑ)``; multiplication `e · e'` yields an element at grade
    ``n_e + n_{e'} − s`` (computed by :func:`zaszlo.algebra.flag_product`).

    Parameters
    ----------
    type_flag:
        The type σ, given as a :class:`Flag` with ``type_size == graph.n``
        (every vertex labeled).
    n:
        Vertex count of each basis flag.  Must satisfy ``n ≥ s``.
    terms:
        Iterable of ``(coefficient, flag)`` pairs.  Each flag must have
        ``type_size == s``, ``graph.n == n``, ``graph.k`` matching σ, and
        induce σ on its first ``s`` vertices.  Duplicates are merged;
        zero-coefficient terms are dropped.  An empty result is valid and
        represents the zero element at this grade.

    Attributes
    ----------
    type:
        The type σ (a Flag).
    n:
        Vertex count of the grade.
    s:
        Type size (``type.graph.n``).
    k:
        Edge uniformity, inherited from σ.
    terms:
        Canonical list of ``(Fraction, Flag)`` pairs with distinct flags and
        nonzero coefficients.
    """

    def __init__(
        self,
        type_flag: Flag,
        n: int,
        terms: Iterable[tuple[_CoefLike, Flag]] = (),
    ):
        if type_flag.type_size != type_flag.graph.n:
            raise ValueError(
                "type_flag must be a type (all vertices labeled): "
                f"got type_size={type_flag.type_size}, graph.n={type_flag.graph.n}"
            )
        s = type_flag.graph.n
        if n < s:
            raise ValueError(f"n={n} must be at least the type size s={s}")

        # Lazy imports to avoid cycles.
        from .graphs import canonical, induce

        merged: dict[Flag, Fraction] = {}
        order: list[Flag] = []
        for pair in terms:
            if not (isinstance(pair, tuple) and len(pair) == 2):
                raise ValueError(
                    f"each term must be a (coefficient, Flag) tuple; got {pair!r}"
                )
            coef, flag = pair
            if isinstance(coef, bool) or not isinstance(coef, (int, Fraction)):
                raise ValueError(
                    f"coefficient must be int or Fraction (got {type(coef).__name__}: {coef!r}). "
                    "Floats are rejected; wrap in Fraction(...) explicitly."
                )
            if not isinstance(flag, Flag):
                raise ValueError(
                    f"each term's second element must be a Flag; got {type(flag).__name__}"
                )
            if flag.type_size != s:
                raise ValueError(
                    f"flag has type_size={flag.type_size}, expected s={s} (matching σ)"
                )
            if flag.graph.n != n:
                raise ValueError(
                    f"flag has graph.n={flag.graph.n}, expected n={n} (grade of this element)"
                )
            if flag.graph.k != type_flag.graph.k:
                raise ValueError(
                    f"flag has k={flag.graph.k}, expected k={type_flag.graph.k} (matching σ)"
                )

            # Canonicalize: permute the m − s unlabeled vertices only.
            f_canon = canonical(flag)

            # Verify the labeled substructure is σ.
            sub = induce(f_canon.graph, list(range(1, s + 1)))
            if sub.edges != type_flag.graph.edges:
                raise ValueError(
                    "flag's labeled substructure does not match the type σ: "
                    f"expected edges on {{1..{s}}} = {type_flag.graph.edges}, got {sub.edges}"
                )

            c_frac = Fraction(coef)
            if f_canon in merged:
                merged[f_canon] += c_frac
            else:
                merged[f_canon] = c_frac
                order.append(f_canon)

        self.type: Flag = type_flag
        self.n: int = n
        self.s: int = s
        self.k: int = type_flag.graph.k
        self.terms: list[tuple[Fraction, Flag]] = [
            (merged[f], f) for f in order if merged[f] != 0
        ]

    # --- semantic queries ---

    @property
    def is_zero(self) -> bool:
        """True iff this element is the zero of A^σ at grade n."""
        return len(self.terms) == 0

    def _require_same_grade(self, other: "FlagAlgebraElement") -> None:
        if self.type != other.type:
            raise ValueError(
                "flag-algebra operations require matching type σ"
            )
        if self.n != other.n:
            raise ValueError(
                f"cannot add/subtract elements of different grades: n={self.n} vs n={other.n}. "
                "Multiplication is the only operation that changes grade."
            )

    # --- arithmetic ---

    def __add__(self, other: object) -> "FlagAlgebraElement":
        if isinstance(other, FlagAlgebraElement):
            self._require_same_grade(other)
            return FlagAlgebraElement(self.type, self.n, self.terms + other.terms)
        return NotImplemented

    def __radd__(self, other: object) -> "FlagAlgebraElement":
        return self.__add__(other)

    def __neg__(self) -> "FlagAlgebraElement":
        return FlagAlgebraElement(self.type, self.n, [(-c, f) for c, f in self.terms])

    def __sub__(self, other: object) -> "FlagAlgebraElement":
        if isinstance(other, FlagAlgebraElement):
            self._require_same_grade(other)
            return self + (-other)
        return NotImplemented

    def __mul__(self, other: object) -> "FlagAlgebraElement":
        # Scalar multiplication (int or Fraction).
        if isinstance(other, (int, Fraction)) and not isinstance(other, bool):
            return FlagAlgebraElement(
                self.type, self.n, [(Fraction(other) * c, f) for c, f in self.terms]
            )
        # Flag-algebra product.
        if isinstance(other, FlagAlgebraElement):
            if self.type != other.type:
                raise ValueError(
                    "flag-algebra product requires matching type σ on both operands"
                )
            m = self.n + other.n - self.s
            if self.is_zero or other.is_zero:
                return FlagAlgebraElement(self.type, m, ())
            from .algebra import flag_product
            result_terms: list[tuple[Fraction, Flag]] = []
            for c1, f1 in self.terms:
                for c2, f2 in other.terms:
                    for pc, pf in flag_product(f1, f2):
                        result_terms.append((c1 * c2 * pc, pf))
            return FlagAlgebraElement(self.type, m, result_terms)
        return NotImplemented

    def __rmul__(self, other: object) -> "FlagAlgebraElement":
        return self.__mul__(other)

    def __pow__(self, exponent: int) -> "FlagAlgebraElement":
        if not isinstance(exponent, int) or exponent < 1:
            raise ValueError(f"exponent must be a positive integer, got {exponent!r}")
        result = self
        for _ in range(exponent - 1):
            result = result * self
        return result

    # --- identity / display ---

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FlagAlgebraElement):
            return NotImplemented
        return (
            self.type == other.type
            and self.n == other.n
            and self.terms == other.terms
        )

    def __hash__(self) -> int:
        return hash((self.type, self.n, tuple(self.terms)))

    def __repr__(self) -> str:
        if self.is_zero:
            return f"FlagAlgebraElement[0]  (type σ={self.s}v, n={self.n})"
        return (
            f"FlagAlgebraElement[{len(self.terms)} term"
            f"{'s' if len(self.terms) != 1 else ''}, "
            f"type σ={self.s}v, n={self.n}, k={self.k}]"
        )

    def explain(self) -> str:
        """Return a plain-text description of this algebra element."""
        from .explain import explain_flag_algebra_element
        return explain_flag_algebra_element(self)

    def _repr_html_(self) -> str:
        from .explain import html_flag_algebra_element
        return html_flag_algebra_element(self)

    def _repr_mimebundle_(self, include=None, exclude=None, **kwargs) -> dict:
        bundle = {"text/html": self._repr_html_(), "text/plain": repr(self)}
        if include:
            bundle = {k: v for k, v in bundle.items() if k in include}
        if exclude:
            bundle = {k: v for k, v in bundle.items() if k not in exclude}
        return bundle


# ---------------------------------------------------------------------------
# UnlabeledExpr  —  homogeneous elements of A^∅
# ---------------------------------------------------------------------------
#
# A^∅ is the flag algebra with trivial (empty) type: basis = admissible-graph
# isomorphism classes.  Elements are what :func:`zaszlo.algebra.unlabel`
# produces from A^σ elements.  A basis element H on n vertices has "value on
# a large graph G" equal to induced_density(G, H) — so evaluation on a specific
# admissible graph is a rational linear combination of induced densities.
# ---------------------------------------------------------------------------

class UnlabeledExpr:
    """Element of the flag algebra A^∅ at a fixed vertex count.

    Represents a rational linear combination

        u  =  Σᵢ cᵢ · Hᵢ

    where every basis graph Hᵢ is admissible-on-nₑ-vertices (canonical iso class).
    Produced by :func:`zaszlo.algebra.unlabel` applied to a FlagAlgebraElement,
    or constructed directly for testing / explicit constraints.

    Parameters
    ----------
    k:
        Edge uniformity of every basis graph.
    n:
        Vertex count of every basis graph.
    terms:
        Iterable of ``(coefficient, hypergraph)`` pairs.  Basis graphs are
        canonicalized and deduplicated; zero-coefficient terms are dropped.
        An empty result is valid and represents the zero of A^∅ at grade n.

    Attributes
    ----------
    k, n:
        Uniformity and vertex count.
    terms:
        Canonical ``(Fraction, Hypergraph)`` pairs with distinct canonical
        graphs and nonzero coefficients.
    """

    def __init__(
        self,
        k: int,
        n: int,
        terms: Iterable[tuple[_CoefLike, "Hypergraph"]] = (),
    ):
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")

        from .graphs import canonical as _canon_flag

        merged: dict[Hypergraph, Fraction] = {}
        order: list[Hypergraph] = []
        for pair in terms:
            if not (isinstance(pair, tuple) and len(pair) == 2):
                raise ValueError(
                    f"each term must be a (coefficient, Hypergraph) tuple; got {pair!r}"
                )
            coef, graph = pair
            if isinstance(coef, bool) or not isinstance(coef, (int, Fraction)):
                raise ValueError(
                    f"coefficient must be int or Fraction (got {type(coef).__name__}: {coef!r})"
                )
            if not isinstance(graph, Hypergraph):
                raise ValueError(
                    f"each term's second element must be a Hypergraph; got {type(graph).__name__}"
                )
            if graph.n != n:
                raise ValueError(
                    f"graph has n={graph.n}, expected n={n} (grade of this element)"
                )
            if graph.k != k:
                raise ValueError(
                    f"graph has k={graph.k}, expected k={k}"
                )

            # Canonicalize as an admissible flag (type_size=0).
            g_canon = _canon_flag(Flag(graph, 0)).graph

            c_frac = Fraction(coef)
            if g_canon in merged:
                merged[g_canon] += c_frac
            else:
                merged[g_canon] = c_frac
                order.append(g_canon)

        self.k: int = k
        self.n: int = n
        self.terms: list[tuple[Fraction, Hypergraph]] = [
            (merged[g], g) for g in order if merged[g] != 0
        ]

    # --- semantic queries ---

    @property
    def is_zero(self) -> bool:
        """True iff this element is the zero of A^∅ at grade n."""
        return len(self.terms) == 0

    def evaluate(self, H: "Hypergraph") -> Fraction:
        """Evaluate  Σᵢ cᵢ · induced_density(H, Hᵢ)  in exact rational arithmetic.

        Requires ``H.n ≥ self.n`` and ``H.k == self.k``.  For ``H.n == self.n``
        this reduces to picking out the coefficient of the canonical form of H.
        """
        if H.k != self.k:
            raise ValueError(
                f"uniformity mismatch: H.k={H.k}, this expression has k={self.k}"
            )
        if H.n < self.n:
            raise ValueError(
                f"H has n={H.n} < expression grade n={self.n}; evaluation undefined"
            )
        from .densities import induced_density
        total = Fraction(0)
        for coef, basis in self.terms:
            total += coef * induced_density(H, basis)
        return total

    def _require_same_grade(self, other: "UnlabeledExpr") -> None:
        if self.k != other.k:
            raise ValueError(f"uniformity mismatch: k={self.k} vs k={other.k}")
        if self.n != other.n:
            raise ValueError(
                f"cannot add/subtract UnlabeledExpr of different grades: n={self.n} vs n={other.n}"
            )

    # --- arithmetic ---

    def __add__(self, other: object) -> "UnlabeledExpr":
        if isinstance(other, UnlabeledExpr):
            self._require_same_grade(other)
            return UnlabeledExpr(self.k, self.n, self.terms + other.terms)
        return NotImplemented

    def __neg__(self) -> "UnlabeledExpr":
        return UnlabeledExpr(self.k, self.n, [(-c, g) for c, g in self.terms])

    def __sub__(self, other: object) -> "UnlabeledExpr":
        if isinstance(other, UnlabeledExpr):
            self._require_same_grade(other)
            return self + (-other)
        return NotImplemented

    def __mul__(self, other: object) -> "UnlabeledExpr":
        # Scalar multiplication only; product in A^∅ is out of scope here.
        if isinstance(other, (int, Fraction)) and not isinstance(other, bool):
            return UnlabeledExpr(
                self.k, self.n, [(Fraction(other) * c, g) for c, g in self.terms]
            )
        return NotImplemented

    def __rmul__(self, other: object) -> "UnlabeledExpr":
        return self.__mul__(other)

    # --- identity / display ---

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UnlabeledExpr):
            return NotImplemented
        return self.k == other.k and self.n == other.n and self.terms == other.terms

    def __hash__(self) -> int:
        return hash((self.k, self.n, tuple(self.terms)))

    def __repr__(self) -> str:
        if self.is_zero:
            return f"UnlabeledExpr[0]  (n={self.n}, k={self.k})"
        return (
            f"UnlabeledExpr[{len(self.terms)} term"
            f"{'s' if len(self.terms) != 1 else ''}, n={self.n}, k={self.k}]"
        )

    def explain(self) -> str:
        """Return a plain-text description of this A^∅ element."""
        from .explain import explain_unlabeled_expr
        return explain_unlabeled_expr(self)

    def _repr_html_(self) -> str:
        from .explain import html_unlabeled_expr
        return html_unlabeled_expr(self)

    def _repr_mimebundle_(self, include=None, exclude=None, **kwargs) -> dict:
        bundle = {"text/html": self._repr_html_(), "text/plain": repr(self)}
        if include:
            bundle = {k: v for k, v in bundle.items() if k in include}
        if exclude:
            bundle = {k: v for k, v in bundle.items() if k not in exclude}
        return bundle


# ---------------------------------------------------------------------------
# AuxiliaryConstraint  —  ⟦e⟧ ≥ 0  imposed on the SDP
# ---------------------------------------------------------------------------

class AuxiliaryConstraint:
    """A non-negativity constraint ``⟦e⟧ ≥ 0`` on an unlabeled algebra element.

    Injects an additional inequality into the flag-algebra SDP.  For each
    admissible graph H at the problem's vertex count n, the constraint
    contributes ``μ · c_H`` to the sum-of-squares identity, where ``μ ≥ 0`` is
    a new SDP variable and ``c_H`` is the coefficient of H in the lift of
    ``expr`` to grade n.  Adding valid constraints can never invalidate the
    bound; adding tight ones typically tightens it.

    Parameters
    ----------
    expr:
        An :class:`UnlabeledExpr` at some grade ``n_e ≤ prob.n``.  Assumed to
        satisfy ``expr.evaluate(G) ≥ 0`` for every graphon G in the admissible
        class — this is the user's mathematical claim, not checked by the SDP.

    Attributes
    ----------
    expr:
        The underlying A^∅ element.
    sense:
        Currently always ``'>=0'``.  Equality or ``<=`` constraints can be
        expressed as pairs / negations of ``>=`` constraints.
    """

    def __init__(self, expr: UnlabeledExpr, *, sense: str = ">=0"):
        if not isinstance(expr, UnlabeledExpr):
            raise ValueError(
                f"AuxiliaryConstraint requires an UnlabeledExpr; got {type(expr).__name__}. "
                "Call unlabel() on a FlagAlgebraElement first."
            )
        if sense != ">=0":
            raise ValueError(
                f"only sense='>=0' is currently supported; got {sense!r}. "
                "Express equality as two >=0 constraints and <=0 as -expr >= 0."
            )
        self.expr = expr
        self.sense = sense

    @property
    def k(self) -> int:
        return self.expr.k

    @property
    def n(self) -> int:
        return self.expr.n

    def __repr__(self) -> str:
        return f"AuxiliaryConstraint(⟦e⟧ ≥ 0, n={self.n}, {len(self.expr.terms)} terms)"

    def explain(self) -> str:
        """Return a plain-text description of this constraint."""
        from .explain import explain_aux_constraint
        return explain_aux_constraint(self)

    def _repr_html_(self) -> str:
        from .explain import html_aux_constraint
        return html_aux_constraint(self)

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


# --- Flag arithmetic (bolted on after FlagAlgebraElement is defined) -------
#
# A Flag over σ is a basis element of the flag algebra A^σ, so its arithmetic
# lives naturally in that algebra.  Multiplication requires matching types.
# Scalar and Flag±Flag operations lift the operand(s) into single-term
# FlagAlgebraElements and dispatch there.

def _flag_type(f: Flag) -> Flag:
    """Return the type σ of a flag as a Flag object (labeled substructure on 1..s)."""
    from .graphs import canonical, induce
    f_canon = canonical(f)
    sigma_graph = induce(f_canon.graph, list(range(1, f.type_size + 1)))
    return Flag(sigma_graph, f.type_size)


def _flag_as_element(f: Flag) -> FlagAlgebraElement:
    """Lift a bare Flag into a single-term FlagAlgebraElement with coefficient 1."""
    return FlagAlgebraElement(_flag_type(f), f.graph.n, [(1, f)])


def _flag_add(self: Flag, other: object) -> "FlagAlgebraElement":
    if isinstance(other, Flag):
        return _flag_as_element(self) + _flag_as_element(other)
    if isinstance(other, FlagAlgebraElement):
        return _flag_as_element(self) + other
    return NotImplemented


def _flag_sub(self: Flag, other: object) -> "FlagAlgebraElement":
    if isinstance(other, Flag):
        return _flag_as_element(self) - _flag_as_element(other)
    if isinstance(other, FlagAlgebraElement):
        return _flag_as_element(self) - other
    return NotImplemented


def _flag_neg(self: Flag) -> "FlagAlgebraElement":
    return -_flag_as_element(self)


def _flag_mul(self: Flag, other: object) -> "FlagAlgebraElement":
    if isinstance(other, (int, Fraction)) and not isinstance(other, bool):
        return _flag_as_element(self) * other
    if isinstance(other, Flag):
        return _flag_as_element(self) * _flag_as_element(other)
    if isinstance(other, FlagAlgebraElement):
        return _flag_as_element(self) * other
    return NotImplemented


def _flag_rmul(self: Flag, other: object) -> "FlagAlgebraElement":
    return _flag_mul(self, other)


def _flag_pow(self: Flag, exponent: int) -> "FlagAlgebraElement":
    return _flag_as_element(self) ** exponent


Flag.__add__ = _flag_add
Flag.__radd__ = _flag_add
Flag.__sub__ = _flag_sub
Flag.__neg__ = _flag_neg
Flag.__mul__ = _flag_mul
Flag.__rmul__ = _flag_rmul
Flag.__pow__ = _flag_pow


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
        Hypergraphs whose non-induced copies are forbidden.  Multiple entries
        are supported; a graph is admissible iff it avoids all of them.
    forbidden_induced:
        Hypergraphs whose induced copies are forbidden.
    target:
        What to optimize.  Accepts:

        * ``None`` — edge density (the default).
        * :class:`Hypergraph` — induced density of that graph.
        * :class:`DensityExpr` — an explicit rational linear combination
          Σᵢ cᵢ · d(·, Gᵢ) of induced densities.
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

    Linear-combination target — optimize edge density minus triangle density::

        from fractions import Fraction
        from zaszlo import DensityExpr, complete, FlagProblem
        expr = DensityExpr([
            (1, complete(2)),           # edge
            (-Fraction(3, 2), complete(3)),  # triangle penalty
        ])
        prob = FlagProblem(5, 3, 2, target=expr)
    """

    def __init__(
        self,
        n: int,
        type_order: int,
        k: int,
        *,
        forbidden: list[Hypergraph] | None = None,
        forbidden_induced: list[Hypergraph] | None = None,
        target: Optional[Union[Hypergraph, "DensityExpr"]] = None,
        minimize: bool = False,
        aux_constraints: list["AuxiliaryConstraint"] | None = None,
    ):
        if n < 2:
            raise ValueError(f"n must be at least 2, got {n}")
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
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
        patterns: list[Hypergraph] = list(forbidden or []) + list(forbidden_induced or [])
        if isinstance(target, Hypergraph):
            patterns.append(target)
        elif isinstance(target, DensityExpr):
            patterns.extend(g for _, g in target.terms)
        elif target is not None:
            raise ValueError(
                f"target must be None, a Hypergraph, or a DensityExpr; "
                f"got {type(target).__name__}"
            )
        for pat in patterns:
            if pat.k != k:
                raise ValueError(
                    f"All forbidden and target hypergraphs must have uniformity k={k}; "
                    f"got {pat!r} with k={pat.k}"
                )
        self.n = n
        self.type_order = type_order
        self.k = k
        self.forbidden: list[Hypergraph] = forbidden or []
        self.forbidden_induced: list[Hypergraph] = forbidden_induced or []
        self.target = target
        self.minimize = minimize
        self.aux_constraints: list["AuxiliaryConstraint"] = []
        for c in aux_constraints or []:
            self.add_constraint(c)

    def add_constraint(self, constraint: "AuxiliaryConstraint") -> None:
        """Register an auxiliary flag-algebra inequality ``⟦e⟧ ≥ 0`` on the SDP.

        The constraint's underlying :class:`UnlabeledExpr` must have uniformity
        matching this problem's ``k`` and vertex grade ``≤ n``.  Lifting to
        grade ``n`` happens automatically when the SDP data is built.

        Parameters
        ----------
        constraint:
            An :class:`AuxiliaryConstraint`.  Bare :class:`UnlabeledExpr` or
            :class:`FlagAlgebraElement` are also accepted for convenience —
            they will be wrapped / unlabeled as needed.
        """
        if isinstance(constraint, FlagAlgebraElement):
            from .algebra import unlabel
            constraint = AuxiliaryConstraint(unlabel(constraint))
        elif isinstance(constraint, UnlabeledExpr):
            constraint = AuxiliaryConstraint(constraint)
        elif not isinstance(constraint, AuxiliaryConstraint):
            raise ValueError(
                f"add_constraint expects an AuxiliaryConstraint, UnlabeledExpr, "
                f"or FlagAlgebraElement; got {type(constraint).__name__}"
            )
        if constraint.k != self.k:
            raise ValueError(
                f"constraint has k={constraint.k}, expected k={self.k}"
            )
        if constraint.n > self.n:
            raise ValueError(
                f"constraint at grade n={constraint.n} exceeds problem n={self.n}; "
                "aux constraints must be at grade ≤ problem n so they can be lifted."
            )
        self.aux_constraints.append(constraint)

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
        aux_coefficients: list[list[Fraction]] | None = None,
    ):
        """Store precomputed flag algebra data produced by build_flag_algebra_data.

        ``aux_coefficients[j][H_idx]`` gives the H-coefficient of the j-th
        auxiliary constraint after lifting to the problem's grade n.  Empty
        when the problem has no auxiliary constraints.
        """
        self.problem = problem
        self.types = types
        self.flags = flags
        self.admissible = admissible
        self.densities = densities
        self.pair_dens = pair_dens  # pair_dens[H_idx][sigma] = 2D list of Fraction
        self.aux_coefficients: list[list[Fraction]] = list(aux_coefficients or [])

    def __repr__(self) -> str:
        flag_counts = [len(fs) for fs in self.flags]
        return (
            f"FlagAlgebraData("
            f"types={len(self.types)}, "
            f"flags={flag_counts}, "
            f"admissible={len(self.admissible)})"
        )

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
    data:
        The originating FlagAlgebraData. Set automatically by solve_sdp and
        propagated through round_certificate; used by explain() and
        _repr_html_() to name sharp graphs and show densities.
    certificate:
        Exact rational certificate, populated by solve_and_certify(). None
        when the result comes from solve_sdp() or solve() directly.
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
        data: Optional["FlagAlgebraData"] = None,
        certificate: Optional["Certificate"] = None,
        mu: Optional[list[float]] = None,
        mu_exact: Optional[list[Fraction]] = None,
    ):
        """Store the output of solve_sdp. Constructed internally; not called directly.

        ``mu`` and ``mu_exact`` hold the auxiliary-constraint dual variables
        (nonneg per-constraint weights) — nonempty when the problem carries
        auxiliary flag-algebra inequalities via ``prob.aux_constraints``.
        """
        self.problem = problem
        self.status = status
        self.bound = bound
        self.Q = Q
        self.slacks = slacks
        self.cholesky_factors = cholesky_factors
        self.Q_exact = Q_exact
        self.bound_exact = bound_exact
        self.data = data
        self.certificate = certificate
        self.mu: list[float] = list(mu) if mu is not None else []
        self.mu_exact: Optional[list[Fraction]] = mu_exact

    def __repr__(self) -> str:
        q_desc = "not extracted" if self.Q is None else f"{len(self.Q)} matrices"
        l_desc = "" if self.cholesky_factors is None else "  (Cholesky factors stored)"
        if self.certificate is None:
            cert_desc = ""
        else:
            cert_status = "valid" if self.certificate.valid else "INVALID"
            cert_desc = f"\n  certified: {self.certificate.bound}  [{cert_status}]"
        return (
            f"FlagAlgebraResult\n"
            f"  status : {self.status}\n"
            f"  bound  : {self.bound}\n"
            f"  Q      : {q_desc}{l_desc}"
            f"{cert_desc}"
        )

    def explain(self) -> str:
        """Return a plain-text explanation of this result."""
        from .explain import explain_result
        return explain_result(self)

    def _repr_html_(self) -> str:
        from .explain import html_result
        return html_result(self)

    def _repr_mimebundle_(self, include=None, exclude=None, **kwargs) -> dict:
        bundle = {"text/html": self._repr_html_(), "text/plain": repr(self)}
        if include:
            bundle = {k: v for k, v in bundle.items() if k in include}
        if exclude:
            bundle = {k: v for k, v in bundle.items() if k not in exclude}
        return bundle

    def to_dict(self) -> dict:
        """Structured JSON-ready dict of this result (rationals as {num,den})."""
        from .serialize import result_to_dict
        return result_to_dict(self)

    def to_json(self, indent: int | None = 2) -> str:
        """JSON string of this result (rationals as {num,den})."""
        from .serialize import to_json
        return to_json(self.to_dict(), indent=indent)

    def diagnose(self) -> "DiagnosticReport":
        """Return a structured DiagnosticReport of solver observations.

        Surfaces observations (not interpretations) suitable for an agent to
        reason over.  See :class:`zaszlo.diagnose.DiagnosticReport` for fields.
        """
        from .diagnose import diagnose_result
        return diagnose_result(self)


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

    def to_dict(self) -> dict:
        """Structured JSON-ready dict of this SharpsResult (rationals as {num,den})."""
        from .serialize import sharps_to_dict
        return sharps_to_dict(self)

    def to_json(self, indent: int | None = 2) -> str:
        """JSON string of this SharpsResult (rationals as {num,den})."""
        from .serialize import to_json
        return to_json(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Certificate
# ---------------------------------------------------------------------------

@dataclass
class CertificateProvenance:
    """Record of how a :class:`Certificate` was produced.

    Populated by the certifier that built the certificate so that consumers
    (users, agents, JSON roundtrips) can inspect the pipeline that produced
    the object without re-running it.  Every field is optional; only the
    fields the producing pipeline knows about are set.

    Attributes
    ----------
    pipeline:
        Which entry point produced this certificate: ``'certify'`` (Cholesky
        rounding of a solved SDP) or ``'certify_at_bound'`` (kernel + rational
        correction at a user-supplied exact bound).
    denom_limit:
        Maximum denominator used during rationalisation (for either pipeline).
    chol_reg:
        Cholesky regularisation used by :func:`certify` (None for
        ``certify_at_bound``).
    feasibility_status:
        For ``certify_at_bound``: the Clarabel status of the fixed-λ
        feasibility solve (``'optimal'``, ``'infeasible'``, …).
    kernel_dims:
        For ``certify_at_bound``: dimension of the rational kernel identified
        in each Q_σ (one entry per type).
    correction:
        For ``certify_at_bound``: the ``info`` dict returned by
        :func:`_correct_T_for_sharp_graphs` — ``consistent``, ``rank``,
        ``float_rank``, ``dropped_rows``, ``num_sharp``, ``num_vars``,
        ``max_correction`` (formatted as ``"num/den"`` string).
    psd_safeguard:
        For ``certify_at_bound``: outcome of the post-correction PSD check —
        ``'not_run'``, ``'passed'``, ``'tripped_correction_discarded'``.
    reason:
        On any invalid path, a short human-readable explanation of what went
        wrong (e.g. ``"feasibility solve returned infeasible"``).
    """

    pipeline: Optional[str] = None
    denom_limit: Optional[int] = None
    chol_reg: Optional[float] = None
    feasibility_status: Optional[str] = None
    kernel_dims: Optional[list[int]] = None
    correction: Optional[dict] = None
    psd_safeguard: Optional[str] = None
    reason: Optional[str] = None

    def to_dict(self) -> dict:
        """JSON-ready dict of this provenance record.  Empty values are dropped."""
        out: dict = {}
        for k, v in self.__dict__.items():
            if v is None:
                continue
            out[k] = v
        return out


class Certificate:
    """Exact rational certificate for a flag algebra bound.

    Produced by :func:`certify` or :func:`certify_at_bound`. Consolidates the
    rounded PSD matrices, exact certified bound, per-graph residuals, and
    validity status in one object.

    Attributes
    ----------
    problem:
        The originating FlagProblem.
    bound:
        Exact certified bound as a Fraction.
    valid:
        True iff all residuals are ≥ 0 (Q is PSD by construction).
    Q:
        Exact rational PSD matrices, one per type.
    residuals:
        Exact rational residual for each admissible graph.
    data:
        The originating FlagAlgebraData; used by explain() to name graphs.
    provenance:
        :class:`CertificateProvenance` describing how this certificate was
        produced.  Never ``None``; a bare Certificate constructed by hand has
        an empty-fielded provenance.  Surfaced by ``explain()`` and
        ``diagnose()``.

    Properties
    ----------
    active_constraints:
        Indices of admissible graphs where the certificate identity is exactly
        tight (residual == 0).  These are the graphs where the sum-of-squares
        identity has no slack; they constrain the bound.
    """

    def __init__(
        self,
        problem: "FlagProblem",
        bound: Fraction,
        valid: bool,
        Q: list[list[list[Fraction]]],
        residuals: list[Fraction],
        data: Optional["FlagAlgebraData"] = None,
        mu: Optional[list[Fraction]] = None,
        provenance: Optional["CertificateProvenance"] = None,
    ):
        self.problem = problem
        self.bound = bound
        self.valid = valid
        self.Q = Q
        self.residuals = residuals
        self.data = data
        # Auxiliary-constraint weights (rational, ≥ 0).  Empty when the problem
        # has no aux constraints; otherwise mu[j] is the certified weight on
        # ⟦e_j⟧ ≥ 0 that appears in the residual identity.
        self.mu: list[Fraction] = list(mu) if mu is not None else []
        self.provenance: CertificateProvenance = (
            provenance if provenance is not None else CertificateProvenance()
        )

    @property
    def active_constraints(self) -> list[int]:
        """Admissible graph indices where the certificate identity is exactly tight."""
        return [i for i, r in enumerate(self.residuals) if r == Fraction(0)]

    def __repr__(self) -> str:
        status = "valid" if self.valid else "INVALID"
        return f"Certificate  bound={self.bound}  [{status}]"

    def explain(self) -> str:
        """Return a plain-text description of this certificate."""
        from .explain import explain_certificate
        return explain_certificate(self)

    def _repr_html_(self) -> str:
        from .explain import html_certificate
        return html_certificate(self)

    def _repr_mimebundle_(self, include=None, exclude=None, **kwargs) -> dict:
        bundle = {"text/html": self._repr_html_(), "text/plain": repr(self)}
        if include:
            bundle = {k: v for k, v in bundle.items() if k in include}
        if exclude:
            bundle = {k: v for k, v in bundle.items() if k not in exclude}
        return bundle

    def to_dict(self) -> dict:
        """Structured JSON-ready dict of this certificate (rationals as {num,den})."""
        from .serialize import certificate_to_dict
        return certificate_to_dict(self)

    def to_json(self, indent: int | None = 2) -> str:
        """JSON string of this certificate (rationals as {num,den})."""
        from .serialize import to_json
        return to_json(self.to_dict(), indent=indent)

    def diagnose(self) -> "DiagnosticReport":
        """Return a structured DiagnosticReport of certificate observations.

        Surfaces observations (not interpretations) suitable for an agent to
        reason over.  See :class:`zaszlo.diagnose.DiagnosticReport` for fields.
        """
        from .diagnose import diagnose_certificate
        return diagnose_certificate(self)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class CertificationError(RuntimeError):
    """Raised when rounded Q matrices fail exact rational verification.

    The sum-of-squares identity  bound − Σ_σ⟨Q_σ, P_σ(H)⟩ − density(H) ≥ 0
    must hold for every admissible graph H.  After rounding floating-point Q
    matrices to rationals, this identity can fail for some H (residual < 0),
    meaning the rounded certificate does not actually prove the bound.

    This is a numerical precision issue in the rounding step, not a failure of
    the underlying mathematical bound.  Increasing ``denom_limit`` (finer
    rational approximation) or adjusting ``chol_reg`` usually resolves it.

    Attributes
    ----------
    result:
        The :class:`FlagAlgebraResult` with ``result.certificate`` attached.
        Inspect ``err.result.certificate.residuals`` for the per-graph
        residuals and ``err.result.certificate.explain()`` for a full breakdown.
    """

    def __init__(self, result: "FlagAlgebraResult", denom_limit: int) -> None:
        self.result = result
        cert = result.certificate
        neg = [(i, r) for i, r in enumerate(cert.residuals) if r < 0]
        min_r = min(r for _, r in neg)
        worst_i = min(neg, key=lambda x: x[1])[0]
        msg = (
            f"Certificate invalid after rounding: {len(neg)} of "
            f"{len(cert.residuals)} admissible graph(s) have negative residual. "
            f"Worst offender: graph index {worst_i}, residual = {min_r}. "
            f"The rounded Q matrices fail the sum-of-squares identity for these graphs. "
            f"Try increasing denom_limit (currently {denom_limit}) or adjusting chol_reg. "
            f"Inspect err.result.certificate for full details."
        )
        super().__init__(msg)


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
