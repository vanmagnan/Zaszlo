"""Flag-algebra kernel operations: product in A^σ, unlabeling A^σ → A^∅, and
grade lifting in A^∅.

These functions compute the mathematical primitives underlying Razborov's flag
algebra formalism.  They operate in exact rational (Fraction) arithmetic and
return canonicalized homogeneous elements.

Definitions
-----------

Let σ be a type on s labeled vertices, and let f, g be σ-flags on n_f, n_g
vertices respectively (with the first s vertices carrying the label σ).

**Product in A^σ (`flag_product`).**  The bilinear product yields an element
supported on σ-flags of grade m = n_f + n_g − s.  For each such σ-flag h,

    (f · g)_h  =  P[ random 2-partition of h's m − s unlabeled vertices into
                     sizes (n_f − s, n_g − s) yields  f  on the first side
                     and  g  on the second side ].

Equivalently, the numerator counts ordered pairs (A, B) with |A| = n_f − s
and A ⊔ B = unlabeled vertices of h, such that σ ∪ A induces f and σ ∪ B
induces g; the denominator is the total number of such pairs C(m − s, n_f − s).

**Unlabeling A^σ → A^∅ (`unlabel`).**  For a σ-flag F on n vertices,

    ⟦F⟧  =  Σ_H  ( # σ-injections θ : {1..s} ↪ V(H) inducing F )
              / ( n · (n−1) · … · (n−s+1) )   ·   H,

where the sum is over admissible unlabeled graphs H on n vertices.

**Grade lifting in A^∅ (`lift_to`).**  An element u ∈ A^∅ at grade n' extends
to grade n_target ≥ n' by uniform random extension: for each basis graph
H_small in u and each admissible graph H_big on n_target vertices, the lift
contributes coefficient  c_{H_small} · induced_density(H_big, H_small).

Complexity note
---------------

These operations enumerate σ-flags / admissible graphs at various grades and
loop over vertex permutations or partitions.  Costs grow rapidly with vertex
count — expect subsecond behavior for the sizes zászló typically targets
(n ≤ 6 for k = 2 or 3) and observably slower work beyond that.  No caching or
memoization is attempted here; callers doing repeated computations should
cache their own generate_admissible/generate_flags results externally.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations, permutations
from math import comb
from typing import TYPE_CHECKING

from .densities import induced_density
from .generation import generate_admissible, generate_flags
from .graphs import canonical, induce
from .types import Flag, FlagAlgebraElement, Hypergraph, UnlabeledExpr

if TYPE_CHECKING:
    pass


__all__ = ["flag_product", "unlabel", "lift_to"]


# ---------------------------------------------------------------------------
# Product in A^σ
# ---------------------------------------------------------------------------

def flag_product(f: Flag, g: Flag) -> list[tuple[Fraction, Flag]]:
    """Return the product f · g in A^σ as a list of ``(coefficient, flag)`` pairs.

    The two operand flags must share the same type σ (same ``type_size`` and
    the same induced substructure on their labeled vertices).  The result
    lives on ``m = f.graph.n + g.graph.n − s`` vertices.  Each returned flag
    is canonical; coefficients are nonzero exact ``Fraction`` values.

    Parameters
    ----------
    f, g:
        σ-flags with matching type.

    Returns
    -------
    list of (Fraction, Flag)
        The nonzero terms of ``f · g`` in canonical form.  Sum this list into
        a :class:`FlagAlgebraElement` if you need element-level operations.

    Raises
    ------
    ValueError
        If the two flags disagree on type size, uniformity, or induced type
        substructure.
    """
    if f.type_size != g.type_size:
        raise ValueError(
            f"flag_product requires matching type size; got {f.type_size} vs {g.type_size}"
        )
    if f.graph.k != g.graph.k:
        raise ValueError(
            f"flag_product requires matching uniformity k; got {f.graph.k} vs {g.graph.k}"
        )
    s = f.type_size
    k = f.graph.k
    n_f = f.graph.n
    n_g = g.graph.n
    m = n_f + n_g - s

    # Canonicalize the operands so their edge lists are the comparison keys.
    f_canon = canonical(f)
    g_canon = canonical(g)

    # Verify both flags induce the same σ on their labeled prefix.
    sigma_f_edges = induce(f_canon.graph, list(range(1, s + 1))).edges
    sigma_g_edges = induce(g_canon.graph, list(range(1, s + 1))).edges
    if sigma_f_edges != sigma_g_edges:
        raise ValueError(
            "flag_product operands induce different types σ on labeled vertices"
        )

    f_key = f_canon.graph.edges
    g_key = g_canon.graph.edges

    # Build σ (as a Flag) so we can generate σ-flags on m vertices.
    sigma_flag = Flag(Hypergraph(s, k, list(sigma_f_edges)), s)

    # All σ-flags on m vertices, unrestricted (no forbidden patterns here —
    # the algebra product is a purely combinatorial operation, independent of
    # any forbidden-subgraph constraint the ambient FlagProblem may impose).
    if m == s:
        # Degenerate case: the only "m-vertex" σ-flag is σ itself.
        m_flags = [sigma_flag]
    else:
        m_flags = generate_flags(m, sigma_flag, forbidden=[], forbidden_induced=[])

    total_partitions = comb(m - s, n_f - s)
    unlabeled = list(range(s + 1, m + 1))

    result: list[tuple[Fraction, Flag]] = []
    for h in m_flags:
        h_edges = h.graph.edges
        # Count partitions (A, B) with |A| = n_f − s such that σ ∪ A ≅ f and σ ∪ B ≅ g.
        count = 0
        for A_tup in combinations(unlabeled, n_f - s):
            verts_A = list(range(1, s + 1)) + list(A_tup)
            sub_f_graph = induce(Hypergraph(m, k, h_edges), verts_A)
            if canonical(Flag(sub_f_graph, s)).graph.edges != f_key:
                continue
            B_list = [v for v in unlabeled if v not in A_tup]
            verts_B = list(range(1, s + 1)) + B_list
            sub_g_graph = induce(Hypergraph(m, k, h_edges), verts_B)
            if canonical(Flag(sub_g_graph, s)).graph.edges == g_key:
                count += 1
        if count > 0:
            result.append((Fraction(count, total_partitions), h))
    return result


# ---------------------------------------------------------------------------
# Unlabeling A^σ → A^∅
# ---------------------------------------------------------------------------

def unlabel(element: FlagAlgebraElement) -> UnlabeledExpr:
    """Compute ⟦e⟧_σ : A^σ → A^∅ at the same grade.

    Averages each basis σ-flag over uniformly random σ-injections into an
    admissible graph on the same vertex count.  Returns an
    :class:`UnlabeledExpr` at grade ``element.n`` whose evaluation on a large
    graphon coincides with the flag-algebra value of the original element.

    Parameters
    ----------
    element:
        A :class:`FlagAlgebraElement` (any homogeneous element of A^σ).

    Returns
    -------
    UnlabeledExpr
        The unlabeled element, canonicalized over admissible-graph iso classes.

    Notes
    -----
    - Zero elements unlabel to zero at the same grade.
    - Grade lifting to a larger vertex count is a separate operation; see
      :func:`lift_to`.
    """
    n = element.n
    k = element.k
    s = element.s
    sigma_edges = element.type.graph.edges

    if element.is_zero:
        return UnlabeledExpr(k, n, ())

    # Ordered σ-injections into an n-vertex graph number n · (n−1) · … · (n−s+1).
    injection_denom = 1
    for i in range(s):
        injection_denom *= (n - i)

    all_admissible = generate_admissible(n, k)

    coeff_by_H: dict[Hypergraph, Fraction] = {}
    for coef, F in element.terms:
        F_canon_edges = F.graph.edges  # element.terms stores canonical flags
        for H in all_admissible:
            count = 0
            for theta in permutations(range(1, H.n + 1), s):
                theta_list = list(theta)
                # θ must induce σ on the labeled positions.
                if induce(H, theta_list).edges != sigma_edges:
                    continue
                # The σ-flag structure on V(H) via θ: labeled = θ (in order),
                # unlabeled = the rest (in canonical order).
                rest = [v for v in range(1, H.n + 1) if v not in theta_list]
                sub_g = induce(H, theta_list + rest)
                if canonical(Flag(sub_g, s)).graph.edges == F_canon_edges:
                    count += 1
            if count > 0:
                contribution = coef * Fraction(count, injection_denom)
                coeff_by_H[H] = coeff_by_H.get(H, Fraction(0)) + contribution

    terms = [(c, H) for H, c in coeff_by_H.items() if c != 0]
    return UnlabeledExpr(k, n, terms)


# ---------------------------------------------------------------------------
# Grade lifting in A^∅
# ---------------------------------------------------------------------------

def lift_to(u: UnlabeledExpr, n_target: int) -> UnlabeledExpr:
    """Lift an A^∅ element from grade ``u.n`` to grade ``n_target ≥ u.n``.

    Uses uniform random extension: for each basis graph ``H_small`` in ``u``,
    the coefficient on an admissible ``H_big`` at grade ``n_target`` picks up
    ``induced_density(H_big, H_small)`` times ``u``'s original coefficient.

    Parameters
    ----------
    u:
        The element to lift.
    n_target:
        The target grade.  Must satisfy ``n_target ≥ u.n``.

    Returns
    -------
    UnlabeledExpr
        The lifted element at grade ``n_target``, canonicalized over admissible
        graphs on ``n_target`` vertices.

    Raises
    ------
    ValueError
        If ``n_target < u.n``.
    """
    if n_target < u.n:
        raise ValueError(
            f"lift_to cannot lower grade: expr at n={u.n}, target n={n_target}"
        )
    if n_target == u.n:
        return u
    if u.is_zero:
        return UnlabeledExpr(u.k, n_target, ())

    all_bigger = generate_admissible(n_target, u.k)
    coeff_by_H: dict[Hypergraph, Fraction] = {}
    for c, H_small in u.terms:
        for H_big in all_bigger:
            d = induced_density(H_big, H_small)
            if d == 0:
                continue
            contribution = c * d
            coeff_by_H[H_big] = coeff_by_H.get(H_big, Fraction(0)) + contribution

    terms = [(c, H) for H, c in coeff_by_H.items() if c != 0]
    return UnlabeledExpr(u.k, n_target, terms)
