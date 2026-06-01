# Density computations: edge density, induced subgraph density, pair densities.
#
# Pair density normalization (mirrors Julia implementation):
#   Uses ordered iteration (both orderings of each half-pair), so:
#     total    = 2 × C's tcnum
#     diagonal:     counts[i][i] / total
#     off-diagonal: counts[i][j] / (2 * total)
#   This matches the C binary's output exactly.

from __future__ import annotations

from fractions import Fraction
from itertools import combinations, permutations
from math import comb

from .graphs import canonical, induce
from .types import Flag, Hypergraph


def edge_density(g: Hypergraph) -> Fraction:
    """Edge density of g: number of edges / C(n, k)."""
    denom = comb(g.n, g.k)
    if denom == 0:
        return Fraction(0)
    return Fraction(len(g.edges), denom)


def induced_density(g: Hypergraph, h: Hypergraph) -> Fraction:
    """Density of h as an induced subgraph of g.

    Fraction of k-subsets of V(g) whose induced subgraph is isomorphic to h.
    """
    h_canon = canonical(Flag(h, 0)).graph.edges
    count = sum(
        1
        for verts in combinations(range(1, g.n + 1), h.n)
        if canonical(Flag(induce(g, list(verts)), 0)).graph.edges == h_canon
    )
    denom = comb(g.n, h.n)
    if denom == 0:
        return Fraction(0)
    return Fraction(count, denom)


def compute_pair_densities(
    H: Hypergraph,
    types: list[Flag],
    flags: list[list[Flag]],
) -> list[list[list[Fraction]]]:
    """Compute pair density matrices for one admissible graph H.

    Returns a list (one per type σ) of upper-triangular matrices stored as
    lists-of-lists of Fraction.

    pair_dens[σ][i][j]  (i ≤ j) = probability that a random injection of type
    vertices, extended by two independent random flag halves, yields flags i and
    j over type σ.
    """
    n = H.n
    num_types = len(types)

    # Precompute canonical edge lists once.
    type_canons = [canonical(t).graph.edges for t in types]
    flag_canons = [
        [canonical(f).graph.edges for f in flags[sigma]]
        for sigma in range(num_types)
    ]

    result = []

    for sigma in range(num_types):
        s = types[sigma].type_size
        half = ((n - s) // 2) if (n - s) % 2 == 0 else ((n - s - 1) // 2)
        nf = len(flags[sigma])

        counts = [[0] * nf for _ in range(nf)]
        total = 0

        for type_perm in permutations(range(1, n + 1), s):
            type_sub = induce(H, list(type_perm))
            type_matches = type_sub.edges == type_canons[sigma]

            rest = [v for v in range(1, n + 1) if v not in type_perm]

            for half1 in combinations(rest, half):
                rest2 = [v for v in rest if v not in half1]
                if len(rest2) < half:
                    continue

                for half2 in combinations(rest2, half):
                    # Count ALL (type_perm, half1, half2) regardless of type match.
                    total += 1

                    if not type_matches:
                        continue

                    f1_idx = _find_flag(H, type_perm, half1, flag_canons[sigma], s)
                    f2_idx = _find_flag(H, type_perm, half2, flag_canons[sigma], s)

                    if f1_idx is None or f2_idx is None:
                        continue

                    i, j = min(f1_idx, f2_idx), max(f1_idx, f2_idx)
                    counts[i][j] += 1

        # Normalize into Fraction upper-triangular matrix.
        mat = [[Fraction(0)] * nf for _ in range(nf)]
        if total > 0:
            for i in range(nf):
                for j in range(i, nf):
                    if i == j:
                        mat[i][j] = Fraction(counts[i][j], total)
                    else:
                        mat[i][j] = Fraction(counts[i][j], 2 * total)

        result.append(mat)

    return result


def _find_flag(
    H: Hypergraph,
    type_perm: tuple[int, ...],
    half_verts: tuple[int, ...],
    flag_canons: list[list[tuple[int, ...]]],
    s: int,
) -> int | None:
    """Find index of the flag induced by (type_perm, half_verts) in flag_canons."""
    verts = list(type_perm) + list(half_verts)
    sub = induce(H, verts)
    sub_canon = canonical(Flag(sub, s)).graph.edges
    for idx, fc in enumerate(flag_canons):
        if fc == sub_canon:
            return idx
    return None
