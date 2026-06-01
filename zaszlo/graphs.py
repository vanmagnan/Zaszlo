# Graph operations: induction, canonical forms, subgraph tests.

from __future__ import annotations

from itertools import combinations, permutations

from .types import Flag, Hypergraph


# ---------------------------------------------------------------------------
# Induced subgraph
# ---------------------------------------------------------------------------

def induce(g: Hypergraph, verts: list[int]) -> Hypergraph:
    """Induced subgraph of g on vertex subset verts (1-based indices).

    Vertices are reindexed 1..len(verts) in the order given.
    """
    idx = {v: i + 1 for i, v in enumerate(verts)}
    new_edges = []
    for e in g.edges:
        if all(v in idx for v in e):
            new_edges.append(tuple(idx[v] for v in e))
    return Hypergraph(len(verts), g.k, new_edges)


# ---------------------------------------------------------------------------
# Canonical form
# ---------------------------------------------------------------------------

def canonical(f: Flag) -> Flag:
    """Lex-minimum edge list over all permutations of unlabeled vertices.

    Labeled vertices (1..type_size) are held fixed.
    Two flags are isomorphic iff their canonical forms are equal.
    """
    s = f.type_size
    n = f.graph.n
    best_edges = f.graph.edges  # already normalized on construction

    for perm in permutations(range(s + 1, n + 1)):
        # Build relabeling: labeled vertices stay, unlabeled vertices are permuted.
        relabel = {i: i for i in range(1, s + 1)}
        for j, v in enumerate(perm):
            relabel[s + 1 + j] = v

        new_edges = sorted(
            tuple(sorted(relabel[v] for v in e)) for e in f.graph.edges
        )
        if new_edges < best_edges:
            best_edges = new_edges

    return Flag(Hypergraph(n, f.graph.k, best_edges), s)


def flag_isomorphic(f1: Flag, f2: Flag) -> bool:
    """True iff f1 and f2 are isomorphic as flags (same type_size fixed)."""
    return (
        f1.type_size == f2.type_size
        and f1.graph.n == f2.graph.n
        and canonical(f1).graph.edges == canonical(f2).graph.edges
    )


# ---------------------------------------------------------------------------
# Subgraph tests
# ---------------------------------------------------------------------------

def has_subgraph(g: Hypergraph, sg: Hypergraph) -> bool:
    """True iff g contains sg as a (not necessarily induced) subgraph."""
    edge_set = set(g.edges)
    for perm in permutations(range(1, g.n + 1), sg.n):
        if all(
            tuple(sorted(perm[v - 1] for v in e)) in edge_set
            for e in sg.edges
        ):
            return True
    return False


def has_induced_subgraph(g: Hypergraph, sg: Hypergraph) -> bool:
    """True iff g contains sg as an induced subgraph."""
    sg_canon = canonical(Flag(sg, 0)).graph.edges
    for verts in combinations(range(1, g.n + 1), sg.n):
        sub = induce(g, list(verts))
        if canonical(Flag(sub, 0)).graph.edges == sg_canon:
            return True
    return False
