# Flag and admissible graph generation.

from __future__ import annotations

from itertools import combinations

from .graphs import canonical, has_subgraph, has_induced_subgraph, induce
from .types import Flag, Hypergraph


def _all_possible_edges(n: int, k: int) -> list[tuple[int, ...]]:
    """All sorted K-tuples on vertices 1..n."""
    return list(combinations(range(1, n + 1), k))


def generate_flags(
    n: int,
    type_flag: Flag,
    forbidden: list[Hypergraph],
    forbidden_induced: list[Hypergraph],
) -> list[Flag]:
    """Generate all non-isomorphic K-uniform flags on n vertices over type_flag.

    The seed (type edges embedded in n vertices, unlabeled vertices isolated)
    is always included. Forbidden subgraph checks only examine vertex subsets
    containing the newly added edge (mirrors C optimization).

    Parameters
    ----------
    n:
        Total vertex count of each flag.
    type_flag:
        The type: a Flag with type_size == graph.n (all vertices labeled).
    forbidden:
        Forbidden non-induced subgraphs.
    forbidden_induced:
        Forbidden induced subgraphs (filtered in a second pass).
    """
    s = type_flag.type_size
    k = type_flag.graph.k

    # Candidate edges: those involving at least one unlabeled vertex.
    candidate_edges = [
        e for e in _all_possible_edges(n, k) if any(v > s for v in e)
    ]

    # Seed flag: type edges embedded in n vertices.
    seed_graph = Hypergraph(n, k, list(type_flag.graph.edges))
    seed_flag = canonical(Flag(seed_graph, s))

    # layers[i] = flags with exactly i extra edges beyond the type.
    layers: list[list[Flag]] = [[seed_flag]]

    for _ in range(len(candidate_edges)):
        current_layer: list[Flag] = []
        seen: set[tuple[tuple[int, ...], ...]] = set()

        for f in layers[-1]:
            edge_set = set(f.graph.edges)

            for new_edge in candidate_edges:
                if new_edge in edge_set:
                    continue

                new_edges = sorted(list(f.graph.edges) + [new_edge])
                g = Hypergraph(n, k, new_edges)
                candidate = Flag(g, s)

                # Check forbidden subgraphs only on subsets containing new_edge.
                is_bad = False
                for fg in forbidden:
                    for verts in combinations(range(1, n + 1), fg.n):
                        if not any(v in verts for v in new_edge):
                            continue
                        sub = induce(g, list(verts))
                        if has_subgraph(sub, fg):
                            is_bad = True
                            break
                    if is_bad:
                        break
                if is_bad:
                    continue

                canon = canonical(candidate)
                key = tuple(canon.graph.edges)
                if key not in seen:
                    seen.add(key)
                    current_layer.append(canon)

        if not current_layer:
            break
        layers.append(current_layer)

    all_flags = [f for layer in layers for f in layer]

    # Second pass: remove flags with forbidden induced subgraphs.
    if forbidden_induced:
        all_flags = [
            f for f in all_flags
            if not any(has_induced_subgraph(f.graph, fg) for fg in forbidden_induced)
        ]

    return all_flags


def generate_admissible(
    n: int,
    k: int,
    *,
    forbidden: list[Hypergraph] | None = None,
    forbidden_induced: list[Hypergraph] | None = None,
) -> list[Hypergraph]:
    """Generate all non-isomorphic admissible K-uniform hypergraphs on n vertices."""
    forbidden = forbidden or []
    forbidden_induced = forbidden_induced or []
    empty_type = Flag(Hypergraph(0, k, []), 0)
    flags = generate_flags(n, empty_type, forbidden, forbidden_induced)
    return [f.graph for f in flags]


def generate_types(
    order: int,
    k: int,
    *,
    forbidden: list[Hypergraph] | None = None,
    forbidden_induced: list[Hypergraph] | None = None,
) -> list[Flag]:
    """Generate all non-isomorphic types of a given vertex order.

    A type is an admissible graph with all vertices labeled.
    """
    graphs = generate_admissible(order, k, forbidden=forbidden, forbidden_induced=forbidden_induced)
    return [Flag(g, g.n) for g in graphs]
