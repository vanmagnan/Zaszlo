# High-level pipeline: FlagProblem → FlagAlgebraData.

from __future__ import annotations

from .densities import compute_pair_densities, edge_density, induced_density
from .generation import generate_admissible, generate_flags, generate_types
from .types import FlagAlgebraData, FlagProblem


def build_flag_algebra_data(prob: FlagProblem) -> FlagAlgebraData:
    """Run the full pre-SDP pipeline for a FlagProblem.

    Steps
    -----
    1. generate_types   — all non-isomorphic types at each valid order
    2. generate_flags   — flags over each type at the appropriate flag size
    3. generate_admissible — admissible graphs on n vertices
    4. densities        — edge density (or induced density of target) for each H
    5. compute_pair_densities — pair density matrices for every (H, type) pair

    Valid type orders: all s with n-s even (same parity as n),
    from n%2 up to prob.type_order in steps of 2.
    """
    n = prob.n
    k = prob.k

    # Collect types from all valid orders.
    min_s = n % 2  # 0 if n even, 1 if n odd (ensures n-s is always even)
    types = []
    for s in range(min_s, prob.type_order + 1, 2):
        types.extend(
            generate_types(
                s, k,
                forbidden=prob.forbidden,
                forbidden_induced=prob.forbidden_induced,
            )
        )

    # Flags over each type at flag size m = floor((n+s)/2).
    flags = []
    for t in types:
        s = t.type_size
        m = (n + s) // 2 if (n - s) % 2 == 0 else (n + s - 1) // 2
        flags.append(
            generate_flags(m, t, prob.forbidden, prob.forbidden_induced)
        )

    # Admissible graphs on n vertices.
    admissible = generate_admissible(
        n, k,
        forbidden=prob.forbidden,
        forbidden_induced=prob.forbidden_induced,
    )

    # Density of each admissible graph.
    if prob.target is None:
        densities = [edge_density(H) for H in admissible]
    else:
        densities = [induced_density(H, prob.target) for H in admissible]

    # Pair densities: pair_dens[H_idx][sigma] = upper-triangular Fraction matrix.
    pair_dens = [
        compute_pair_densities(H, types, flags)
        for H in admissible
    ]

    return FlagAlgebraData(prob, types, flags, admissible, densities, pair_dens)
