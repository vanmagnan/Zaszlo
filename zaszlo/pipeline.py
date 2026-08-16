# High-level pipeline: FlagProblem → FlagAlgebraData.

from __future__ import annotations

from fractions import Fraction

from .algebra import lift_to
from .densities import compute_pair_densities, edge_density, induced_density
from .generation import generate_admissible, generate_flags, generate_types
from .types import DensityExpr, FlagAlgebraData, FlagProblem, Hypergraph


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
    target = prob.target
    if target is None:
        densities = [edge_density(H) for H in admissible]
    elif isinstance(target, Hypergraph):
        densities = [induced_density(H, target) for H in admissible]
    elif isinstance(target, DensityExpr):
        densities = [target.evaluate(H) for H in admissible]
    else:
        # FlagProblem.__init__ already validates this; guard here defensively
        # so future callers get an obvious error rather than a silent miscount.
        raise TypeError(
            f"Unsupported target type {type(target).__name__}; expected None, "
            "Hypergraph, or DensityExpr."
        )

    # Pair densities: pair_dens[H_idx][sigma] = upper-triangular Fraction matrix.
    pair_dens = [
        compute_pair_densities(H, types, flags)
        for H in admissible
    ]

    # Auxiliary constraint coefficients (per-H, lifted to problem grade n).
    # aux_coefficients[j][H_idx] = coefficient of admissible[H_idx] in
    # lift_to(constraint_j.expr, n).  Empty when there are no aux constraints.
    aux_coefficients: list[list[Fraction]] = []
    for constraint in prob.aux_constraints:
        lifted = lift_to(constraint.expr, n)
        coef_by_H: dict[Hypergraph, Fraction] = dict(
            (h, c) for c, h in lifted.terms
        )
        row = [coef_by_H.get(H, Fraction(0)) for H in admissible]
        aux_coefficients.append(row)

    return FlagAlgebraData(
        prob, types, flags, admissible, densities, pair_dens,
        aux_coefficients=aux_coefficients,
    )
