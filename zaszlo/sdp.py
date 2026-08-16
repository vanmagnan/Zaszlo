# SDP assembly and solving via Clarabel.
#
# Two modes, controlled by prob.minimize:
#
#   Upper bound (minimize=False, maximize density):
#     Minimize λ  subject to:  λ - Σ_σ <P_σ(H_i), Q_σ> - Σ_j μ_j c_{H_i,j} - s_i = density(H_i)
#     Q_σ ≽ 0, P_σ(H_i) ≽ 0, μ_j ≥ 0  →  effective:  λ ≥ density(H_i) + <Q,P> + Σ_j μ_j c_{H_i,j}
#
#   Lower bound (minimize=True, minimize density):
#     Maximize λ  subject to:  λ + Σ_σ <P_σ(H_i), Q_σ> + Σ_j μ_j c_{H_i,j} - s_i = density(H_i)
#     effective:  λ ≤ density(H_i) − <Q,P> − Σ_j μ_j c_{H_i,j}
#
# The extra μ_j ≥ 0 variables carry auxiliary flag-algebra inequalities
# ⟦e_j⟧ ≥ 0 supplied via prob.aux_constraints.  c_{H_i,j} is the coefficient of
# H_i in the grade-n lift of e_j.  In any density-weighted sum Σ_i ρ_i c_{H_i,j}
# = ⟦e_j⟧(G) → nonneg on admissible graphons — so μ_j c_{H_i,j} contributes a
# nonneg extra term in the weighted identity, and the bound remains valid.
#
# Inner product <P, Q> for symmetric Q and upper-triangular P:
#   Σ_j P[j,j]*Q[j,j]  +  2 * Σ_{j<k} P[j,k]*Q[j,k]
#
# Clarabel standard form:  min (1/2) x'Px + q'x  s.t.  Ax + s = b,  s ∈ K
# Cone ordering: [ZeroCone (equalities) | PSDCone per type | NonnegCone (slacks + μ_j)]
# Q variables stored in x using column-major upper-triangular order; the √2
# off-diagonal scaling required by Clarabel's PSD cone appears only in A.

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any, Optional

import clarabel
import numpy as np
import scipy.sparse as sp

from .types import Certificate, FlagAlgebraData, FlagAlgebraResult, FlagProblem, SharpsResult

_SQRT2 = math.sqrt(2.0)

_STATUS_MAP = {
    "Solved": "optimal",
    "AlmostSolved": "optimal_inaccurate",
    "MaxIterations": "optimal_inaccurate",
    "InsufficientProgress": "optimal_inaccurate",
    "PrimalInfeasible": "infeasible",
    "DualInfeasible": "dual_infeasible",
    "NumericalError": "numerical_error",
}


def _tri_idx(j: int, k: int) -> int:
    """Column-major upper-triangular index for (j, k) with j ≤ k (0-based)."""
    return k * (k + 1) // 2 + j


# ---------------------------------------------------------------------------
# SDP construction
# ---------------------------------------------------------------------------

def build_sdp(
    data: FlagAlgebraData,
    *,
    sharps: list[int] | None = None,
) -> dict[str, Any]:
    """Build the flag algebra SDP in Clarabel's standard conic form.

    Parameters
    ----------
    data:
        Precomputed flag algebra data.  When ``data.aux_coefficients`` is
        non-empty, one extra nonneg SDP variable ``μ_j`` is added per aux
        constraint, with its column set to ``−c_{H,j}`` in each H equation
        (see the module docstring for the sign derivation).
    sharps:
        Indices (0-based) of admissible graphs known to be extremal. These
        get equality constraints with no slack variable.

    Returns
    -------
    dict with keys:
        'P', 'q'       — objective (P is zero; objective is linear in λ)
        'A', 'b'       — constraint matrix and RHS in Ax + s = b, s ∈ K
        'cones'        — Clarabel cone list
        'q_offsets'    — start index of each Q_σ block in x
        'flag_sizes'   — number of flags per type
        'slack_offset' — start index of slack block in x
        'non_sharp'    — ordered list of non-sharp H indices
        'slack_map'    — maps H index → local slack position
        'mu_offset'    — start index of μ block in x (or slack_offset+num_slacks
                          if no aux constraints)
        'num_aux'      — number of auxiliary constraints
    """
    sharps = set(sharps or [])
    num_types = len(data.types)
    num_H = len(data.admissible)
    num_aux = len(data.aux_coefficients)

    flag_sizes = [len(data.flags[sigma]) for sigma in range(num_types)]
    tri_sizes = [n * (n + 1) // 2 for n in flag_sizes]

    non_sharp = [i for i in range(num_H) if i not in sharps]
    num_slacks = len(non_sharp)
    slack_map = {i: loc for loc, i in enumerate(non_sharp)}

    # Variable layout in x: [λ, Q_0_vec, ..., slack_0, ..., μ_0, ..., μ_{num_aux-1}]
    q_offsets: list[int] = []
    offset = 1
    for ts in tri_sizes:
        q_offsets.append(offset)
        offset += ts
    slack_offset = offset
    mu_offset = slack_offset + num_slacks
    total_vars = mu_offset + num_aux

    # Clarabel always minimizes; negate λ for the lower-bound (maximize) case.
    q_obj = np.zeros(total_vars)
    q_obj[0] = 1.0 if not data.problem.minimize else -1.0

    csign = 1 if data.problem.minimize else -1

    num_psd_rows = sum(tri_sizes)
    # Nonneg cone covers both slack variables and μ_j (both ≥ 0).
    num_nonneg = num_slacks + num_aux
    total_rows = num_H + num_psd_rows + num_nonneg

    rows_coo: list[int] = []
    cols_coo: list[int] = []
    vals_coo: list[float] = []
    b = np.zeros(total_rows)

    # --- Equality constraints (ZeroCone, rows 0..num_H-1) ---
    for i in range(num_H):
        rows_coo.append(i); cols_coo.append(0); vals_coo.append(1.0)

        for sigma in range(num_types):
            nf = flag_sizes[sigma]
            P = data.pair_dens[i][sigma]
            q_off = q_offsets[sigma]
            for j in range(nf):
                for kk in range(j, nf):
                    p_val = float(P[j][kk])
                    if p_val == 0.0:
                        continue
                    factor = 1.0 if j == kk else 2.0
                    rows_coo.append(i)
                    cols_coo.append(q_off + _tri_idx(j, kk))
                    vals_coo.append(csign * factor * p_val)

        if i not in sharps:
            rows_coo.append(i)
            cols_coo.append(slack_offset + slack_map[i])
            vals_coo.append(-1.0)

        # Aux constraint columns: coefficient of μ_j in row i is −c_{H_i,j}
        # (upper) or +c_{H_i,j} (lower).  Equivalently: sign = csign, since
        # csign is −1 for upper and +1 for lower — but that's a coincidence
        # of sign conventions.  The DERIVED sign for both modes is the SAME
        # (subtract μ·c from LHS in upper, add in lower — see module docstring).
        # Concretely: aux column entry = csign * c_{H_i,j} makes the effective
        # constraint slack = λ - density - <Q,P> - μ·c (upper) or
        # slack = density - λ - <Q,P> - μ·c (lower), and both variants give
        # rise to a "+μ·⟦e⟧" nonneg contribution in the weighted-density sum
        # that proves the bound.  See the sign-derivation comment in the
        # module docstring above for the algebra.
        for j in range(num_aux):
            c = float(data.aux_coefficients[j][i])
            if c == 0.0:
                continue
            rows_coo.append(i)
            cols_coo.append(mu_offset + j)
            vals_coo.append(csign * c)

        b[i] = float(data.densities[i])

    # --- PSD constraints (one block per type, rows num_H..num_H+Σtri-1) ---
    # s_psd = b - A*x lies in Clarabel's PSD cone (b=0 here).
    # Clarabel uses column-major upper-triangular with √2 scaling on off-diagonals.
    psd_row_base = num_H
    for sigma in range(num_types):
        nf = flag_sizes[sigma]
        q_off = q_offsets[sigma]
        row_base = psd_row_base + sum(tri_sizes[:sigma])
        for k in range(nf):
            for j in range(k + 1):
                idx = _tri_idx(j, k)
                scale = 1.0 if j == k else _SQRT2
                rows_coo.append(row_base + idx)
                cols_coo.append(q_off + idx)
                vals_coo.append(-scale)

    # --- Non-negativity constraints (NonnegCone: slacks then μ_j) ---
    nonneg_base = num_H + num_psd_rows
    for loc in range(num_slacks):
        rows_coo.append(nonneg_base + loc)
        cols_coo.append(slack_offset + loc)
        vals_coo.append(-1.0)
    for j in range(num_aux):
        rows_coo.append(nonneg_base + num_slacks + j)
        cols_coo.append(mu_offset + j)
        vals_coo.append(-1.0)

    A = sp.csc_matrix(
        (vals_coo, (rows_coo, cols_coo)),
        shape=(total_rows, total_vars),
    )
    P_zero = sp.csc_matrix((total_vars, total_vars))

    cones: list[Any] = [clarabel.ZeroConeT(num_H)]
    for nf in flag_sizes:
        cones.append(clarabel.PSDTriangleConeT(nf))  # nf = matrix size n, not n*(n+1)/2
    if num_nonneg > 0:
        cones.append(clarabel.NonnegativeConeT(num_nonneg))

    return {
        "P": P_zero,
        "q": q_obj,
        "A": A,
        "b": b,
        "cones": cones,
        "q_offsets": q_offsets,
        "flag_sizes": flag_sizes,
        "slack_offset": slack_offset,
        "non_sharp": non_sharp,
        "slack_map": slack_map,
        "mu_offset": mu_offset,
        "num_aux": num_aux,
    }


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------

def solve_sdp(
    data: FlagAlgebraData,
    *,
    sharps: list[int] | None = None,
    extract_Q: bool = False,
    verbose: bool = False,
    settings=None,
) -> FlagAlgebraResult:
    """Build and solve the flag algebra SDP using Clarabel.

    Parameters
    ----------
    data:
        Precomputed flag algebra data.
    sharps:
        0-based indices of known extremal admissible graphs.
    extract_Q:
        If True, store the PSD certificate matrices in the result.
    verbose:
        If True, print Clarabel solver output.
    settings:
        Optional ``clarabel.DefaultSettings`` instance for fine-grained solver
        control (tolerances, iteration limits, etc.).

    Returns
    -------
    FlagAlgebraResult
    """
    built = build_sdp(data, sharps=sharps)

    if settings is None:
        settings = clarabel.DefaultSettings()
    settings.verbose = verbose

    solver = clarabel.DefaultSolver(
        built["P"], built["q"], built["A"], built["b"], built["cones"], settings
    )
    sol = solver.solve()

    status_key = str(sol.status).split(".")[-1]
    status = _STATUS_MAP.get(status_key, "unknown")

    x = np.asarray(sol.x) if sol.x is not None else None
    bound = float(x[0]) if x is not None else float("nan")

    Q_vals: Optional[list[np.ndarray]] = None
    if extract_Q and x is not None:
        Q_vals = []
        for sigma, nf in enumerate(built["flag_sizes"]):
            q_off = built["q_offsets"][sigma]
            Q = np.zeros((nf, nf))
            for k in range(nf):
                for j in range(k + 1):
                    val = float(x[q_off + _tri_idx(j, k)])
                    Q[j, k] = val
                    Q[k, j] = val
            Q_vals.append(Q)

    num_H = len(data.admissible)
    sharps_set = set(sharps or [])
    slack_map = built["slack_map"]
    slack_offset = built["slack_offset"]
    mu_offset = built["mu_offset"]
    num_aux = built["num_aux"]

    slacks: list[float] = []
    for i in range(num_H):
        if i in sharps_set:
            slacks.append(0.0)
        elif x is not None:
            slacks.append(float(x[slack_offset + slack_map[i]]))
        else:
            slacks.append(float("nan"))

    mu: list[float] = []
    if x is not None and num_aux > 0:
        mu = [float(x[mu_offset + j]) for j in range(num_aux)]

    return FlagAlgebraResult(
        data.problem, status, bound, Q_vals, slacks, data=data, mu=mu,
    )


# ---------------------------------------------------------------------------
# Certificate verification
# ---------------------------------------------------------------------------

def verify_certificate(
    data: FlagAlgebraData,
    result: FlagAlgebraResult,
    *,
    rat_tol: float = 1e-6,
    psd_tol: float = 1e-6,
) -> dict[str, Any]:
    """Verify a flag algebra certificate in exact rational arithmetic.

    Steps
    -----
    1. Rationalize λ and each Q matrix (float → Fraction).
    2. Check each Q is numerically PSD (min eigenvalue ≥ -psd_tol).
    3. For each admissible graph H_i, compute the exact rational residual:
         upper bound:  residual[i] = λ_rat - Σ_σ <Q_σ, P_σ(H_i)> - density[i]
         lower bound:  residual[i] = density[i] - λ_rat - Σ_σ <Q_σ, P_σ(H_i)>
       A non-negative residual certifies the bound for H_i.

    Parameters
    ----------
    rat_tol:
        Tolerance passed to Fraction.limit_denominator for rationalizing floats.
        Smaller = more exact but larger denominators.
    psd_tol:
        Minimum allowed eigenvalue for Q matrices to be considered PSD.

    Returns
    -------
    dict with keys:
        'valid'          — True iff all Q PSD and all residuals ≥ 0
        'lam_certified'  — rationalized bound (Fraction)
        'min_psd_eigval' — minimum eigenvalue across all Q matrices (float)
        'residuals'      — list of Fraction residuals per admissible graph
        'min_residual'   — tightest (smallest) residual
    """
    if result.Q is None:
        raise ValueError("No Q matrices in result — rerun solve_sdp with extract_Q=True")

    minimize = data.problem.minimize
    denom_limit = round(1 / rat_tol)

    # Step 1: rationalize λ, Q matrices, and μ (aux constraint weights).
    # Use exact Fraction values from round_certificate when available; this avoids
    # re-rationalizing floats from L_rat @ L_rat.T which introduces small errors.
    if result.bound_exact is not None:
        lam_rat = result.bound_exact
    else:
        lam_rat = Fraction(result.bound).limit_denominator(denom_limit)

    if result.Q_exact is not None:
        Q_rat = result.Q_exact
    else:
        Q_rat = [
            [[Fraction(q[j, kk]).limit_denominator(denom_limit)
              for kk in range(q.shape[1])]
             for j in range(q.shape[0])]
            for q in result.Q
        ]

    # μ: use pre-rounded exact values when available, else rationalize (floored
    # to nonneg to keep the certificate valid — μ ≥ 0 is a hard requirement).
    if result.mu_exact is not None:
        mu_rat = result.mu_exact
    else:
        mu_rat = [
            max(Fraction(0), Fraction(m).limit_denominator(denom_limit))
            for m in result.mu
        ]

    # Step 2: PSD check using float eigenvalues.
    min_psd_eigval = min(
        float(np.linalg.eigvalsh(q).min())
        for q in result.Q
    )
    # Q is PSD by construction when built from Cholesky factors (L @ L.T).
    if result.cholesky_factors is not None:
        psd_ok = True
    else:
        psd_ok = min_psd_eigval >= -psd_tol

    # Step 3: exact rational residuals.
    #   Upper bound: residual[i] = λ - <Q,P(H_i)> - Σ μ_j·c_{H_i,j} - density(H_i) ≥ 0
    #   Lower bound: residual[i] = density(H_i) - λ - <Q,P(H_i)> - Σ μ_j·c_{H_i,j} ≥ 0
    # (both derived from the equation slack ≥ 0 in the corresponding sign convention).
    flag_sums = _compute_flag_sums(data, Q_rat)
    aux_sums = _compute_aux_sums(data, mu_rat)
    residuals = [
        (data.densities[i] - lam_rat - flag_sums[i] - aux_sums[i]) if minimize
        else (lam_rat - flag_sums[i] - aux_sums[i] - data.densities[i])
        for i in range(len(data.admissible))
    ]

    min_residual = min(residuals)
    valid = psd_ok and min_residual >= 0

    return {
        "valid": valid,
        "lam_certified": lam_rat,
        "min_psd_eigval": min_psd_eigval,
        "residuals": residuals,
        "min_residual": min_residual,
        "mu": mu_rat,
    }


# ---------------------------------------------------------------------------
# Rational rounding of SDP certificates
# ---------------------------------------------------------------------------

def _compute_flag_sums(
    data: FlagAlgebraData,
    Q_rat: list[list[list[Fraction]]],
) -> list[Fraction]:
    """Compute Σ_σ <Q_σ, P_σ(H_i)> for each admissible graph H_i in exact arithmetic."""
    flag_sums: list[Fraction] = []
    for i in range(len(data.admissible)):
        flag_sum = Fraction(0)
        for sigma in range(len(data.types)):
            nf = len(data.flags[sigma])
            P = data.pair_dens[i][sigma]
            Q_s = Q_rat[sigma]
            for j in range(nf):
                for kk in range(j, nf):
                    if P[j][kk] == 0:
                        continue
                    factor = 1 if j == kk else 2
                    flag_sum += P[j][kk] * factor * Q_s[j][kk]
        flag_sums.append(flag_sum)
    return flag_sums


def _compute_aux_sums(
    data: FlagAlgebraData,
    mu_rat: list[Fraction],
) -> list[Fraction]:
    """Compute Σ_j μ_j · c_{H_i, j} for each admissible graph H_i in exact arithmetic.

    Returns a zero list when the problem has no aux constraints or μ is empty.
    """
    num_H = len(data.admissible)
    if not mu_rat or not data.aux_coefficients:
        return [Fraction(0) for _ in range(num_H)]
    aux_sums = [Fraction(0) for _ in range(num_H)]
    for j, mu_j in enumerate(mu_rat):
        if mu_j == 0:
            continue
        row = data.aux_coefficients[j]
        for i in range(num_H):
            if row[i] != 0:
                aux_sums[i] += mu_j * row[i]
    return aux_sums


def round_certificate(
    data: FlagAlgebraData,
    result: FlagAlgebraResult,
    *,
    denom_limit: int = 1000,
    chol_reg: float = 1e-10,
) -> FlagAlgebraResult:
    """Round floating-point Q matrices to rationals via Cholesky factor rounding.

    For each Q matrix, computes the Cholesky factor L (so Q = L Lᵀ), rounds
    each entry of L to a rational with denominator ≤ denom_limit, then
    reconstructs Q_rat = L_rat L_ratᵀ exactly in rational arithmetic.
    The result is PSD by construction.

    The certified bound is computed as the tightest lambda compatible with the
    rounded Q matrices — the maximum over all admissible graphs of
    (density + flag_sum) for upper bounds, or the minimum for lower bounds.
    This guarantees all residuals in verify_certificate() are ≥ 0.

    Parameters
    ----------
    data:
        Precomputed flag algebra data (same as passed to solve_sdp).
    result:
        SDP result with floating-point Q matrices (extract_Q=True required).
    denom_limit:
        Maximum denominator used when rounding each entry of the Cholesky
        factor.  Larger values give a tighter rational certificate at the cost
        of larger numerators.
    chol_reg:
        Small diagonal regularization added to Q before Cholesky to handle
        near-singular matrices.

    Returns
    -------
    FlagAlgebraResult with exact rational Q matrices (Q_exact), certified bound
    (bound_exact), float approximations (Q, bound), and Cholesky factors.
    The certified bound is the tightest lambda certifiable from the rounded Q.
    """
    if result.Q is None:
        raise ValueError("No Q matrices in result — rerun solve_sdp with extract_Q=True")

    rounded_Qs = []
    cholesky_factors = []
    Q_exact_list: list[list[list[Fraction]]] = []
    for Q in result.Q:
        n = Q.shape[0]
        Q_reg = Q + chol_reg * np.eye(n)
        try:
            L = np.linalg.cholesky(Q_reg)
        except np.linalg.LinAlgError:
            # Fallback: eigendecomposition for matrices that are not PD even
            # after regularization (e.g. badly conditioned solver output).
            eigvals, eigvecs = np.linalg.eigh(Q_reg)
            L = eigvecs * np.sqrt(np.maximum(eigvals, 0.0))

        # Exact rational Cholesky factor.
        L_frac = [[Fraction(x).limit_denominator(denom_limit) for x in row] for row in L]

        # Exact rational Q = L_frac @ L_frac.T — PSD by construction.
        Q_frac: list[list[Fraction]] = [
            [sum(L_frac[i][t] * L_frac[j][t] for t in range(n)) for j in range(n)]
            for i in range(n)
        ]
        Q_exact_list.append(Q_frac)

        # Float versions kept for numpy operations and eigenvalue display.
        L_rat = np.array([[float(x) for x in row] for row in L_frac])
        cholesky_factors.append(L_rat)
        rounded_Qs.append(L_rat @ L_rat.T)

    # Round μ_j to rationals (floored to nonneg so the certificate remains valid).
    mu_exact: list[Fraction] = [
        max(Fraction(0), Fraction(m).limit_denominator(denom_limit))
        for m in result.mu
    ]

    # Compute the tightest certified bound compatible with the rounded Q and μ.
    # Rounding L (and μ) changes the flag / aux sums, so the solver's bound may
    # not be achievable; we take the max/min over all admissible graphs instead.
    #   Upper: bound_exact = max_i (density(H_i) + flag_sum(i) + aux_sum(i))
    #   Lower: bound_exact = min_i (density(H_i) - flag_sum(i) - aux_sum(i))
    flag_sums = _compute_flag_sums(data, Q_exact_list)
    aux_sums = _compute_aux_sums(data, mu_exact)
    minimize = data.problem.minimize
    n_admissible = len(data.admissible)
    if not minimize:
        bound_exact = max(
            data.densities[i] + flag_sums[i] + aux_sums[i]
            for i in range(n_admissible)
        )
    else:
        bound_exact = min(
            data.densities[i] - flag_sums[i] - aux_sums[i]
            for i in range(n_admissible)
        )

    bound_rat = float(bound_exact)
    return FlagAlgebraResult(
        result.problem, result.status, bound_rat, rounded_Qs, result.slacks,
        cholesky_factors=cholesky_factors,
        Q_exact=Q_exact_list,
        bound_exact=bound_exact,
        data=data,
        mu=[float(m) for m in mu_exact],
        mu_exact=mu_exact,
    )


# ---------------------------------------------------------------------------
# Sharp graph identification
# ---------------------------------------------------------------------------

def identify_sharps(
    data: FlagAlgebraData,
    result: FlagAlgebraResult,
    *,
    sharp_tol: float = 1e-4,
    rat_tol: float = 1e-6,
    psd_tol: float = 1e-6,
) -> SharpsResult:
    """Identify sharp (extremal) admissible graphs from a certificate.

    Uses float slack values to identify sharps (avoiding precision issues),
    then runs verify_certificate for the residuals.

    Returns
    -------
    SharpsResult with attributes:
        .indices   — 0-based positions of sharp graphs in data.admissible
        .graphs    — corresponding Hypergraph objects
        .densities — their Fraction densities
        .residuals — Fraction residuals for all admissible graphs
    """
    if result.Q is None:
        raise ValueError("No Q matrices in result — rerun solve_sdp with extract_Q=True")

    cert = verify_certificate(data, result, rat_tol=rat_tol, psd_tol=psd_tol)
    indices = [i for i, s in enumerate(result.slacks) if s < sharp_tol]

    return SharpsResult(
        problem=data.problem,
        indices=indices,
        graphs=[data.admissible[i] for i in indices],
        densities=[data.densities[i] for i in indices],
        residuals=cert["residuals"],
    )


# ---------------------------------------------------------------------------
# Certified proof workflow
# ---------------------------------------------------------------------------

def certify(
    result: FlagAlgebraResult,
    data: Optional[FlagAlgebraData] = None,
    *,
    denom_limit: int = 1000,
    chol_reg: float = 1e-10,
) -> Certificate:
    """Round and verify a flag algebra certificate in one call.

    Combines :func:`round_certificate` and :func:`verify_certificate` into a
    single step that returns a :class:`Certificate` with an exact rational
    bound, PSD matrices, per-graph residuals, and validity status.

    Parameters
    ----------
    result:
        SDP result with Q matrices (``solve_sdp(extract_Q=True)`` required).
    data:
        Precomputed flag algebra data.  If ``None``, uses ``result.data``
        (set automatically by :func:`solve_sdp`).
    denom_limit:
        Maximum denominator used when rounding each entry of the Cholesky
        factor.  Passed to :func:`round_certificate`.
    chol_reg:
        Diagonal regularization before Cholesky decomposition.  Passed to
        :func:`round_certificate`.

    Returns
    -------
    Certificate
        Exact rational certificate ready for inspection, verification, and
        sharing.

    Examples
    --------
    ::

        result = solve_sdp(data, extract_Q=True)
        proof  = certify(result)

        print(proof.bound)   # Fraction(1, 2)
        print(proof.valid)   # True
        proof.explain()
    """
    if data is None:
        data = result.data
    if data is None:
        raise ValueError(
            "No data attached to result — pass data explicitly or use solve_sdp, "
            "which attaches it automatically."
        )
    rounded = round_certificate(data, result, denom_limit=denom_limit, chol_reg=chol_reg)
    cert_dict = verify_certificate(data, rounded)
    return Certificate(
        problem=data.problem,
        bound=cert_dict["lam_certified"],
        valid=cert_dict["valid"],
        Q=rounded.Q_exact,
        residuals=cert_dict["residuals"],
        data=data,
        mu=cert_dict["mu"],
    )
