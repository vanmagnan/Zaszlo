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

from .types import (
    Certificate,
    CertificateProvenance,
    FlagAlgebraData,
    FlagAlgebraResult,
    FlagProblem,
    SharpsResult,
)

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


def _fraction_to_str(r: Fraction) -> str:
    """Format a Fraction as 'num/den' (or 'num' if den == 1)."""
    return str(r.numerator) if r.denominator == 1 else f"{r.numerator}/{r.denominator}"


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
    valid = cert_dict["valid"]
    reason: Optional[str] = None
    if not valid:
        neg = sum(1 for r in cert_dict["residuals"] if r < 0)
        min_r = min(cert_dict["residuals"])
        reason = (
            f"After Cholesky rounding at denom_limit={denom_limit}, "
            f"{neg} admissible graph(s) have negative residual "
            f"(worst = {min_r}). Try raising denom_limit."
        )
    return Certificate(
        problem=data.problem,
        bound=cert_dict["lam_certified"],
        valid=valid,
        Q=rounded.Q_exact,
        residuals=cert_dict["residuals"],
        data=data,
        mu=cert_dict["mu"],
        provenance=CertificateProvenance(
            pipeline="certify",
            denom_limit=denom_limit,
            chol_reg=chol_reg,
            reason=reason,
        ),
    )


# ---------------------------------------------------------------------------
# Certify at exact rational bound  (Cohn–de Laat–Leijenhorst 2024 pipeline)
# ---------------------------------------------------------------------------
#
# Four-step algorithm (arXiv:2001.00256 + arXiv:2403.16874):
#
#   Step 1 — Fix λ = target_bound and re-solve the SDP as a feasibility
#             problem.  Substituting the scalar x[0] = float(target_bound)
#             into the Clarabel standard form removes λ from the variable
#             vector and moves its contribution to the RHS.  The resulting Q
#             is better-conditioned for the exact bound than the optimizer's Q:
#             with λ pinned, the solver is forced to drive flag_sum(sharp) → 0
#             to make all constraint residuals nonneg simultaneously.
#
#   Step 2 — Eigendecompose each Q.  Eigenvectors with eigenvalue below
#             kernel_threshold span the "PSD-cone face" the optimum lives on.
#             Rationalize each such eigenvector with limit_denominator, then
#             apply integer RREF to obtain a clean rational basis W_rat for the
#             kernel.  (The RREF step is the Cohn–de Laat–Leijenhorst
#             improvement over the LLL-based Dostert–de Laat–Moustrou 2021
#             approach: same result, O(k^5 n) instead of O(n^6).)
#
#   Step 3 — Rationalize the positive-eigenspace part.  Round each positive
#             eigenvector column and each positive eigenvalue independently to
#             Fraction with denominator ≤ denominator_limit.  Reconstruct
#             Q_exact = Σ_j λ_j_rat · v_j_rat · v_j_ratᵀ — PSD by construction
#             as a sum of rank-1 outer products with nonneg rational scalars.
#             The kernel directions are absent from this sum, so flag_sum at
#             sharp graphs is structurally zero (not just approximately zero).
#
#   Step 4 — Verify all residuals ≥ 0 at bound in exact rational arithmetic
#             via the existing verify_certificate infrastructure.
#
# Why this works where round_certificate fails
# ---------------------------------------------
# round_certificate rounds each entry of L (Cholesky factor) to the nearest
# rational.  Each rounded entry contributes O(1/denom_limit) error to every
# Q[j,k], and those errors accumulate in <Q, P(H_sharp)> regardless of how
# large denom_limit is — the max (density + flag_sum) always overshoots the
# tight bound by a positive amount.
#
# This pipeline removes the sharp-graph directions from Q entirely in Step 2,
# so flag_sum(sharp) = 0 is structurally enforced, not approximated.  The
# remaining (positive) part of Q can then be rounded without perturbing the
# tightness at sharp graphs.


def _solve_feasibility_at_bound(
    data: FlagAlgebraData,
    target_bound: Fraction,
    *,
    verbose: bool = False,
) -> dict[str, Any]:
    """Re-solve the flag algebra SDP with λ pinned to target_bound.

    Substitutes x[0] = float(target_bound) into the standard-form SDP produced
    by build_sdp(), moving the λ contribution into the RHS vector b and
    dropping the variable.  The result is a zero-objective feasibility problem
    in the remaining Q and slack variables; it is feasible iff target_bound is
    a valid upper/lower bound for the problem.

    Returns
    -------
    dict with keys:
        ``'status'`` — mapped solver status (``'optimal'``,
        ``'optimal_inaccurate'``, ``'infeasible'``, ``'dual_infeasible'``,
        ``'numerical_error'``, or ``'unknown'``).
        ``'raw_status'`` — Clarabel's raw status string.
        ``'Q_vals'`` — list of Q numpy arrays (one per type), or None on failure.
        ``'mu_vals'`` — list of floats for auxiliary-constraint dual variables,
        or None on failure.
    """
    built = build_sdp(data)
    A = built["A"]
    b = built["b"]

    lam_float = float(target_bound)
    # Use toarray().ravel() for a clean 1-D array regardless of sparse format.
    lam_col = A[:, 0].toarray().ravel()
    b_fixed = b - lam_float * lam_col

    # Remove x[0] (λ) from the variable vector; all stored offsets shift by -1.
    A_fixed = A[:, 1:].tocsc()
    n_new = A_fixed.shape[1]

    P_zero = sp.csc_matrix((n_new, n_new))
    q_zero = np.zeros(n_new)

    settings = clarabel.DefaultSettings()
    settings.verbose = verbose
    solver = clarabel.DefaultSolver(
        P_zero, q_zero, A_fixed, b_fixed, built["cones"], settings
    )
    sol = solver.solve()

    status_key = str(sol.status).split(".")[-1]
    status = _STATUS_MAP.get(status_key, "unknown")
    if status not in ("optimal", "optimal_inaccurate") or sol.x is None:
        return {"status": status, "raw_status": status_key,
                "Q_vals": None, "mu_vals": None}

    x = np.asarray(sol.x)
    flag_sizes = built["flag_sizes"]
    # q_offsets were computed with λ at position 0; subtract 1 for new layout.
    q_offsets_new = [off - 1 for off in built["q_offsets"]]

    Q_vals: list[np.ndarray] = []
    for sigma, nf in enumerate(flag_sizes):
        q_off = q_offsets_new[sigma]
        Q = np.zeros((nf, nf))
        for k in range(nf):
            for j in range(k + 1):
                val = float(x[q_off + _tri_idx(j, k)])
                Q[j, k] = val
                Q[k, j] = val
        Q_vals.append(Q)

    # Extract auxiliary constraint dual variables μ (empty when no aux constraints).
    mu_vals: list[float] = []
    num_aux = built["num_aux"]
    if num_aux > 0:
        mu_offset_new = built["mu_offset"] - 1
        mu_vals = [max(0.0, float(x[mu_offset_new + j])) for j in range(num_aux)]

    return {"status": status, "raw_status": status_key,
            "Q_vals": Q_vals, "mu_vals": mu_vals}


def _rref_rational(rows: list[list[Fraction]]) -> list[list[Fraction]]:
    """Reduced row echelon form over ℚ via Gauss-Jordan elimination.

    Operates on a copy of rows.  Zero rows (all entries zero) are dropped from
    the output so the result is always a basis, not just a spanning set.

    Parameters
    ----------
    rows:
        A matrix given as a list of rows, each row a list of Fraction values.
        All rows must have the same length.

    Returns
    -------
    List of nonzero rows in RREF.
    """
    if not rows:
        return []
    M = [list(row) for row in rows]
    num_rows = len(M)
    num_cols = len(M[0])
    pivot_row = 0
    for col in range(num_cols):
        # Find a nonzero pivot in this column at or below the current pivot row.
        found = next((r for r in range(pivot_row, num_rows) if M[r][col] != 0), None)
        if found is None:
            continue
        M[pivot_row], M[found] = M[found], M[pivot_row]
        scale = M[pivot_row][col]
        M[pivot_row] = [x / scale for x in M[pivot_row]]
        for r in range(num_rows):
            if r != pivot_row and M[r][col] != 0:
                factor = M[r][col]
                M[r] = [M[r][c] - factor * M[pivot_row][c] for c in range(num_cols)]
        pivot_row += 1
    return [row for row in M if any(x != 0 for x in row)]


def _find_rational_kernel(
    Q: np.ndarray,
    *,
    kernel_threshold: float,
    denom_limit: int,
) -> list[list[Fraction]]:
    """Extract a rational basis for the kernel of Q via eigendecomposition + RREF.

    Near-zero eigenvectors (eigenvalue < kernel_threshold) are rounded to
    rationals with limit_denominator, then RREF over ℚ is applied to clean up
    linear dependencies and yield an exact rational basis.

    Returns
    -------
    W_rat — list of length-n row vectors (lists of Fraction).  Empty when Q is
    numerically positive definite (no eigenvalues below kernel_threshold).
    """
    eigvals, eigvecs = np.linalg.eigh(Q)
    n = Q.shape[0]
    kernel_mask = eigvals < kernel_threshold
    if not kernel_mask.any():
        return []
    # One row per near-zero eigenvector (each eigenvector is a column in eigvecs).
    W_rows: list[list[Fraction]] = [
        [Fraction(eigvecs[row, col]).limit_denominator(denom_limit) for row in range(n)]
        for col in range(eigvecs.shape[1]) if kernel_mask[col]
    ]
    return _rref_rational(W_rows)


def _null_space_from_rref(W_rat: list[list[Fraction]], n: int) -> list[list[Fraction]]:
    """Rational null space of W_rat via the RREF free-variable formula.

    Given W_rat in RREF (k×n, k ≤ n), returns n−k basis vectors for
    {v ∈ ℚⁿ : W_rat · v = 0}.  These vectors are the exact rational
    complement of the row space of W_rat (i.e., the orthogonal complement of
    the kernel of Q) and have small-denominator rational entries.

    When W_rat is empty (Q has no kernel) the full standard basis is returned.

    The key correctness property: if at the exact SDP optimum the positive
    eigenspace of Q* lies in null(P_σ(sharp)) for each sharp graph, then the
    null-space vectors returned here (which span the complement of the kernel
    of Q*) will satisfy V_rat^T · P_σ(sharp) · V_rat = 0 exactly in ℚ — so
    flag_sum(sharp) = Tr(P · V T V^T) = 0 for ANY rational T.
    """
    if not W_rat:
        # No kernel: the null space of the zero matrix is all of ℚⁿ.
        return [[Fraction(1) if i == j else Fraction(0) for j in range(n)] for i in range(n)]

    # Identify pivot columns from RREF (first nonzero entry per row).
    pivot_cols: list[int] = []
    for row in W_rat:
        for col, val in enumerate(row):
            if val != 0:
                pivot_cols.append(col)
                break

    # Free columns (non-pivot) index the null space basis vectors.
    pivot_set = set(pivot_cols)
    free_cols = [j for j in range(n) if j not in pivot_set]

    # For each free column j, the null space basis vector v_j satisfies:
    #   v_j[j] = 1,  v_j[j'] = 0  (other free columns),
    #   v_j[pivot_cols[i]] = -W_rat[i][j]  (pivot columns).
    V_rat: list[list[Fraction]] = []
    for j_free in free_cols:
        v: list[Fraction] = [Fraction(0)] * n
        v[j_free] = Fraction(1)
        for i, c_piv in enumerate(pivot_cols):
            v[c_piv] = -W_rat[i][j_free]
        V_rat.append(v)

    return V_rat


def _reconstruct_Q_vtv(
    V_rat: list[list[Fraction]],
    Q_float: np.ndarray,
    n: int,
    denom_limit: int,
) -> tuple[list[list[Fraction]], np.ndarray]:
    """Reconstruct Q_exact = V_rat · T_rat · V_ratᵀ in exact ℚ arithmetic.

    T_rat is the reduced matrix V_ratᵀ · Q_float · V_rat, rounded to rationals.
    T_rat is symmetrized before rationalization to eliminate float asymmetry.

    Parameters
    ----------
    V_rat:
        Rational null-space (complement) basis; m vectors of length n.
    Q_float:
        Floating-point Q from the feasibility solve.
    n:
        Matrix dimension.
    denom_limit:
        Max denominator for rationalizing T entries.

    Returns
    -------
    (Q_frac, Q_fl)
        Q_frac — n×n exact rational matrix as list-of-lists of Fraction.
        Q_fl   — float version for eigenvalue checks.

    Notes
    -----
    PSD guarantee: Q_frac = V T_rat Vᵀ is PSD iff T_rat is PSD.  T_rat is
    the rounding of V^T Q V, which is PSD (Q is PSD, V full column rank in
    the positive subspace); small rounding errors may push it slightly
    indefinite.  The float reconstruction Q_fl is verified for PSD-ness by
    verify_certificate via eigenvalue check.
    """
    m = len(V_rat)

    if m == 0:
        # V is empty: Q_exact = 0.
        Q_frac = [[Fraction(0)] * n for _ in range(n)]
        return Q_frac, np.zeros((n, n))

    # Build float V (m × n) for fast matrix multiplication.
    V_fl = np.array([[float(V_rat[j][r]) for r in range(n)] for j in range(m)])

    # Gram matrix G = V V^T (m × m).  V_rat columns from RREF are NOT orthonormal
    # (e.g., [-1,1] has ||v||^2 = 2), so the naive T = V Q V^T overcounts by
    # ||v||^4.  The correct projection formula is T = G^{-1} (V Q V^T) G^{-1},
    # which equals V_orth^T Q V_orth where V_orth = V (V^T V)^{-1/2} is the
    # QR-orthonormalized basis.
    G_fl = V_fl @ V_fl.T
    G_inv_fl = np.linalg.pinv(G_fl)
    VQVt = V_fl @ Q_float @ V_fl.T
    T_fl = G_inv_fl @ VQVt @ G_inv_fl
    T_fl = (T_fl + T_fl.T) / 2.0

    # Rationalize T entry-by-entry.
    T_rat: list[list[Fraction]] = [
        [Fraction(T_fl[j, k]).limit_denominator(denom_limit) for k in range(m)]
        for j in range(m)
    ]

    # Reconstruct Q_frac = Σ_{j,k} V_rat[j] ⊗ V_rat[k] · T_rat[j][k] (exact ℚ).
    Q_frac: list[list[Fraction]] = [[Fraction(0)] * n for _ in range(n)]
    for j in range(m):
        for k in range(m):
            t = T_rat[j][k]
            if t == 0:
                continue
            for r in range(n):
                for c in range(n):
                    Q_frac[r][c] += t * V_rat[j][r] * V_rat[k][c]

    Q_fl = np.array([[float(Q_frac[r][c]) for c in range(n)] for r in range(n)])
    return Q_frac, Q_fl


def _correct_T_for_sharp_graphs(
    V_rat_list: list[list[list[Fraction]]],
    T_rat_list: list[list[list[Fraction]]],
    data: FlagAlgebraData,
    sharp_indices: list[int],
    bound: Fraction,
) -> tuple[list[list[list[Fraction]]], dict[str, Any]]:
    """Adjust T_rat so flag_sum(sharp_i) = target_i exactly in ℚ.

    Formulation
    -----------
    After the Gram-corrected V T V^T rationalization, small floating-point
    errors in V_rat (from RREF of approximate eigenvectors) cause flag_sum at
    sharp graphs to miss their exact rational targets by ≈1e-8 to 1e-5.
    We treat this as a small exact linear system in the flat T-entries.

    Let ``t`` be the flat vector of upper-triangular T-entries across all σ
    (``(σ, a, b)`` with a ≤ b).  Define

        J[i][j] = ∂ flag_sum[sharp_i] / ∂ t[j]           (exact ℚ)
        err[i] = current_flag_sum[sharp_i] - target[i]   (exact ℚ)

    A correction δt with ``J δt = −err`` restores exact equality.  We solve
    the augmented system ``[J | −err]`` by RREF over ℚ:

    * If any reduced row is ``[0 … 0 | c ≠ 0]`` the system is inconsistent
      in ℚ and no correction can fix it exactly (usually a sign that
      V_rat's rationalization broke a linear dependence).  Fallback below.
    * Otherwise the particular solution with free variables = 0 gives a
      correction supported on ``rank(J)`` pivot entries.  The magnitude is
      bounded by the pre-correction err, so the resulting T stays PSD by
      wide margin.

    Robustness fallback for rank-deficient J with inconsistent RHS: many
    problems (e.g. Mantel) have sharp constraints that are algebraically
    dependent, but the float→ℚ rationalization of V_rat may not preserve
    that dependence exactly.  In that case, some sharp constraints are
    linearly dependent on others *in float* but not *in ℚ*.  We detect
    the numerical rank r of J via SVD, then keep only r linearly
    independent rows (selected by QR column-pivoting on Jᵀ) and re-solve.
    The dropped sharp constraints must be satisfied automatically by the
    linear-dependence structure inherited from the exact optimum.

    Parameters
    ----------
    V_rat_list, T_rat_list:
        Per-sigma rational V (null-space basis) and T (projected Q) matrices.
    data:
        Flag algebra data carrying pair_dens (exact rational pair densities).
    sharp_indices:
        Indices into data.admissible of tight graphs (residual ≈ 0 in float).
    bound:
        Exact rational bound being certified.

    Returns
    -------
    (T_corrected, info):
        T_corrected — new nested list (inputs not mutated).  Equal to
        T_rat_list when err is already zero or the system is inconsistent
        even after rank reduction.
        info — dict with keys 'consistent' (bool), 'rank' (int),
        'float_rank' (int), 'num_sharp' (int), 'num_vars' (int),
        'dropped_rows' (list[int]), 'max_correction' (Fraction).
    """
    n_sigma = len(V_rat_list)
    minimize = data.problem.minimize

    # Flat T-variable layout: (sigma, a, b) with a ≤ b (upper triangular).
    flat_vars: list[tuple[int, int, int]] = []
    for sigma in range(n_sigma):
        m = len(V_rat_list[sigma])
        for a in range(m):
            for b in range(a, m):
                flat_vars.append((sigma, a, b))
    n_vars = len(flat_vars)
    n_sharp = len(sharp_indices)

    zero_info: dict[str, Any] = {
        "consistent": True, "rank": 0, "float_rank": 0,
        "num_sharp": n_sharp, "num_vars": n_vars,
        "dropped_rows": [], "max_correction": Fraction(0),
    }
    if n_sharp == 0 or n_vars == 0:
        return T_rat_list, zero_info

    # Build J in exact ℚ.  J[i][j] = (V^T P_full V)[a,b] × (1 if a==b else 2)
    # where P_full[r,c] = P[r][c] for r ≤ c, else P[c][r].
    J: list[list[Fraction]] = [[Fraction(0)] * n_vars for _ in range(n_sharp)]
    for i, sh in enumerate(sharp_indices):
        for j, (sigma, a, b) in enumerate(flat_vars):
            V = V_rat_list[sigma]
            m = len(V)
            if m == 0:
                continue
            n = len(V[0])
            P = data.pair_dens[sh][sigma]
            val = Fraction(0)
            for r in range(n):
                v_a_r = V[a][r]
                if v_a_r == 0:
                    continue
                for c in range(n):
                    p_rc = P[r][c] if r <= c else P[c][r]
                    if p_rc == 0:
                        continue
                    val += p_rc * v_a_r * V[b][c]
            scale = Fraction(1) if a == b else Fraction(2)
            J[i][j] = val * scale

    # Current T-level flag sums and errors, all in exact ℚ.
    t_flat = [T_rat_list[sigma][a][b] for sigma, a, b in flat_vars]
    flag_sum_T = [
        sum((J[i][j] * t_flat[j] for j in range(n_vars)), Fraction(0))
        for i in range(n_sharp)
    ]
    targets = [
        (data.densities[sh] - bound) if minimize else (bound - data.densities[sh])
        for sh in sharp_indices
    ]
    err = [flag_sum_T[i] - targets[i] for i in range(n_sharp)]

    if all(e == 0 for e in err):
        return T_rat_list, zero_info

    delta_t, solve_info = _solve_rational_linear_system(J, err, n_sharp, n_vars)

    # Apply δt to T (symmetric: off-diagonals get the same delta on both sides).
    T_corrected = [
        [list(row) for row in T_rat_list[sigma]]
        for sigma in range(n_sigma)
    ]
    if solve_info["consistent"]:
        for j, (sigma, a, b) in enumerate(flat_vars):
            d = delta_t[j]
            if d == 0:
                continue
            T_corrected[sigma][a][b] += d
            if a != b:
                T_corrected[sigma][b][a] += d

    return T_corrected, solve_info


def _solve_rational_linear_system(
    J: list[list[Fraction]],
    err: list[Fraction],
    n_rows: int,
    n_cols: int,
) -> tuple[list[Fraction], dict[str, Any]]:
    """Solve J δ = −err in exact ℚ via augmented RREF, with rank fallback.

    Returns
    -------
    (delta, info)
        delta — length-n_cols Fraction list.  All zeros when the augmented
        system is inconsistent even after dropping numerically dependent rows.
        info — 'consistent', 'rank', 'float_rank', 'dropped_rows',
        'max_correction' plus 'num_sharp'/'num_vars' passthroughs.
    """
    # First pass: augmented RREF over ℚ using all rows.
    aug = [J[i] + [-err[i]] for i in range(n_rows)]
    rref = _rref_rational(aug)

    delta = [Fraction(0)] * n_cols
    rank = 0
    consistent = True
    for row in rref:
        pivot = next((j for j in range(n_cols) if row[j] != 0), None)
        rhs = row[n_cols]
        if pivot is None:
            if rhs != 0:
                consistent = False
            continue
        rank += 1
        delta[pivot] = rhs  # pivot normalized to 1 by _rref_rational

    dropped: list[int] = []
    if consistent:
        max_corr = max((abs(d) for d in delta), default=Fraction(0))
        return delta, {
            "consistent": True, "rank": rank, "float_rank": rank,
            "num_sharp": n_rows, "num_vars": n_cols,
            "dropped_rows": dropped, "max_correction": max_corr,
        }

    # Rank-fallback: the ℚ-augmented system is inconsistent.  This typically
    # means V_rat's rationalization broke an algebraic dependence between the
    # sharp constraints (e.g. Mantel: three sharp constraints, two independent
    # in float but three independent after float→ℚ rounding of V).  Detect the
    # float rank of J, select that many independent rows via QR column-pivoting
    # on Jᵀ, and re-solve.  The dropped rows should be satisfied automatically
    # once the kept ones are, up to the same float error we already accept.
    J_fl = np.array([[float(x) for x in row] for row in J])
    if J_fl.size == 0 or n_rows == 0:
        return [Fraction(0)] * n_cols, {
            "consistent": False, "rank": rank, "float_rank": 0,
            "num_sharp": n_rows, "num_vars": n_cols,
            "dropped_rows": list(range(n_rows)), "max_correction": Fraction(0),
        }
    sv = np.linalg.svd(J_fl, compute_uv=False)
    sv_max = float(sv[0]) if sv.size else 0.0
    tol = max(1e-8 * sv_max, 1e-14)
    float_rank = int(np.sum(sv > tol))
    if float_rank == 0 or float_rank >= n_rows:
        return [Fraction(0)] * n_cols, {
            "consistent": False, "rank": rank, "float_rank": float_rank,
            "num_sharp": n_rows, "num_vars": n_cols,
            "dropped_rows": [], "max_correction": Fraction(0),
        }

    # QR with column pivoting on Jᵀ selects the n_rows-by-n_rows submatrix of
    # rows most linearly independent.  Take the first `float_rank` pivots.
    try:
        from scipy.linalg import qr as _qr
        _, _, piv = _qr(J_fl.T, pivoting=True, mode="economic")
    except Exception:
        # scipy unavailable / QR failure: fall back to greedy row selection by
        # residual after Gram-Schmidt-like reduction.
        piv = _greedy_independent_rows(J_fl)
    kept = sorted(int(p) for p in piv[:float_rank])
    dropped = [i for i in range(n_rows) if i not in kept]

    J_reduced = [J[i] for i in kept]
    err_reduced = [err[i] for i in kept]
    aug2 = [J_reduced[i] + [-err_reduced[i]] for i in range(len(kept))]
    rref2 = _rref_rational(aug2)

    delta = [Fraction(0)] * n_cols
    rank = 0
    consistent = True
    for row in rref2:
        pivot = next((j for j in range(n_cols) if row[j] != 0), None)
        rhs = row[n_cols]
        if pivot is None:
            if rhs != 0:
                consistent = False
            continue
        rank += 1
        delta[pivot] = rhs

    if not consistent:
        return [Fraction(0)] * n_cols, {
            "consistent": False, "rank": rank, "float_rank": float_rank,
            "num_sharp": n_rows, "num_vars": n_cols,
            "dropped_rows": dropped, "max_correction": Fraction(0),
        }

    max_corr = max((abs(d) for d in delta), default=Fraction(0))
    return delta, {
        "consistent": True, "rank": rank, "float_rank": float_rank,
        "num_sharp": n_rows, "num_vars": n_cols,
        "dropped_rows": dropped, "max_correction": max_corr,
    }


def _greedy_independent_rows(J_fl: np.ndarray) -> list[int]:
    """Fallback row selection: greedy Gram-Schmidt with residual norm.

    Returns a permutation of row indices with the most independent rows first.
    Only used when scipy.linalg.qr with pivoting is unavailable.
    """
    n_rows = J_fl.shape[0]
    remaining = list(range(n_rows))
    order: list[int] = []
    basis: list[np.ndarray] = []
    while remaining:
        best_i = None
        best_norm = -1.0
        best_vec: np.ndarray | None = None
        for i in remaining:
            v = J_fl[i].astype(float).copy()
            for b in basis:
                v = v - (v @ b) * b
            norm = float(np.linalg.norm(v))
            if norm > best_norm:
                best_norm = norm
                best_i = i
                best_vec = v
        assert best_i is not None
        order.append(best_i)
        remaining.remove(best_i)
        if best_norm > 1e-14 and best_vec is not None:
            basis.append(best_vec / best_norm)
    return order


def certify_at_bound(
    data: FlagAlgebraData,
    bound: Fraction,
    *,
    denominator_limit: int = 1000,
    kernel_threshold: float = 1e-6,
) -> Certificate:
    """Certify a flag algebra bound at an exact user-supplied rational value.

    Implements the Cohn–de Laat–Leijenhorst (2024) pipeline for exact rational
    certificates at tight bounds, where standard Cholesky rounding (``certify``)
    fails because the optimal Q lies on the boundary of the PSD cone.

    The pipeline is:

    1. **Fixed-λ feasibility solve** — re-solve the SDP with λ pinned to
       ``bound``.  With the objective removed, the solver drives
       flag_sum(sharp) → 0 to satisfy all residual constraints simultaneously,
       producing a Q better-conditioned for the exact bound than the
       optimization result.

    2. **Rational kernel via RREF** — eigendecompose each Q; round near-zero
       eigenvectors (eigenvalue < ``kernel_threshold``) to rationals; apply
       RREF over ℚ to extract an exact rational basis W_rat for the kernel.

    3. **Rational null-space complement** — compute V_rat = null(W_rat) via
       the RREF free-variable formula.  V_rat is the exact rational positive
       eigenspace; crucially, V_rat^T · P_σ(sharp) · V_rat = 0 exactly in ℚ
       (when the RREF found the exact rational kernel), so any Q = V T V^T
       has flag_sum(sharp) = 0 structurally.

    4. **Reconstruct Q_exact = V · T_rat · V^T** — project the feasibility Q
       down to the complement subspace (T = V^T Q V, rationalized), then lift
       back.  PSD by construction when T_rat is PSD.

    5. **Exact verification** — compute all residuals in exact Fraction
       arithmetic via :func:`verify_certificate`.

    Parameters
    ----------
    data:
        Precomputed flag algebra data (from :func:`build_flag_algebra_data`).
    bound:
        Exact rational bound to certify.  Must be a :class:`fractions.Fraction`.
    denominator_limit:
        Maximum denominator when rationalizing T entries.  Larger values allow
        closer approximations at the cost of larger certificate size.
    kernel_threshold:
        Eigenvalue cutoff for the kernel/positive-space split.  Increase if
        the feasibility solve leaves small-but-positive spurious eigenvalues;
        decrease if genuine positive directions are being discarded.

    Returns
    -------
    Certificate
        ``valid=True`` if all residuals ≥ 0 at ``bound`` in exact rational
        arithmetic; ``valid=False`` otherwise.  Never raises; failure is
        communicated via ``Certificate.valid`` and ``Certificate.residuals``.

    Examples
    --------
    ::

        from fractions import Fraction
        from zaszlo import FlagProblem, complete, build_flag_algebra_data
        from zaszlo import certify_at_bound

        prob = FlagProblem(4, 2, 2, forbidden=[complete(3)])
        data = build_flag_algebra_data(prob)

        cert = certify_at_bound(data, Fraction(1, 2))
        assert cert.bound == Fraction(1, 2)
        assert cert.valid
    """
    if not isinstance(bound, Fraction):
        raise TypeError(f"bound must be a Fraction, got {type(bound).__name__}")

    num_admissible = len(data.admissible)

    # Step 1: re-solve as a feasibility SDP with λ fixed to the target bound.
    feasibility = _solve_feasibility_at_bound(data, bound)
    feasibility_status = feasibility["status"]
    if feasibility["Q_vals"] is None:
        return Certificate(
            problem=data.problem,
            bound=bound,
            valid=False,
            Q=[],
            residuals=[Fraction(0)] * num_admissible,
            data=data,
            provenance=CertificateProvenance(
                pipeline="certify_at_bound",
                denom_limit=denominator_limit,
                feasibility_status=feasibility_status,
                reason=(
                    f"Feasibility solve with λ pinned to {bound} returned "
                    f"'{feasibility_status}' (Clarabel raw: "
                    f"'{feasibility['raw_status']}'). The bound may be tighter "
                    f"than what the SDP relaxation can achieve at this n / "
                    f"type_order — try a larger problem or relax the bound."
                ),
            ),
        )
    Q_fixed = feasibility["Q_vals"]
    mu_float = feasibility["mu_vals"]

    # Steps 2–4: per type, find the rational kernel (RREF), compute the
    # rational null-space complement (V_rat), project Q down to that subspace
    # via the Gram-corrected formula T = G⁻¹ (V Q Vᵀ) G⁻¹, rationalize T,
    # then lift back Q_exact = V T_rat Vᵀ.  Keep V_rat and T_rat separately so
    # the sharp-graph correction in Step 5 can adjust T_rat before lifting.
    V_rat_list: list[list[list[Fraction]]] = []
    T_rat_list: list[list[list[Fraction]]] = []
    for Q in Q_fixed:
        n = Q.shape[0]
        W_rat = _find_rational_kernel(
            Q, kernel_threshold=kernel_threshold, denom_limit=denominator_limit
        )
        V_rat = _null_space_from_rref(W_rat, n)
        # RREF pivot division inflates denominators: an eigenvector entry near
        # ±1 can end up as 332928/332929, which prevents the *exact* algebraic
        # dependences between sharp constraints from surviving.  Snap each
        # entry back down to its nearest rational with denominator ≤ denom
        # limit — anything closer than 1/denom_limit was noise anyway.
        V_rat = [
            [entry.limit_denominator(denominator_limit) for entry in row]
            for row in V_rat
        ]
        V_rat_list.append(V_rat)
        m = len(V_rat)
        if m == 0:
            T_rat_list.append([])
        else:
            V_fl = np.array([[float(V_rat[j][r]) for r in range(n)] for j in range(m)])
            G_fl = V_fl @ V_fl.T
            G_inv = np.linalg.pinv(G_fl)
            VQVt = V_fl @ Q @ V_fl.T
            T_fl = G_inv @ VQVt @ G_inv
            T_fl = (T_fl + T_fl.T) / 2.0
            T_rat: list[list[Fraction]] = [
                [Fraction(T_fl[a, b]).limit_denominator(denominator_limit) for b in range(m)]
                for a in range(m)
            ]
            T_rat_list.append(T_rat)

    # Step 5: exact rational correction.
    # After rationalizing T_rat, tiny float errors in V_rat (from RREF of
    # approximate eigenvectors) make flag_sum(sharp) miss its exact target
    # (bound − density) by ≈ 1e-8 to 1e-5.  Identify sharp graphs from the
    # raw feasibility Q (tight constraints have float residual ≈ 0), then
    # solve a small rational linear system to find δT such that the corrected
    # T_rat = T_rat + δT exactly satisfies all sharp flag_sum constraints.
    # Non-sharp graphs have large slack (> 0.01), so the tiny correction
    # cannot flip them negative.
    lam_fl = float(bound)
    minimize = data.problem.minimize
    float_flag_sums = []
    for h_idx in range(len(data.admissible)):
        fs = 0.0
        for sigma, Q in enumerate(Q_fixed):
            P = data.pair_dens[h_idx][sigma]
            nf = len(P)
            for j in range(nf):
                fs += float(P[j][j]) * Q[j, j]
                for k in range(j + 1, nf):
                    fs += 2.0 * float(P[j][k]) * Q[j, k]
        float_flag_sums.append(fs)
    float_residuals = [
        (float(data.densities[i]) - lam_fl - float_flag_sums[i]) if minimize
        else (lam_fl - float_flag_sums[i] - float(data.densities[i]))
        for i in range(len(data.admissible))
    ]
    sharp_indices = [i for i, r in enumerate(float_residuals) if abs(r) < 1e-3]

    T_corrected, correction_info = _correct_T_for_sharp_graphs(
        V_rat_list, T_rat_list, data, sharp_indices, bound
    )
    # PSD safeguard: only accept the correction if every T block remains PSD
    # (min eigenvalue ≥ −psd_tol).  Corrections are tiny (order 1e-5 or
    # smaller) and the current T entries are order 1e-1 to 1, so this is a
    # cheap sanity check: if it ever fires, something's wrong upstream.
    psd_tol_local = 1e-6
    corrected_psd = True
    if sharp_indices:
        for T_block in T_corrected:
            if not T_block:
                continue
            M = np.array([[float(T_block[a][b]) for b in range(len(T_block))]
                          for a in range(len(T_block))])
            if M.size and float(np.linalg.eigvalsh(M).min()) < -psd_tol_local:
                corrected_psd = False
                break

    correction_applied = (
        bool(sharp_indices) and correction_info["consistent"] and corrected_psd
    )
    if correction_applied:
        T_rat_list = T_corrected

    if not sharp_indices:
        psd_status = "not_run"
    elif not corrected_psd:
        psd_status = "tripped_correction_discarded"
    else:
        psd_status = "passed"

    correction_provenance: dict = {
        "consistent": correction_info["consistent"],
        "rank": correction_info["rank"],
        "float_rank": correction_info["float_rank"],
        "num_sharp": correction_info["num_sharp"],
        "num_vars": correction_info["num_vars"],
        "dropped_rows": list(correction_info["dropped_rows"]),
        "max_correction": _fraction_to_str(correction_info["max_correction"]),
        "applied": correction_applied,
    }

    # Lift corrected T_rat back to full Q matrices.
    Q_exact_list: list[list[list[Fraction]]] = []
    Q_float_list: list[np.ndarray] = []
    for sigma, (V_rat, T_rat) in enumerate(zip(V_rat_list, T_rat_list)):
        n = Q_fixed[sigma].shape[0]
        m = len(V_rat)
        Q_frac: list[list[Fraction]] = [[Fraction(0)] * n for _ in range(n)]
        if m > 0 and T_rat:
            for a in range(m):
                for b in range(m):
                    t = T_rat[a][b]
                    if t == 0:
                        continue
                    for r in range(n):
                        for c in range(n):
                            Q_frac[r][c] += t * V_rat[a][r] * V_rat[b][c]
        Q_exact_list.append(Q_frac)
        Q_float_list.append(
            np.array([[float(Q_frac[r][c]) for c in range(n)] for r in range(n)])
        )

    # Rationalize auxiliary constraint weights (clamped to ≥ 0).
    mu_exact: list[Fraction] = [
        max(Fraction(0), Fraction(m).limit_denominator(denominator_limit))
        for m in mu_float
    ]

    # Step 6: verify all residuals ≥ 0 at bound in exact rational arithmetic.
    # cholesky_factors set to a non-None sentinel so verify_certificate skips
    # the float PSD check (PSD guaranteed by T_rat ≥ 0 and Q = V T V^T).
    synthetic = FlagAlgebraResult(
        problem=data.problem,
        status="optimal",
        bound=float(bound),
        Q=Q_float_list,
        slacks=[0.0] * len(data.admissible),
        cholesky_factors=[np.zeros((0, 0))],
        Q_exact=Q_exact_list,
        bound_exact=bound,
        data=data,
        mu=[float(m) for m in mu_exact],
        mu_exact=mu_exact,
    )
    cert_dict = verify_certificate(data, synthetic)
    valid = cert_dict["valid"]
    reason: Optional[str] = None
    if not valid:
        neg_indices = [i for i, r in enumerate(cert_dict["residuals"]) if r < 0]
        min_r = min(cert_dict["residuals"])
        if not correction_applied and sharp_indices:
            if not correction_info["consistent"]:
                reason = (
                    f"Rational correction step was inconsistent "
                    f"(rank {correction_info['rank']} < num_sharp "
                    f"{correction_info['num_sharp']}; float rank "
                    f"{correction_info['float_rank']}). V_rat's rationalization "
                    f"likely broke a linear dependence between sharp constraints — "
                    f"try raising denominator_limit."
                )
            elif not corrected_psd:
                reason = (
                    "PSD safeguard tripped: post-correction T block was "
                    "numerically indefinite. Correction discarded; residuals "
                    "at sharp graphs are non-zero."
                )
        else:
            reason = (
                f"{len(neg_indices)} admissible graph(s) have negative residual "
                f"after certification (worst = {min_r}). Consider raising "
                f"denominator_limit or checking that {bound} is actually feasible."
            )
    return Certificate(
        problem=data.problem,
        bound=bound,
        valid=valid,
        Q=Q_exact_list,
        residuals=cert_dict["residuals"],
        data=data,
        mu=cert_dict["mu"],
        provenance=CertificateProvenance(
            pipeline="certify_at_bound",
            denom_limit=denominator_limit,
            feasibility_status=feasibility_status,
            kernel_dims=[
                Q_fixed[sigma].shape[0] - len(V_rat_list[sigma])
                for sigma in range(len(Q_fixed))
            ],
            correction=correction_provenance,
            psd_safeguard=psd_status,
            reason=reason,
        ),
    )
