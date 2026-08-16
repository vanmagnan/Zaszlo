# High-level one-call entry point for common workflows.
#
# solve() wraps the full pipeline: build_flag_algebra_data + solve_sdp,
# and optionally certify(). Both underlying functions remain public for
# step-by-step control.

from __future__ import annotations

import warnings
from typing import Optional

from .pipeline import build_flag_algebra_data
from .sdp import certify as _certify, solve_sdp
from .types import CertificationError, FlagAlgebraResult, FlagProblem

_OPTIMAL_STATUSES = {"optimal", "optimal_inaccurate"}


def solve(
    prob: FlagProblem,
    *,
    certify: bool = False,
    settings=None,
    denom_limit: int = 1000,
    chol_reg: float = 1e-10,
    verbose: bool = False,
) -> FlagAlgebraResult:
    """Build flag algebra data and solve the SDP in one call.

    Wraps :func:`build_flag_algebra_data`, :func:`solve_sdp`, and optionally
    :func:`certify`.  All three public functions remain available for
    step-by-step control.

    Parameters
    ----------
    prob:
        A fully specified :class:`FlagProblem`.
    certify:
        If True, round the floating-point certificate to exact rational
        arithmetic and verify it.  Populates ``result.certificate`` with a
        :class:`Certificate` whose ``.valid`` and ``.bound`` (a
        ``Fraction``) can be inspected directly.
    settings:
        Optional ``clarabel.DefaultSettings`` for fine-grained solver control
        (tolerances, iteration limits, etc.).
    denom_limit:
        Maximum denominator when rounding Cholesky entries to rationals.
        Larger values give tighter certificates with bigger numerators.
        Only used when ``certify=True``.
    chol_reg:
        Diagonal regularization before Cholesky decomposition.
        Only used when ``certify=True``.
    verbose:
        If True, print a progress summary at each step.

    Returns
    -------
    FlagAlgebraResult
        ``result.bound``        — float SDP objective value.
        ``result.data``         — the :class:`FlagAlgebraData` used.
        ``result.certificate``  — :class:`Certificate` if ``certify=True``,
                                  otherwise ``None``.

    Raises
    ------
    RuntimeError
        If ``certify=True`` and the SDP solver returns a non-optimal status.
    CertificationError
        If ``certify=True`` and the rounded certificate fails exact rational
        verification.  ``err.result.certificate`` holds the invalid certificate
        with full per-graph residuals for diagnosis.

    Examples
    --------
    Quick numerical bound::

        from zaszlo import FlagProblem, complete, solve

        prob   = FlagProblem(4, 2, 2, forbidden=[complete(3)])
        result = solve(prob)
        print(result.bound)      # ≈ 0.5
        print(result.explain())

    Exact rational certificate::

        result = solve(prob, certify=True)
        print(result.certificate.bound)   # Fraction(1, 2)
        print(result.certificate.valid)   # True
        print(result.certificate.explain())
    """
    # --- Step 1: build flag algebra data ---
    if verbose:
        print("Building flag algebra data...")
    data = build_flag_algebra_data(prob)
    if verbose:
        flag_counts = [len(fs) for fs in data.flags]
        print(
            f"  {len(data.types)} type(s), "
            f"flags per type: {flag_counts}, "
            f"{len(data.admissible)} admissible graph(s)"
        )

    # --- Step 2: solve SDP ---
    if verbose:
        print("Solving SDP...")
    result = solve_sdp(data, extract_Q=certify, settings=settings)
    if verbose:
        print(f"  Status: {result.status}   bound ≈ {result.bound:.6f}")

    if not certify:
        return result

    # --- Step 3 (certify=True): guardrails + rational certificate ---
    if result.status not in _OPTIMAL_STATUSES:
        raise RuntimeError(
            f"SDP solver returned status '{result.status}'. "
            "The bound is unreliable. "
            "Try adjusting solver settings (tolerances, max iterations) "
            "or increasing n for a tighter formulation."
        )
    if result.status == "optimal_inaccurate":
        warnings.warn(
            "SDP solver returned 'optimal_inaccurate'. "
            "The bound may be slightly off; consider tightening solver tolerances "
            "via a clarabel.DefaultSettings instance.",
            stacklevel=2,
        )

    if verbose:
        print("Certifying (rounding to exact rational arithmetic)...")
    proof = _certify(result, data, denom_limit=denom_limit, chol_reg=chol_reg)
    result.certificate = proof
    if verbose:
        cert_status = "valid" if proof.valid else "INVALID"
        print(f"  Certificate: {cert_status}   certified bound = {proof.bound}")

    if not proof.valid:
        raise CertificationError(result, denom_limit)

    return result
