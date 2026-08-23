"""Structured diagnostic reports for solver / certificate observations.

The :class:`DiagnosticReport` returned by :func:`diagnose_result` and
:func:`diagnose_certificate` surfaces **observations, not interpretations** —
a machine-readable snapshot of a solved SDP suitable for an AI agent (or a
downstream automation) to reason over.  There are no heuristic hints ("try
raising n"); the report just makes it cheap to notice things like
``not is_valid`` or ``not max_sharp_density_matches_bound``.

Design
------
Every field is either a primitive (bool, int, float, str), None, or a list
of dicts.  Exact rationals are formatted as strings ("num/den") so the report
survives JSON round-trips without a custom decoder.

To interpret the report, the caller consults the plan or a skill file.  This
module intentionally holds no domain knowledge beyond field extraction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

if TYPE_CHECKING:
    from .types import Certificate, FlagAlgebraResult


_OPTIMAL_STATUSES = {"optimal", "optimal_inaccurate"}
_SHARP_TOL = 1e-4  # matches explain._SHARP_TOL


def _rat_str(r: Optional[Fraction]) -> Optional[str]:
    """Format an exact Fraction as 'num/den' (or 'num' if den==1)."""
    if r is None:
        return None
    if r.denominator == 1:
        return str(r.numerator)
    return f"{r.numerator}/{r.denominator}"


def _rank_estimate(Q: np.ndarray, tol: float = 1e-8) -> int:
    """Approximate rank of Q via eigenvalue threshold (Q is PSD)."""
    eigs = np.linalg.eigvalsh(Q)
    return int((eigs > tol).sum())


# ---------------------------------------------------------------------------
# DiagnosticReport
# ---------------------------------------------------------------------------

@dataclass
class DiagnosticReport:
    """Structured observations from a solved FlagAlgebraResult or Certificate.

    Attributes
    ----------
    kind:
        Either ``'FlagAlgebraResult'`` or ``'Certificate'`` — the source object.
    status:
        Solver status string, verbatim.
    is_optimal:
        True iff ``status`` is ``'optimal'`` or ``'optimal_inaccurate'``.
    is_valid:
        True iff the source is a valid Certificate; None if no Certificate
        is attached (e.g. plain FlagAlgebraResult).
    bound_float:
        Numerical objective value from the solver.
    bound_exact:
        Exact rational bound as a "num/den" string, if available.
    num_admissible, num_types, flag_counts:
        Combinatorial size of the SDP.
    num_active_constraints, active_constraint_indices:
        Admissible-graph indices where the certificate identity is tight
        (residual exactly 0).  Available only when a Certificate is attached
        or the source *is* a Certificate.
    sharp_indices, sharp_densities:
        Admissible-graph indices whose SDP slack is below ``sharp_tol``, and
        their exact rational densities (as "num/den" strings).
    max_sharp_density_matches_bound:
        Bool: does the largest (upper-bound problem) / smallest (lower-bound
        problem) sharp density equal the certified bound exactly?  A quick
        proxy for "does the extremal graph attain the bound?".  None if no
        sharps or no bound_exact.
    aux_summary:
        One dict per aux constraint: ``{grade, num_terms, mu_str, mu_float, mu_active}``.
    Q_summary:
        One dict per PSD block: ``{size, min_eigenvalue, rank_estimate}``.
    min_residual, worst_residual_index:
        Minimum residual across admissible graphs (as "num/den" string) and
        its index.  None if no residuals available.
    provenance:
        For Certificates: a dict of how the certificate was produced (pipeline,
        denom_limit, feasibility_status, kernel_dims, correction, etc.).  See
        :class:`zaszlo.types.CertificateProvenance`.  Empty dict when the
        source is a FlagAlgebraResult without an attached certificate.
    """

    kind: str
    status: str
    is_optimal: bool
    is_valid: Optional[bool]
    bound_float: float
    bound_exact: Optional[str]
    num_admissible: int
    num_types: int
    flag_counts: list[int]
    num_active_constraints: int
    active_constraint_indices: list[int]
    sharp_indices: list[int]
    sharp_densities: list[str]
    max_sharp_density_matches_bound: Optional[bool]
    aux_summary: list[dict[str, Any]]
    Q_summary: list[dict[str, Any]]
    min_residual: Optional[str]
    worst_residual_index: Optional[int]
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return this report as a plain dict (JSON-ready)."""
        return asdict(self)

    def to_json(self, indent: int | None = 2) -> str:
        """Return this report as a JSON string."""
        from .serialize import to_json
        return to_json(self.to_dict(), indent=indent)

    def __repr__(self) -> str:
        valid_desc = "valid" if self.is_valid else ("INVALID" if self.is_valid is False else "no-cert")
        return (
            f"DiagnosticReport[{self.kind}]  status={self.status}  "
            f"bound≈{self.bound_float:.6f}  [{valid_desc}]  "
            f"{self.num_active_constraints} active, {len(self.sharp_indices)} sharp"
        )

    def explain(self) -> str:
        """Return a plain-text summary of the observations."""
        lines = [
            f"DiagnosticReport  ({self.kind})",
            "",
            f"  Status              : {self.status}   optimal={self.is_optimal}",
        ]
        if self.is_valid is not None:
            lines.append(f"  Certificate valid   : {self.is_valid}")
        lines += [
            f"  Bound (float)       : {self.bound_float:.10f}",
            f"  Bound (exact)       : {self.bound_exact or 'n/a'}",
            "",
            f"  Admissible graphs   : {self.num_admissible}",
            f"  Types (SDP blocks)  : {self.num_types}   flag counts: {self.flag_counts}",
            "",
            f"  Active constraints  : {self.num_active_constraints}  "
            f"at indices {self.active_constraint_indices}",
            f"  Sharp graphs        : {len(self.sharp_indices)}  "
            f"at indices {self.sharp_indices}",
        ]
        if self.max_sharp_density_matches_bound is not None:
            lines.append(
                f"  Extremal attains bound: {self.max_sharp_density_matches_bound}"
            )
        if self.sharp_densities:
            lines.append(f"  Sharp densities     : {self.sharp_densities}")
        if self.aux_summary:
            lines += ["", f"  Auxiliary constraints ({len(self.aux_summary)}):"]
            for i, a in enumerate(self.aux_summary):
                lines.append(
                    f"    [{i}]  grade={a['grade']}, terms={a['num_terms']}, "
                    f"μ={a['mu_str']} (float≈{a['mu_float']:.6g}, active={a['mu_active']})"
                )
        if self.Q_summary:
            lines += ["", "  PSD certificate blocks:"]
            for i, q in enumerate(self.Q_summary):
                lines.append(
                    f"    Q[{i}]  size={q['size']}  min_eigval≈{q['min_eigenvalue']:.2e}  "
                    f"rank≈{q['rank_estimate']}"
                )
        if self.min_residual is not None:
            lines += [
                "",
                f"  Min residual        : {self.min_residual} "
                f"(at graph index {self.worst_residual_index})",
            ]
        if self.provenance:
            lines += ["", "  Certificate provenance:"]
            for key in (
                "pipeline", "denom_limit", "chol_reg", "feasibility_status",
                "kernel_dims", "psd_safeguard", "reason",
            ):
                if key in self.provenance:
                    lines.append(f"    {key:<19}: {self.provenance[key]}")
            corr = self.provenance.get("correction")
            if corr:
                lines.append(
                    f"    {'correction':<19}: rank {corr['rank']}/{corr['num_sharp']} "
                    f"(float rank {corr['float_rank']}), "
                    f"max δT = {corr['max_correction']}, applied = {corr['applied']}"
                )
                if corr.get("dropped_rows"):
                    lines.append(
                        f"    {'  dropped rows':<19}: {corr['dropped_rows']}"
                    )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------

def _extract_common_fields(problem, data) -> dict[str, Any]:
    """Pull combinatorial size + type/flag info from a problem + data pair."""
    return {
        "num_admissible": len(data.admissible) if data is not None else 0,
        "num_types": len(data.types) if data is not None else 0,
        "flag_counts": (
            [len(fs) for fs in data.flags] if data is not None else []
        ),
    }


def _sharp_fields(
    slacks: list[float],
    data,
    bound_exact: Optional[Fraction],
    minimize: bool,
    sharp_tol: float = _SHARP_TOL,
) -> tuple[list[int], list[str], Optional[bool]]:
    """Extract sharp indices/densities and the extremal-attainment flag."""
    if data is None or not slacks:
        return [], [], None
    sharp_idx = [i for i, s in enumerate(slacks) if s < sharp_tol]
    sharp_dens: list[Fraction] = [data.densities[i] for i in sharp_idx]
    sharp_dens_str = [_rat_str(d) or "" for d in sharp_dens]
    if bound_exact is None or not sharp_dens:
        return sharp_idx, sharp_dens_str, None
    # For a lower-bound problem the extremal density is the smallest; else largest.
    extreme = min(sharp_dens) if minimize else max(sharp_dens)
    return sharp_idx, sharp_dens_str, (extreme == bound_exact)


def _aux_summary(
    result_or_cert_mu: list[Fraction],
    problem,
    tol_active: Fraction = Fraction(0),
) -> list[dict[str, Any]]:
    """Per-aux-constraint summary: grade, num_terms, μ, active flag."""
    if not problem.aux_constraints:
        return []
    summary = []
    for j, c in enumerate(problem.aux_constraints):
        mu_j = result_or_cert_mu[j] if j < len(result_or_cert_mu) else Fraction(0)
        summary.append({
            "grade": c.n,
            "num_terms": len(c.expr.terms),
            "mu_str": _rat_str(mu_j),
            "mu_float": float(mu_j),
            "mu_active": mu_j > tol_active,
        })
    return summary


def _Q_summary_from_floats(Q_list) -> list[dict[str, Any]]:
    if Q_list is None:
        return []
    out = []
    for Q in Q_list:
        eigs = np.linalg.eigvalsh(Q)
        out.append({
            "size": int(Q.shape[0]),
            "min_eigenvalue": float(eigs.min()),
            "rank_estimate": _rank_estimate(Q),
        })
    return out


def diagnose_result(r: "FlagAlgebraResult") -> DiagnosticReport:
    """Build a DiagnosticReport from a FlagAlgebraResult."""
    minimize = r.problem.minimize
    is_valid: Optional[bool] = None
    active: list[int] = []
    residuals_frac: Optional[list[Fraction]] = None
    bound_exact = r.bound_exact
    mu_frac: list[Fraction] = []

    # Certificate is the richest source when attached.
    if r.certificate is not None:
        is_valid = r.certificate.valid
        active = list(r.certificate.active_constraints)
        residuals_frac = list(r.certificate.residuals)
        bound_exact = r.certificate.bound
        mu_frac = list(r.certificate.mu)
    elif r.mu_exact is not None:
        mu_frac = list(r.mu_exact)
    elif r.mu:
        # Best-effort rational conversion of float mu (for display).
        mu_frac = [Fraction(m).limit_denominator(10**6) for m in r.mu]

    sharp_idx, sharp_dens_str, extreme_matches = _sharp_fields(
        r.slacks, r.data, bound_exact, minimize
    )
    aux = _aux_summary(mu_frac, r.problem)
    Q_sum = _Q_summary_from_floats(r.Q)

    min_res_str: Optional[str] = None
    worst_idx: Optional[int] = None
    if residuals_frac:
        worst_idx = min(range(len(residuals_frac)), key=lambda i: residuals_frac[i])
        min_res_str = _rat_str(residuals_frac[worst_idx])

    provenance = (
        r.certificate.provenance.to_dict()
        if r.certificate is not None else {}
    )
    common = _extract_common_fields(r.problem, r.data)
    return DiagnosticReport(
        kind="FlagAlgebraResult",
        status=r.status,
        is_optimal=(r.status in _OPTIMAL_STATUSES),
        is_valid=is_valid,
        bound_float=r.bound,
        bound_exact=_rat_str(bound_exact),
        num_admissible=common["num_admissible"],
        num_types=common["num_types"],
        flag_counts=common["flag_counts"],
        num_active_constraints=len(active),
        active_constraint_indices=active,
        sharp_indices=sharp_idx,
        sharp_densities=sharp_dens_str,
        max_sharp_density_matches_bound=extreme_matches,
        aux_summary=aux,
        Q_summary=Q_sum,
        min_residual=min_res_str,
        worst_residual_index=worst_idx,
        provenance=provenance,
    )


def diagnose_certificate(c: "Certificate") -> DiagnosticReport:
    """Build a DiagnosticReport from a Certificate.

    Uses the certificate's exact rationals for bound, residuals, and mu.
    Sharps are inferred from the certificate's active_constraints when data
    is available (no float slacks — the report degrades to active_constraints
    only for the sharp fields).
    """
    minimize = c.problem.minimize
    data = c.data

    # From a Certificate alone we don't have float slacks, so "sharp" is
    # equated with "residual is exactly zero" — the algebraic notion of
    # tightness — which coincides with active_constraints.
    active = list(c.active_constraints)
    if data is not None:
        sharp_dens = [data.densities[i] for i in active]
        sharp_dens_str = [_rat_str(d) or "" for d in sharp_dens]
        if sharp_dens:
            extreme = min(sharp_dens) if minimize else max(sharp_dens)
            extreme_matches: Optional[bool] = (extreme == c.bound)
        else:
            extreme_matches = None
    else:
        sharp_dens_str = []
        extreme_matches = None

    aux = _aux_summary(list(c.mu), c.problem)

    Q_sum = []
    for Q_rat in c.Q:
        n = len(Q_rat)
        Q_float = np.array([[float(x) for x in row] for row in Q_rat])
        eigs = np.linalg.eigvalsh(Q_float)
        Q_sum.append({
            "size": n,
            "min_eigenvalue": float(eigs.min()),
            "rank_estimate": _rank_estimate(Q_float),
        })

    worst_idx = min(range(len(c.residuals)), key=lambda i: c.residuals[i]) if c.residuals else None
    min_res_str = _rat_str(c.residuals[worst_idx]) if worst_idx is not None else None

    common = _extract_common_fields(c.problem, data)
    return DiagnosticReport(
        kind="Certificate",
        status="certificate",
        is_optimal=True,
        is_valid=c.valid,
        bound_float=float(c.bound),
        bound_exact=_rat_str(c.bound),
        num_admissible=common["num_admissible"],
        num_types=common["num_types"],
        flag_counts=common["flag_counts"],
        num_active_constraints=len(active),
        active_constraint_indices=active,
        sharp_indices=active,
        sharp_densities=sharp_dens_str,
        max_sharp_density_matches_bound=extreme_matches,
        aux_summary=aux,
        Q_summary=Q_sum,
        min_residual=min_res_str,
        worst_residual_index=worst_idx,
        provenance=c.provenance.to_dict(),
    )
