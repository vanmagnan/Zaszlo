# Explanation and HTML rendering for zaszlo types.
#
# Design principle: every zaszlo object is self-describing.
#   .explain()            → plain-text mathematical description, readable in any
#                           context: scripts, REPLs, Jupyter, CI logs.
#   _repr_html_()         → annotated HTML card for Jupyter rich display.
#   _repr_mimebundle_()   → both at once, keyed by MIME type; Jupyter picks this
#                           up automatically so objects render without display().
#   __repr__()            → compact one-liner for repr() / text/plain fallback.
#
# The contract for adding a new zaszlo type:
#   1. Implement __repr__ (compact, no newlines).
#   2. Add explain_<type>() and html_<type>() here in explain.py.
#   3. Delegate from the class via lazy imports (avoids circular imports with types.py).
#   4. Implement _repr_mimebundle_ returning {"text/html": ..., "text/plain": repr(self)}.
#
# All rendering logic lives here; types.py holds thin wrappers that
# call into this module via lazy imports to avoid circular imports.

from __future__ import annotations

import math
from fractions import Fraction
from math import comb
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .types import Certificate, Flag, FlagAlgebraData, FlagAlgebraResult, FlagProblem, Hypergraph, SharpsResult

_SHARP_TOL = 1e-4

# Vertex colour palette — orange for labeled, blue for unlabeled.
_LABELED_FILL = "#e07b39"
_LABELED_STROKE = "#a04e1f"
_UNLABELED_FILL = "#4a90d9"
_UNLABELED_STROKE = "#2c5f8a"
_VR = 11  # vertex circle radius in SVG


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _fmt_edges(edges: list[tuple[int, ...]]) -> str:
    if not edges:
        return "(none)"
    return "  ".join("{" + ",".join(str(v) for v in e) + "}" for e in edges)


def _density_frac(num_edges: int, n: int, k: int) -> Fraction:
    total = comb(n, k)
    return Fraction(0) if total == 0 else Fraction(num_edges, total)


def _uniformity_name(k: int) -> str:
    return "graph" if k == 2 else f"{k}-uniform hypergraph"


def _sharp_indices(r, tol: float = _SHARP_TOL) -> list[int]:
    return [i for i, s in enumerate(r.slacks) if not math.isnan(s) and s < tol]


# ---------------------------------------------------------------------------
# SVG generation (k=2 or k=3, n ≤ 8 only)
# ---------------------------------------------------------------------------

def _vertex_coords(
    n: int, cx: float, cy: float, r: float
) -> list[tuple[float, float]]:
    if n == 1:
        return [(cx, cy)]
    return [
        (cx + r * math.cos(2 * math.pi * i / n - math.pi / 2),
         cy + r * math.sin(2 * math.pi * i / n - math.pi / 2))
        for i in range(n)
    ]


def _svg_graph(graph, labeled_count: int = 0, width: int = 160, height: int = 160) -> str:
    """Return an SVG string for the hypergraph, or '' if not renderable."""
    n = graph.n
    if n == 0 or graph.k > 3 or n > 8:
        return ""

    cx, cy = width / 2.0, height / 2.0
    r = min(width, height) / 2.0 - _VR - 5
    coords = _vertex_coords(n, cx, cy, r)

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
    ]

    if graph.k == 2:
        for (u, v) in graph.edges:
            x1, y1 = coords[u - 1]
            x2, y2 = coords[v - 1]
            parts.append(
                f'  <line x1="{x1:.1f}" y1="{y1:.1f}" '
                f'x2="{x2:.1f}" y2="{y2:.1f}" stroke="#888" stroke-width="2.5"/>'
            )
    elif graph.k == 3:
        for (u, v, w) in graph.edges:
            pts = " ".join(
                f"{coords[i - 1][0]:.1f},{coords[i - 1][1]:.1f}" for i in (u, v, w)
            )
            parts.append(
                f'  <polygon points="{pts}" fill="rgba(74,144,217,0.2)" '
                f'stroke="#4a90d9" stroke-width="1.5"/>'
            )

    for i, (x, y) in enumerate(coords):
        labeled = i < labeled_count
        fill = _LABELED_FILL if labeled else _UNLABELED_FILL
        stroke = _LABELED_STROKE if labeled else _UNLABELED_STROKE
        parts.append(
            f'  <circle cx="{x:.1f}" cy="{y:.1f}" r="{_VR}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
        )
        parts.append(
            f'  <text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle" '
            f'font-size="10" fill="white" font-family="monospace" font-weight="bold">'
            f'{i + 1}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# HTML card helper
# ---------------------------------------------------------------------------

_CARD_STYLE = (
    "display:inline-block; font-family:monospace; font-size:13px; "
    "border:1px solid #ddd; border-radius:6px; padding:12px; "
    "background:#fafafa; vertical-align:top;"
)
_TITLE_STYLE = "font-weight:bold; color:#333; font-size:14px; margin-bottom:4px;"
_META_STYLE = "color:#888; font-size:12px; margin-bottom:8px;"
_BODY_STYLE = "color:#555; font-size:12px; margin-top:4px; line-height:1.6;"


def _card(title: str, meta: str, body: str, svg: str = "") -> str:
    svg_block = f'<div style="margin:8px 0;">{svg}</div>' if svg else ""
    return (
        f'<div style="{_CARD_STYLE}">'
        f'<div style="{_TITLE_STYLE}">{title}</div>'
        f'<div style="{_META_STYLE}">{meta}</div>'
        f"{svg_block}"
        f'<div style="{_BODY_STYLE}">{body}</div>'
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Hypergraph
# ---------------------------------------------------------------------------

def explain_hypergraph(g) -> str:
    ne = len(g.edges)
    total = comb(g.n, g.k)
    d = _density_frac(ne, g.n, g.k)
    lines = [
        f"Hypergraph  n={g.n}, k={g.k}, {ne} edge{'s' if ne != 1 else ''}",
        "",
        f"Edges: {_fmt_edges(g.edges)}",
        "",
        f"Edge density: {ne}/{total} = {float(d):.4f}  ({d})",
        "",
    ]
    if g.k == 2:
        lines.append("This is an ordinary graph (2-uniform).")
    elif g.k == 3:
        lines.append(
            "This is a 3-uniform hypergraph: every edge is a triple of vertices."
        )
    else:
        lines.append(
            f"This is a {g.k}-uniform hypergraph: every edge contains {g.k} vertices."
        )
    return "\n".join(lines)


def html_hypergraph(g) -> str:
    ne = len(g.edges)
    total = comb(g.n, g.k)
    d = _density_frac(ne, g.n, g.k)
    kind = _uniformity_name(g.k)
    meta = f"n={g.n}, k={g.k}, {ne}/{total} edges"
    svg = _svg_graph(g)
    body = (
        f"<b>Edges:</b> {_fmt_edges(g.edges)}<br>"
        f"<b>Density:</b> {ne}/{total} = {float(d):.4f}"
    )
    title = f"Hypergraph &nbsp;<span style='font-weight:normal;color:#888;'>({kind})</span>"
    return _card(title, meta, body, svg)


# ---------------------------------------------------------------------------
# Flag
# ---------------------------------------------------------------------------

def _flag_role(f) -> str:
    if f.type_size == 0:
        return "admissible"
    if f.type_size == f.graph.n:
        return "type"
    return "flag"


def explain_flag(f) -> str:
    role = _flag_role(f)
    lines = [
        f"Flag  [{role}]  n={f.graph.n}, k={f.graph.k}, type_size={f.type_size}",
        "",
    ]
    if f.type_size == 0:
        lines += [
            "This is an admissible graph: no labeled vertices.",
            "In the flag algebra SDP, admissible graphs are the 'large' objects",
            "whose densities are bounded by the optimization.",
        ]
    elif f.type_size == f.graph.n:
        lines += [
            f"This is a type: all {f.graph.n} vertices are labeled (fixed).",
            "Types parameterize the flag algebra — each type σ indexes a block",
            "of the SDP and a family of flags built on top of it.",
        ]
    else:
        labeled = list(range(1, f.type_size + 1))
        unlabeled = list(range(f.type_size + 1, f.graph.n + 1))
        lines += [
            f"Labeled vertices:   {labeled}  (fixed in the host graph)",
            f"Unlabeled vertices: {unlabeled}  (averaged over all embeddings)",
            "",
            "When two flags over the same type are multiplied and the unlabeled",
            "vertices averaged, the result is a density of an admissible graph.",
        ]
    lines += ["", f"Edges: {_fmt_edges(f.graph.edges)}"]
    return "\n".join(lines)


def html_flag(f) -> str:
    role = _flag_role(f)
    meta = f"n={f.graph.n}, k={f.graph.k}, type_size={f.type_size}"
    svg = _svg_graph(f.graph, labeled_count=f.type_size)

    if f.type_size == 0:
        desc = "No labeled vertices — an admissible graph."
    elif f.type_size == f.graph.n:
        desc = "All vertices labeled — a type."
    else:
        labeled = ", ".join(str(i) for i in range(1, f.type_size + 1))
        unlabeled = ", ".join(str(i) for i in range(f.type_size + 1, f.graph.n + 1))
        desc = (
            f"<span style='color:{_LABELED_FILL};'>&#9679;</span> Labeled: {labeled}"
            f" &nbsp; "
            f"<span style='color:{_UNLABELED_FILL};'>&#9679;</span> Unlabeled: {unlabeled}"
        )

    body = f"{desc}<br><b>Edges:</b> {_fmt_edges(f.graph.edges)}"
    title = f"Flag &nbsp;<span style='font-weight:normal;color:#888;'>({role})</span>"
    return _card(title, meta, body, svg)


# ---------------------------------------------------------------------------
# FlagProblem
# ---------------------------------------------------------------------------

def explain_problem(p) -> str:
    bound_kind = "lower bound" if p.minimize else "upper bound"
    direction = "minimize" if p.minimize else "maximize"
    if p.target is None:
        target_desc = "edge density"
    else:
        target_desc = f"induced density of {p.target}"

    lines = [
        f"FlagProblem  ({bound_kind} on {target_desc})",
        "",
        f"Goal: {direction} the {target_desc}",
        f"      of {p.k}-uniform hypergraphs on n={p.n} vertices.",
        "",
        f"  k          = {p.k}   (edge uniformity)",
        f"  n          = {p.n}   (admissible graph vertex count)",
        f"  type_order = {p.type_order}   (max labeled vertices per type)",
    ]

    if p.forbidden:
        lines += ["", f"Forbidden subgraphs ({len(p.forbidden)}):"]
        for g in p.forbidden:
            lines.append(f"  {g}")
    else:
        lines += ["", "No forbidden subgraphs."]

    if p.forbidden_induced:
        lines += ["", f"Forbidden induced subgraphs ({len(p.forbidden_induced)}):"]
        for g in p.forbidden_induced:
            lines.append(f"  {g}")

    if p.target is not None:
        lines += ["", f"Target: {p.target}"]

    return "\n".join(lines)


def html_problem(p) -> str:
    bound_kind = "lower bound" if p.minimize else "upper bound"
    direction = "minimize" if p.minimize else "maximize"
    target_desc = (
        "edge density"
        if p.target is None
        else f"induced density of {p.target.n}-vertex target"
    )

    forbidden_html = ""
    if p.forbidden:
        items = "".join(f"<li>{g}</li>" for g in p.forbidden)
        forbidden_html += (
            f"<b>Forbidden:</b>"
            f"<ul style='margin:2px 0 2px 16px;'>{items}</ul>"
        )
    if p.forbidden_induced:
        items = "".join(f"<li>{g} (induced)</li>" for g in p.forbidden_induced)
        forbidden_html += (
            f"<b>Forbidden induced:</b>"
            f"<ul style='margin:2px 0 2px 16px;'>{items}</ul>"
        )
    if not p.forbidden and not p.forbidden_induced:
        forbidden_html = "No constraints.<br>"

    body = (
        f"<b>Goal:</b> {direction} {target_desc}<br>"
        f"<b>n</b>={p.n} &nbsp; <b>k</b>={p.k} &nbsp; "
        f"<b>type_order</b>={p.type_order}<br>"
        f"{forbidden_html}"
    )
    return _card("FlagProblem", f"k={p.k}, n={p.n}, {bound_kind}", body)


# ---------------------------------------------------------------------------
# FlagAlgebraData
# ---------------------------------------------------------------------------

def explain_data(d) -> str:
    flag_counts = [len(fs) for fs in d.flags]
    total_flags = sum(flag_counts)
    dens_floats = [float(x) for x in d.densities]
    dmin = min(dens_floats) if dens_floats else 0.0
    dmax = max(dens_floats) if dens_floats else 0.0

    lines = [
        "FlagAlgebraData  (pre-SDP combinatorial data)",
        "",
        f"  Types:             {len(d.types)}  (one Q matrix / SDP block each)",
        f"  Flags per type:    {flag_counts}  ({total_flags} total)",
        f"  Admissible graphs: {len(d.admissible)}  (objects whose densities are bounded)",
        f"  Density range:     [{dmin:.4f}, {dmax:.4f}]",
        "",
        "This is the input to the SDP solver. The solver searches for PSD",
        "matrices Q_σ (one per type σ) satisfying, for every admissible H:",
        "",
        "  bound − density(H)  =  Σ_σ ⟨Q_σ, P_σ(H)⟩  +  slack(H)",
        "",
        "The PSD matrices Q_σ define a flag-algebra sum-of-squares expression.",
        "Together with the verified coefficient inequalities over admissible",
        "graphs, this yields the stated bound.",
        "",
        "  Admissible graphs — the feasible graphs on n vertices; the SDP",
        "    bounds their density.  Forbidden subgraphs have already been",
        "    filtered out.",
        "  Types — all-labeled graphs; each type σ indexes one SDP block Q_σ.",
        "  Flags — graphs with a type embedded; pairs of flags over σ average",
        "    to admissible densities, giving the constraint matrix P_σ(H).",
    ]
    return "\n".join(lines)


def html_data(d) -> str:
    flag_counts = [len(fs) for fs in d.flags]
    total_flags = sum(flag_counts)

    rows = "".join(
        f"<tr>"
        f"<td style='padding:2px 8px;'>σ={i}</td>"
        f"<td style='padding:2px 8px; color:#666;'>{t.graph.n}v, {t.graph.k}-unif</td>"
        f"<td style='padding:2px 8px; text-align:right;'>{flag_counts[i]}</td>"
        f"</tr>"
        for i, t in enumerate(d.types)
    )
    table = (
        "<table style='border-collapse:collapse; font-size:12px; margin-top:4px;'>"
        "<thead><tr style='border-bottom:1px solid #ccc;'>"
        "<th style='padding:2px 8px; text-align:left;'>Type</th>"
        "<th style='padding:2px 8px; text-align:left;'>Graph</th>"
        "<th style='padding:2px 8px; text-align:right;'>Flags</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
    )

    body = (
        f"<b>Admissible graphs:</b> {len(d.admissible)}<br>"
        f"<b>Total flags:</b> {total_flags}<br>"
        f"{table}"
    )
    meta = f"{len(d.types)} type{'s' if len(d.types) != 1 else ''}, {len(d.admissible)} admissible"
    return _card("FlagAlgebraData", meta, body)


# ---------------------------------------------------------------------------
# FlagAlgebraResult
# ---------------------------------------------------------------------------

def explain_result(r) -> str:
    bound_kind = "lower" if r.problem.minimize else "upper"
    ineq = "≥" if r.problem.minimize else "≤"
    density_name = "edge density" if r.problem.target is None else "target density"
    kind = "graph" if r.problem.k == 2 else f"{r.problem.k}-uniform hypergraph"

    # Proof claim in plain language.
    n_forb = len(r.problem.forbidden) + len(r.problem.forbidden_induced)
    if n_forb:
        claim = (
            f"Every {kind} avoiding the {n_forb} forbidden pattern(s) "
            f"has {density_name} {ineq} {r.bound:.6f}."
        )
    else:
        claim = f"Every admissible {kind} has {density_name} {ineq} {r.bound:.6f}."

    lines = [
        "FlagAlgebraResult",
        "",
        f"  Bound  : {r.bound:.8f}  ({bound_kind} bound)",
        f"  Status : {r.status}",
        "",
        f"Proof claim: {claim}",
        "",
    ]

    # --- Sharp graphs (active constraints) ---
    sharps = _sharp_indices(r)
    lines.append("Sharp graphs (active constraints):")
    if sharps:
        lines += [
            f"  {len(sharps)} graph(s) with near-zero SDP slack.",
            "  These graphs have tight certificate constraints. Note: zero slack",
            "  does not by itself mean a graph attains the density bound.",
        ]
        if r.data is not None:
            lines.append("")
            for i in sharps:
                g = r.data.admissible[i]
                lines.append(f"    [{i}]  {g}  (density = {float(r.data.densities[i]):.4f})")
        else:
            lines.append(f"  indices: {sharps}")
    else:
        lines.append("  No sharp graphs found (all slacks above threshold).")
    lines.append("")

    # --- Proof certificate ---
    lines.append("Proof certificate:")
    if r.Q is None:
        lines += [
            "  Q matrices not extracted — rerun solve_sdp(extract_Q=True) to",
            "  access the full proof certificate.",
            "",
            "  The certificate consists of PSD matrices Q_σ (one per type σ)",
            "  witnessing a sum-of-squares identity that holds for every admissible",
            "  graph. It can be verified in exact rational arithmetic.",
        ]
    else:
        import numpy as np

        lhs = "density(H) − bound" if r.problem.minimize else "bound − density(H)"
        lines += [
            "  The bound is certified by PSD matrices Q_σ, one per type σ.",
            "  For every admissible H the following identity holds and is ≥ 0:",
            "",
            f"    {lhs}  =  Σ_σ ⟨Q_σ, P_σ(H)⟩  +  slack(H)",
            "",
            "  where:",
            "    P_σ(H) — pair density matrix of H over type σ",
            "    Q_σ    — PSD certificate matrix (one per type, found by SDP)",
            "    ⟨A, B⟩ — matrix inner product  Σ_{ij} A_{ij} B_{ij}",
            "",
            "  The PSD matrices Q_σ define a flag-algebra sum-of-squares expression.",
            "  Together with the verified coefficient inequalities over admissible",
            "  graphs, this yields the stated bound.",
            "",
        ]

        lines.append(f"  Q matrices ({len(r.Q)} total, one per type):")
        for i, Q in enumerate(r.Q):
            eigvals = np.linalg.eigvalsh(Q)
            min_eig = float(eigvals.min())
            note = "  ← near PSD boundary" if min_eig < 1e-6 else ""
            lines.append(
                f"    Q[{i}]  {Q.shape[0]}×{Q.shape[0]}   "
                f"min eigenvalue: {min_eig:.2e}{note}"
            )
        lines.append("")

        if r.cholesky_factors is not None:
            bound_str = (
                str(r.bound_exact) if r.bound_exact is not None else f"{r.bound:.8f}"
            )
            lines += [
                f"  Rounded to exact rationals.  Certified bound: {bound_str}",
                "  Run verify_certificate() to confirm the identity in exact arithmetic.",
            ]
        else:
            lines += [
                "  Floating-point certificate (raw solver output).",
                "  Run round_certificate() then verify_certificate() for an exact proof.",
            ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SharpsResult
# ---------------------------------------------------------------------------

def explain_sharps(s: "SharpsResult") -> str:
    bound_word = "lower" if s.problem.minimize else "upper"
    ineq = "≥" if s.problem.minimize else "≤"
    lhs = "density(H) − bound" if s.problem.minimize else "bound − density(H)"

    lines = [
        f"SharpsResult  (n={s.problem.n}, k={s.problem.k})",
        "",
        f"{len(s.indices)} sharp graph(s) found.",
        "",
        "A graph is sharp when its SDP slack is near zero — the certificate",
        "constraint is tight at that graph. Zero slack does not imply the graph",
        "attains the density bound; that is a separate (extremal) condition.",
        "",
        "The residual is the exact rational value of",
        f"  {lhs} − Σ_σ ⟨Q_σ, P_σ(H)⟩",
        "computed from the rounded certificate.  A residual of 0 means the",
        "certificate identity is tight at H; all residuals ≥ 0 verifies the proof.",
        "",
    ]

    if s.indices:
        lines.append("Sharp graphs:")
        for idx, g, d in zip(s.indices, s.graphs, s.densities):
            res = s.residuals[idx]
            lines.append(f"  [{idx}]  {g}   density = {d}   residual = {res}")
    else:
        lines.append("No sharp graphs found — all slacks are above the threshold.")

    return "\n".join(lines)


def html_sharps(s: "SharpsResult") -> str:
    bound_word = "lower" if s.problem.minimize else "upper"

    if not s.indices:
        body = "<span style='color:#aaa;'>No sharp graphs found.</span>"
        meta = f"n={s.problem.n}, k={s.problem.k}"
        return _card("SharpsResult", meta, body)

    cards_html = ""
    for idx, g, d in zip(s.indices, s.graphs, s.densities):
        res = s.residuals[idx]
        svg = _svg_graph(g, width=120, height=120)
        svg_block = f'<div style="margin:4px 0;">{svg}</div>' if svg else ""
        card_body = (
            f"<b>density</b> = {d}<br>"
            f"<b>residual</b> = {res}"
        )
        cards_html += (
            f'<div style="display:inline-block; vertical-align:top; '
            f'margin:4px; padding:8px; border:1px solid #ddd; border-radius:4px; '
            f'background:#fafafa; font-family:monospace; font-size:12px;">'
            f'<div style="font-weight:bold; color:#333; margin-bottom:2px;">H{idx}</div>'
            f'{svg_block}'
            f'<div style="color:#555;">{card_body}</div>'
            f'</div>'
        )

    meta = f"{len(s.indices)} sharp graph(s) · n={s.problem.n}, k={s.problem.k}"
    body = (
        f"<div style='color:#666; font-size:12px; margin-bottom:6px;'>"
        f"Sharp graphs — active certificate constraints ({bound_word}-bound problem):</div>"
        f"{cards_html}"
    )
    return _card("SharpsResult", meta, body)


def html_result(r) -> str:
    bound_kind = "lower bound" if r.problem.minimize else "upper bound"
    ineq = "≥" if r.problem.minimize else "≤"
    status_color = "#2a9d5c" if "optimal" in r.status else "#c0392b"
    sharps = _sharp_indices(r)

    if sharps:
        if r.data is not None:
            items = "".join(
                f"<li>[{i}] {r.data.admissible[i]}"
                f" &nbsp; (density={float(r.data.densities[i]):.4f})</li>"
                for i in sharps
            )
        else:
            items = "".join(f"<li>index {i}</li>" for i in sharps)
        sharp_html = (
            f"<b>Sharp graphs</b> ({len(sharps)}):"
            f"<ul style='margin:2px 0 2px 16px;'>{items}</ul>"
        )
    else:
        sharp_html = (
            "<span style='color:#aaa;'>No sharp graphs above threshold.</span><br>"
        )

    if r.Q is not None:
        q_sizes_str = ", ".join(str(q.shape[0]) for q in r.Q)
        cert_status = (
            "rounded &amp; exact"
            if r.cholesky_factors is not None
            else "floating-point — run round_certificate()"
        )
        q_html = (
            f"<b>Certificate:</b> {len(r.Q)} Q matrices ({q_sizes_str})"
            f" &nbsp;<span style='color:#888;font-size:11px;'>{cert_status}</span><br>"
        )
    else:
        q_html = (
            "<span style='color:#aaa;'>Certificate not extracted — "
            "rerun solve_sdp(extract_Q=True)</span><br>"
        )

    body = (
        f"<div style='font-size:26px; font-weight:bold; color:{status_color}; margin:6px 0;'>"
        f"{r.bound:.6f}</div>"
        f"<div style='color:#888; font-size:12px; margin-bottom:8px;'>"
        f"{bound_kind} &nbsp;·&nbsp; status: {r.status}</div>"
        f"{sharp_html}"
        f"{q_html}"
    )
    return _card("FlagAlgebraResult", f"k={r.problem.k}, n={r.problem.n}", body)


# ---------------------------------------------------------------------------
# Certificate
# ---------------------------------------------------------------------------

def explain_certificate(c: "Certificate") -> str:
    bound_kind = "lower" if c.problem.minimize else "upper"
    status = "VALID" if c.valid else "INVALID"

    lines = [
        f"Certificate  [{status}]  ({bound_kind} bound)",
        "",
        f"  Certified bound : {c.bound}  =  {float(c.bound):.8f}",
        f"  Valid           : {c.valid}",
        f"  Admissible graphs: {len(c.residuals)}",
        "",
        "The PSD matrices Q_σ define a flag-algebra sum-of-squares expression.",
        "Together with the verified coefficient inequalities over admissible",
        "graphs, this yields the stated bound.",
        "",
    ]

    # Active constraints
    ac = c.active_constraints
    lines.append(f"Active constraints — {len(ac)} graph(s) with residual = 0:")
    if ac:
        for i in ac:
            if c.data is not None:
                g = c.data.admissible[i]
                d = c.data.densities[i]
                lines.append(f"  [{i}]  {g}   density = {d}")
            else:
                lines.append(f"  [{i}]")
    else:
        lines.append("  none")
    lines.append("")

    # Q matrices
    lines.append(f"PSD certificate matrices ({len(c.Q)} total, one per type):")
    for i, Q in enumerate(c.Q):
        n = len(Q)
        lines.append(f"  Q[{i}]  {n}×{n}  (rational entries, PSD by construction)")

    if not c.valid:
        min_res = min(c.residuals)
        worst = c.residuals.index(min_res)
        lines += [
            "",
            "Certificate is INVALID.",
            f"  Minimum residual : {min_res}",
            f"  Worst graph index: {worst}",
        ]
        if c.data is not None:
            lines.append(f"  Worst graph      : {c.data.admissible[worst]}")

    return "\n".join(lines)


def html_certificate(c: "Certificate") -> str:
    bound_kind = "lower bound" if c.problem.minimize else "upper bound"
    valid_color = "#2a9d5c" if c.valid else "#c0392b"
    valid_label = "valid" if c.valid else "INVALID"

    ac = c.active_constraints

    def _graph_items(indices):
        if not indices:
            return "<span style='color:#aaa;'>none</span>"
        if c.data is not None:
            items = "".join(
                f"<li>[{i}] {c.data.admissible[i]}"
                + f" &nbsp; density={c.data.densities[i]}"
                + "</li>"
                for i in indices
            )
        else:
            items = "".join(f"<li>index {i}</li>" for i in indices)
        return f"<ul style='margin:2px 0 2px 16px;'>{items}</ul>"

    q_sizes = ", ".join(str(len(Q)) for Q in c.Q)
    body = (
        f"<div style='font-size:26px; font-weight:bold; color:{valid_color}; margin:6px 0;'>"
        f"{c.bound} &nbsp;"
        f"<span style='font-size:14px;'>[{valid_label}]</span></div>"
        f"<div style='color:#888; font-size:12px; margin-bottom:8px;'>"
        f"{bound_kind} &nbsp;·&nbsp; "
        f"Q matrices: {len(c.Q)} ({q_sizes})</div>"
        f"<b>Active constraints</b> ({len(ac)}):{_graph_items(ac)}"
    )
    return _card("Certificate", f"k={c.problem.k}, n={c.problem.n}", body)
