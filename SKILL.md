---
name: zaszlo
description: Prove extremal-combinatorics bounds via the flag-algebra semidefinite method. Given a problem statement about hypergraph density (Turán-type, target-density, hypergraph, linear-combination target), translate it into a FlagProblem, run solve+certify, inspect the DiagnosticReport, iterate as needed, and return the exact rational certificate.
allowed-tools: Bash, Read, Write, Edit
---

# Zászló — Agent Guide

Zászló is a Python library that turns extremal-combinatorics questions into semidefinite programs (SDPs) via Razborov's flag-algebra method, solves them numerically, and produces exact rational proof certificates. This document is a **map** of the library for AI agents. It is not a tutorial — for that, see `tutorial.ipynb` and `Short_Tutorial.ipynb`.

## What Zászló proves

A `FlagProblem` specifies:
- a class of admissible k-uniform hypergraphs (defined by forbidden non-induced and induced subgraphs),
- a target quantity to bound (edge density by default; any single induced density; or an explicit rational linear combination of induced densities), and
- an optimization direction (upper or lower bound).

`solve(prob, certify=True)` runs the full pipeline: enumerate the flag algebra, build and solve the SDP with Clarabel, round the floating-point PSD matrices to exact rationals, and verify the sum-of-squares identity in exact `Fraction` arithmetic. When it succeeds it returns a `FlagAlgebraResult` whose `.certificate` is a valid `Certificate` — the bound holds for every admissible graphon.

The library can also inject auxiliary flag-algebra inequalities `⟦e⟧ ≥ 0` via `prob.add_constraint(...)`, which corresponds to standard flag-algebra proof techniques where extra known SoS inequalities tighten the SDP bound.

## Problem taxonomy

Map a natural-language problem statement to one of these shapes.

| Problem shape                                     | `k` | `target=`                     | `forbidden=`     | Notes |
|---------------------------------------------------|-----|-------------------------------|------------------|-------|
| "Max edge density of {F}-free graphs" (Turán)     | 2   | `None` (edge density)         | `[F]`            | Set `n ≥ |V(F)| + 2`, `type_order = n − 2`. |
| "Max edge density of {F}-free k-uniform hypergraphs" | k   | `None`                       | `[F]`            | Set `k=3` for 3-graphs; forbidden shapes must have `k=3` too. |
| "Max density of induced-H in {F}-free graphs"     | k   | `H` (a `Hypergraph`)          | `[F]`            | For non-induced density in a special setting where every copy is automatically induced, use the same setup. |
| "Multiple forbidden patterns"                     | k   | `None` or `H`                 | `[F1, F2, ...]`  | Both `forbidden` and `forbidden_induced` accept lists. |
| "Optimize a rational combination of densities"    | k   | `DensityExpr([...])`          | `[F]`            | Use `Fraction` coefficients; floats are rejected. |
| "Prove a bound tighter than the raw SDP"          | any | any                           | any              | Add `AuxiliaryConstraint(unlabel(e * e))` for a flag-algebra element `e`; the SDP will pick nonneg weights μ_j. |
| "Lower bound on density" (rather than upper)      | any | any                           | any              | Set `minimize=True`. |

## Recommended workflow

```python
from zaszlo import FlagProblem, complete, solve

# 1. State the problem as a FlagProblem.
prob = FlagProblem(4, 2, 2, forbidden=[complete(3)])

# 2. Solve + certify in one call.
result = solve(prob, certify=True)

# 3. Inspect the diagnostic report (structured, machine-readable).
diag = result.diagnose()             # DiagnosticReport dataclass
print(diag.to_json(indent=2))         # or diag.to_dict() for a Python dict

# 4. Decide next action based on the report:
#    - `is_valid=True`  and `max_sharp_density_matches_bound=True`  → done.
#    - `is_valid=False`                                             → increase `denom_limit` on solve(...).
#    - `max_sharp_density_matches_bound=False` with valid cert       → try larger `n`, or add an auxiliary constraint.
#    - `status != "optimal"`                                        → tune Clarabel settings or check problem well-posedness.

# 5. Emit the certificate as JSON for reporting / handoff.
print(result.certificate.to_json())
```

## Parameter selection guide

- **`k`**: edge uniformity. `k=2` = ordinary graphs, `k=3` = 3-uniform hypergraphs, etc. Every graph in `forbidden` and any `target` must share this `k`.
- **`n`**: number of vertices in the admissible graphs enumerated by the SDP. Larger `n` gives tighter bounds but grows the flag counts quickly. Start with `n = max(|V(F)| for F in forbidden) + 1` and raise as needed.
- **`type_order`**: maximum type-vertex count. Constraints:
  - `type_order ≤ n − 2` (else the flags fill the admissible graph, trivializing the SDP).
  - `type_order` and `n` must have the same parity (else flag sizes `(n + s) / 2` are non-integer).
  - **Default rule**: pick `type_order = n − 2`. That's the largest legal value.
- **When to raise `n`**: if `diagnose()` reports `is_valid=True` but `max_sharp_density_matches_bound=False`, the bound is loose. Try `n = n + 2` (preserving parity) and rerun.
- **When to add an auxiliary constraint**: if raising `n` becomes expensive and the bound is stuck, an SoS inequality `⟦(e)²⟧ ≥ 0` for a well-chosen algebra element `e` may tighten the bound. Start with `e = f − g` where `f, g` are flags over the same type that "should be equal" at the extremal graphon.

## Common pitfalls

- **Parity mismatch**: `FlagProblem(5, 2, 2)` raises because `n=5` is odd and `type_order=2` is even. The error message lists the valid `type_order` values for the given `n`.
- **`type_order > n − 2`**: e.g. `FlagProblem(4, 4, 2)` — flags at grade `n`, SDP trivial.
- **Uniformity mismatch**: passing a `k=2` graph in `forbidden=[...]` for a `k=3` problem raises immediately.
- **Floats in coefficients**: `DensityExpr([(0.5, G)])` is rejected. Wrap in `Fraction(1, 2)`. This is deliberate — floats would silently corrupt the exact certificate.
- **"Non-induced" vs "induced"**: `forbidden=[F]` forbids **any** copy of F (non-induced). Use `forbidden_induced=[F]` for induced copies only. For a graph like `pentagon_c5_density`, in a triangle-free host every C₅ is automatically induced so both give the same class.
- **`certify=True` but certificate invalid**: usually a rounding issue. Increase `denom_limit` on `solve(prob, certify=True, denom_limit=10000)`; default is 1000.

## Reference corpus

`zaszlo.corpus` holds 10 curated problems with known bounds. Each entry is validated by `tests/test_corpus.py` and pairs a fresh `FlagProblem` factory with a citation and an expected bound.

```python
from zaszlo import corpus

for e in corpus.list_entries():
    print(f"{e.slug:32s}  {e.expected_bound}  ({e.citation})")

# Look up by slug:
entry = corpus.get("mantel")
prob = entry.problem_factory()

# Filter by tag:
turan_problems = corpus.search("turan")
```

Current corpus (10 entries): `mantel`, `turan_k4`, `turan_k5`, `pentagon_c5_density`, `k4_minus_free_3graphs`, `f32_free_3graphs`, `c5_tight_free_3graphs`, `affine_mantel`, `mantel_multi_forbidden`, `mantel_with_aux_sos`.

**When translating a new problem, first check whether it matches a corpus entry** — the entry's setup is a proven-good starting point.

## Key APIs at a glance

```python
# Types
FlagProblem(n, type_order, k, *, forbidden, forbidden_induced,
            target, minimize, aux_constraints)
Hypergraph(n, k, edges)                # 1-based vertices
Flag(graph, type_size)                  # type_size labeled leading vertices
DensityExpr([(coef, graph), ...])       # rational linear combo of induced densities
FlagAlgebraElement(type, n, terms)     # element of A^σ (via Flag arithmetic)
UnlabeledExpr(k, n, terms)             # element of A^∅
AuxiliaryConstraint(unlabeled_expr)    # ⟦e⟧ ≥ 0 injected into SDP

# Named constructors
complete(n, k=2)                        # complete k-uniform on n
k4_minus(); c5_3uniform(); f32()        # standard hypergraph patterns

# Entry points
solve(prob, certify=True)               # end-to-end
build_flag_algebra_data(prob)           # step 1: enumerate
solve_sdp(data, extract_Q=True)         # step 2: numerical SDP
certify(result)                         # step 3: round + verify → Certificate
identify_sharps(data, result)           # extremal (sharp) graphs
flag_product(f, g); unlabel(e); lift_to(u, n)   # algebra kernel

# Reporting / introspection
obj.explain()                           # plain-text description (works on every zaszlo type)
result.diagnose()                       # structured DiagnosticReport
result.to_dict(); result.to_json()      # machine-readable serialization
result.certificate.to_json()            # exact rational proof certificate
```

## Where to look next

- `tutorial.ipynb` — annotated walkthrough of Mantel, rational certificates, pentagon, K₄⁻-free, plus multi-forbidden and DensityExpr sections.
- `Short_Tutorial.ipynb` — condensed intro focused on the workflow.
- `CLAUDE.md` — architecture and design principles for humans/agents modifying the library.
- `zaszlo/corpus/entries.py` — the source of truth for corpus content; add new entries here.
- `tests/test_sdp.py` — every end-to-end test class demonstrates a problem shape.

## When to hand back to the user

- Certificate is valid and matches a corpus entry's expected bound → **done**, return `certificate.to_json()`.
- Certificate is valid but the numerical bound is loose (extremal not attained) → report the diagnostic, ask whether to spend compute on larger `n` / additional aux constraints.
- Solver returns non-optimal status → surface the error message and the diagnostic; do not silently retry.
- Problem statement is ambiguous about induced vs. non-induced, or about upper vs. lower bound → ask before setting up the `FlagProblem`.
