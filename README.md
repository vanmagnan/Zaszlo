# Zászló

A Python library for flag algebra computations in extremal combinatorics. It supports graphs and uniform hypergraphs through a single interface, compiles problem specifications into semidefinite programs, and turns numerical solutions into exact rational certificates that can be independently verified.

## Tutorials

| Notebook | Description | |
|---|---|---|
| `tutorial.ipynb` | Full walkthrough: Mantel, rational certificates, pentagon problem, K₄⁻-free 3-graphs | [![Launch Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/vanmagnan/zaszlo/HEAD?labpath=tutorial.ipynb) |
| `Short_Tutorial.ipynb` | Condensed intro to the core workflow via Mantel's theorem | [![Launch Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/vanmagnan/zaszlo/HEAD?labpath=Short_Tutorial.ipynb) |


## Installation

```bash
git clone https://github.com/vanmagnan/zaszlo.git
cd zaszlo
pip install -e ".[dev]"           # development + tests
pip install -e ".[tutorial]"      # tutorial notebooks
pip install -e ".[dev,tutorial]"  # everything
```

Requires Python ≥ 3.10.  Core dependencies are [Clarabel](https://clarabel.org/), SciPy, and NumPy;
the `[dev]` extra adds pytest for running the test suite;
the `[tutorial]` extra adds matplotlib, networkx, and jupyter for the tutorial notebooks.

## Quick example: Mantel's theorem

The maximum edge density in a triangle-free graph is 1/2 (Mantel, 1907).

```python
from zaszlo import FlagProblem, complete, solve

prob   = FlagProblem(4, 2, 2, forbidden=[complete(3)])
result = solve(prob)
print(result.bound)               # ≈ 0.5

result = solve(prob, certify=True)
print(result.certificate.valid)   # True
print(result.certificate.bound)   # Fraction close to 1/2
```

For a step-by-step walkthrough — including rational certificates, the pentagon problem, and
K₄⁻-free 3-uniform hypergraphs — see the tutorial notebooks above.

## API overview

`solve(prob)` is the main entry point.  Pass `certify=True` to also produce
an exact rational certificate:

| Call | `result.bound` | `result.certificate` |
|---|---|---|
| `solve(prob)` | float | `None` |
| `solve(prob, certify=True)` | float | `Certificate` with exact `Fraction` bound |

The full step-by-step pipeline is also public for fine-grained control:

```python
from zaszlo import build_flag_algebra_data, solve_sdp, certify, identify_sharps

data   = build_flag_algebra_data(prob)   # types, flags, pair densities
result = solve_sdp(data, extract_Q=True) # float SDP solution
proof  = certify(result)                 # exact rational Certificate
```

Every object in this library has an `.explain()` method that returns a
plain-English description of its mathematical content — useful for
interactive inspection or when working with an LLM.

```python
prob.explain()                    # what problem is being solved
result.explain()                  # bound, status, sharp graphs
result.certificate.explain()      # full rational proof breakdown
```

## Running the tests

```bash
pytest tests/ -v
```

## Solver

zászló uses [Clarabel](https://clarabel.org/) directly as its SDP solver.  For fine-grained
control over tolerances or iteration limits, pass a `clarabel.DefaultSettings` instance:

```python
import clarabel
settings = clarabel.DefaultSettings()
settings.eps_abs = 1e-9
result = solve_sdp(data, extract_Q=True, settings=settings)
```

## References

- Razborov, A. (2007). Flag algebras. *Journal of Symbolic Logic*, 72(4), 1239–1282.
- Mantel, W. (1907). Problem 28. *Wiskundige Opgaven*, 10, 60–61.
- Grzesik, A. (2012). On the maximum number of five-cycles in a triangle-free graph.
  *Journal of Combinatorial Theory, Series B*, 102(5), 1061–1066.
- Hatami, H., Hladký, J., Král', D., Norine, S., & Razborov, A. (2013).
  On the number of pentagons in triangle-free graphs.
  *Journal of Combinatorial Theory, Series B*, 103(3), 452–465.
- Frankl, P. & Füredi, Z. (1984). A new generalization of the Erdős–Ko–Rado theorem.
  *Combinatorica*, 4(4), 341–349.
- Keevash, P. & Sudakov, B. (2005). The Turán number of the Fano plane.
  *Combinatorica*, 25(5), 561–574.

## License

MIT
