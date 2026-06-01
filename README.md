# Zászló

A Python library implementing the **flag algebra method** for extremal combinatorics.

Given a problem specification — which subgraphs are forbidden, which pattern density to
bound — zászló enumerates combinatorial structures, builds and solves a semidefinite program
(SDP), and verifies the resulting certificate in exact rational arithmetic.

## Tutorials

| Notebook | Description | |
|---|---|---|
| `tutorial.ipynb` | Full walkthrough: Mantel, rational certificates, pentagon problem, K₄⁻-free 3-graphs | [![Launch Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/vanmagnan/zaszlo/HEAD?labpath=tutorial.ipynb) |
| `Short_Tutorial.ipynb` | Condensed intro to the core workflow via Mantel's theorem | [![Launch Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/vanmagnan/zaszlo/HEAD?labpath=Short_Tutorial.ipynb) |


## Installation

```bash
pip install -e ".[dev]"
```

Requires Python ≥ 3.10.  Core dependencies are [Clarabel](https://clarabel.org/), SciPy, and NumPy;
the `[dev]` extra adds pytest for running the test suite;
the `[tutorial]' extra adds matplotlib, networkx, and jupyter for the tutorial notebooks.

## Quick example: Mantel's theorem

The maximum edge density in a triangle-free graph is 1/2 (Mantel, 1907).

```python
from zaszlo import FlagProblem, Hypergraph, build_flag_algebra_data, solve_sdp, verify_certificate

K3 = Hypergraph(3, 2, [(1, 2), (1, 3), (2, 3)])

prob = FlagProblem(
    n=4,
    type_order=2,
    k=2,
    forbidden=[K3],
    minimize=False,
)

data   = build_flag_algebra_data(prob)
result = solve_sdp(data, extract_Q=True)
cert   = verify_certificate(data, result)

print(cert["lam_certified"])   # 1/2
```

For a step-by-step walkthrough — including rational certificates, the pentagon problem, and
K₄⁻-free 3-uniform hypergraphs — see the tutorial notebooks above.

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
