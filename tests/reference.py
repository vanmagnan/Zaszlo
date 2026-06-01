# Utilities for parsing C flagmatic reference output.
# Used only in tests — not part of the zaszlo package.
#
# Mirrors flagmatic_julia/test/reference.jl exactly.

from __future__ import annotations

import re
from fractions import Fraction
from pathlib import Path

from zaszlo.types import Flag, Hypergraph


# ---------------------------------------------------------------------------
# C graph string parser
# ---------------------------------------------------------------------------

def parse_c_graph(s: str, k: int) -> Hypergraph:
    """Parse a C flagmatic graph string into a Hypergraph.

    Format: "n:e1e2...em" where each edge is k consecutive decimal digit
    characters (1-based vertex indices).

    Examples
    --------
    >>> parse_c_graph("4:121314", 2)   # 4v graph with edges (1,2),(1,3),(1,4)
    >>> parse_c_graph("5:123234", 3)   # 5v 3-graph with edges (1,2,3),(2,3,4)
    >>> parse_c_graph("0:", 2)         # empty hypergraph
    """
    colon = s.index(":")
    n = int(s[:colon])
    edge_str = s[colon + 1:]
    if not edge_str:
        return Hypergraph(n, k, [])
    if len(edge_str) % k != 0:
        raise ValueError(f"Edge string length not divisible by k={k}: {s!r}")
    edges = [
        tuple(int(edge_str[i + j]) for j in range(k))
        for i in range(0, len(edge_str), k)
    ]
    return Hypergraph(n, k, edges)


def parse_c_flag(s: str, k: int, type_size: int) -> Flag:
    """Parse a C graph string into a Flag with the given type_size."""
    return Flag(parse_c_graph(s, k), type_size)


# ---------------------------------------------------------------------------
# flags.py parser
# ---------------------------------------------------------------------------

def _extract_list(text: str, name: str) -> list[str]:
    """Extract a named list of quoted strings from flags.py."""
    pattern = re.compile(
        rf"{re.escape(name)}\s*=\s*\[([^\[\]]*?)\]", re.DOTALL
    )
    m = pattern.search(text)
    if m is None:
        raise ValueError(f"Could not find '{name}' in flags.py")
    return re.findall(r'"([^"]*)"', m.group(1))


def _extract_int(text: str, name: str) -> int:
    m = re.search(rf"{re.escape(name)}\s*=\s*(\d+)", text)
    if m is None:
        raise ValueError(f"Could not find '{name}' in flags.py")
    return int(m.group(1))


def _extract_int_list(text: str, name: str) -> list[int]:
    m = re.search(rf"{re.escape(name)}\s*=\s*\[([^\]]*)\]", text)
    if m is None:
        raise ValueError(f"Could not find '{name}' in flags.py")
    return [int(s.strip()) for s in m.group(1).split(",") if s.strip()]


def _extract_nested_list(text: str) -> list[list[str]]:
    """Extract the nested flags list from flags.py.

    The flags variable has the form:
        flags = [
            [
                "...",
            ],
            [
                "...",
            ],
        ]
    """
    result: list[list[str]] = []
    current: list[str] = []
    in_flags = False
    in_inner = False

    for line in text.splitlines():
        s = line.strip()
        if not in_flags:
            if s == "flags = [" or s.startswith("flags = ["):
                in_flags = True
        elif s == "[" or s == "],[":
            in_inner = True
            current = []
        elif (s in ("]", "],")) and in_inner:
            result.append(current)
            current = []
            in_inner = False
        elif s == "]":
            break
        elif in_inner:
            m = re.search(r'"([^"]*)"', s)
            if m:
                current.append(m.group(1))

    return result


def parse_flags_py(path: str | Path, k: int) -> dict:
    """Parse a complete flags.py file.

    Returns a dict with keys:
        n, num_types, num_flags, types, flags, H
    where types is a list of Flag, flags is a list of lists of Flag,
    and H is a list of Hypergraph.
    """
    text = Path(path).read_text()
    n         = _extract_int(text, "n")
    num_types = _extract_int(text, "num_types")
    num_flags = _extract_int_list(text, "num_flags")
    type_strs = _extract_list(text, "types")
    flag_strs = _extract_nested_list(text)
    H_strs    = _extract_list(text, "H")

    types = [parse_c_flag(s, k, parse_c_graph(s, k).n) for s in type_strs]
    flags = [
        [parse_c_flag(s, k, types[sigma].type_size) for s in flag_strs[sigma]]
        for sigma in range(num_types)
    ]
    H = [parse_c_graph(s, k) for s in H_strs]

    return dict(n=n, num_types=num_types, num_flags=num_flags,
                types=types, flags=flags, H=H)


# ---------------------------------------------------------------------------
# flags.rat parser
# ---------------------------------------------------------------------------

def parse_flags_rat(path: str | Path) -> dict[tuple[int, int, int, int], Fraction]:
    """Parse flags.rat into a lookup dict.

    Format per line: H_idx block_idx flag_i flag_j numer denom
    block_idx starts at 2 in the C output (block 1 is reserved).

    Returns
    -------
    dict mapping (H_idx, sigma, i, j) → Fraction
    where H_idx and i, j are 1-based (matching C convention),
    and sigma is 0-based type index (block_idx - 2).
    """
    result: dict[tuple[int, int, int, int], Fraction] = {}
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) != 6:
            continue
        H_idx, block, i, j, numer, denom = map(int, parts)
        sigma = block - 2  # C blocks start at 2; sigma=0 is the first type
        result[(H_idx, sigma, i, j)] = Fraction(numer, denom)
    return result
