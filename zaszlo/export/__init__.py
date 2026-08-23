"""Certificate exporters to external toolchains.

Currently supports:

- :mod:`zaszlo.export.lean` — Flagmatic-format JSON consumed by
  the ``flag_certificate`` tactic of the taeyool
  ``lean-flag-algebras-release`` Lean 4 library
  (Jeong, Park, Hyun, Oum, Yang; arXiv:2607.23500; Apache-2.0).
  Turns a :class:`zaszlo.types.Certificate` into a JSON blob whose
  elaboration by the Lean tactic yields an axiom-free proof of the bound.
"""

from .lean import (
    LeanExportError,
    flagmatic_bound,
    flagmatic_flag_string,
    flagmatic_graph_string,
    to_flagmatic_certificate,
    write_flagmatic_certificate,
)

__all__ = [
    "LeanExportError",
    "flagmatic_bound",
    "flagmatic_flag_string",
    "flagmatic_graph_string",
    "to_flagmatic_certificate",
    "write_flagmatic_certificate",
]
