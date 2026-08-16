"""A minimal reference corpus of extremal-combinatorics problems with known
flag-algebra bounds.

Each entry pairs a canonical :class:`FlagProblem` factory with the expected
bound (as an exact :class:`~fractions.Fraction` when a closed form is known,
or a close rational approximation) and a citation.  Every entry is validated
by ``tests/test_corpus.py``: running the factory + solve + certify must
produce a bound within tolerance of ``expected_bound``.

The corpus exists to give an AI agent (or a human user) something to grep and
pattern-match against when translating a natural-language problem into a
:class:`FlagProblem`.  It is intentionally curated and small; a much larger
corpus harvested from the literature is planned for a future project.

Public API
----------
- :func:`list_entries`  — all entries in registration order.
- :func:`get`           — look up an entry by slug.
- :func:`search`        — filter entries by tag.

Every entry is a :class:`CorpusEntry` with fields documented on that class.
"""

from __future__ import annotations

from .entries import ALL_ENTRIES, CorpusEntry


_BY_SLUG: dict[str, CorpusEntry] = {e.slug: e for e in ALL_ENTRIES}


def list_entries() -> list[CorpusEntry]:
    """Return all corpus entries in their registration order."""
    return list(ALL_ENTRIES)


def get(slug: str) -> CorpusEntry:
    """Return the entry with ``slug``, or raise :class:`KeyError` if not found."""
    try:
        return _BY_SLUG[slug]
    except KeyError:
        known = ", ".join(sorted(_BY_SLUG))
        raise KeyError(
            f"no corpus entry with slug {slug!r}. Known slugs: {known}"
        ) from None


def search(tag: str) -> list[CorpusEntry]:
    """Return all entries whose ``tags`` include ``tag``."""
    return [e for e in ALL_ENTRIES if tag in e.tags]


__all__ = ["CorpusEntry", "list_entries", "get", "search"]
