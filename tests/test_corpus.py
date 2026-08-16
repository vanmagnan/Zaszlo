# Corpus tests — self-validating.
#
# For every entry in zaszlo.corpus.list_entries():
#   1. problem_factory() must return a valid FlagProblem
#   2. solve(prob, certify=True) must succeed
#   3. certified bound must match expected_bound within tolerance
#
# Also tests the registry API (get, search) and metadata invariants.

from __future__ import annotations

from fractions import Fraction

import pytest

from zaszlo import FlagProblem, corpus, solve
from zaszlo.corpus import CorpusEntry


# ---------------------------------------------------------------------------
# Per-entry SDP validation
# ---------------------------------------------------------------------------

# The c5_tight_free_3graphs entry only converges to ~1/2 at n=5 — allow a
# looser tolerance for it as documented in its notes.
_LOOSE_TOLERANCE_ENTRIES = {"c5_tight_free_3graphs"}
_DEFAULT_TOLERANCE = 1e-3
_LOOSE_TOLERANCE = 1e-2


@pytest.mark.parametrize("entry", corpus.list_entries(), ids=lambda e: e.slug)
class TestEntrySolves:
    def test_factory_returns_flag_problem(self, entry):
        prob = entry.problem_factory()
        assert isinstance(prob, FlagProblem)

    def test_solve_and_certify(self, entry):
        prob = entry.problem_factory()
        result = solve(prob, certify=True)
        tolerance = (
            _LOOSE_TOLERANCE if entry.slug in _LOOSE_TOLERANCE_ENTRIES
            else _DEFAULT_TOLERANCE
        )
        # Numerical bound close to expected.
        assert abs(result.bound - float(entry.expected_bound)) < tolerance, (
            f"[{entry.slug}] bound {result.bound} vs expected {float(entry.expected_bound)}"
        )
        # Certificate is valid.
        assert result.certificate.valid, (
            f"[{entry.slug}] certificate was invalid"
        )


# ---------------------------------------------------------------------------
# Registry API
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_list_entries_nonempty(self):
        assert len(corpus.list_entries()) >= 1

    def test_all_entries_are_corpus_entries(self):
        for e in corpus.list_entries():
            assert isinstance(e, CorpusEntry)

    def test_slugs_are_unique(self):
        slugs = [e.slug for e in corpus.list_entries()]
        assert len(slugs) == len(set(slugs)), "duplicate slugs in corpus"

    def test_get_by_slug(self):
        e = corpus.get("mantel")
        assert e.slug == "mantel"
        assert e.expected_bound == Fraction(1, 2)

    def test_get_unknown_slug_raises(self):
        with pytest.raises(KeyError, match="no corpus entry"):
            corpus.get("this_does_not_exist")

    def test_search_by_tag(self):
        turan = corpus.search("turan")
        assert len(turan) >= 3
        assert all("turan" in e.tags for e in turan)

    def test_search_by_missing_tag_returns_empty(self):
        assert corpus.search("nonexistent-tag") == []


# ---------------------------------------------------------------------------
# Entry metadata sanity
# ---------------------------------------------------------------------------

class TestEntryMetadata:
    @pytest.mark.parametrize("entry", corpus.list_entries(), ids=lambda e: e.slug)
    def test_required_fields_nonempty(self, entry):
        assert entry.slug
        assert entry.name
        assert entry.citation
        assert entry.description
        assert entry.tags
        assert callable(entry.problem_factory)
        assert isinstance(entry.expected_bound, Fraction)

    @pytest.mark.parametrize("entry", corpus.list_entries(), ids=lambda e: e.slug)
    def test_factory_is_repeatable(self, entry):
        # Calling the factory twice must yield two independent FlagProblem
        # instances with matching parameters.  Guards against accidentally
        # sharing mutable state (like the aux_constraints list).
        p1 = entry.problem_factory()
        p2 = entry.problem_factory()
        assert p1 is not p2
        assert p1.n == p2.n and p1.k == p2.k and p1.type_order == p2.type_order
        assert len(p1.aux_constraints) == len(p2.aux_constraints)
