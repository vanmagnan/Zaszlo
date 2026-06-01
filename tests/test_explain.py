# Tests for .explain() and _repr_html_() on zaszlo types.
#
# Fast tests (no SDP solving) cover Hypergraph, Flag, FlagProblem.
# Module-scoped fixtures solve Mantel once for FlagAlgebraData /
# FlagAlgebraResult tests, keeping the suite reasonably fast.

from __future__ import annotations

import pytest

from zaszlo import (
    Flag,
    FlagAlgebraResult,
    FlagProblem,
    Hypergraph,
    build_flag_algebra_data,
    solve_sdp,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mantel_data():
    k3 = Hypergraph(3, 2, [(1, 2), (1, 3), (2, 3)])
    prob = FlagProblem(4, 2, 2, forbidden=[k3], minimize=False)
    return build_flag_algebra_data(prob)


@pytest.fixture(scope="module")
def mantel_result(mantel_data):
    return solve_sdp(mantel_data, extract_Q=True)


# ---------------------------------------------------------------------------
# Hypergraph
# ---------------------------------------------------------------------------

class TestHypergraphExplain:
    def test_returns_string(self):
        g = Hypergraph(3, 2, [(1, 2), (2, 3), (1, 3)])
        assert isinstance(g.explain(), str)

    def test_contains_n_k(self):
        g = Hypergraph(4, 3, [(1, 2, 3), (1, 2, 4)])
        text = g.explain()
        assert "n=4" in text
        assert "k=3" in text

    def test_contains_edge_count(self):
        g = Hypergraph(4, 3, [(1, 2, 3), (1, 2, 4)])
        assert "2 edge" in g.explain()

    def test_contains_density(self):
        g = Hypergraph(3, 2, [(1, 2), (2, 3), (1, 3)])
        text = g.explain()
        assert "density" in text.lower()
        assert "3/3" in text or "1.0000" in text

    def test_empty_edges(self):
        g = Hypergraph(4, 2, [])
        text = g.explain()
        assert "0 edge" in text
        assert "(none)" in text

    def test_graph_description_k2(self):
        g = Hypergraph(3, 2, [(1, 2)])
        assert "graph" in g.explain().lower()

    def test_graph_description_k3(self):
        g = Hypergraph(4, 3, [(1, 2, 3)])
        assert "3-uniform" in g.explain()

    def test_single_vertex(self):
        g = Hypergraph(1, 2, [])
        text = g.explain()
        assert isinstance(text, str)
        assert "n=1" in text


class TestHypergraphMimebundle:
    def test_has_both_keys(self):
        g = Hypergraph(3, 2, [(1, 2)])
        bundle = g._repr_mimebundle_()
        assert "text/html" in bundle
        assert "text/plain" in bundle

    def test_html_matches_repr_html(self):
        g = Hypergraph(3, 2, [(1, 2)])
        assert g._repr_mimebundle_()["text/html"] == g._repr_html_()

    def test_plain_matches_repr(self):
        g = Hypergraph(3, 2, [(1, 2)])
        assert g._repr_mimebundle_()["text/plain"] == repr(g)

    def test_include_filter(self):
        g = Hypergraph(3, 2, [(1, 2)])
        bundle = g._repr_mimebundle_(include={"text/plain"})
        assert "text/plain" in bundle
        assert "text/html" not in bundle

    def test_exclude_filter(self):
        g = Hypergraph(3, 2, [(1, 2)])
        bundle = g._repr_mimebundle_(exclude={"text/html"})
        assert "text/html" not in bundle
        assert "text/plain" in bundle


class TestHypergraphHtml:
    def test_returns_string(self):
        g = Hypergraph(3, 2, [(1, 2), (2, 3), (1, 3)])
        assert isinstance(g._repr_html_(), str)

    def test_is_html(self):
        g = Hypergraph(3, 2, [(1, 2), (2, 3)])
        html = g._repr_html_()
        assert "<" in html and ">" in html

    def test_svg_for_k2_small(self):
        g = Hypergraph(4, 2, [(1, 2), (2, 3)])
        assert "<svg" in g._repr_html_()

    def test_svg_for_k3_small(self):
        g = Hypergraph(4, 3, [(1, 2, 3), (1, 2, 4)])
        assert "<svg" in g._repr_html_()

    def test_no_svg_for_large_k(self):
        g = Hypergraph(5, 4, [(1, 2, 3, 4)])
        html = g._repr_html_()
        assert isinstance(html, str)
        assert "<svg" not in html

    def test_no_svg_for_large_n(self):
        edges = [(i, i + 1) for i in range(1, 9)]
        g = Hypergraph(9, 2, edges)
        assert "<svg" not in g._repr_html_()

    def test_contains_density(self):
        g = Hypergraph(3, 2, [(1, 2)])
        assert "1/3" in g._repr_html_() or "0.3333" in g._repr_html_()


# ---------------------------------------------------------------------------
# Flag
# ---------------------------------------------------------------------------

class TestFlagExplain:
    def test_admissible_role(self):
        g = Hypergraph(3, 2, [(1, 2)])
        f = Flag(g, 0)
        text = f.explain()
        assert "admissible" in text.lower()

    def test_type_role(self):
        g = Hypergraph(2, 2, [])
        f = Flag(g, 2)
        text = f.explain()
        assert "type" in text.lower()

    def test_flag_mentions_labeled_unlabeled(self):
        g = Hypergraph(4, 2, [(1, 2), (3, 4)])
        f = Flag(g, 2)
        text = f.explain()
        assert "labeled" in text.lower()
        assert "unlabeled" in text.lower()

    def test_flag_vertex_lists(self):
        g = Hypergraph(4, 2, [(1, 2)])
        f = Flag(g, 2)
        text = f.explain()
        assert "[1, 2]" in text
        assert "[3, 4]" in text

    def test_contains_edges(self):
        g = Hypergraph(3, 2, [(1, 3)])
        f = Flag(g, 1)
        text = f.explain()
        assert "{1,3}" in text

    def test_empty_edges(self):
        g = Hypergraph(3, 2, [])
        f = Flag(g, 0)
        assert "(none)" in f.explain()


class TestFlagHtml:
    def test_is_html(self):
        g = Hypergraph(4, 2, [(1, 2), (3, 4)])
        f = Flag(g, 2)
        html = f._repr_html_()
        assert "<" in html

    def test_svg_present_for_small(self):
        g = Hypergraph(4, 2, [(1, 2)])
        f = Flag(g, 1)
        assert "<svg" in f._repr_html_()

    def test_labeled_color_in_svg(self):
        g = Hypergraph(4, 2, [(1, 2)])
        f = Flag(g, 2)
        html = f._repr_html_()
        # Orange fill for labeled vertices should appear in the SVG
        assert "#e07b39" in html

    def test_unlabeled_color_in_svg(self):
        g = Hypergraph(4, 2, [(1, 2)])
        f = Flag(g, 1)
        html = f._repr_html_()
        assert "#4a90d9" in html

    def test_type_all_labeled(self):
        g = Hypergraph(2, 2, [])
        f = Flag(g, 2)
        html = f._repr_html_()
        assert "type" in html.lower() or "labeled" in html.lower()


# ---------------------------------------------------------------------------
# FlagProblem
# ---------------------------------------------------------------------------

class TestFlagProblemExplain:
    def test_returns_string(self):
        prob = FlagProblem(4, 2, 2)
        assert isinstance(prob.explain(), str)

    def test_contains_n_k_type_order(self):
        prob = FlagProblem(5, 3, 3)
        text = prob.explain()
        assert "n=5" in text
        assert "k" in text and "3" in text
        assert "type_order" in text

    def test_forbidden_mentioned(self):
        k3 = Hypergraph(3, 2, [(1, 2), (1, 3), (2, 3)])
        prob = FlagProblem(4, 2, 2, forbidden=[k3])
        assert "forbidden" in prob.explain().lower()

    def test_no_forbidden_mentioned(self):
        prob = FlagProblem(4, 2, 2)
        assert "no forbidden" in prob.explain().lower()

    def test_minimize_mode(self):
        prob = FlagProblem(4, 2, 2, minimize=True)
        text = prob.explain().lower()
        assert "lower" in text or "minim" in text

    def test_maximize_mode(self):
        prob = FlagProblem(4, 2, 2, minimize=False)
        text = prob.explain().lower()
        assert "upper" in text or "maxim" in text

    def test_target_mentioned(self):
        c5 = Hypergraph(5, 2, [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)])
        k3 = Hypergraph(3, 2, [(1, 2), (1, 3), (2, 3)])
        prob = FlagProblem(5, 3, 2, forbidden=[k3], target=c5)
        assert "target" in prob.explain().lower()

    def test_forbidden_induced_mentioned(self):
        k3 = Hypergraph(3, 2, [(1, 2), (1, 3), (2, 3)])
        prob = FlagProblem(4, 2, 2, forbidden_induced=[k3])
        assert "induced" in prob.explain().lower()


class TestFlagProblemHtml:
    def test_is_html(self):
        prob = FlagProblem(4, 2, 2)
        assert "<" in prob._repr_html_()

    def test_contains_n_k(self):
        prob = FlagProblem(5, 3, 3)
        html = prob._repr_html_()
        assert "5" in html
        assert "3" in html

    def test_forbidden_in_html(self):
        k3 = Hypergraph(3, 2, [(1, 2), (1, 3), (2, 3)])
        prob = FlagProblem(4, 2, 2, forbidden=[k3])
        html = prob._repr_html_()
        assert "forbidden" in html.lower() or "Forbidden" in html


# ---------------------------------------------------------------------------
# FlagAlgebraData
# ---------------------------------------------------------------------------

class TestFlagAlgebraDataExplain:
    def test_returns_string(self, mantel_data):
        assert isinstance(mantel_data.explain(), str)

    def test_type_count(self, mantel_data):
        text = mantel_data.explain()
        assert "3" in text  # 3 types in Mantel

    def test_admissible_count(self, mantel_data):
        text = mantel_data.explain()
        assert "7" in text  # 7 admissible K3-free graphs on 4 vertices

    def test_mentions_pair_densities(self, mantel_data):
        assert "pair" in mantel_data.explain().lower() or "dens" in mantel_data.explain().lower()

    def test_mentions_flags(self, mantel_data):
        assert "flag" in mantel_data.explain().lower()


class TestFlagAlgebraDataHtml:
    def test_is_html(self, mantel_data):
        assert "<" in mantel_data._repr_html_()

    def test_has_table(self, mantel_data):
        assert "<table" in mantel_data._repr_html_()

    def test_admissible_count_in_html(self, mantel_data):
        assert "7" in mantel_data._repr_html_()

    def test_type_count_in_html(self, mantel_data):
        assert "3" in mantel_data._repr_html_()


# ---------------------------------------------------------------------------
# FlagAlgebraResult
# ---------------------------------------------------------------------------

class TestFlagAlgebraResultExplain:
    def test_returns_string(self, mantel_result):
        assert isinstance(mantel_result.explain(), str)

    def test_contains_bound(self, mantel_result):
        text = mantel_result.explain()
        # Bound should be close to 0.5
        assert "0.5" in text or "0.50" in text

    def test_contains_status(self, mantel_result):
        assert mantel_result.status in mantel_result.explain()

    def test_contains_bound_kind(self, mantel_result):
        text = mantel_result.explain().lower()
        assert "upper" in text or "lower" in text

    def test_with_data_shows_graphs(self, mantel_result, mantel_data):
        text = mantel_result.explain(data=mantel_data)
        assert isinstance(text, str)
        # Sharp graphs should be described with their structure
        assert "hypergraph" in text.lower() or "density" in text.lower()

    def test_without_data_shows_indices(self, mantel_result):
        text = mantel_result.explain()
        assert "indices" in text.lower() or "index" in text.lower() or "[" in text

    def test_q_matrices_mentioned(self, mantel_result):
        assert "Q" in mantel_result.explain()

    def test_no_q_says_so(self):
        prob = FlagProblem(4, 2, 2)
        result = FlagAlgebraResult(prob, "optimal", 0.5, None, [0.0, 0.1])
        assert "not extracted" in result.explain() or "extract_Q" in result.explain()


class TestFlagAlgebraResultExplainCertificate:
    def test_returns_string(self, mantel_result):
        assert isinstance(mantel_result.explain_certificate(), str)

    def test_contains_q_matrices(self, mantel_result):
        text = mantel_result.explain_certificate()
        assert "Q[0]" in text

    def test_contains_eigenvalue_info(self, mantel_result):
        text = mantel_result.explain_certificate()
        assert "eigenvalue" in text.lower()

    def test_contains_identity_description(self, mantel_result):
        text = mantel_result.explain_certificate()
        assert "P_σ" in text or "Q_σ" in text

    def test_q_sizes_listed(self, mantel_result):
        text = mantel_result.explain_certificate()
        # Mantel has 3 types with flag counts [2, 4, 3]
        assert "2\xd72" in text or "4\xd74" in text or "3\xd73" in text

    def test_raises_without_q(self):
        prob = FlagProblem(4, 2, 2)
        result = FlagAlgebraResult(prob, "optimal", 0.5, None, [])
        with pytest.raises(ValueError, match="extract_Q"):
            result.explain_certificate()


class TestMimebundleOtherTypes:
    def test_flag_mimebundle(self):
        g = Hypergraph(3, 2, [(1, 2)])
        f = Flag(g, 1)
        bundle = f._repr_mimebundle_()
        assert "text/html" in bundle and "text/plain" in bundle

    def test_problem_mimebundle(self):
        prob = FlagProblem(4, 2, 2)
        bundle = prob._repr_mimebundle_()
        assert "text/html" in bundle and "text/plain" in bundle

    def test_data_mimebundle(self, mantel_data):
        bundle = mantel_data._repr_mimebundle_()
        assert "text/html" in bundle and "text/plain" in bundle

    def test_result_mimebundle(self, mantel_result):
        bundle = mantel_result._repr_mimebundle_()
        assert "text/html" in bundle and "text/plain" in bundle

    def test_result_mimebundle_plain_is_repr(self, mantel_result):
        bundle = mantel_result._repr_mimebundle_()
        assert bundle["text/plain"] == repr(mantel_result)


class TestFlagAlgebraResultHtml:
    def test_is_html(self, mantel_result):
        assert "<" in mantel_result._repr_html_()

    def test_contains_bound(self, mantel_result):
        html = mantel_result._repr_html_()
        assert "0.5" in html or "0.50" in html

    def test_with_data(self, mantel_result, mantel_data):
        html = mantel_result._repr_html_(data=mantel_data)
        assert "<" in html
        assert "density" in html.lower()

    def test_no_q_shows_message(self):
        prob = FlagProblem(4, 2, 2)
        result = FlagAlgebraResult(prob, "optimal", 0.5, None, [])
        html = result._repr_html_()
        assert "extract_Q" in html or "not extracted" in html.lower()
