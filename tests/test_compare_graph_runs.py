from __future__ import annotations

from evaluation.analysis.compare_graph_runs import compare_bundles


def _bundle(edges):
    return {
        "metadata": {},
        "skills": [{"name": "a"}, {"name": "b"}, {"name": "c"}],
        "edges": edges,
    }


def _edge(source, target, edge_type, provenance):
    return {
        "source": source,
        "target": target,
        "type": edge_type,
        "provenance": provenance,
    }


def test_compare_bundles_separates_deterministic_and_llm_stability():
    deterministic = _edge("a", "b", "dependency", "deterministic_io")
    first = _bundle([deterministic, _edge("b", "c", "semantic", "llm_validated")])
    second = _bundle([deterministic, _edge("c", "b", "semantic", "llm_validated")])

    result = compare_bundles({"run-a": first, "run-b": second})
    pair = result["pairwise"][0]

    assert pair["left"] == "run-a"
    assert pair["right"] == "run-b"
    assert pair["typed_directed_jaccard"] == 1 / 3
    assert pair["deterministic_typed_directed_jaccard"] == 1.0
    assert pair["deterministic_candidate_jaccard"] == 1.0
    assert pair["llm_typed_directed_jaccard"] == 0.0
    assert pair["reversed_same_type_count"] == 1
    assert result["consensus"]["present_in_all_runs"] == 1
    assert result["consensus"]["present_in_one_run"] == 2


def test_compare_bundles_reports_normalized_node_field_stability():
    first = _bundle([])
    second = _bundle([])
    first["skills"][0]["inputs"] = ["alpha", "beta"]
    second["skills"][0]["inputs"] = ["alpha", "gamma"]

    result = compare_bundles({"run-a": first, "run-b": second})
    inputs = result["node_normalization"]["fields"]["inputs"]

    assert result["node_normalization"]["common_node_count"] == 3
    assert inputs["pairwise_mean_jaccard"] == 7 / 9
    assert inputs["pairwise_exact_fraction"] == 2 / 3
    assert inputs["all_runs_exact_count"] == 2
