from __future__ import annotations

from evaluation.analysis.aggregate_retrieval_runs import aggregate_retrieval_runs


def _row(task, condition, recall):
    return {
        "task_id": task,
        "condition": condition,
        "available_oracle_count": 1,
        "full_oracle_recall": recall,
        "available_oracle_recall": recall,
        "bundle_precision": recall,
        "full_bundle_complete": recall == 1.0,
        "available_bundle_complete": recall == 1.0,
        "oracle_dependency_pair_count": 1,
        "recovered_oracle_dependency_pair_count": int(recall == 1.0),
        "retrieved_skill_count": 5,
        "rendered_context_chars": 100,
        "latency_ms": 1.0,
    }


def test_aggregate_retrieval_runs_keeps_graph_replicates_as_paired_units():
    runs = {
        "B": [_row("task", "reverse-ppr", 1.0), _row("task", "no-graph", 0.0)],
        "C": [_row("task", "reverse-ppr", 0.0), _row("task", "no-graph", 0.0)],
    }

    result = aggregate_retrieval_runs(runs)

    assert result["run_count"] == 2
    assert (
        result["pooled_aggregates"]["reverse-ppr"]["macro_available_oracle_recall"]
        == 0.5
    )
    comparison = result["pooled_paired_comparisons"]["reverse-ppr-minus-no-graph"][
        "available_oracle_recall"
    ]
    assert comparison["task_count"] == 2
    assert comparison["wins"] == 1
    assert comparison["ties"] == 1
    task_averaged = result["task_averaged_paired_comparisons"][
        "reverse-ppr-minus-no-graph"
    ]["available_oracle_recall"]
    assert task_averaged["task_count"] == 1
    assert task_averaged["mean_delta"] == 0.5
    assert task_averaged["wins"] == 1
