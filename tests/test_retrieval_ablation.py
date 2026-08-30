from __future__ import annotations

import sys

from evaluation.analysis.run_retrieval_ablation import (
    aggregate_rows,
    apply_edge_view,
    parse_args,
    paired_comparisons,
    score_retrieval_row,
)


def test_cli_defaults_match_paper_bundle_budget(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_retrieval_ablation.py",
            "--bundle",
            "bundle.json",
            "--coverage",
            "coverage.json",
            "--skillset-name",
            "skills_200",
            "--output-dir",
            "output",
        ],
    )

    args = parse_args()

    assert args.top_n == 8
    assert args.seed_top_k == 5
    assert args.max_skill_chars == 2400
    assert args.max_context_chars == 12000


def test_deterministic_core_edge_view_drops_llm_relations_and_restores_shadowed_io():
    bundle = {
        "metadata": {"dependency_match_threshold": 0.4},
        "skills": [
            {
                "name": "producer",
                "inputs": [],
                "outputs": ["normalized seismic catalog"],
            },
            {"name": "consumer", "inputs": ["seismic catalog"], "outputs": []},
        ],
        "edges": [
            {
                "source": "producer",
                "target": "consumer",
                "type": "dependency",
                "provenance": "llm_validated",
            },
            {
                "source": "consumer",
                "target": "producer",
                "type": "semantic",
                "provenance": "llm_validated",
            },
        ],
    }

    filtered = apply_edge_view(bundle, "deterministic-core")

    assert len(filtered["edges"]) == 1
    assert filtered["edges"][0]["provenance"] == "deterministic_io"
    assert filtered["edges"][0]["source"] == "producer"


def test_score_retrieval_reports_full_and_available_oracle_metrics():
    bundle = {
        "skills": [
            {
                "name": "Alpha",
                "skill_dir": "alpha",
                "source_path": "/opt/graphskills/skills/alpha/SKILL.md",
            },
            {
                "name": "Beta",
                "skill_dir": "beta_alias",
                "source_path": "/opt/graphskills/skills/beta_alias/SKILL.md",
            },
            {
                "name": "Distractor",
                "skill_dir": "distractor",
                "source_path": "/opt/graphskills/skills/distractor/SKILL.md",
            },
        ],
        "edges": [{"source": "Alpha", "target": "Beta", "type": "dependency"}],
    }
    coverage = {
        "oracle_count": 3,
        "exact": ["alpha"],
        "aliases": {"beta": "beta_alias"},
        "absent": ["missing"],
        "available_count": 2,
        "available_fraction": 2 / 3,
        "all_available": False,
    }
    retrieval = {
        "skills": [
            {"source_path": "/opt/graphskills/skills/alpha/SKILL.md"},
            {"source_path": "/opt/graphskills/skills/distractor/SKILL.md"},
        ],
        "rendered_context": "abcd",
    }

    row = score_retrieval_row(
        task_id="task-a",
        condition="reverse-ppr",
        coverage=coverage,
        retrieval=retrieval,
        bundle=bundle,
        latency_ms=2.5,
    )

    assert row["oracle_hit_count"] == 1
    assert row["full_oracle_recall"] == 1 / 3
    assert row["available_oracle_recall"] == 1 / 2
    assert row["bundle_precision"] == 1 / 2
    assert row["full_bundle_complete"] is False
    assert row["available_bundle_complete"] is False
    assert row["oracle_dependency_pair_count"] == 1
    assert row["recovered_oracle_dependency_pair_count"] == 0
    assert row["rendered_context_chars"] == 4


def test_aggregate_rows_reports_macro_and_latency_percentiles():
    rows = [
        {
            "condition": "reverse-ppr",
            "available_oracle_count": 1,
            "full_oracle_recall": 0.5,
            "available_oracle_recall": 1.0,
            "bundle_precision": 0.4,
            "full_bundle_complete": False,
            "available_bundle_complete": True,
            "oracle_dependency_pair_count": 1,
            "recovered_oracle_dependency_pair_count": 1,
            "retrieved_skill_count": 5,
            "rendered_context_chars": 100,
            "latency_ms": 1.0,
        },
        {
            "condition": "reverse-ppr",
            "available_oracle_count": 1,
            "full_oracle_recall": 0.0,
            "available_oracle_recall": 0.0,
            "bundle_precision": 0.0,
            "full_bundle_complete": False,
            "available_bundle_complete": False,
            "oracle_dependency_pair_count": 1,
            "recovered_oracle_dependency_pair_count": 0,
            "retrieved_skill_count": 4,
            "rendered_context_chars": 80,
            "latency_ms": 3.0,
        },
    ]

    aggregate = aggregate_rows(rows)["reverse-ppr"]

    assert aggregate["task_count"] == 2
    assert aggregate["macro_full_oracle_recall"] == 0.25
    assert aggregate["macro_available_oracle_recall"] == 0.5
    assert aggregate["available_bundle_complete_rate"] == 0.5
    assert aggregate["oracle_dependency_pair_recovery"] == 0.5
    assert aggregate["latency_ms_p50"] == 2.0
    assert aggregate["latency_ms_p95"] == 2.9


def test_aggregate_rows_excludes_empty_available_oracle_tasks_from_conditioned_metrics():
    rows = [
        {
            "condition": "reverse-ppr",
            "available_oracle_count": 2,
            "full_oracle_recall": 0.5,
            "available_oracle_recall": 0.5,
            "bundle_precision": 0.4,
            "full_bundle_complete": False,
            "available_bundle_complete": False,
            "oracle_dependency_pair_count": 1,
            "recovered_oracle_dependency_pair_count": 1,
            "retrieved_skill_count": 5,
            "rendered_context_chars": 100,
            "latency_ms": 1.0,
        },
        {
            "condition": "reverse-ppr",
            "available_oracle_count": 0,
            "full_oracle_recall": 0.0,
            "available_oracle_recall": 1.0,
            "bundle_precision": 0.0,
            "full_bundle_complete": False,
            "available_bundle_complete": True,
            "oracle_dependency_pair_count": 0,
            "recovered_oracle_dependency_pair_count": 0,
            "retrieved_skill_count": 5,
            "rendered_context_chars": 100,
            "latency_ms": 1.0,
        },
    ]

    aggregate = aggregate_rows(rows)["reverse-ppr"]

    assert aggregate["task_count"] == 2
    assert aggregate["available_task_count"] == 1
    assert aggregate["macro_full_oracle_recall"] == 0.25
    assert aggregate["macro_available_oracle_recall"] == 0.5
    assert aggregate["macro_bundle_precision"] == 0.4
    assert aggregate["available_bundle_complete_rate"] == 0.0


def test_paired_comparison_reports_wins_ties_losses_and_mean_delta():
    rows = [
        {
            "task_id": "a",
            "condition": "reverse-ppr",
            "available_oracle_count": 1,
            "available_oracle_recall": 1.0,
        },
        {
            "task_id": "b",
            "condition": "reverse-ppr",
            "available_oracle_count": 1,
            "available_oracle_recall": 0.5,
        },
        {
            "task_id": "c",
            "condition": "reverse-ppr",
            "available_oracle_count": 1,
            "available_oracle_recall": 0.0,
        },
        {
            "task_id": "a",
            "condition": "no-graph",
            "available_oracle_count": 1,
            "available_oracle_recall": 0.5,
        },
        {
            "task_id": "b",
            "condition": "no-graph",
            "available_oracle_count": 1,
            "available_oracle_recall": 0.5,
        },
        {
            "task_id": "c",
            "condition": "no-graph",
            "available_oracle_count": 1,
            "available_oracle_recall": 0.5,
        },
    ]

    comparison = paired_comparisons(
        rows,
        reference="reverse-ppr",
        metrics=["available_oracle_recall"],
        bootstrap_samples=100,
    )["reverse-ppr-minus-no-graph"]["available_oracle_recall"]

    assert comparison["task_count"] == 3
    assert comparison["wins"] == 1
    assert comparison["ties"] == 1
    assert comparison["losses"] == 1
    assert comparison["mean_delta"] == 0.0


def test_paired_comparison_excludes_empty_available_oracle_tasks():
    rows = [
        {
            "task_id": "a",
            "condition": "reverse-ppr",
            "available_oracle_count": 1,
            "available_oracle_recall": 1.0,
        },
        {
            "task_id": "a",
            "condition": "no-graph",
            "available_oracle_count": 1,
            "available_oracle_recall": 0.5,
        },
        {
            "task_id": "empty",
            "condition": "reverse-ppr",
            "available_oracle_count": 0,
            "available_oracle_recall": 1.0,
        },
        {
            "task_id": "empty",
            "condition": "no-graph",
            "available_oracle_count": 0,
            "available_oracle_recall": 1.0,
        },
    ]

    comparison = paired_comparisons(
        rows,
        reference="reverse-ppr",
        metrics=["available_oracle_recall"],
        bootstrap_samples=100,
    )["reverse-ppr-minus-no-graph"]["available_oracle_recall"]

    assert comparison["task_count"] == 1
    assert comparison["mean_delta"] == 0.5
