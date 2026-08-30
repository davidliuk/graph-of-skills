from __future__ import annotations

import sys

from evaluation.analysis.deterministic_edges import recompute_deterministic_edges
from evaluation.analysis.graph_diagnostics import (
    graph_statistics,
    parse_args,
    stratified_edge_sample,
    threshold_analysis,
    write_annotation_packet,
)


def _bundle():
    return {
        "metadata": {},
        "skills": [
            {
                "name": "producer",
                "inputs": [],
                "outputs": ["normalized seismic catalog"],
                "domain_tags": ["seismology"],
            },
            {
                "name": "consumer",
                "inputs": ["raw catalog"],
                "outputs": [],
                "domain_tags": ["seismology"],
            },
            {
                "name": "isolated",
                "inputs": [],
                "outputs": [],
                "domain_tags": [],
            },
        ],
        "edges": [
            {
                "source": "producer",
                "target": "consumer",
                "type": "dependency",
                "provenance": "deterministic_io",
                "weight": 0.5,
                "confidence": 0.5,
                "description": "producer produces a catalog",
                "evidence": "catalog",
                "validator_model": "",
            },
            {
                "source": "consumer",
                "target": "producer",
                "type": "semantic",
                "provenance": "llm_validated",
                "weight": 0.3,
                "confidence": 0.8,
                "description": "related catalog skills",
                "evidence": "catalog",
                "validator_model": "model",
            },
        ],
    }


def test_cli_defaults_match_main_retrieval_budget(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "graph_diagnostics.py",
            "--bundle",
            "bundle.json",
            "--construction-report",
            "construction.json",
            "--output-dir",
            "output",
        ],
    )

    args = parse_args()

    assert args.top_n == 8
    assert args.seed_top_k == 5
    assert args.max_skill_chars == 2400
    assert args.max_context_chars == 12000


def test_recompute_deterministic_edges_respects_zeta():
    low = recompute_deterministic_edges(_bundle()["skills"], threshold=0.4)
    high = recompute_deterministic_edges(_bundle()["skills"], threshold=0.6)

    assert [(edge["source"], edge["target"]) for edge in low] == [
        ("producer", "consumer")
    ]
    assert low[0]["weight"] == 0.5
    assert high == []


def test_recompute_uses_conservative_common_format_gate():
    skills = [
        {
            "name": "pcap-analysis",
            "description": "Analyze network captures.",
            "inputs": [],
            "outputs": ["network statistics CSV report"],
            "domain_tags": ["cybersecurity"],
        },
        {
            "name": "flight-search",
            "description": "Search route data.",
            "inputs": ["airport routes CSV file"],
            "outputs": [],
            "domain_tags": ["travel"],
        },
    ]

    assert recompute_deterministic_edges(skills, threshold=0.6) == []


def test_graph_statistics_reports_directed_integrity_and_isolates():
    stats = graph_statistics(_bundle())

    assert stats["nodes"] == 3
    assert stats["edges"] == 2
    assert stats["directed"] is True
    assert stats["isolates"] == 1
    assert stats["weak_components"] == 2
    assert stats["duplicate_typed_directed"] == 0
    assert stats["by_type"] == {"dependency": 1, "semantic": 1}
    assert stats["by_provenance"] == {
        "deterministic_io": 1,
        "llm_validated": 1,
    }


def test_stratified_sample_is_reproducible_and_balanced():
    edges = _bundle()["edges"]
    first = stratified_edge_sample(edges, per_stratum=1, salt="fixed")
    second = stratified_edge_sample(list(reversed(edges)), per_stratum=1, salt="fixed")

    assert first == second
    assert len(first) == 2
    assert {row["stratum"] for row in first} == {
        "deterministic_io/dependency",
        "llm_validated/semantic",
    }


def test_threshold_analysis_distinguishes_llm_shadowing_from_missing_edges():
    bundle = _bundle()
    bundle["edges"][0] = {
        **bundle["edges"][0],
        "provenance": "llm_validated",
        "confidence": 0.9,
    }

    rows, _ = threshold_analysis(bundle, [0.4])
    row = rows["0.400"]

    assert row["llm_shadowed_deterministic_count"] == 1
    assert row["missing_from_persisted_dependencies"] == 0
    assert row["coverage_by_any_persisted_dependency"] == 1.0


def test_annotation_packet_includes_blind_labeling_instructions(tmp_path):
    sample = stratified_edge_sample(_bundle()["edges"], per_stratum=1, salt="fixed")

    write_annotation_packet(_bundle(), sample, tmp_path)

    instructions = (tmp_path / "annotation_instructions.md").read_text(encoding="utf-8")
    assert "without opening `edge_annotation_key.json`" in instructions
    assert "valid_relation" in instructions
    assert "uncertain" in instructions
    assert "aggregate_edge_annotations" in instructions
