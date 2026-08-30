from __future__ import annotations

import gzip
import json
import pickle
from dataclasses import asdict
from pathlib import Path

import igraph as ig
import pytest

from evaluation.analysis.scale_summary import summarize_scale_workspace
from gos.core.schema import SkillNode


def test_scale_summary_combines_metadata_preparation_and_relink_cost(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    graph = ig.Graph(directed=True)
    nodes = [
        SkillNode.from_lists(
            name="complete",
            description="complete",
            inputs=["csv"],
            outputs=["report"],
            domain_tags=["analysis"],
        ),
        SkillNode.from_lists(name="sparse", description="sparse"),
    ]
    graph.add_vertices(2)
    for index, node in enumerate(nodes):
        graph.vs[index].update_attributes(**asdict(node))
    graph.add_edge(0, 1, type="dependency", provenance="deterministic_io")
    with gzip.open(workspace / "graph_igraph_data.pklz", "wb") as stream:
        pickle.dump(graph, stream)
    (workspace / "scale_preparation_report.json").write_text(
        json.dumps(
            {
                "label": "frozen nodes",
                "wall_seconds": 2.0,
                "embedding_usage": {
                    "embedding": {"calls": 1, "input_tokens": 10, "cost_usd": 0.1}
                },
            }
        ),
        encoding="utf-8",
    )
    (workspace / "construction_report.json").write_text(
        json.dumps(
            {
                "timing": {"wall_seconds": 5.0, "preparation_seconds": 1.5},
                "usage_totals": {
                    "calls": 3,
                    "input_tokens": 20,
                    "output_tokens": 4,
                    "cost_usd": 0.2,
                    "cache_hits": 0,
                    "failures": 1,
                },
                "construction": {
                    "validator_requests": 2,
                    "submitted_candidates": 7,
                },
                "edges": {
                    "total": 1,
                    "by_type": {"dependency": 1},
                    "by_provenance": {"deterministic_io": 1},
                },
                "configuration": {"link_top_k": 8},
                "relink": {"failed_focus": {}},
            }
        ),
        encoding="utf-8",
    )

    result = summarize_scale_workspace("fixture", workspace)

    assert result["nodes"] == 2
    assert result["io_semantic_complete_nodes"] == 1
    assert result["io_semantic_complete_fraction"] == 0.5
    assert result["validator_requests"] == 2
    assert result["submitted_candidate_utilization"] == 7 / 16
    assert result["total_calls"] == 4
    assert result["total_input_tokens"] == 30
    assert result["total_cost_usd"] == pytest.approx(0.3)
    assert result["total_wall_seconds"] == 7.0
