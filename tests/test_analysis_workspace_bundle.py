from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import hnswlib
import igraph as ig

from evaluation.analysis.workspace_bundle import (
    load_workspace_bundle,
    write_workspace_vector_store,
)


def _write_skill(root: Path, directory: str, name: str) -> None:
    path = root / directory / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nname: {name}\ndescription: {name} description\n---\n# {name}\n",
        encoding="utf-8",
    )


def _write_workspace(workspace: Path) -> None:
    workspace.mkdir(parents=True)
    graph = ig.Graph(directed=True)
    # Intentionally not in lexical file order.
    graph.add_vertices(2)
    graph.vs[0]["name"] = "Beta Skill"
    graph.vs[0]["source_path"] = "data/skillsets/skills_fixture/beta/SKILL.md"
    graph.vs[0]["description"] = "Beta description"
    graph.vs[0]["inputs"] = "alpha artifact"
    graph.vs[0]["outputs"] = "beta artifact"
    graph.vs[1]["name"] = "Alpha Skill"
    graph.vs[1]["source_path"] = "data/skillsets/skills_fixture/alpha/SKILL.md"
    graph.vs[1]["description"] = "Alpha description"
    graph.vs[1]["inputs"] = "source file"
    graph.vs[1]["outputs"] = "alpha artifact"
    graph.add_edge(
        1,
        0,
        type="dependency",
        weight=1.0,
        confidence=1.0,
        provenance="deterministic_io",
        evidence="alpha artifact",
        description="Alpha produces the artifact consumed by Beta.",
        validator_model="",
    )
    with gzip.open(workspace / "graph_igraph_data.pklz", "wb") as stream:
        pickle.dump(graph, stream)
    (workspace / "construction_report.json").write_text(
        json.dumps({"graph": {"directed": True}, "nodes": {"total": 2}}),
        encoding="utf-8",
    )
    (workspace / "relink_progress.json").write_text(
        json.dumps({"fingerprint": "sha256:fixture"}), encoding="utf-8"
    )


def test_workspace_bundle_preserves_vertex_order_direction_and_provenance(
    tmp_path: Path,
):
    skills_root = tmp_path / "skills_fixture"
    _write_skill(skills_root, "alpha", "Alpha Skill")
    _write_skill(skills_root, "beta", "Beta Skill")
    workspace = tmp_path / "workspace"
    _write_workspace(workspace)

    bundle = load_workspace_bundle(workspace, skills_root)

    assert bundle["metadata"]["graph_source"] == "persisted_directed_workspace"
    assert bundle["metadata"]["graph_fingerprint"] == "sha256:fixture"
    assert [skill["name"] for skill in bundle["skills"]] == [
        "Beta Skill",
        "Alpha Skill",
    ]
    assert [skill["graph_vertex_id"] for skill in bundle["skills"]] == [0, 1]
    assert bundle["skills"][0]["source_path"] == (
        "/opt/graphskills/skills/beta/SKILL.md"
    )
    assert bundle["skills"][0]["inputs"] == ["alpha artifact"]
    assert bundle["edges"] == [
        {
            "source": "Alpha Skill",
            "target": "Beta Skill",
            "type": "dependency",
            "weight": 1.0,
            "confidence": 1.0,
            "provenance": "deterministic_io",
            "evidence": "alpha artifact",
            "description": "Alpha produces the artifact consumed by Beta.",
            "validator_model": "",
        }
    ]


def test_workspace_bundle_reports_canonicalization_and_deduplication(tmp_path: Path):
    skills_root = tmp_path / "skills_fixture"
    _write_skill(skills_root, "alpha", "Alpha Skill")
    _write_skill(skills_root, "beta", "Beta Skill")
    workspace = tmp_path / "workspace"
    _write_workspace(workspace)
    graph_path = workspace / "graph_igraph_data.pklz"
    with gzip.open(graph_path, "rb") as stream:
        graph = pickle.load(stream)
    graph.add_edge(
        0,
        1,
        type="semantic",
        weight=0.5,
        confidence=0.8,
        provenance="llm_validated",
        evidence="shared domain",
        description="same capability",
        validator_model="model",
    )
    graph.add_edge(
        1,
        0,
        type="semantic",
        weight=0.6,
        confidence=0.9,
        provenance="llm_validated",
        evidence="stronger shared domain",
        description="same capability",
        validator_model="model",
    )
    with gzip.open(graph_path, "wb") as stream:
        pickle.dump(graph, stream)

    bundle = load_workspace_bundle(workspace, skills_root)

    assert bundle["metadata"]["raw_persisted_edge_count"] == 3
    assert bundle["metadata"]["canonicalized_undirected_edge_count"] == 1
    assert bundle["metadata"]["canonical_deduplicated_edge_count"] == 1
    assert bundle["metadata"]["edge_count"] == 2
    semantic = [edge for edge in bundle["edges"] if edge["type"] == "semantic"]
    assert semantic[0]["source"] == "Alpha Skill"
    assert semantic[0]["target"] == "Beta Skill"
    assert semantic[0]["confidence"] == 0.9


def test_vector_export_maps_hnsw_vertex_ids_to_bundle_rows(tmp_path: Path):
    skills_root = tmp_path / "skills_fixture"
    _write_skill(skills_root, "alpha", "Alpha Skill")
    _write_skill(skills_root, "beta", "Beta Skill")
    workspace = tmp_path / "workspace"
    _write_workspace(workspace)

    index = hnswlib.Index(space="cosine", dim=2)
    index.init_index(max_elements=2, ef_construction=20, M=8)
    # Vertex 0 is Beta; vertex 1 is Alpha.
    index.add_items([[1.0, 0.0], [0.0, 1.0]], [0, 1])
    index.save_index(str(workspace / "entities_hnsw_index_2.bin"))

    bundle = load_workspace_bundle(workspace, skills_root)
    # Exercise a different bundle order to prove that labels are not assumed to be
    # sorted skill-file positions.
    bundle["skills"] = [bundle["skills"][1], bundle["skills"][0]]
    output = tmp_path / "vectors.pkl"
    write_workspace_vector_store(bundle, workspace, output)

    with output.open("rb") as stream:
        payload = pickle.load(stream)
    rows = list(memoryview(payload["vectors_f32_le"]).cast("f"))
    assert payload["ids"] == [0, 1]
    assert rows == [0.0, 1.0, 1.0, 0.0]
