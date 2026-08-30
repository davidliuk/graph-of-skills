import igraph as ig
import pytest

from scripts.validate_graph_construction import analyze_workspace


def _write_graph(path, *, directed):
    path.mkdir(parents=True)
    graph = ig.Graph(directed=directed)
    graph.add_vertices(["producer", "consumer"])
    graph.add_edge(
        "producer",
        "consumer",
        description="producer emits catalog consumed by consumer",
        type="dependency",
        weight=1.0,
        confidence=1.0,
        provenance="deterministic_io",
        evidence="catalog",
        validator_model="",
        chunks=[],
    )
    ig.Graph.write_picklez(graph, str(path / "graph_igraph_data.pklz"))


def test_validation_report_accepts_typed_directed_graph(tmp_path):
    workspace = tmp_path / "workspace"
    _write_graph(workspace, directed=True)

    report = analyze_workspace(workspace)

    assert report["integrity"]["valid"] is True
    assert report["graph"]["directed"] is True
    assert report["edges"]["by_type"] == {"dependency": 1}
    assert report["edges"]["duplicate_typed_directed"] == 0


def test_validation_report_rejects_undirected_graph(tmp_path):
    workspace = tmp_path / "workspace"
    _write_graph(workspace, directed=False)

    with pytest.raises(ValueError, match="directed"):
        analyze_workspace(workspace)


def test_prerequisite_audit_ignores_skills_outside_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    _write_graph(workspace, directed=True)
    skillset = tmp_path / "skillset"
    for name, body in {
        "producer": "Producer skill.",
        "consumer": "Consumer skill.",
        "outsider": "This skill requires producer as a prerequisite.",
    }.items():
        path = skillset / name / "SKILL.md"
        path.parent.mkdir(parents=True)
        path.write_text(
            f"---\nname: {name}\ndescription: {body}\n---\n{body}\n",
            encoding="utf-8",
        )

    report = analyze_workspace(workspace, skillset)

    assert report["explicit_prerequisites"]["mentions"] == 0
