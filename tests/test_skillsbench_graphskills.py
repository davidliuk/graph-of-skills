from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATOR_PATH = REPO_ROOT / "evaluation" / "skillsbench" / "graphskills_benchmark.py"
RUNTIME_PATH = (
    REPO_ROOT / "evaluation" / "skillsbench" / "graphskills_assets" / "query.py"
)


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_skill(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_graphskills_bundle_builds_dependency_edges(tmp_path):
    generator = _load_module("skillsbench_graph_builder", GENERATOR_PATH)

    skills_root = tmp_path / "all_skills"
    _write_skill(
        skills_root / "read_csv" / "SKILL.md",
        """---
name: read_csv
description: Read a CSV file into a dataset.
inputs:
  - csv_path
outputs:
  - dataset
---
# Usage
Load a CSV file.
""",
    )
    _write_skill(
        skills_root / "analyze_trend" / "SKILL.md",
        """---
name: analyze_trend
description: Analyze a dataset and produce a trend report.
inputs:
  - dataset
outputs:
  - trend_report
---
# Usage
Analyze a dataset.
""",
    )
    _write_skill(
        skills_root / "render_chart" / "SKILL.md",
        """---
name: render_chart
description: Render a chart from a trend report.
inputs:
  - trend_report
outputs:
  - chart
---
# Usage
Render a chart.
""",
    )

    bundle = generator.build_graph_bundle(skills_root)

    assert bundle["metadata"]["skill_count"] == 3
    skill_names = {skill["name"] for skill in bundle["skills"]}
    assert skill_names == {"read_csv", "analyze_trend", "render_chart"}
    assert any(
        edge["source"] == "read_csv" and edge["target"] == "analyze_trend"
        for edge in bundle["edges"]
    )
    assert any(
        edge["source"] == "analyze_trend" and edge["target"] == "render_chart"
        for edge in bundle["edges"]
    )
    assert all(
        skill["source_path"].startswith("/opt/graphskills/skills/")
        for skill in bundle["skills"]
    )


def test_generator_prefers_persisted_repaired_workspace(monkeypatch, tmp_path):
    generator = _load_module("skillsbench_graph_builder_persisted", GENERATOR_PATH)
    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "construction_report.json").write_text(
        '{"graph":{"directed":true},"nodes":{"total":0}}', encoding="utf-8"
    )
    expected = {
        "metadata": {"graph_source": "persisted_directed_workspace"},
        "skills": [],
        "edges": [],
    }
    calls = []

    def fake_loader(received_workspace, received_skills_root):
        calls.append((received_workspace, received_skills_root))
        return expected

    monkeypatch.setattr(generator, "load_workspace_bundle", fake_loader)

    bundle = generator.load_generation_bundle(skills_root, workspace)

    assert bundle is expected
    assert calls == [(workspace, skills_root)]


def test_graphskills_runtime_retrieval_respects_context_budget(tmp_path):
    generator = _load_module("skillsbench_graph_builder_runtime", GENERATOR_PATH)
    runtime = _load_module("skillsbench_graph_runtime", RUNTIME_PATH)

    skills_root = tmp_path / "all_skills"
    _write_skill(
        skills_root / "read_csv" / "SKILL.md",
        """---
name: read_csv
description: Read a CSV file into a dataset.
inputs:
  - csv_path
outputs:
  - dataset
---
# Usage
Load a CSV file and return a dataset object.
""",
    )
    _write_skill(
        skills_root / "analyze_trend" / "SKILL.md",
        """---
name: analyze_trend
description: Analyze a dataset and produce a trend report.
inputs:
  - dataset
outputs:
  - trend_report
---
# Usage
Compute summary statistics over a dataset.
""",
    )
    _write_skill(
        skills_root / "render_chart" / "SKILL.md",
        """---
name: render_chart
description: Render a chart from a trend report.
inputs:
  - trend_report
outputs:
  - chart
---
# Usage
Render a chart image from a trend report.
""",
    )

    bundle = generator.build_graph_bundle(skills_root)
    result = runtime.retrieve(
        bundle,
        "Analyze sales csv trends and make a chart",
        top_n=3,
        seed_top_k=3,
        max_skill_chars=160,
        max_context_chars=520,
        seed_mode="lexical",
        propagation_mode="ppr",
        vector_store_path=None,
    )

    retrieved_names = [skill["name"] for skill in result["skills"]]
    assert "analyze_trend" in retrieved_names
    assert "render_chart" in retrieved_names
    assert len(result["rendered_context"]) <= 520
    assert result["relations"]


def test_graphskills_forward_only_transition_omits_reverse_edges():
    runtime = _load_module("skillsbench_graph_runtime_forward_only", RUNTIME_PATH)
    skills = [{"name": "producer"}, {"name": "consumer"}]
    edges = [
        {
            "source": "producer",
            "target": "consumer",
            "type": "dependency",
            "weight": 1.0,
        }
    ]

    full = runtime.build_transition(skills, edges, reverse_mode="full")
    forward_only = runtime.build_transition(skills, edges, reverse_mode="none")

    assert full[1] == {0: 1.0}
    assert forward_only[1] == {1: 1.0}
    assert forward_only[0] == {1: 1.0}


def test_graphskills_one_hop_adds_incoming_dependency_prerequisite():
    runtime = _load_module("skillsbench_graph_runtime_one_hop", RUNTIME_PATH)
    skills = [
        {
            "name": "read_csv",
            "description": "Read CSV data.",
            "source_path": "/skills/read_csv/SKILL.md",
            "raw_content": "Read CSV data.",
        },
        {
            "name": "analyze_trend",
            "description": "Analyze a dataset.",
            "source_path": "/skills/analyze_trend/SKILL.md",
            "raw_content": "Analyze a dataset.",
        },
    ]
    edges = [
        {
            "source": "read_csv",
            "target": "analyze_trend",
            "type": "dependency",
            "weight": 1.0,
        }
    ]

    ranked = runtime.rank_one_hop(
        skills,
        edges,
        seed_entries=[(1, 1.0, 1)],
        max_skill_chars=200,
    )

    assert [skill["name"] for skill in ranked] == ["analyze_trend", "read_csv"]
    assert ranked[1]["semantic_rank"] is None


def test_no_graph_fills_matched_top_n_beyond_ppr_seed_count():
    runtime = _load_module("skillsbench_graph_runtime_matched_none", RUNTIME_PATH)
    skills = [
        {
            "name": f"skill-{index}",
            "description": f"shared query capability {index}",
            "source_path": f"/skills/skill-{index}/SKILL.md",
            "raw_content": f"shared query capability {index}",
        }
        for index in range(6)
    ]
    bundle = {"metadata": {}, "skills": skills, "edges": []}

    result = runtime.retrieve(
        bundle,
        "shared query capability",
        top_n=5,
        seed_top_k=4,
        max_skill_chars=100,
        max_context_chars=5000,
        seed_mode="lexical",
        propagation_mode="none",
        reverse_mode="full",
    )

    assert len(result["skills"]) == 5
    assert result["budget"]["seed_top_k"] == 4


def test_build_task_list_accepts_current_task_markdown_schema(tmp_path):
    generator = _load_module("skillsbench_graph_builder_task_md", GENERATOR_PATH)
    task = tmp_path / "dialogue-parser"
    task.mkdir()
    (task / "task.md").write_text("---\nschema_version: '1.3'\n---\nSolve it.\n")

    tasks = generator.build_task_list(tmp_path, ["dialogue-parser"])

    assert tasks == [task]


def test_normalize_task_layout_converts_current_skillsbench_schema(tmp_path):
    generator = _load_module("skillsbench_graph_builder_normalize", GENERATOR_PATH)
    task = tmp_path / "dialogue-parser"
    (task / "verifier").mkdir(parents=True)
    (task / "oracle").mkdir()
    (task / "task.md").write_text(
        """---
schema_version: '1.3'
metadata:
  author_name: reviewer
  difficulty: easy
  category: software-engineering
  tags: [parsing, json]
verifier:
  timeout_sec: 900.0
agent:
  timeout_sec: 800.0
environment:
  build_timeout_sec: 600.0
  cpus: 2
  memory_mb: 4096
  storage_mb: 10240
  gpus: 0
  network_mode: public
---

Implement the parser.
""",
        encoding="utf-8",
    )
    (task / "verifier" / "test.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (task / "oracle" / "solve.sh").write_text("#!/bin/sh\n", encoding="utf-8")

    generator.normalize_task_layout(task)

    assert (task / "instruction.md").read_text(
        encoding="utf-8"
    ) == "Implement the parser.\n"
    task_toml = (task / "task.toml").read_text(encoding="utf-8")
    assert 'author_name = "reviewer"' in task_toml
    assert "timeout_sec = 900.0" in task_toml
    assert "memory_mb = 4096" in task_toml
    assert "allow_internet = true" in task_toml
    assert (task / "tests" / "test.sh").exists()
    assert (task / "solution" / "solve.sh").exists()


def test_graphskills_compose_includes_harbor_runtime_mounts():
    generator = _load_module("skillsbench_graph_builder_compose", GENERATOR_PATH)
    compose = generator.GOS_TEMPLATE_DIR.joinpath("docker-compose.yaml").read_text(
        encoding="utf-8"
    )

    assert "context: ${CONTEXT_DIR}" in compose
    assert "image: ${MAIN_IMAGE_NAME}" in compose
    assert "${HOST_AGENT_LOGS_PATH}:${ENV_AGENT_LOGS_PATH}" in compose
    assert "${HOST_VERIFIER_LOGS_PATH}:${ENV_VERIFIER_LOGS_PATH}" in compose
