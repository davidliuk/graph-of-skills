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
        skill["source_path"].startswith("/opt/graphskills/library/")
        for skill in bundle["skills"]
    )


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
    )

    retrieved_names = [skill["name"] for skill in result["skills"]]
    assert "analyze_trend" in retrieved_names
    assert "render_chart" in retrieved_names
    assert len(result["rendered_context"]) <= 520
    assert result["relations"]
