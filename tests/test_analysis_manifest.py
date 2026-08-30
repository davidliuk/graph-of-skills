from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.analysis.manifest import (
    ExperimentManifest,
    atomic_write_json,
    sha256_tree,
    validate_workspace_report,
)


def test_sha256_tree_is_content_and_path_stable(tmp_path: Path):
    root = tmp_path / "corpus"
    (root / "b").mkdir(parents=True)
    (root / "a.txt").write_text("alpha", encoding="utf-8")
    (root / "b" / "c.txt").write_text("gamma", encoding="utf-8")

    first = sha256_tree(root)
    second = sha256_tree(root)
    assert first == second
    assert first.startswith("sha256:")

    (root / "b" / "c.txt").write_text("changed", encoding="utf-8")
    assert sha256_tree(root) != first


def test_atomic_write_json_replaces_complete_payload(tmp_path: Path):
    output = tmp_path / "manifest.json"
    atomic_write_json(output, {"run_id": "first"})
    atomic_write_json(output, {"run_id": "second", "complete": True})

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "complete": True,
        "run_id": "second",
    }
    assert not list(tmp_path.glob(".*.tmp"))


def test_workspace_report_must_be_directed_and_match_skill_count(tmp_path: Path):
    report = tmp_path / "construction_report.json"
    report.write_text(
        json.dumps(
            {
                "graph": {"directed": False},
                "nodes": {"count": 2},
                "relink": {"fingerprint": "sha256:abc"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="directed"):
        validate_workspace_report(report.parent, expected_skill_count=2)

    report.write_text(
        json.dumps(
            {
                "graph": {"directed": True},
                "nodes": {"count": 2},
                "relink": {"fingerprint": "sha256:abc"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="skill count"):
        validate_workspace_report(report.parent, expected_skill_count=3)

    validated = validate_workspace_report(report.parent, expected_skill_count=2)
    assert validated["relink"]["fingerprint"] == "sha256:abc"


def test_manifest_serialization_excludes_secret_values():
    manifest = ExperimentManifest(
        run_id="coverage-test",
        experiment="coverage",
        corpus_path="/tmp/skills",
        corpus_sha256="sha256:corpus",
        task_path="/tmp/tasks",
        task_sha256="sha256:tasks",
        git_commit="abc123",
        dirty_paths=["tests/example.py"],
        configuration={"model": "minimax/minimax-m2.7"},
    )

    payload = manifest.to_dict()
    assert payload["schema_version"] == 1
    assert "api_key" not in json.dumps(payload).lower()

