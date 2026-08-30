from __future__ import annotations

from pathlib import Path

from evaluation.analysis.skillsbench_oracles import (
    audit_skillset_coverage,
    normalize_skill_name,
    render_coverage_markdown,
)


TASK_MARKDOWN = """---
schema_version: '1.3'
metadata:
  category: test
---

Build the requested artifact.
"""


def _write_skill(root: Path, directory_name: str, skill_name: str | None = None) -> None:
    skill_dir = root / directory_name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {skill_name or directory_name}\ndescription: test skill\n---\n",
        encoding="utf-8",
    )


def test_normalize_skill_name_resolves_punctuation_only_aliases():
    assert normalize_skill_name("FFmpeg Audio Processing") == normalize_skill_name(
        "ffmpeg-audio-processing"
    )
    assert normalize_skill_name("image_editing") == "imageediting"


def test_coverage_separates_exact_alias_and_absent(tmp_path: Path):
    tasks = tmp_path / "tasks"
    task = tasks / "task-a"
    (task / "environment" / "skills").mkdir(parents=True)
    (task / "task.md").write_text(TASK_MARKDOWN, encoding="utf-8")
    _write_skill(task / "environment" / "skills", "exact-skill")
    _write_skill(task / "environment" / "skills", "image-editing")
    _write_skill(task / "environment" / "skills", "missing-skill")

    library = tmp_path / "skills_200"
    _write_skill(library, "exact-skill")
    _write_skill(library, "image_editing")

    result = audit_skillset_coverage(tasks, {"skills_200": library})
    summary = result["skillsets"]["skills_200"]
    task_row = result["tasks"][0]["skillsets"]["skills_200"]

    assert summary["unique_oracle_count"] == 3
    assert summary["library_path"] == str(library.resolve())
    assert summary["library_sha256"].startswith("sha256:")
    assert summary["exact_count"] == 1
    assert summary["alias_count"] == 1
    assert summary["absent_count"] == 1
    assert task_row["exact"] == ["exact-skill"]
    assert task_row["aliases"] == {"image-editing": "image_editing"}
    assert task_row["absent"] == ["missing-skill"]
    assert task_row["available_fraction"] == 2 / 3
    assert task_row["all_available"] is False


def test_markdown_reports_attainable_task_ceiling(tmp_path: Path):
    tasks = tmp_path / "tasks"
    for task_name, oracle_name in (("covered", "present"), ("uncovered", "absent")):
        task = tasks / task_name
        (task / "environment" / "skills").mkdir(parents=True)
        (task / "task.md").write_text(TASK_MARKDOWN, encoding="utf-8")
        _write_skill(task / "environment" / "skills", oracle_name)

    library = tmp_path / "skills_200"
    _write_skill(library, "present")
    result = audit_skillset_coverage(tasks, {"skills_200": library})

    markdown = render_coverage_markdown(result)
    assert "1/2" in markdown
    assert "50.0%" in markdown
    assert "absent" in markdown
