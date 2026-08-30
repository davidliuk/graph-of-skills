from __future__ import annotations

from pathlib import Path

from evaluation.analysis.alfworld_metadata_audit import (
    audit_raw_skillset,
    render_markdown,
)


def _write_skill(root: Path, name: str, frontmatter: str, body: str = "") -> None:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\n{frontmatter}---\n{body}", encoding="utf-8"
    )


def test_raw_audit_separates_formal_fields_from_prose(tmp_path: Path):
    _write_skill(
        tmp_path,
        "alfworld-clean-object",
        "description: Cleans an object and produces a clean state.\n",
        "The input object must be held. Output is a clean object.",
    )
    _write_skill(
        tmp_path,
        "alfworld-heat-object",
        "description: Heat an object.\ninputs: [object]\noutputs: [heated object]\n"
        "domain: [alfworld]\n",
    )
    _write_skill(tmp_path, "unrelated", "description: Other.\n")

    result = audit_raw_skillset(tmp_path)

    assert result["skill_count"] == 3
    assert result["alfworld_skill_count"] == 2
    assert result["formal_inputs_count"] == 1
    assert result["formal_outputs_count"] == 1
    assert result["formal_domain_count"] == 1
    assert result["prose_io_mentions_count"] == 2


def test_markdown_explains_query_adaptation_and_metadata_layers(tmp_path: Path):
    _write_skill(
        tmp_path,
        "alfworld-clean-object",
        "description: Cleans an object.\n",
    )
    markdown = render_markdown({"skills_500": audit_raw_skillset(tmp_path)})
    assert "raw SKILL.md frontmatter" in markdown
    assert "extracted graph-node metadata" in markdown
    assert "structured retrieval query" in markdown
