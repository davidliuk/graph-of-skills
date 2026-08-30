from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml

from .manifest import atomic_write_json, sha256_tree
from .workspace_bundle import load_workspace_bundle


_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.DOTALL)


def _frontmatter(content: str) -> dict[str, Any]:
    match = _FRONTMATTER.match(content)
    if not match:
        return {}
    payload = yaml.safe_load(match.group(1)) or {}
    return payload if isinstance(payload, dict) else {}


def _nonempty_listish(value: Any) -> bool:
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return bool(str(value or "").strip())


def audit_raw_skillset(root: Path) -> dict[str, Any]:
    root = root.resolve()
    skill_files = sorted(root.glob("*/SKILL.md"))
    rows: list[dict[str, Any]] = []
    for path in skill_files:
        if not path.parent.name.casefold().startswith("alfworld-"):
            continue
        content = path.read_text(encoding="utf-8", errors="ignore")
        metadata = _frontmatter(content)
        inputs = metadata.get("inputs", metadata.get("input_schema"))
        outputs = metadata.get("outputs", metadata.get("output_schema"))
        domain = metadata.get("domain", metadata.get("domain_tags"))
        rows.append(
            {
                "name": path.parent.name,
                "formal_inputs": _nonempty_listish(inputs),
                "formal_outputs": _nonempty_listish(outputs),
                "formal_domain": _nonempty_listish(domain),
                "description": bool(str(metadata.get("description", "")).strip()),
                "prose_io_mentions": bool(
                    re.search(
                        r"\b(inputs?|outputs?|prerequisites?|post-conditions?|results?)\b",
                        content,
                        re.I,
                    )
                ),
            }
        )
    return {
        "root": str(root),
        "sha256": sha256_tree(root),
        "skill_count": len(skill_files),
        "alfworld_skill_count": len(rows),
        "formal_inputs_count": sum(row["formal_inputs"] for row in rows),
        "formal_outputs_count": sum(row["formal_outputs"] for row in rows),
        "formal_domain_count": sum(row["formal_domain"] for row in rows),
        "description_count": sum(row["description"] for row in rows),
        "prose_io_mentions_count": sum(row["prose_io_mentions"] for row in rows),
        "skills": rows,
    }


def audit_workspace(workspace: Path, skillset: Path) -> dict[str, Any]:
    bundle = load_workspace_bundle(workspace, skillset)
    skills = [
        skill
        for skill in bundle["skills"]
        if str(skill["name"]).casefold().startswith("alfworld-")
    ]
    names = {str(skill["name"]) for skill in skills}
    incident_edges = [
        edge
        for edge in bundle["edges"]
        if edge["source"] in names or edge["target"] in names
    ]
    internal_edges = [
        edge
        for edge in incident_edges
        if edge["source"] in names and edge["target"] in names
    ]
    return {
        "workspace": str(workspace.resolve()),
        "alfworld_node_count": len(skills),
        "inputs_count": sum(bool(skill["inputs"]) for skill in skills),
        "outputs_count": sum(bool(skill["outputs"]) for skill in skills),
        "domain_tags_count": sum(bool(skill["domain_tags"]) for skill in skills),
        "description_count": sum(bool(skill["description"]) for skill in skills),
        "inputs_and_outputs_count": sum(
            bool(skill["inputs"] and skill["outputs"]) for skill in skills
        ),
        "incident_edge_count": len(incident_edges),
        "internal_edge_count": len(internal_edges),
    }


def render_markdown(
    raw: dict[str, dict[str, Any]],
    workspaces: dict[str, dict[str, Any]] | None = None,
) -> str:
    lines = [
        "# ALFWorld Skill Metadata Audit",
        "",
        "The audit distinguishes raw SKILL.md frontmatter from extracted graph-node metadata; prose descriptions of pre/postconditions are not counted as formal schema fields.",
        "",
        "| Skillset | Skills | ALFWorld | Raw inputs | Raw outputs | Raw domain | Prose I/O mentions |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in sorted(raw.items()):
        lines.append(
            f"| {name} | {result['skill_count']} | {result['alfworld_skill_count']} | "
            f"{result['formal_inputs_count']} | {result['formal_outputs_count']} | "
            f"{result['formal_domain_count']} | {result['prose_io_mentions_count']} |"
        )

    if workspaces:
        lines.extend(
            [
                "",
                "## Extracted graph-node metadata",
                "",
                "| Skillset | ALFWorld nodes | Inputs | Outputs | Domain tags | Both I/O | Incident edges | Internal edges |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name, result in sorted(workspaces.items()):
            lines.append(
                f"| {name} | {result['alfworld_node_count']} | {result['inputs_count']} | "
                f"{result['outputs_count']} | {result['domain_tags_count']} | "
                f"{result['inputs_and_outputs_count']} | {result['incident_edge_count']} | "
                f"{result['internal_edge_count']} |"
            )

    lines.extend(
        [
            "",
            "## Retrieval adaptation",
            "",
            "ALFWorld does not rely on the raw instruction alone. The evaluator converts each goal into a structured retrieval query with task type, ALFWorld/household domains, operations, object/receptacle/device artifacts, required state, count, and an action-sequence constraint. This query adaptation mitigates missing query-side schemas, but it does not repair missing skill-side metadata; the latter is measured separately above and by the metadata stress test.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ALFWorld skill metadata")
    parser.add_argument("--skillset", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--workspace", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    skillsets = dict(item.split("=", 1) for item in args.skillset)
    workspace_paths = dict(item.split("=", 1) for item in args.workspace)
    raw = {name: audit_raw_skillset(Path(path)) for name, path in skillsets.items()}
    workspaces = {
        name: audit_workspace(Path(path), Path(skillsets[name]))
        for name, path in workspace_paths.items()
    }
    payload = {"schema_version": 1, "raw_skillsets": raw, "workspaces": workspaces}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / "summary.json", payload)
    (args.output_dir / "results.md").write_text(
        render_markdown(raw, workspaces), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
