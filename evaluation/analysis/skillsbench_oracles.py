from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from .manifest import atomic_write_json, sha256_tree


def normalize_skill_name(name: str) -> str:
    """Normalize only case and punctuation; do not infer semantic equivalence."""
    return re.sub(r"[^a-z0-9]+", "", name.casefold())


def _skill_directory_names(root: Path, *, immediate: bool = False) -> list[str]:
    if immediate:
        candidates = sorted(root.glob("*/SKILL.md"))
    else:
        candidates = sorted(root.rglob("SKILL.md"))
    return sorted({path.parent.name for path in candidates})


def _task_instruction(task_dir: Path) -> str:
    task_markdown = task_dir / "task.md"
    if not task_markdown.is_file():
        instruction = task_dir / "instruction.md"
        return instruction.read_text(encoding="utf-8") if instruction.is_file() else ""
    content = task_markdown.read_text(encoding="utf-8")
    match = re.match(r"\A---\s*\n.*?\n---\s*\n(.*)\Z", content, re.DOTALL)
    return (match.group(1) if match else content).lstrip("\n")


def _resolve_oracles(
    oracle_names: list[str],
    library_names: list[str],
) -> dict[str, Any]:
    exact_library = set(library_names)
    normalized_library: dict[str, list[str]] = {}
    for name in library_names:
        normalized_library.setdefault(normalize_skill_name(name), []).append(name)

    exact: list[str] = []
    aliases: dict[str, str] = {}
    absent: list[str] = []
    ambiguous_aliases: dict[str, list[str]] = {}
    for oracle in oracle_names:
        if oracle in exact_library:
            exact.append(oracle)
            continue
        candidates = normalized_library.get(normalize_skill_name(oracle), [])
        if len(candidates) == 1:
            aliases[oracle] = candidates[0]
        elif len(candidates) > 1:
            ambiguous_aliases[oracle] = sorted(candidates)
            absent.append(oracle)
        else:
            absent.append(oracle)

    available_count = len(exact) + len(aliases)
    oracle_count = len(oracle_names)
    return {
        "oracle_count": oracle_count,
        "exact": sorted(exact),
        "aliases": dict(sorted(aliases.items())),
        "ambiguous_aliases": dict(sorted(ambiguous_aliases.items())),
        "absent": sorted(absent),
        "available_count": available_count,
        "available_fraction": available_count / oracle_count if oracle_count else 1.0,
        "all_available": available_count == oracle_count,
    }


def audit_skillset_coverage(
    tasks_root: Path,
    skillsets: dict[str, Path],
) -> dict[str, Any]:
    tasks_root = tasks_root.resolve()
    task_rows: list[dict[str, Any]] = []
    all_oracles: set[str] = set()

    task_dirs = sorted(
        task
        for task in tasks_root.iterdir()
        if task.is_dir() and (task / "environment" / "skills").is_dir()
    )
    task_oracles: dict[str, list[str]] = {}
    for task_dir in task_dirs:
        oracle_names = _skill_directory_names(
            task_dir / "environment" / "skills", immediate=True
        )
        task_oracles[task_dir.name] = oracle_names
        all_oracles.update(oracle_names)

    resolved_skillsets = {
        name: path.resolve() for name, path in sorted(skillsets.items())
    }
    library_names_by_skillset = {
        name: _skill_directory_names(path)
        for name, path in resolved_skillsets.items()
    }

    for task_dir in task_dirs:
        row = {
            "task_id": task_dir.name,
            "instruction": _task_instruction(task_dir),
            "oracle_skills": task_oracles[task_dir.name],
            "skillsets": {},
        }
        for name, library_names in library_names_by_skillset.items():
            row["skillsets"][name] = _resolve_oracles(
                task_oracles[task_dir.name], library_names
            )
        task_rows.append(row)

    unique_oracles = sorted(all_oracles)
    summaries: dict[str, Any] = {}
    for name, library_names in library_names_by_skillset.items():
        unique_resolution = _resolve_oracles(unique_oracles, library_names)
        task_results = [row["skillsets"][name] for row in task_rows]
        library_path = resolved_skillsets[name]
        summaries[name] = {
            "library_path": str(library_path),
            "library_sha256": sha256_tree(library_path),
            "library_skill_count": len(library_names),
            "unique_oracle_count": len(unique_oracles),
            "exact_count": len(unique_resolution["exact"]),
            "alias_count": len(unique_resolution["aliases"]),
            "absent_count": len(unique_resolution["absent"]),
            "exact": unique_resolution["exact"],
            "aliases": unique_resolution["aliases"],
            "absent": unique_resolution["absent"],
            "ambiguous_aliases": unique_resolution["ambiguous_aliases"],
            "all_available_task_count": sum(
                bool(result["all_available"]) for result in task_results
            ),
            "task_count": len(task_rows),
        }

    return {
        "schema_version": 1,
        "tasks_root": str(tasks_root),
        "tasks_sha256": sha256_tree(tasks_root),
        "task_count": len(task_rows),
        "unique_oracle_skills": unique_oracles,
        "skillsets": summaries,
        "tasks": task_rows,
    }


def render_coverage_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# SkillsBench Skillset Coverage Audit",
        "",
        f"Tasks: {result['task_count']}",
        f"Unique oracle skills: {len(result['unique_oracle_skills'])}",
        "",
        "| Skillset | Library skills | Exact | Punctuation aliases | Absent | Fully attainable tasks |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, summary in sorted(result["skillsets"].items()):
        attainable = int(summary["all_available_task_count"])
        task_count = int(summary["task_count"])
        percent = 100.0 * attainable / task_count if task_count else 100.0
        lines.append(
            f"| {name} | {summary['library_skill_count']} | {summary['exact_count']} | "
            f"{summary['alias_count']} | {summary['absent_count']} | "
            f"{attainable}/{task_count} ({percent:.1f}%) |"
        )

    lines.extend(["", "## Missing and aliased oracle names", ""])
    for name, summary in sorted(result["skillsets"].items()):
        lines.append(f"### {name}")
        lines.append("")
        alias_text = ", ".join(
            f"`{source}` -> `{target}`"
            for source, target in summary["aliases"].items()
        ) or "None"
        absent_text = ", ".join(f"`{item}`" for item in summary["absent"]) or "None"
        lines.append(f"Aliases: {alias_text}")
        lines.append("")
        lines.append(f"Absent: {absent_text}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit SkillsBench oracle coverage.")
    parser.add_argument("--tasks-root", type=Path, required=True)
    parser.add_argument(
        "--skillset",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Named skillset. Repeat for multiple libraries.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    skillsets: dict[str, Path] = {}
    for item in args.skillset:
        name, separator, raw_path = item.partition("=")
        if not separator or not name or not raw_path:
            raise SystemExit(f"Invalid --skillset {item!r}; expected NAME=PATH")
        skillsets[name] = Path(raw_path)
    if not skillsets:
        raise SystemExit("At least one --skillset NAME=PATH is required")

    result = audit_skillset_coverage(args.tasks_root, skillsets)
    atomic_write_json(args.output_json, result)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(
        render_coverage_markdown(result), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
