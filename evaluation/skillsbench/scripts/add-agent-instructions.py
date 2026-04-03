#!/usr/bin/env python3
"""
Add AGENTS.md and GEMINI.md to all tasks_gos task environments.

For each task in tasks_gos/:
  1. Copy _gos_template/AGENTS.md  → environment/AGENTS.md
  2. Copy _gos_template/GEMINI.md  → environment/GEMINI.md
  3. Append COPY instructions to environment/Dockerfile (idempotent)

Usage:
  python scripts/add-agent-instructions.py [--dry-run] [--task <name>]
"""

import argparse
import shutil
import sys
from pathlib import Path

SKILLSBENCH = Path(__file__).parent.parent
TEMPLATE_DIR = SKILLSBENCH / "_gos_template"
TASKS_DIR = SKILLSBENCH / "tasks_gos"

FILES = {
    "AGENTS.md": "/root/AGENTS.md",   # Codex
    "GEMINI.md": "/root/GEMINI.md",   # Gemini CLI
}

DOCKERFILE_MARKER = "# ── GoS instrumentation ─"


def process_task(task_dir: Path, dry_run: bool) -> str:
    """Process a single task. Returns status: 'updated' | 'skipped' | 'error'"""
    env_dir = task_dir / "environment"
    dockerfile = env_dir / "Dockerfile"

    if not dockerfile.exists():
        return "no-dockerfile"

    dockerfile_content = dockerfile.read_text()

    # Check which files still need to be added
    missing_copies = {}
    for src_name, dest_path in FILES.items():
        copy_line = f"COPY {src_name} {dest_path}"
        if copy_line not in dockerfile_content:
            missing_copies[src_name] = (dest_path, copy_line)

    if not missing_copies:
        return "skipped"

    if dry_run:
        for src_name in missing_copies:
            print(f"  [dry-run] would add: COPY {src_name} {FILES[src_name]}")
        return "would-update"

    # Copy template files into environment/
    for src_name in missing_copies:
        src = TEMPLATE_DIR / src_name
        dst = env_dir / src_name
        if not src.exists():
            print(f"  WARNING: template {src} not found, skipping", file=sys.stderr)
            continue
        shutil.copy2(src, dst)

    # Inject COPY lines into Dockerfile, just before the GoS marker block
    # (so they're grouped with other agent instruction files like CLAUDE.md)
    lines_to_inject = []
    for src_name, (dest_path, copy_line) in missing_copies.items():
        lines_to_inject.append(copy_line)

    # Find the COPY CLAUDE.md line and insert after it
    new_lines = []
    inserted = False
    for line in dockerfile_content.splitlines():
        new_lines.append(line)
        if not inserted and line.strip().startswith("COPY CLAUDE.md"):
            for inject in lines_to_inject:
                new_lines.append(inject)
            inserted = True

    if not inserted:
        # Fallback: append before the closing marker comment if present
        final_lines = []
        for line in new_lines:
            if not inserted and line.strip().startswith("# ────"):
                for inject in lines_to_inject:
                    final_lines.append(inject)
                inserted = True
            final_lines.append(line)
        if not inserted:
            final_lines.extend(lines_to_inject)
        new_lines = final_lines

    dockerfile.write_text("\n".join(new_lines) + "\n")
    return "updated"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    parser.add_argument("--task", help="Process only this task name")
    args = parser.parse_args()

    if not TEMPLATE_DIR.exists():
        print(f"ERROR: template dir not found: {TEMPLATE_DIR}", file=sys.stderr)
        sys.exit(1)

    for src_name in FILES:
        if not (TEMPLATE_DIR / src_name).exists():
            print(f"ERROR: {TEMPLATE_DIR / src_name} not found", file=sys.stderr)
            sys.exit(1)

    task_dirs = sorted(TASKS_DIR.iterdir()) if not args.task else [TASKS_DIR / args.task]

    counts = {"updated": 0, "skipped": 0, "would-update": 0, "error": 0}

    for task_dir in task_dirs:
        if not task_dir.is_dir():
            continue
        status = process_task(task_dir, args.dry_run)
        counts[status] = counts.get(status, 0) + 1
        if status not in ("skipped",):
            print(f"  [{status:12s}] {task_dir.name}")

    print()
    if args.dry_run:
        print(f"Dry run: {counts.get('would-update', 0)} would be updated, {counts.get('skipped', 0)} already up to date")
    else:
        print(f"Done: {counts.get('updated', 0)} updated, {counts.get('skipped', 0)} skipped")


if __name__ == "__main__":
    main()
