#!/usr/bin/env python3
"""
Migrate tasks_gos/<task>/environment/ to the GoS evaluation pattern.

For each task (skipping already-migrated ones):
  1. Replace environment/skills/ with only graph-skills-retriever
  2. Write environment/CLAUDE.md
  3. Write/merge environment/docker-compose.yaml with GoS volumes
  4. Append GoS instrumentation block to environment/Dockerfile

Usage:
    # Dry run (show what would change, write nothing)
    python scripts/migrate-to-gos.py --dry-run

    # Migrate all tasks
    python scripts/migrate-to-gos.py

    # Migrate specific tasks
    python scripts/migrate-to-gos.py powerlifting-coef-calc weighted-gdp-calc

    # Force re-migrate already-migrated tasks
    python scripts/migrate-to-gos.py --force

Run from: evaluation/skillsbench/
"""

import argparse
import shutil
import sys
from pathlib import Path

import yaml  # pip install pyyaml

# ── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = Path(__file__).parent
BENCH_DIR    = SCRIPT_DIR.parent                          # evaluation/skillsbench/
TASKS_DIR    = BENCH_DIR / "tasks_gos"
TEMPLATE_DIR = BENCH_DIR / "_gos_template"
AGENT_SKILLS = BENCH_DIR.parent.parent / "agent_skills"  # graph-skills/agent_skills/
GOS_RETRIEVER = AGENT_SKILLS / "graph-skills-retriever"

# ── Templates ────────────────────────────────────────────────────────────────

CLAUDE_MD = (TEMPLATE_DIR / "CLAUDE.md").read_text()

COMPOSE_VOLUMES = [
    "../../../../../gos:/opt/graphskills/gos:ro",
    "${GOS_PREBUILT_HOST_WORKSPACE:-../../../../data/gos_workspace/all_skills_v1}:/opt/graphskills/prebuilt:ro",
    "../../../all_skills:/opt/graphskills/skills:ro",
]
COMPOSE_ENV = [
    "GOS_PREBUILT_WORKING_DIR=/opt/graphskills/prebuilt",
    "GOS_WORKING_DIR=/opt/graphskills/runtime",
    "GOS_SKILLS_DIR=/opt/graphskills/skills",
    "GOS_EMBEDDING_MODEL",
    "GOS_EMBEDDING_DIM",
    "GOS_LLM_MODEL",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
]

DOCKERFILE_APPEND = (TEMPLATE_DIR / "Dockerfile.gos-append").read_text()

ALREADY_MIGRATED_MARKER = "GoS instrumentation"

# Tasks where specific skills are referenced by the verifier (keep them in skills/).
# These skills will be preserved alongside graph-skills-retriever.
VERIFIER_SKILL_EXCEPTIONS: dict[str, list[str]] = {
    "scheduling-email-assistant": ["gmail-skill"],
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def log(msg: str, dry_run: bool = False) -> None:
    prefix = "[DRY RUN] " if dry_run else ""
    print(f"{prefix}{msg}")


def merge_gos_into_compose(existing_path: Path, dry_run: bool) -> None:
    """Add GoS volumes and environment into an existing docker-compose.yaml."""
    with open(existing_path) as f:
        data = yaml.safe_load(f)

    main = data.setdefault("services", {}).setdefault("main", {})

    # Volumes
    existing_vols: list = main.setdefault("volumes", [])
    existing_vol_strs = [str(v) for v in existing_vols]
    added_vols = []
    for vol in COMPOSE_VOLUMES:
        if not any(vol.split(":")[1] in ev for ev in existing_vol_strs):
            existing_vols.append(vol)
            added_vols.append(vol)

    # Environment
    existing_env: list = main.setdefault("environment", [])
    existing_env_keys = {(e.split("=")[0] if isinstance(e, str) else e) for e in existing_env}
    added_env = []
    for env in COMPOSE_ENV:
        key = env.split("=")[0]
        if key not in existing_env_keys:
            existing_env.append(env)
            added_env.append(key)

    if not added_vols and not added_env:
        log(f"  docker-compose.yaml: already has GoS volumes/env, skipping merge")
        return

    log(f"  docker-compose.yaml: merging GoS additions (volumes: {len(added_vols)}, env: {len(added_env)})", dry_run)
    if not dry_run:
        with open(existing_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def write_gos_compose(path: Path, dry_run: bool) -> None:
    """Write a fresh GoS docker-compose.yaml."""
    content = (TEMPLATE_DIR / "docker-compose.yaml").read_text()
    log(f"  docker-compose.yaml: writing GoS template", dry_run)
    if not dry_run:
        path.write_text(content)


def migrate_skills(env_dir: Path, task_name: str, dry_run: bool) -> None:
    """Replace skills/ with graph-skills-retriever (+ any verifier exceptions)."""
    skills_dir = env_dir / "skills"
    preserve = VERIFIER_SKILL_EXCEPTIONS.get(task_name, [])

    # Collect dirs to remove (everything except preserved ones)
    if skills_dir.exists():
        to_remove = [
            d for d in skills_dir.iterdir()
            if d.is_dir()
            and d.name != "graph-skills-retriever"
            and d.name not in preserve
        ]
        if to_remove:
            log(f"  skills/: removing {[d.name for d in to_remove]}", dry_run)
            if not dry_run:
                for d in to_remove:
                    shutil.rmtree(d)
    else:
        log(f"  skills/: creating directory", dry_run)
        if not dry_run:
            skills_dir.mkdir(parents=True)

    gos_dest = skills_dir / "graph-skills-retriever"
    if not gos_dest.exists():
        log(f"  skills/: copying graph-skills-retriever", dry_run)
        if not dry_run:
            shutil.copytree(GOS_RETRIEVER, gos_dest)
    else:
        log(f"  skills/graph-skills-retriever: already present")


def append_gos_to_dockerfile(dockerfile: Path, dry_run: bool) -> None:
    """Append the GoS instrumentation block if not already present."""
    content = dockerfile.read_text()

    if ALREADY_MIGRATED_MARKER in content:
        log(f"  Dockerfile: GoS block already present, skipping")
        return

    # Remove existing bare COPY skills lines that the GoS append block will replace.
    # We keep them if they have extra destinations (verifier paths etc.) - the append
    # block adds the standard agent dirs on top.
    lines = content.splitlines(keepends=True)
    # Mark lines that are simple "COPY skills /xxx/.yyy/skills" to drop
    # (they become redundant once we append the GoS block with its own COPY skills)
    bare_skill_paths = {
        "/root/.claude/skills", "/root/.codex/skills", "/root/.opencode/skill",
        "/root/.opencode/skills", "/root/.goose/skills", "/root/.factory/skills",
        "/root/.agents/skills", "/root/.gemini/skills",
        "/etc/claude-code/.claude/skills",
        "/app/.agents/skills", "/app/.cursor/skills", "/app/.gemini/skills",
    }
    filtered = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("COPY skills ") or stripped.startswith("COPY skills/ "):
            dest = stripped.split()[-1].rstrip("/")
            if dest in bare_skill_paths:
                continue  # drop – GoS append block re-adds them
        filtered.append(line)

    new_content = "".join(filtered).rstrip() + "\n\n" + DOCKERFILE_APPEND
    log(f"  Dockerfile: appending GoS block", dry_run)
    if not dry_run:
        dockerfile.write_text(new_content)


def migrate_task(task_dir: Path, dry_run: bool, force: bool) -> None:
    name = task_dir.name
    env_dir = task_dir / "environment"

    if not env_dir.exists():
        log(f"[SKIP] {name}: no environment/ directory")
        return

    dockerfile = env_dir / "Dockerfile"
    if not dockerfile.exists():
        log(f"[SKIP] {name}: no Dockerfile")
        return

    # Check if already migrated
    if not force and ALREADY_MIGRATED_MARKER in dockerfile.read_text():
        log(f"[OK]   {name}: already migrated")
        return

    log(f"\n[MIGRATE] {name}")

    # 1. Skills directory
    migrate_skills(env_dir, name, dry_run)

    # 2. CLAUDE.md
    claude_md = env_dir / "CLAUDE.md"
    if not claude_md.exists() or force:
        log(f"  CLAUDE.md: writing", dry_run)
        if not dry_run:
            claude_md.write_text(CLAUDE_MD)
    else:
        log(f"  CLAUDE.md: already exists")

    # 3. docker-compose.yaml
    compose = env_dir / "docker-compose.yaml"
    if compose.exists():
        existing_text = compose.read_text()
        if any(v.split(":")[1] in existing_text for v in COMPOSE_VOLUMES):
            log(f"  docker-compose.yaml: already has GoS volumes")
        else:
            merge_gos_into_compose(compose, dry_run)
    else:
        write_gos_compose(compose, dry_run)

    # 4. Dockerfile
    append_gos_to_dockerfile(dockerfile, dry_run)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tasks", nargs="*", help="Task names to migrate (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing files")
    parser.add_argument("--force", action="store_true", help="Re-migrate already-migrated tasks")
    args = parser.parse_args()

    if not TASKS_DIR.exists():
        print(f"ERROR: tasks_gos not found at {TASKS_DIR}", file=sys.stderr)
        sys.exit(1)

    if not GOS_RETRIEVER.exists():
        print(f"ERROR: graph-skills-retriever not found at {GOS_RETRIEVER}", file=sys.stderr)
        sys.exit(1)

    if args.tasks:
        task_dirs = []
        for name in args.tasks:
            td = TASKS_DIR / name
            if not td.exists():
                print(f"WARNING: task '{name}' not found in {TASKS_DIR}", file=sys.stderr)
            else:
                task_dirs.append(td)
    else:
        task_dirs = sorted(p for p in TASKS_DIR.iterdir() if p.is_dir())

    skip_already_done = {"3d-scan-calc"}  # reference task, already migrated manually
    task_dirs = [t for t in task_dirs if t.name not in skip_already_done or args.force]

    print(f"Migrating {len(task_dirs)} task(s) to GoS pattern...")
    if args.dry_run:
        print("DRY RUN — no files will be written.\n")

    migrated = skipped = 0
    for td in task_dirs:
        dockerfile = td / "environment" / "Dockerfile"
        if (
            not args.force
            and dockerfile.exists()
            and ALREADY_MIGRATED_MARKER in dockerfile.read_text()
        ):
            skipped += 1
            continue
        migrate_task(td, dry_run=args.dry_run, force=args.force)
        migrated += 1

    print(f"\nDone. migrated={migrated} skipped(already done)={skipped}")


if __name__ == "__main__":
    main()
