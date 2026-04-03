#!/usr/bin/env python3
"""Add all-skills agent instruction files to task environments and patch Dockerfiles."""

from pathlib import Path
import shutil


SKILLSBENCH = Path(__file__).parent.parent
TEMPLATE_DIR = SKILLSBENCH / "_allskills_template"
TASKS_DIR = SKILLSBENCH / "tasks_all_skills"
FILES = ("AGENTS.md", "CLAUDE.md", "GEMINI.md")
COPY_BLOCK = [
    "COPY CLAUDE.md /root/CLAUDE.md",
    "COPY AGENTS.md /root/AGENTS.md",
    "COPY GEMINI.md /root/GEMINI.md",
]


def main() -> None:
    for task_dir in sorted(TASKS_DIR.iterdir()):
        env_dir = task_dir / "environment"
        dockerfile = env_dir / "Dockerfile"
        if not dockerfile.exists():
            continue

        for name in FILES:
            shutil.copy2(TEMPLATE_DIR / name, env_dir / name)

        content = dockerfile.read_text(encoding="utf-8")
        if COPY_BLOCK[0] in content:
            continue

        anchor = "COPY skills/ /root/.gemini/skills/"
        if anchor in content:
            content = content.replace(anchor, anchor + "\n" + "\n".join(COPY_BLOCK), 1)
        else:
            content = content.rstrip() + "\n\n" + "\n".join(COPY_BLOCK) + "\n"
        dockerfile.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
