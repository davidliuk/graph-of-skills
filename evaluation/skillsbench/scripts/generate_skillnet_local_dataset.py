#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SKILLSBENCH_ROOT = REPO_ROOT / "evaluation" / "skillsbench"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gos.core.parsing import parse_skill_document  # noqa: E402

GRAPH_MARKER_START = "# BEGIN GRAPH SKILLS BENCHMARK"
GRAPH_MARKER_END = "# END GRAPH SKILLS BENCHMARK"
GRAPH_LIBRARY_PATH = "/opt/graphskills/skills"
SKILL_FILENAME = "SKILL.md"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-large"
DEFAULT_EMBEDDING_DIM = 3072
BATCH_SIZE = 64

TEMPLATE_DIR = SKILLSBENCH_ROOT / "_skillnetskills_template"
BOOTSTRAP_SKILL_DIR = REPO_ROOT / "agent_skills" / "skillnet-skills-retriever"
QUERY_SCRIPT = SKILLSBENCH_ROOT / "scripts" / "skillnet_local_query.py"


def resolve_api_key() -> str:
    base_url = resolve_base_url()
    if "openrouter.ai" in base_url:
        key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    else:
        key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY or OPENROUTER_API_KEY must be set")
    return key


def resolve_base_url() -> str:
    configured = (os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL).strip().rstrip("/")
    if "openrouter.ai/api" in configured and not configured.endswith("/v1"):
        return configured + "/v1"
    return configured


def embed_batch(texts: list[str], model: str) -> list[list[float]]:
    endpoint = resolve_base_url() + "/embeddings"
    payload = json.dumps({"model": model, "input": texts, "encoding_format": "float"}).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={
            "Authorization": f"Bearer {resolve_api_key()}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Embedding request failed: {exc.code} {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Embedding request failed: {exc}") from exc

    data = body.get("data") or []
    if len(data) != len(texts):
        raise RuntimeError(f"Expected {len(texts)} embeddings, got {len(data)}")
    return [item["embedding"] for item in data]


def embed_texts(texts: list[str], model: str) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        vectors.extend(embed_batch(texts[start : start + BATCH_SIZE], model))
    return vectors


def clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def scan_skills(skills_root: Path) -> list[dict[str, Any]]:
    skills: list[dict[str, Any]] = []
    for skill_file in sorted(skills_root.rglob(SKILL_FILENAME)):
        content = skill_file.read_text(encoding="utf-8", errors="ignore")
        rel_dir = skill_file.parent.relative_to(skills_root)
        source_path = f"{GRAPH_LIBRARY_PATH}/{rel_dir.as_posix()}/{SKILL_FILENAME}"
        parsed = parse_skill_document(content, source_path=source_path, snippet_chars=900)
        if parsed is None:
            continue
        card_text = "\n\n".join(
            part
            for part in [
                parsed.name,
                parsed.description,
                parsed.rendered_snippet,
            ]
            if part
        )
        skills.append(
            {
                "name": parsed.name,
                "description": parsed.description,
                "source_path": source_path,
                "rendered_snippet": parsed.rendered_snippet,
                "card_text": clip(card_text, 4000),
            }
        )
    return skills


def build_index(skills_root: Path, shared_dir: Path) -> Path:
    model = os.getenv("GOS_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
    dim = int(os.getenv("GOS_EMBEDDING_DIM") or DEFAULT_EMBEDDING_DIM)
    skills = scan_skills(skills_root)
    vectors = embed_texts([skill["card_text"] for skill in skills], model)
    for skill, vector in zip(skills, vectors, strict=True):
        skill["embedding"] = vector
        del skill["card_text"]
    shared_dir.mkdir(parents=True, exist_ok=True)
    index_path = shared_dir / "skillnet_local_index.json"
    index_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "type": "skillnet-local-index",
                    "embedding_model": model,
                    "embedding_dim": dim,
                    "skill_count": len(skills),
                    "skills_root": str(skills_root),
                },
                "skills": skills,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return index_path


def hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def replace_task_skills(task_dir: Path) -> None:
    skills_dir = task_dir / "environment" / "skills"
    if skills_dir.exists():
        shutil.rmtree(skills_dir)
    copy_tree(BOOTSTRAP_SKILL_DIR, skills_dir)


def render_compose(destination_env_dir: Path, skills_root: Path) -> str:
    rendered = (TEMPLATE_DIR / "docker-compose.yaml").read_text(encoding="utf-8")
    skills_relative = os.path.relpath(skills_root, destination_env_dir)
    rendered = rendered.replace("../../../../data/skillsets/skills_200", skills_relative)
    return rendered.replace("../../../skillsets/skills_200", skills_relative)


def patch_dockerfile(dockerfile_path: Path) -> None:
    original = dockerfile_path.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"{re.escape(GRAPH_MARKER_START)}.*?{re.escape(GRAPH_MARKER_END)}\n?",
        re.DOTALL,
    )
    updated = re.sub(pattern, "", original)
    updated = re.sub(r"(?m)^\s*COPY\s+skills(?:[^\n]*)\n?", "", updated)
    updated = updated.rstrip()
    block = "\n".join(
        [
            GRAPH_MARKER_START,
            "RUN if command -v python3 >/dev/null 2>&1; then :; elif command -v apt-get >/dev/null 2>&1; then apt-get update && apt-get install -y python3 && rm -rf /var/lib/apt/lists/*; else echo \"python3 is required for skillnet-query\" >&2; exit 1; fi",
            "COPY skillnet_local_query.py /usr/local/bin/skillnet-query",
            "RUN chmod +x /usr/local/bin/skillnet-query",
            "COPY CLAUDE.md /root/CLAUDE.md",
            "COPY AGENTS.md /root/AGENTS.md",
            "COPY GEMINI.md /root/GEMINI.md",
            GRAPH_MARKER_END,
        ]
    )
    dockerfile_path.write_text(f"{updated}\n\n{block}\n", encoding="utf-8")


def prepare_task(source_task_dir: Path, destination_task_dir: Path, skills_root: Path) -> None:
    copy_tree(source_task_dir, destination_task_dir)
    replace_task_skills(destination_task_dir)

    env_dir = destination_task_dir / "environment"
    for name in ("AGENTS.md", "CLAUDE.md", "GEMINI.md"):
        hardlink_or_copy(TEMPLATE_DIR / name, env_dir / name)
    hardlink_or_copy(QUERY_SCRIPT, env_dir / "skillnet_local_query.py")
    (env_dir / "docker-compose.yaml").write_text(
        render_compose(env_dir, skills_root),
        encoding="utf-8",
    )
    patch_dockerfile(env_dir / "Dockerfile")


def build_task_list(tasks_root: Path, selected_tasks: list[str]) -> list[Path]:
    tasks = sorted(path for path in tasks_root.iterdir() if path.is_dir() and (path / "task.toml").exists())
    if not selected_tasks:
        return tasks
    selected = set(selected_tasks)
    filtered = [path for path in tasks if path.name in selected]
    missing = selected - {path.name for path in filtered}
    if missing:
        raise FileNotFoundError(f"Unknown task ids: {', '.join(sorted(missing))}")
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SkillNet-local SkillsBench tasks")
    parser.add_argument("--tasks-root", type=Path, default=SKILLSBENCH_ROOT / "tasks")
    parser.add_argument("--skills-root", type=Path, default=REPO_ROOT / "data" / "skillsets" / "skills_1000")
    parser.add_argument("--output-root", type=Path, default=SKILLSBENCH_ROOT / "generated_skills1000")
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()

    tasks_root = args.tasks_root.resolve()
    skills_root = args.skills_root.resolve()
    output_root = args.output_root.resolve()
    tasks_dir = output_root / "tasks_skillnet_skills"
    shared_dir = output_root / "shared"

    if args.clear and tasks_dir.exists():
        shutil.rmtree(tasks_dir)
    tasks_dir.mkdir(parents=True, exist_ok=True)

    index_path = build_index(skills_root, shared_dir)
    tasks = build_task_list(tasks_root, args.task)
    for task in tasks:
        prepare_task(task, tasks_dir / task.name, skills_root)

    print(f"SkillNet-local index: {index_path}")
    print(f"Tasks generated: {len(tasks)}")
    print(f"Dataset: {tasks_dir}")


if __name__ == "__main__":
    main()
