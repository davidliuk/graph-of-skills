#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any


SKILLSBENCH_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SKILLSBENCH_ROOT.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gos.core.parsing import parse_skill_document  # noqa: E402


TOKEN_STOPWORDS = {
    "a",
    "an",
    "and",
    "any",
    "arg",
    "args",
    "array",
    "bool",
    "boolean",
    "data",
    "dict",
    "file",
    "float",
    "for",
    "from",
    "in",
    "input",
    "int",
    "json",
    "list",
    "object",
    "of",
    "on",
    "or",
    "output",
    "path",
    "record",
    "result",
    "set",
    "str",
    "string",
    "text",
    "the",
    "to",
    "value",
}

GRAPH_QUERY_PATH = "/opt/graphskills/query.py"
GRAPH_QUERY_COMMAND = "graphskills-query"
GRAPH_LIBRARY_PATH = "/opt/graphskills/skills"
GRAPH_MARKER_START = "# BEGIN GRAPH SKILLS BENCHMARK"
GRAPH_MARKER_END = "# END GRAPH SKILLS BENCHMARK"
SKILLS_HOST_ENV_VAR = "SKILLSBENCH_SKILLS_HOST_DIR"
GOS_WORKSPACE_ENV_VAR = "GOS_PREBUILT_HOST_WORKSPACE"

ASSETS_DIR = SKILLSBENCH_ROOT / "graphskills_assets"
BOOTSTRAP_SKILL_DIR = REPO_ROOT / "agent_skills" / "graph-skills-retriever"
VECTOR_BOOTSTRAP_SKILL_DIR = REPO_ROOT / "agent_skills" / "vector-skills-retriever"
QUERY_TEMPLATE = ASSETS_DIR / "query.py"
VECTOR_QUERY_TEMPLATE = ASSETS_DIR / "vector_query.py"
ALLSKILLS_TEMPLATE_DIR = SKILLSBENCH_ROOT / "_allskills_template"
GOS_TEMPLATE_DIR = SKILLSBENCH_ROOT / "_gos_template"
VECTORSKILLS_TEMPLATE_DIR = SKILLSBENCH_ROOT / "_vectorskills_template"
SKILLSETS_ROOT = REPO_ROOT / "data" / "skillsets"
DEFAULT_SKILLSET_NAME = "skills_200"

DEFAULT_TASKS_ROOT = SKILLSBENCH_ROOT / "tasks"
DEFAULT_SKILLS_ROOT = SKILLSETS_ROOT / DEFAULT_SKILLSET_NAME
DEFAULT_OUTPUT_ROOT = SKILLSBENCH_ROOT / "generated"


def signature_tokens(values: list[str]) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        lowered = value.lower()
        normalized = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
        if normalized:
            tokens.add(normalized)
        for token in re.findall(r"[a-z0-9]+", lowered):
            token = token.rstrip("s")
            if len(token) < 3 or token in TOKEN_STOPWORDS:
                continue
            tokens.add(token)
    return tokens


def schema_overlap_score(
    producer_values: list[str],
    consumer_values: list[str],
) -> tuple[float, list[str]]:
    best_score = 0.0
    best_evidence: set[str] = set()

    for producer in producer_values:
        producer_norm = re.sub(r"[^a-z0-9]+", "_", producer.lower()).strip("_")
        producer_tokens = signature_tokens([producer])
        for consumer in consumer_values:
            consumer_norm = re.sub(r"[^a-z0-9]+", "_", consumer.lower()).strip("_")
            consumer_tokens = signature_tokens([consumer])

            if producer_norm and producer_norm == consumer_norm:
                return 1.0, [producer_norm]

            if producer_norm and consumer_norm and (
                producer_norm in consumer_norm or consumer_norm in producer_norm
            ):
                candidate = {producer_norm, consumer_norm}
                return 0.85, sorted(candidate)

            overlap = producer_tokens & consumer_tokens
            if not overlap:
                continue

            union = producer_tokens | consumer_tokens
            score = max(0.5, len(overlap) / max(len(union), 1))
            if score > best_score:
                best_score = score
                best_evidence = overlap

    return best_score, sorted(best_evidence)


def semantic_similarity(
    left: dict[str, Any],
    right: dict[str, Any],
) -> tuple[float, list[str]]:
    left_tokens = signature_tokens(
        [
            left["name"],
            left["description"],
            left["rendered_snippet"],
            "\n".join(left["compatibility"]),
            "\n".join(left["allowed_tools"]),
        ]
    )
    right_tokens = signature_tokens(
        [
            right["name"],
            right["description"],
            right["rendered_snippet"],
            "\n".join(right["compatibility"]),
            "\n".join(right["allowed_tools"]),
        ]
    )
    overlap = left_tokens & right_tokens
    if len(overlap) < 2:
        return 0.0, []

    union = left_tokens | right_tokens
    score = len(overlap) / max(len(union), 1)
    if score < 0.18:
        return 0.0, []

    return min(0.6, round(score * 2.0, 3)), sorted(overlap)[:5]


def load_skill_library(skills_root: Path) -> list[dict[str, Any]]:
    skills: list[dict[str, Any]] = []

    for skill_file in sorted(skills_root.rglob("SKILL.md")):
        content = skill_file.read_text(encoding="utf-8", errors="ignore")
        relative_dir = skill_file.parent.relative_to(skills_root)
        container_path = f"{GRAPH_LIBRARY_PATH}/{relative_dir.as_posix()}/SKILL.md"
        parsed = parse_skill_document(
            content,
            source_path=container_path,
            snippet_chars=900,
        )
        if parsed is None:
            continue

        skills.append(
            {
                "name": parsed.name,
                "description": parsed.description,
                "inputs": parsed.inputs,
                "outputs": parsed.outputs,
                "compatibility": parsed.compatibility,
                "allowed_tools": parsed.allowed_tools,
                "source_path": container_path,
                "skill_dir": relative_dir.as_posix(),
                "rendered_snippet": parsed.rendered_snippet,
                "raw_content": parsed.raw_content,
                "skill_id": parsed.skill_id,
            }
        )

    return skills


def build_graph_bundle(
    skills_root: Path,
    *,
    dependency_threshold: float = 0.6,
) -> dict[str, Any]:
    skills = load_skill_library(skills_root)
    edges: list[dict[str, Any]] = []
    edge_keys: set[tuple[str, str, str]] = set()

    for index, skill in enumerate(skills):
        for candidate in skills[index + 1 :]:
            forward_score, forward_evidence = schema_overlap_score(
                skill["outputs"],
                candidate["inputs"],
            )
            if forward_score >= dependency_threshold:
                key = (skill["name"], candidate["name"], "dependency")
                if key not in edge_keys:
                    edge_keys.add(key)
                    edges.append(
                        {
                            "source": skill["name"],
                            "target": candidate["name"],
                            "type": "dependency",
                            "weight": float(forward_score),
                            "confidence": float(forward_score),
                            "description": (
                                f"{skill['name']} produces data that {candidate['name']} "
                                f"consumes: {', '.join(forward_evidence) or 'compatible I/O'}."
                            ),
                        }
                    )

            reverse_score, reverse_evidence = schema_overlap_score(
                candidate["outputs"],
                skill["inputs"],
            )
            if reverse_score >= dependency_threshold:
                key = (candidate["name"], skill["name"], "dependency")
                if key not in edge_keys:
                    edge_keys.add(key)
                    edges.append(
                        {
                            "source": candidate["name"],
                            "target": skill["name"],
                            "type": "dependency",
                            "weight": float(reverse_score),
                            "confidence": float(reverse_score),
                            "description": (
                                f"{candidate['name']} produces data that {skill['name']} "
                                f"consumes: {', '.join(reverse_evidence) or 'compatible I/O'}."
                            ),
                        }
                    )

            semantic_score, semantic_evidence = semantic_similarity(skill, candidate)
            if semantic_score > 0:
                source_name, target_name = sorted((skill["name"], candidate["name"]))
                key = (source_name, target_name, "semantic")
                if key not in edge_keys:
                    edge_keys.add(key)
                    edges.append(
                        {
                            "source": source_name,
                            "target": target_name,
                            "type": "semantic",
                            "weight": float(semantic_score),
                            "confidence": float(semantic_score),
                            "description": (
                                "They share a narrow capability cluster through "
                                f"{', '.join(semantic_evidence)}."
                            ),
                        }
                    )

    return {
        "metadata": {
            "version": 1,
            "skill_count": len(skills),
            "edge_count": len(edges),
            "library_root": GRAPH_LIBRARY_PATH,
            "query_command": GRAPH_QUERY_COMMAND,
            "ppr_damping": 0.2,
            "ppr_max_iter": 50,
            "ppr_tolerance": 1e-6,
        },
        "skills": skills,
        "edges": edges,
    }


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def hardlink_or_copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def link_or_copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    for item in source.rglob("*"):
        relative = item.relative_to(source)
        target = destination / relative
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        hardlink_or_copy_file(item, target)


def replace_task_skills(task_dir: Path, replacement_dir: Path) -> None:
    skills_dir = task_dir / "environment" / "skills"
    if skills_dir.exists():
        shutil.rmtree(skills_dir)
    link_or_copy_tree(replacement_dir, skills_dir)


def render_compose_template(
    template_path: Path,
    *,
    destination_env_dir: Path,
    skills_root: Path,
    gos_workspace: Path | None = None,
) -> str:
    rendered = template_path.read_text(encoding="utf-8")
    skills_relative = os.path.relpath(skills_root, destination_env_dir)
    rendered = rendered.replace("../../../../data/skillsets/skills_200", skills_relative)
    rendered = rendered.replace("../../../skillsets/skills_200", skills_relative)
    rendered = rendered.replace("../../../all_skills", skills_relative)

    if gos_workspace is not None:
        workspace_relative = os.path.relpath(gos_workspace, destination_env_dir)
        rendered = rendered.replace("../../../../data/gos_workspace/all_skills_v1", workspace_relative)
        rendered = rendered.replace("../../../gos_workspace/all_skills_v1", workspace_relative)
        rendered = re.sub(
            r"(\$\{GOS_PREBUILT_HOST_WORKSPACE:-)([^}]+)(\})",
            lambda m: f"{m.group(1)}{workspace_relative}{m.group(3)}",
            rendered,
        )

    return rendered


def build_docker_block(variant: str) -> str:
    lines = [GRAPH_MARKER_START]

    instruction_block = [
        "COPY CLAUDE.md /root/CLAUDE.md",
        "COPY AGENTS.md /root/AGENTS.md",
        "COPY GEMINI.md /root/GEMINI.md",
        'RUN for d in /app /app/workspace /app/video /root /workspace /repo; do '
        'if [ -d "$d" ]; then '
        'cp /root/CLAUDE.md "$d/CLAUDE.md" 2>/dev/null || true; '
        'cp /root/AGENTS.md "$d/AGENTS.md" 2>/dev/null || true; '
        'cp /root/GEMINI.md "$d/GEMINI.md" 2>/dev/null || true; '
        'fi; '
        'done',
    ]

    if variant in {"graphskills", "vectorskills"}:
        command_name = "graphskills-query" if variant == "graphskills" else "vectorskills-query"
        lines.append(
            "RUN if command -v python3 >/dev/null 2>&1; then :; "
            "elif command -v apt-get >/dev/null 2>&1; then "
            "apt-get update && apt-get install -y python3 && rm -rf /var/lib/apt/lists/*; "
            f"else echo \"python3 is required for {command_name}\" >&2; exit 1; fi"
        )

    if variant == "graphskills":
        lines.extend(
            [
                "COPY graphskills /opt/graphskills",
                "RUN chmod +x /opt/graphskills/query.py && ln -sf /opt/graphskills/query.py /usr/local/bin/graphskills-query",
                "ENV GRAPHSKILLS_ROOT=/opt/graphskills",
                *instruction_block,
            ]
        )
    elif variant == "vectorskills":
        lines.extend(
            [
                "RUN if command -v python3 >/dev/null 2>&1; then :; elif command -v apt-get >/dev/null 2>&1; then apt-get update && apt-get install -y python3 && rm -rf /var/lib/apt/lists/*; else echo \"python3 is required for vectorskills-query\" >&2; exit 1; fi",
                "COPY vectorskills /opt/graphskills/vectorskills",
                "RUN chmod +x /opt/graphskills/vectorskills/query.py && ln -sf /opt/graphskills/vectorskills/query.py /usr/local/bin/vectorskills-query",
                "ENV GOS_PREBUILT_WORKING_DIR=/opt/graphskills/prebuilt",
                "ENV GOS_WORKING_DIR=/opt/graphskills/runtime",
                *instruction_block,
            ]
        )
    else:
        lines.extend(instruction_block)

    lines.append(GRAPH_MARKER_END)
    return "\n".join(lines)


def patch_dockerfile(dockerfile_path: Path, variant: str) -> None:
    original = dockerfile_path.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"{re.escape(GRAPH_MARKER_START)}.*?{re.escape(GRAPH_MARKER_END)}\n?",
        re.DOTALL,
    )
    updated = re.sub(pattern, "", original)
    updated = re.sub(r"(?m)^\s*COPY\s+skills(?:[^\n]*)\n?", "", updated)
    updated = updated.rstrip()
    block = build_docker_block(variant)
    updated = f"{updated}\n\n{block}\n"
    dockerfile_path.write_text(updated, encoding="utf-8")


def copy_task_tree(source_task_dir: Path, destination_task_dir: Path) -> None:
    if destination_task_dir.exists():
        shutil.rmtree(destination_task_dir)
    shutil.copytree(source_task_dir, destination_task_dir)


def canonical_task_source(task_dir: Path) -> Path:
    """Prefer the stable canonical task under evaluation/skillsbench/tasks.

    Older generated task directories may have stale or partially-corrupted top-level
    files. The canonical task source keeps task.toml/instruction/tests/solution
    authoritative while the generator still rewrites the environment for each method.
    """
    candidate = DEFAULT_TASKS_ROOT / task_dir.name
    if candidate.exists() and (candidate / "task.toml").exists():
        return candidate
    return task_dir


def prepare_allskills_task(
    source_task_dir: Path,
    destination_task_dir: Path,
    skills_root: Path,
) -> None:
    copy_task_tree(source_task_dir, destination_task_dir)

    skills_dir = destination_task_dir / "environment" / "skills"
    if skills_dir.exists():
        shutil.rmtree(skills_dir)

    env_dir = destination_task_dir / "environment"
    for name in ("AGENTS.md", "CLAUDE.md", "GEMINI.md"):
        hardlink_or_copy_file(ALLSKILLS_TEMPLATE_DIR / name, env_dir / name)
    (env_dir / "docker-compose.yaml").write_text(
        render_compose_template(
            ALLSKILLS_TEMPLATE_DIR / "docker-compose.yaml",
            destination_env_dir=env_dir,
            skills_root=skills_root,
        ),
        encoding="utf-8",
    )

    patch_dockerfile(destination_task_dir / "environment" / "Dockerfile", "allskills")


def prepare_graphskills_task(
    source_task_dir: Path,
    destination_task_dir: Path,
    bundle_path: Path,
    vector_store_path: Path,
    skills_root: Path,
    gos_workspace: Path,
) -> None:
    copy_task_tree(source_task_dir, destination_task_dir)
    replace_task_skills(destination_task_dir, BOOTSTRAP_SKILL_DIR)

    env_dir = destination_task_dir / "environment"
    for name in ("AGENTS.md", "CLAUDE.md", "GEMINI.md"):
        hardlink_or_copy_file(GOS_TEMPLATE_DIR / name, env_dir / name)
    (env_dir / "docker-compose.yaml").write_text(
        render_compose_template(
            GOS_TEMPLATE_DIR / "docker-compose.yaml",
            destination_env_dir=env_dir,
            skills_root=skills_root,
            gos_workspace=gos_workspace,
        ),
        encoding="utf-8",
    )

    graphskills_dir = destination_task_dir / "environment" / "graphskills"
    if graphskills_dir.exists():
        shutil.rmtree(graphskills_dir)
    graphskills_dir.mkdir(parents=True, exist_ok=True)

    hardlink_or_copy_file(QUERY_TEMPLATE, graphskills_dir / "query.py")
    hardlink_or_copy_file(bundle_path, graphskills_dir / "bundle.json")
    hardlink_or_copy_file(vector_store_path, graphskills_dir / "vectors.pkl")

    patch_dockerfile(destination_task_dir / "environment" / "Dockerfile", "graphskills")


def prepare_vectorskills_task(
    source_task_dir: Path,
    destination_task_dir: Path,
    metadata_path: Path,
    vector_store_path: Path,
    skills_root: Path,
    gos_workspace: Path,
) -> None:
    copy_task_tree(source_task_dir, destination_task_dir)
    copy_task_tree(VECTOR_BOOTSTRAP_SKILL_DIR, destination_task_dir / "environment" / "skills")

    env_dir = destination_task_dir / "environment"
    for name in ("AGENTS.md", "CLAUDE.md", "GEMINI.md"):
        hardlink_or_copy_file(VECTORSKILLS_TEMPLATE_DIR / name, env_dir / name)
    (env_dir / "docker-compose.yaml").write_text(
        render_compose_template(
            VECTORSKILLS_TEMPLATE_DIR / "docker-compose.yaml",
            destination_env_dir=env_dir,
            skills_root=skills_root,
            gos_workspace=gos_workspace,
        ),
        encoding="utf-8",
    )

    vectorskills_dir = destination_task_dir / "environment" / "vectorskills"
    if vectorskills_dir.exists():
        shutil.rmtree(vectorskills_dir)
    vectorskills_dir.mkdir(parents=True, exist_ok=True)

    hardlink_or_copy_file(VECTOR_QUERY_TEMPLATE, vectorskills_dir / "query.py")
    hardlink_or_copy_file(metadata_path, vectorskills_dir / "metadata.json")
    hardlink_or_copy_file(vector_store_path, vectorskills_dir / "vectors.pkl")

    patch_dockerfile(destination_task_dir / "environment" / "Dockerfile", "vectorskills")


def build_task_list(tasks_root: Path, selected_tasks: list[str]) -> list[Path]:
    tasks = sorted(
        path for path in tasks_root.iterdir() if path.is_dir() and (path / "task.toml").exists()
    )
    if not selected_tasks:
        return tasks

    selected = set(selected_tasks)
    filtered = [path for path in tasks if path.name in selected]
    missing = selected - {path.name for path in filtered}
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise FileNotFoundError(f"Unknown task ids: {missing_text}")
    return filtered


def write_bundle(bundle: dict[str, Any], output_root: Path) -> Path:
    bundle_dir = output_root / "shared"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = bundle_dir / "graphskills_bundle.json"
    bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return bundle_path


def write_vector_metadata(bundle: dict[str, Any], output_root: Path) -> Path:
    bundle_dir = output_root / "shared"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = bundle_dir / "vectorskills_metadata.json"
    metadata = {"skills": bundle["skills"]}
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def write_vector_store(bundle: dict[str, Any], skills_root: Path, gos_workspace: Path, output_root: Path) -> Path:
    try:
        import hnswlib  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "hnswlib is required in the host environment to export vectorskills store"
        ) from exc

    index_path = None
    for candidate in sorted(gos_workspace.glob("entities_hnsw_index_*.bin")):
        index_path = candidate
        break
    if index_path is None:
        raise FileNotFoundError(f"No entities_hnsw_index_*.bin found under {gos_workspace}")

    dim_text = index_path.stem.removeprefix("entities_hnsw_index_")
    dim = int(dim_text)
    index = hnswlib.Index(space="cosine", dim=dim)
    index.load_index(str(index_path))

    id_list = [int(item) for item in index.get_ids_list()]
    vectors = index.get_items(id_list)

    skill_by_path = {
        f"{GRAPH_LIBRARY_PATH}/{skill_file.parent.relative_to(skills_root).as_posix()}/SKILL.md": position
        for position, skill_file in enumerate(sorted(skills_root.rglob("SKILL.md")))
    }

    ordered_rows: list[int] = []
    ordered_ids: list[int] = []
    for skill in bundle["skills"]:
        source_path = skill.get("source_path", "")
        row_idx = skill_by_path.get(source_path)
        if row_idx is None:
            raise KeyError(f"Missing vector row for skill source path: {source_path}")
        ordered_rows.append(row_idx)
        ordered_ids.append(row_idx)

    # The current workspace indexing order matches the sorted SKILL.md order used above.
    # We still reorder explicitly so the exported rows align exactly with metadata.json indices.
    row_to_vector = {row_id: vectors[pos] for pos, row_id in enumerate(id_list)}
    ordered_vectors = [row_to_vector[row_id] for row_id in ordered_rows]

    store_path = output_root / "shared" / "vectorskills_vectors.pkl"
    payload = {
        "dim": dim,
        "ids": ordered_ids,
        "vectors_f32_le": b"".join(memoryview(vector).cast("B") for vector in ordered_vectors),
    }
    import pickle

    with store_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return store_path


def write_manifest(
    output_root: Path,
    *,
    tasks: list[Path],
    bundle: dict[str, Any],
    allskills_dir: Path,
    graphskills_dir: Path,
    vectorskills_dir: Path,
    skills_root: Path,
    skillset_name: str | None,
    gos_workspace: Path,
) -> None:
    manifest = {
        "source_tasks_root": str(DEFAULT_TASKS_ROOT),
        "skills_root": str(skills_root),
        "skillset_name": skillset_name,
        "gos_workspace": str(gos_workspace),
        "task_count": len(tasks),
        "task_ids": [task.name for task in tasks],
        "bundle_metadata": bundle["metadata"],
        "datasets": {
            "allskills": str(allskills_dir),
            "graphskills": str(graphskills_dir),
            "vectorskills": str(vectorskills_dir),
        },
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SkillsBench datasets for all-skills, graph-skills, and vector-only evaluation."
    )
    parser.add_argument(
        "--skillset-name",
        default=DEFAULT_SKILLSET_NAME,
        help=(
            "Named skillset under data/skillsets/. "
            "Ignored when --skills-root is explicitly provided."
        ),
    )
    parser.add_argument(
        "--tasks-root",
        type=Path,
        default=DEFAULT_TASKS_ROOT,
        help="Source tasks directory, typically tasks-no-skills.",
    )
    parser.add_argument(
        "--skills-root",
        type=Path,
        default=DEFAULT_SKILLS_ROOT,
        help="Root directory containing the external all_skills library.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where generated datasets will be written.",
    )
    parser.add_argument(
        "--gos-workspace",
        type=Path,
        default=None,
        help="Prebuilt GoS workspace to mount into generated graphskills tasks.",
    )
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="Optional task id to generate. Repeat to include multiple tasks.",
    )
    parser.add_argument(
        "--skip-allskills",
        action="store_true",
        help="Do not generate the all-skills full-context dataset.",
    )
    parser.add_argument(
        "--skip-graphskills",
        action="store_true",
        help="Do not generate the graph-skills retrieval dataset.",
    )
    parser.add_argument(
        "--skip-vectorskills",
        action="store_true",
        help="Do not generate the vector-only retrieval dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    skills_root = args.skills_root
    skillset_name = None
    if args.skills_root == DEFAULT_SKILLS_ROOT and args.skillset_name:
        skills_root = SKILLSETS_ROOT / args.skillset_name
        skillset_name = args.skillset_name

    skills_root = skills_root.resolve()
    gos_workspace = (
        args.gos_workspace.resolve()
        if args.gos_workspace is not None
        else (REPO_ROOT / "data" / "gos_workspace" / f"{(skillset_name or skills_root.name)}_v1").resolve()
    )

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    source_tasks_root = args.tasks_root.resolve()
    tasks = build_task_list(source_tasks_root, args.task)

    # Some older generated graphskills directories were used as source roots for
    # ablation variants. A subset of those directories ended up with zero-byte
    # top-level task files such as instruction.md/task.toml while their original
    # non-ablation counterparts still had valid task payloads. Repair that here by
    # sourcing each task from the canonical sibling directory when available.
    canonical_tasks_root = source_tasks_root
    source_name = source_tasks_root.name
    if source_name == "tasks_graph_skills":
        parent_name = source_tasks_root.parent.name
        for suffix in ("_noppr", "_embedseed"):
            if parent_name.endswith(suffix):
                candidate_root = source_tasks_root.parent.parent / parent_name[: -len(suffix)] / source_name
                if candidate_root.exists():
                    canonical_tasks_root = candidate_root.resolve()
                break

    repaired_tasks: list[Path] = []
    for task in tasks:
        candidate = canonical_tasks_root / task.name
        instruction_path = task / "instruction.md"
        task_toml_path = task / "task.toml"
        needs_repair = (
            canonical_tasks_root != source_tasks_root
            and candidate.exists()
            and (
                not instruction_path.exists()
                or instruction_path.stat().st_size == 0
                or not task_toml_path.exists()
                or task_toml_path.stat().st_size == 0
            )
        )
        repaired = candidate if needs_repair else task
        repaired_tasks.append(canonical_task_source(repaired.resolve()))

    tasks = repaired_tasks
    bundle = build_graph_bundle(skills_root)
    bundle_path = write_bundle(bundle, output_root)
    vector_metadata_path = None
    vector_store_path = None
    if not args.skip_graphskills or not args.skip_vectorskills:
        vector_store_path = write_vector_store(bundle, skills_root, gos_workspace, output_root)
    if not args.skip_vectorskills:
        vector_metadata_path = write_vector_metadata(bundle, output_root)

    allskills_dir = output_root / "tasks_all_skills"
    graphskills_dir = output_root / "tasks_graph_skills"
    vectorskills_dir = output_root / "tasks_vector_skills"

    if not args.skip_allskills:
        ensure_clean_dir(allskills_dir)
        for task in tasks:
            prepare_allskills_task(
                task,
                allskills_dir / task.name,
                skills_root,
            )

    if not args.skip_graphskills:
        ensure_clean_dir(graphskills_dir)
        for task in tasks:
            prepare_graphskills_task(
                task,
                graphskills_dir / task.name,
                bundle_path,
                vector_store_path,
                skills_root,
                gos_workspace,
            )

    if not args.skip_vectorskills:
        assert vector_metadata_path is not None
        assert vector_store_path is not None
        ensure_clean_dir(vectorskills_dir)
        for task in tasks:
            prepare_vectorskills_task(
                task,
                vectorskills_dir / task.name,
                vector_metadata_path,
                vector_store_path,
                skills_root,
                gos_workspace,
            )

    write_manifest(
        output_root,
        tasks=tasks,
        bundle=bundle,
        allskills_dir=allskills_dir,
        graphskills_dir=graphskills_dir,
        vectorskills_dir=vectorskills_dir,
        skills_root=skills_root,
        skillset_name=skillset_name,
        gos_workspace=gos_workspace,
    )

    print(f"Tasks generated: {len(tasks)}")
    print(f"Graph bundle: {bundle_path}")
    if not args.skip_allskills:
        print(f"All-skills dataset: {allskills_dir}")
    if not args.skip_graphskills:
        print(f"Graph-skills dataset: {graphskills_dir}")
    if not args.skip_vectorskills:
        print(f"Vector-skills dataset: {vectorskills_dir}")


if __name__ == "__main__":
    main()
