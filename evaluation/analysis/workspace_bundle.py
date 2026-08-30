from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path
from typing import Any

import hnswlib
import igraph as ig

from .manifest import validate_workspace_report


GRAPH_LIBRARY_PATH = "/opt/graphskills/skills"


def _attribute(attributes: set[str], item: Any, name: str, default: Any) -> Any:
    return item[name] if name in attributes and item[name] is not None else default


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        candidates = value
    elif isinstance(value, str):
        candidates = value.splitlines()
    elif value is None:
        candidates = []
    else:
        candidates = [value]
    return [str(item).strip() for item in candidates if str(item).strip()]


def _resolve_skill_file(
    source_path: str,
    skills_root: Path,
    files_by_directory: dict[str, list[Path]],
) -> Path:
    parts = Path(source_path).parts
    if skills_root.name in parts:
        offset = parts.index(skills_root.name)
        candidate = skills_root.joinpath(*parts[offset + 1 :])
        if candidate.is_file():
            return candidate

    directory_name = Path(source_path).parent.name
    matches = files_by_directory.get(directory_name, [])
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(
            f"Graph node source {source_path!r} is absent from skillset {skills_root}"
        )
    raise ValueError(
        f"Graph node source {source_path!r} is ambiguous in skillset {skills_root}: "
        f"{[str(path) for path in matches]}"
    )


def _load_graph(workspace: Path) -> ig.Graph:
    graph_path = workspace / "graph_igraph_data.pklz"
    if not graph_path.is_file():
        raise FileNotFoundError(f"Missing graph storage: {graph_path}")
    with gzip.open(graph_path, "rb") as stream:
        graph = pickle.load(stream)  # noqa: S301 - trusted local experiment artifact
    if not isinstance(graph, ig.Graph):
        raise TypeError(f"Unexpected graph payload in {graph_path}: {type(graph)}")
    if not graph.is_directed():
        raise ValueError(f"Legacy undirected graph cannot be exported: {workspace}")
    return graph


def load_workspace_bundle(
    workspace: Path,
    skills_root: Path,
    *,
    library_path: str = GRAPH_LIBRARY_PATH,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    skills_root = skills_root.resolve()
    report = validate_workspace_report(workspace)
    graph = _load_graph(workspace)
    if graph.vcount() != len(list(skills_root.rglob("SKILL.md"))):
        raise ValueError(
            "Persisted graph and selected skillset have different skill counts: "
            f"graph={graph.vcount()}, skillset={len(list(skills_root.rglob('SKILL.md')))}"
        )

    files_by_directory: dict[str, list[Path]] = {}
    for skill_file in sorted(skills_root.rglob("SKILL.md")):
        files_by_directory.setdefault(skill_file.parent.name, []).append(skill_file)

    vertex_attributes = set(graph.vs.attributes())
    skills: list[dict[str, Any]] = []
    for vertex in graph.vs:
        source_path = str(
            _attribute(vertex_attributes, vertex, "source_path", "") or ""
        )
        skill_file = _resolve_skill_file(source_path, skills_root, files_by_directory)
        relative = skill_file.relative_to(skills_root).as_posix()
        raw_content = skill_file.read_text(encoding="utf-8", errors="ignore")
        skills.append(
            {
                "name": str(_attribute(vertex_attributes, vertex, "name", "")),
                "description": str(
                    _attribute(vertex_attributes, vertex, "description", "")
                ),
                "inputs": _string_list(
                    _attribute(vertex_attributes, vertex, "inputs", [])
                ),
                "outputs": _string_list(
                    _attribute(vertex_attributes, vertex, "outputs", [])
                ),
                "compatibility": _string_list(
                    _attribute(vertex_attributes, vertex, "compatibility", [])
                ),
                "allowed_tools": _string_list(
                    _attribute(vertex_attributes, vertex, "allowed_tools", [])
                ),
                "script_entrypoints": _string_list(
                    _attribute(vertex_attributes, vertex, "script_entrypoints", [])
                ),
                "domain_tags": _string_list(
                    _attribute(vertex_attributes, vertex, "domain_tags", [])
                ),
                "tooling": _string_list(
                    _attribute(vertex_attributes, vertex, "tooling", [])
                ),
                "example_tasks": _string_list(
                    _attribute(vertex_attributes, vertex, "example_tasks", [])
                ),
                "source_path": f"{library_path}/{relative}",
                "skill_dir": skill_file.parent.relative_to(skills_root).as_posix(),
                "rendered_snippet": str(
                    _attribute(vertex_attributes, vertex, "rendered_snippet", "")
                ),
                "raw_content": raw_content,
                "skill_id": str(
                    _attribute(vertex_attributes, vertex, "skill_id", source_path)
                ),
                "graph_vertex_id": int(vertex.index),
            }
        )

    edge_attributes = set(graph.es.attributes())
    edge_map: dict[tuple[str, str, str], dict[str, Any]] = {}
    canonicalized_undirected_edge_count = 0
    for edge in graph.es:
        source = str(graph.vs[edge.source]["name"])
        target = str(graph.vs[edge.target]["name"])
        relation_type = str(_attribute(edge_attributes, edge, "type", ""))
        if relation_type in {"semantic", "alternative"}:
            canonical_source, canonical_target = sorted((source, target))
            canonicalized_undirected_edge_count += int(
                (canonical_source, canonical_target) != (source, target)
            )
            source, target = canonical_source, canonical_target
        payload = {
            "source": source,
            "target": target,
            "type": relation_type,
            "weight": float(_attribute(edge_attributes, edge, "weight", 1.0)),
            "confidence": float(_attribute(edge_attributes, edge, "confidence", 0.0)),
            "provenance": str(_attribute(edge_attributes, edge, "provenance", "")),
            "evidence": str(_attribute(edge_attributes, edge, "evidence", "")),
            "description": str(_attribute(edge_attributes, edge, "description", "")),
            "validator_model": str(
                _attribute(edge_attributes, edge, "validator_model", "")
            ),
        }
        key = (source, target, relation_type)
        existing = edge_map.get(key)
        if existing is None or (payload["confidence"], payload["weight"]) > (
            existing["confidence"],
            existing["weight"],
        ):
            edge_map[key] = payload
    edges = [edge_map[key] for key in sorted(edge_map)]

    progress_path = workspace / "relink_progress.json"
    progress = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if progress_path.is_file()
        else {}
    )
    configuration = report.get("configuration", {})
    return {
        "metadata": {
            "version": 3,
            "graph_source": "persisted_directed_workspace",
            "graph_fingerprint": str(progress.get("fingerprint", "")),
            "skill_count": len(skills),
            "edge_count": len(edges),
            "raw_persisted_edge_count": graph.ecount(),
            "canonicalized_undirected_edge_count": (
                canonicalized_undirected_edge_count
            ),
            "canonical_deduplicated_edge_count": graph.ecount() - len(edges),
            "undirected_relation_export_policy": (
                "lexicographically_sort_semantic_and_alternative_endpoints"
            ),
            "library_root": library_path,
            "ppr_damping": float(configuration.get("ppr_damping", 0.2)),
            "ppr_max_iter": int(configuration.get("ppr_max_iter", 50)),
            "ppr_tolerance": float(configuration.get("ppr_tolerance", 1e-6)),
            "dependency_match_threshold": float(
                configuration.get("dependency_match_threshold", 0.6)
            ),
        },
        "skills": skills,
        "edges": edges,
    }


def write_workspace_vector_store(
    bundle: dict[str, Any],
    workspace: Path,
    output_path: Path,
) -> Path:
    index_paths = sorted(workspace.glob("entities_hnsw_index_*.bin"))
    if len(index_paths) != 1:
        raise FileNotFoundError(
            f"Expected exactly one entities_hnsw_index_*.bin under {workspace}; "
            f"found {len(index_paths)}"
        )
    index_path = index_paths[0]
    dim = int(index_path.stem.removeprefix("entities_hnsw_index_"))
    index = hnswlib.Index(space="cosine", dim=dim)
    index.load_index(str(index_path))

    graph_ids = [int(skill["graph_vertex_id"]) for skill in bundle["skills"]]
    available_ids = {int(item) for item in index.get_ids_list()}
    missing = sorted(set(graph_ids) - available_ids)
    if missing:
        raise ValueError(f"HNSW index is missing graph vertex IDs: {missing}")
    vectors = index.get_items(graph_ids)

    payload = {
        "dim": dim,
        # Runtime IDs index the bundle rows, not graph vertex IDs.
        "ids": list(range(len(graph_ids))),
        "graph_vertex_ids": graph_ids,
        "vectors_f32_le": b"".join(memoryview(vector).cast("B") for vector in vectors),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return output_path
