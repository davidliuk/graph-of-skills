from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import os
import pickle
import shutil
import time
from pathlib import Path
from typing import Any

import hnswlib
import igraph as ig

from gos.core.engine import build_default_embedding_service
from gos.core.schema import SkillNode

from .manifest import atomic_write_json


PERSISTED_AUXILIARY_FILES = (
    "chunks_kv_data.pkl",
    "map_e2r_blob_data.pkl",
    "map_r2c_blob_data.pkl",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _load_legacy_graph(path: Path) -> ig.Graph:
    with gzip.open(path, "rb") as stream:
        graph = pickle.load(stream)  # noqa: S301 - trusted local release artifact
    if not isinstance(graph, ig.Graph):
        raise TypeError(f"Unexpected graph payload: {type(graph)}")
    return graph


def _node_texts(graph: ig.Graph) -> list[str]:
    skill_fields = set(SkillNode.__dataclass_fields__)
    graph_attributes = set(graph.vs.attributes())
    texts = []
    for vertex in graph.vs:
        values = {
            field: vertex[field]
            for field in skill_fields & graph_attributes
            if vertex[field] is not None
        }
        texts.append(SkillNode(**values).to_str())
    return texts


async def prepare_scale_workspace(
    source_workspace: Path,
    target_workspace: Path,
    *,
    embedding_service: Any,
    expected_node_count: int | None = None,
) -> dict[str, Any]:
    source_workspace = source_workspace.resolve()
    target_workspace = target_workspace.resolve()
    if target_workspace.exists():
        raise FileExistsError(f"Target workspace already exists: {target_workspace}")
    staging = target_workspace.parent / f".{target_workspace.name}.preparing"
    if staging.exists():
        raise FileExistsError(
            f"Preserved interrupted preparation exists: {staging}. Inspect it before retrying."
        )
    staging.mkdir(parents=True)

    started = time.perf_counter()
    source_graph_path = source_workspace / "graph_igraph_data.pklz"
    legacy = _load_legacy_graph(source_graph_path)
    if expected_node_count is not None and legacy.vcount() != expected_node_count:
        raise ValueError(
            f"Legacy node count mismatch: expected={expected_node_count}, actual={legacy.vcount()}"
        )

    directed = ig.Graph(directed=True)
    directed.add_vertices(legacy.vcount())
    for attribute in legacy.vs.attributes():
        directed.vs[attribute] = legacy.vs[attribute]
    target_graph_path = staging / "graph_igraph_data.pklz"
    with gzip.open(target_graph_path, "wb") as stream:
        pickle.dump(directed, stream, protocol=pickle.HIGHEST_PROTOCOL)

    for filename in PERSISTED_AUXILIARY_FILES:
        source = source_workspace / filename
        if not source.is_file():
            raise FileNotFoundError(f"Missing legacy workspace artifact: {source}")
        shutil.copy2(source, staging / filename)

    texts = _node_texts(directed)
    embedding_started = time.perf_counter()
    vectors = await embedding_service.encode(texts)
    embedding_seconds = time.perf_counter() - embedding_started
    if len(vectors) != directed.vcount():
        raise ValueError(
            f"Embedding row count mismatch: nodes={directed.vcount()}, rows={len(vectors)}"
        )
    dim = int(getattr(embedding_service, "embedding_dim", vectors.shape[1]))
    if vectors.shape[1] != dim:
        raise ValueError(
            f"Embedding dimension mismatch: configured={dim}, rows={vectors.shape[1]}"
        )

    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(
        max_elements=max(directed.vcount(), 1),
        ef_construction=200,
        M=16,
    )
    if directed.vcount():
        index.add_items(vectors, list(range(directed.vcount())))
    index.save_index(str(staging / f"entities_hnsw_index_{dim}.bin"))
    with (staging / "entities_hnsw_metadata.pkl").open("wb") as stream:
        pickle.dump({}, stream, protocol=pickle.HIGHEST_PROTOCOL)

    usage = getattr(embedding_service, "usage", None)
    report = {
        "schema_version": 1,
        "label": "relation rebuild from frozen normalized nodes",
        "source_workspace": str(source_workspace),
        "source_graph_sha256": _file_sha256(source_graph_path),
        "source_graph_directed": legacy.is_directed(),
        "reused_normalized_nodes": legacy.vcount(),
        "discarded_legacy_edges": legacy.ecount(),
        "target_graph_directed": True,
        "target_initial_edges": 0,
        "embedding_model": str(getattr(embedding_service, "model", "")),
        "embedding_dim": dim,
        "embedding_seconds": embedding_seconds,
        "embedding_usage": usage.to_dict() if usage and hasattr(usage, "to_dict") else {},
        "wall_seconds": time.perf_counter() - started,
    }
    atomic_write_json(staging / "scale_preparation_report.json", report)
    os.replace(staging, target_workspace)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a legacy workspace into a directed, edge-free scale workspace."
    )
    parser.add_argument("--source-workspace", type=Path, required=True)
    parser.add_argument("--target-workspace", type=Path, required=True)
    parser.add_argument("--expected-node-count", type=int)
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    report = await prepare_scale_workspace(
        args.source_workspace,
        args.target_workspace,
        embedding_service=build_default_embedding_service(),
        expected_node_count=args.expected_node_count,
    )
    print(
        "Prepared directed scale workspace: "
        f"nodes={report['reused_normalized_nodes']} "
        f"embedding_seconds={report['embedding_seconds']:.1f} "
        f"wall_seconds={report['wall_seconds']:.1f}"
    )


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()

