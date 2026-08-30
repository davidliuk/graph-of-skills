from __future__ import annotations

import asyncio
import gzip
import pickle
from dataclasses import asdict
from pathlib import Path

import hnswlib
import igraph as ig
import numpy as np

from evaluation.analysis.prepare_scale_workspace import prepare_scale_workspace
from gos.core.schema import SkillNode


class FakeEmbeddingService:
    model = "fake-embedding"
    embedding_dim = 3

    async def encode(self, texts, model=None):
        return np.array(
            [[float(index + 1), 1.0, 0.0] for index, _ in enumerate(texts)],
            dtype=np.float32,
        )


def _write_legacy_workspace(path: Path) -> None:
    path.mkdir()
    graph = ig.Graph(directed=False)
    nodes = [
        SkillNode.from_lists(name="beta", description="Beta skill"),
        SkillNode.from_lists(name="alpha", description="Alpha skill"),
    ]
    graph.add_vertices(2)
    for index, node in enumerate(nodes):
        graph.vs[index].update_attributes(**asdict(node))
    graph.add_edge(0, 1)
    with gzip.open(path / "graph_igraph_data.pklz", "wb") as stream:
        pickle.dump(graph, stream)
    for name in (
        "chunks_kv_data.pkl",
        "map_e2r_blob_data.pkl",
        "map_r2c_blob_data.pkl",
    ):
        with (path / name).open("wb") as stream:
            pickle.dump({}, stream)


def test_prepare_scale_workspace_converts_nodes_and_rebuilds_vector_ids(tmp_path: Path):
    source = tmp_path / "legacy"
    target = tmp_path / "repaired"
    _write_legacy_workspace(source)

    report = asyncio.run(
        prepare_scale_workspace(
            source,
            target,
            embedding_service=FakeEmbeddingService(),
            expected_node_count=2,
        )
    )

    with gzip.open(target / "graph_igraph_data.pklz", "rb") as stream:
        graph = pickle.load(stream)
    assert graph.is_directed()
    assert graph.vcount() == 2
    assert graph.ecount() == 0
    assert graph.vs[0]["name"] == "beta"

    index = hnswlib.Index(space="cosine", dim=3)
    index.load_index(str(target / "entities_hnsw_index_3.bin"))
    assert sorted(index.get_ids_list()) == [0, 1]
    assert report["source_graph_directed"] is False
    assert report["reused_normalized_nodes"] == 2
    assert report["discarded_legacy_edges"] == 1
    assert report["embedding_model"] == "fake-embedding"

