import asyncio

import igraph as ig
import pytest

from fast_graphrag._storage._gdb_igraph import IGraphStorageConfig
from fast_graphrag._storage._namespace import Workspace

from gos.core.schema import SkillEdge, SkillNode
from gos.core.storage import DirectedIGraphStorage, LegacyGraphError


def test_directed_edge_round_trip_preserves_source_target(tmp_path):
    async def scenario():
        workspace = Workspace(str(tmp_path))
        storage = DirectedIGraphStorage(
            config=IGraphStorageConfig(SkillNode, SkillEdge),
            namespace=workspace.make_for("graph"),
        )

        await storage.insert_start()
        await storage.upsert_node(SkillNode(name="producer"), None)
        await storage.upsert_node(SkillNode(name="consumer"), None)
        await storage.insert_edges(
            [
                SkillEdge(
                    source="producer",
                    target="consumer",
                    description="producer emits an artifact consumed by consumer",
                    type="dependency",
                )
            ]
        )
        await storage.insert_done()

        reloaded = DirectedIGraphStorage(
            config=IGraphStorageConfig(SkillNode, SkillEdge),
            namespace=Workspace(str(tmp_path)).make_for("graph"),
        )
        await reloaded.query_start()
        edge = await reloaded.get_edge_by_index(0)

        assert reloaded._graph is not None
        assert reloaded._graph.is_directed()
        assert edge is not None
        assert (edge.source, edge.target) == ("producer", "consumer")

    asyncio.run(scenario())


def test_undirected_legacy_graph_requires_rebuild(tmp_path):
    graph = ig.Graph(directed=False)
    graph.add_vertices(["a", "b"])
    graph.add_edge("a", "b", description="is")
    ig.Graph.write_picklez(graph, str(tmp_path / "graph_igraph_data.pklz"))

    async def scenario():
        storage = DirectedIGraphStorage(
            config=IGraphStorageConfig(SkillNode, SkillEdge),
            namespace=Workspace(str(tmp_path)).make_for("graph"),
        )
        with pytest.raises(LegacyGraphError, match="[Rr]ebuild"):
            await storage.query_start()

    asyncio.run(scenario())
