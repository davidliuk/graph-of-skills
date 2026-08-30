from typing import Iterable
from dataclasses import dataclass
from fast_graphrag._policies._graph_upsert import (
    DefaultGraphUpsertPolicy,
    DefaultNodeUpsertPolicy,
    DefaultEdgeUpsertPolicy,
)
from fast_graphrag._storage._base import BaseGraphStorage
from fast_graphrag._llm import BaseLLMService
from fast_graphrag._types import TIndex, TId
from .schema import SkillNode, SkillEdge


@dataclass
class SkillGraphUpsertPolicy(DefaultGraphUpsertPolicy[SkillNode, SkillEdge, TId]):
    async def __call__(
        self,
        llm: BaseLLMService,
        target: BaseGraphStorage[SkillNode, SkillEdge, TId],
        source_nodes: Iterable[SkillNode],
        source_edges: Iterable[SkillEdge],
    ) -> tuple[
        BaseGraphStorage[SkillNode, SkillEdge, TId],
        Iterable[tuple[TIndex, SkillNode]],
        Iterable[tuple[TIndex, SkillEdge]],
    ]:
        # 1. Filter source_edges to ensure source and target exist
        # Get existing node names for validation
        node_names = {n.name for n in source_nodes}
        existing_node_names = set()
        node_count = await target.node_count()
        for i in range(node_count):
            node = await target.get_node_by_index(i)
            if node:
                existing_node_names.add(node.name)

        all_valid_node_names = node_names | existing_node_names

        valid_source_edges = [
            e
            for e in source_edges
            if e.source in all_valid_node_names and e.target in all_valid_node_names
        ]

        # 2. Standard Upsert for extracted nodes and edges
        target, upserted_nodes = await self._nodes_upsert(llm, target, source_nodes)
        target, upserted_edges = await self._edges_upsert(
            llm, target, valid_source_edges
        )

        return target, upserted_nodes, upserted_edges


@dataclass
class SkillNodeUpsertPolicy(DefaultNodeUpsertPolicy[SkillNode, TId]):
    pass


@dataclass
class SkillEdgeUpsertPolicy(DefaultEdgeUpsertPolicy[SkillEdge, TId]):
    async def __call__(
        self,
        llm: BaseLLMService,
        target: BaseGraphStorage[SkillNode, SkillEdge, TId],
        source_edges: Iterable[SkillEdge],
    ) -> tuple[
        BaseGraphStorage[SkillNode, SkillEdge, TId],
        Iterable[tuple[TIndex, SkillEdge]],
    ]:
        upserted: list[tuple[TIndex, SkillEdge]] = []

        for edge in source_edges:
            pair_edges = list(await target.get_edges(edge.source, edge.target))
            if edge.type == "workflow" and any(
                existing.type == "dependency" for existing, _ in pair_edges
            ):
                continue

            if edge.type == "dependency":
                dominated_indices = [
                    index
                    for existing, index in pair_edges
                    if existing.type == "workflow"
                ]
                if dominated_indices:
                    await target.delete_edges_by_index(dominated_indices)
                    pair_edges = list(await target.get_edges(edge.source, edge.target))

            matching = [
                (existing, index)
                for existing, index in pair_edges
                if existing.type == edge.type
            ]
            if not matching:
                index = await target.upsert_edge(edge=edge, edge_index=None)
                upserted.append((index, edge))
                continue

            strongest, index = max(
                matching,
                key=lambda item: (item[0].confidence, item[0].weight),
            )
            if (edge.confidence, edge.weight) > (
                strongest.confidence,
                strongest.weight,
            ):
                await target.upsert_edge(edge=edge, edge_index=index)
                upserted.append((index, edge))
            else:
                upserted.append((index, strongest))

        return target, upserted
