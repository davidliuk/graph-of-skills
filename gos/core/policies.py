import asyncio
from typing import Iterable
from dataclasses import dataclass
from fast_graphrag._policies._graph_upsert import (
    DefaultGraphUpsertPolicy,
    DefaultNodeUpsertPolicy,
    DefaultEdgeUpsertPolicy,
)
from fast_graphrag._storage._base import BaseGraphStorage
from fast_graphrag._llm import BaseLLMService, format_and_send_prompt
from fast_graphrag._types import TIndex, TId
from .schema import SkillNode, SkillEdge, GOSRelation, GOSRelationList
from .prompts import PROMPTS


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
    pass
