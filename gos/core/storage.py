from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import igraph as ig
import numpy as np
from scipy.sparse import csr_matrix

from fast_graphrag._exceptions import InvalidStorageError
from fast_graphrag._storage._gdb_igraph import IGraphStorage
from fast_graphrag._types import TIndex
from fast_graphrag._utils import logger


class LegacyGraphError(InvalidStorageError):
    """Raised when an old graph has already lost edge direction or type data."""


@dataclass
class DirectedIGraphStorage(IGraphStorage):
    """igraph storage with a strict directed-graph persistence contract."""

    def _validate_loaded_graph(self, graph: ig.Graph, path: str) -> ig.Graph:
        if not graph.is_directed():
            raise LegacyGraphError(
                f"Legacy undirected GoS graph at `{path}` cannot preserve dependency "
                "direction. Rebuild the workspace with `gos index ... --clear`."
            )
        return graph

    def _load_directed_or_empty(self, *, operation: str) -> ig.Graph:
        if not self.namespace:
            logger.debug(f"Creating new volatile directed graph for {operation}.")
            return ig.Graph(directed=True)

        graph_file_name = self.namespace.get_load_path(self.RESOURCE_NAME)
        if not graph_file_name:
            logger.info("No graph data file found; loading an empty directed graph.")
            return ig.Graph(directed=True)

        try:
            graph = ig.Graph.Read_Picklez(graph_file_name)
        except Exception as exc:
            message = f"Error loading graph from `{graph_file_name}`: {exc}"
            logger.error(message)
            raise InvalidStorageError(message) from exc

        logger.debug(f"Loaded directed graph storage `{graph_file_name}`.")
        return self._validate_loaded_graph(graph, graph_file_name)

    async def _insert_start(self) -> None:
        self._graph = self._load_directed_or_empty(operation="insert")

    async def _query_start(self) -> None:
        if not self.namespace:
            raise InvalidStorageError("Loading a persisted graph requires a namespace.")
        self._graph = self._load_directed_or_empty(operation="query")

    async def are_neighbours(self, source_node: Any, target_node: Any) -> bool:
        return (
            self._graph.get_eid(  # type: ignore[union-attr]
                self._vertex_index(source_node),
                self._vertex_index(target_node),
                directed=True,
                error=False,
            )
            != -1
        )

    def _vertex_index(self, node: Any) -> int:
        if isinstance(node, int):
            return node
        return self._graph.vs.find(name=node).index  # type: ignore[union-attr]

    async def _get_edge_indices(
        self,
        source_node: Any,
        target_node: Any,
    ) -> Iterable[TIndex]:
        source_index = self._vertex_index(source_node)
        target_index = self._vertex_index(target_node)
        edges = self._graph.es.select(  # type: ignore[union-attr]
            _source=source_index,
            _target=target_index,
        )
        return (edge.index for edge in edges)

    async def score_nodes(self, initial_weights: csr_matrix | None) -> csr_matrix:
        if self._graph is None or self._graph.vcount() == 0:
            logger.info("Trying to score nodes in an empty graph.")
            return csr_matrix((1, 0))

        reset_prob = (
            initial_weights.toarray().flatten() if initial_weights is not None else None
        )
        scores = self._graph.personalized_pagerank(
            damping=self.config.ppr_damping,
            directed=True,
            reset=reset_prob,
        )
        return csr_matrix(np.asarray(scores, dtype=np.float32).reshape(1, -1))
