import unittest
from dataclasses import dataclass

from gos.core.retrieval import (
    build_personalization,
    build_transition_matrix,
    personalized_pagerank,
)


@dataclass
class Node:
    name: str


@dataclass
class Edge:
    source: str
    target: str
    type: str
    weight: float


class RetrievalTest(unittest.TestCase):
    def test_ppr_recovers_prerequisite_via_reverse_dependency_weight(self) -> None:
        nodes = [Node("read_csv"), Node("analyze_trend"), Node("plot_chart")]
        edges = [
            Edge("read_csv", "analyze_trend", "dependency", 1.0),
            Edge("analyze_trend", "plot_chart", "workflow", 0.8),
        ]

        transition, _ = build_transition_matrix(nodes, edges)
        personalization = build_personalization(len(nodes), [1])
        scores = personalized_pagerank(transition, personalization)

        self.assertGreater(scores[0], 0.0)
        self.assertGreater(scores[1], scores[2])


if __name__ == "__main__":
    unittest.main()
