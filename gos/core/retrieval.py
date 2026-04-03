from __future__ import annotations

from typing import Mapping, Protocol, Sequence

import numpy as np


class NodeLike(Protocol):
    name: str


class EdgeLike(Protocol):
    source: str
    target: str
    type: str
    weight: float


DEFAULT_REVERSE_WEIGHTS: dict[str, float] = {
    "dependency": 1.0,
    "workflow": 0.5,
    "semantic": 0.2,
    "alternative": 0.1,
}


def build_rank_distribution(count: int) -> np.ndarray:
    if count <= 0:
        return np.array([], dtype=float)

    weights = np.array([1.0 / float(index + 1) for index in range(count)], dtype=float)
    return weights / weights.sum()


def build_personalization(
    node_count: int,
    seed_indices: Sequence[int],
    seed_weights: Sequence[float] | None = None,
) -> np.ndarray:
    personalization = np.zeros(node_count, dtype=float)
    if node_count == 0 or not seed_indices:
        return personalization

    if seed_weights is None or len(seed_weights) != len(seed_indices):
        normalized_weights = build_rank_distribution(len(seed_indices))
    else:
        normalized_weights = np.array(seed_weights, dtype=float)
        total = normalized_weights.sum()
        if total <= 0:
            normalized_weights = build_rank_distribution(len(seed_indices))
        else:
            normalized_weights = normalized_weights / total

    for index, weight in zip(seed_indices, normalized_weights):
        if 0 <= index < node_count:
            personalization[index] += float(weight)

    total = personalization.sum()
    if total > 0:
        personalization = personalization / total

    return personalization


def build_transition_matrix(
    nodes: Sequence[NodeLike],
    edges: Sequence[EdgeLike],
    reverse_weights: Mapping[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    node_count = len(nodes)
    transition = np.zeros((node_count, node_count), dtype=float)
    name_to_index = {node.name: index for index, node in enumerate(nodes)}

    if node_count == 0:
        return transition, name_to_index

    reverse_weights = reverse_weights or DEFAULT_REVERSE_WEIGHTS

    for edge in edges:
        source_index = name_to_index.get(edge.source)
        target_index = name_to_index.get(edge.target)
        if source_index is None or target_index is None:
            continue

        forward_weight = max(float(getattr(edge, "weight", 1.0) or 1.0), 0.0)
        transition[source_index, target_index] += forward_weight

        reverse_weight = reverse_weights.get(getattr(edge, "type", "semantic"), 0.0)
        if reverse_weight > 0:
            transition[target_index, source_index] += forward_weight * reverse_weight

    row_sums = transition.sum(axis=1)
    for index in range(node_count):
        if row_sums[index] > 0:
            transition[index, :] = transition[index, :] / row_sums[index]
        else:
            transition[index, index] = 1.0

    return transition, name_to_index


def personalized_pagerank(
    transition: np.ndarray,
    personalization: np.ndarray,
    *,
    damping: float = 0.2,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> np.ndarray:
    if transition.size == 0 or personalization.size == 0:
        return np.array([], dtype=float)

    scores = personalization.copy()
    for _ in range(max_iter):
        next_scores = damping * personalization + (1.0 - damping) * transition.T.dot(scores)
        if np.linalg.norm(next_scores - scores, ord=1) <= tol:
            scores = next_scores
            break
        scores = next_scores

    total = scores.sum()
    if total > 0:
        scores = scores / total

    return scores

