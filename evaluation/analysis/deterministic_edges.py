from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from gos.core.engine import SkillGraphRAG
from gos.core.schema import SkillNode


def _node(skill: dict[str, Any]) -> SkillNode:
    return SkillNode.from_lists(
        name=str(skill["name"]),
        description=str(skill.get("description", "")),
        one_line_capability=str(skill.get("one_line_capability", "")),
        inputs=list(skill.get("inputs", [])),
        outputs=list(skill.get("outputs", [])),
        domain_tags=list(skill.get("domain_tags", [])),
        tooling=list(skill.get("tooling", [])),
        example_tasks=list(skill.get("example_tasks", [])),
        script_entrypoints=list(skill.get("script_entrypoints", [])),
        compatibility=list(skill.get("compatibility", [])),
        allowed_tools=list(skill.get("allowed_tools", [])),
        source_path=str(skill.get("source_path", "")),
        rendered_snippet=str(skill.get("rendered_snippet", "")),
        raw_content=str(skill.get("raw_content", "")),
        skill_id=str(skill.get("skill_id", "")),
    )


def recompute_deterministic_edges(
    skills: list[dict[str, Any]],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    """Re-evaluate the conservative I/O rule over every unordered node pair."""
    engine = object.__new__(SkillGraphRAG)
    engine.config = SimpleNamespace(dependency_match_threshold=threshold)
    nodes = [_node(skill) for skill in skills]
    edges: dict[tuple[str, str, str], dict[str, Any]] = {}
    for index, node in enumerate(nodes):
        for candidate in nodes[index + 1 :]:
            for relation in engine._dependency_edges_for_pair(node, candidate):
                edge = {
                    "source": relation.source,
                    "target": relation.target,
                    "type": relation.type,
                    "weight": float(relation.weight),
                    "confidence": float(relation.confidence),
                    "provenance": relation.provenance,
                    "evidence": relation.evidence,
                    "description": relation.description,
                    "validator_model": relation.validator_model,
                }
                edges[(edge["source"], edge["target"], edge["type"])] = edge
    return [edges[key] for key in sorted(edges)]
