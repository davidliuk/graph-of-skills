from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from gos.core.engine import SkillGraphRAG

from .deterministic_edges import _node, recompute_deterministic_edges
from .manifest import atomic_write_json


def _edge_key(edge: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(edge.get("source", "")),
        str(edge.get("target", "")),
        str(edge.get("type", "")),
    )


def _keep_cached_llm_edge(
    engine: SkillGraphRAG,
    edge: dict[str, Any],
    nodes: dict[str, Any],
) -> bool:
    source = str(edge.get("source", ""))
    target = str(edge.get("target", ""))
    if source not in nodes or target not in nodes or source == target:
        return False

    relation_type = str(edge.get("type", "")).strip().lower()
    description = str(edge.get("description", ""))
    evidence = [str(edge.get("evidence", ""))]
    if relation_type == "dependency":
        return engine._llm_dependency_direction_supported(
            source,
            target,
            description,
            nodes,
            evidence,
        )
    if relation_type == "workflow":
        return engine._workflow_direction_supported(
            source,
            target,
            description,
            nodes,
            evidence,
        )
    if relation_type == "semantic":
        return engine._semantic_relation_supported(nodes[source], nodes[target])
    if relation_type == "alternative":
        return engine._alternative_relation_supported(nodes[source], nodes[target])
    return False


def _counts(edges: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(edge.get(field, "")) for edge in edges).items()))


def revalidate_bundle(
    bundle: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replay current deterministic and quality gates over a cached bundle.

    This is deliberately a repair diagnostic, not an independent LLM graph run:
    node extraction and accepted LLM proposals come from the parent bundle.
    """
    metadata = dict(bundle.get("metadata", {}))
    threshold = float(metadata.get("dependency_match_threshold", 0.6))
    engine = object.__new__(SkillGraphRAG)
    engine.config = SimpleNamespace(dependency_match_threshold=threshold)
    nodes = {
        str(skill["name"]): _node(skill)
        for skill in bundle.get("skills", [])
        if str(skill.get("name", ""))
    }

    deterministic = recompute_deterministic_edges(
        bundle.get("skills", []),
        threshold=threshold,
    )
    cached_llm = [
        dict(edge)
        for edge in bundle.get("edges", [])
        if edge.get("provenance") == "llm_validated"
    ]
    accepted_llm: list[dict[str, Any]] = []
    rejected_llm: list[dict[str, Any]] = []
    for edge in cached_llm:
        (accepted_llm if _keep_cached_llm_edge(engine, edge, nodes) else rejected_llm).append(
            edge
        )

    deduplicated: dict[tuple[str, str, str], dict[str, Any]] = {}
    for edge in [*deterministic, *accepted_llm]:
        key = _edge_key(edge)
        existing = deduplicated.get(key)
        if existing is None or (
            float(edge.get("confidence", 0.0)),
            float(edge.get("weight", 0.0)),
        ) > (
            float(existing.get("confidence", 0.0)),
            float(existing.get("weight", 0.0)),
        ):
            deduplicated[key] = edge
    edges = [deduplicated[key] for key in sorted(deduplicated)]

    construction_code_sha256 = engine._construction_code_sha256()
    fingerprint_payload = {
        "parent": metadata.get("graph_fingerprint", ""),
        "construction": construction_code_sha256,
        "edges": [_edge_key(edge) for edge in edges],
    }
    repair_fingerprint = "sha256:" + hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    repaired_metadata = {
        **metadata,
        "graph_source": "cached_llm_quality_gate_replay",
        "graph_fingerprint": repair_fingerprint,
        "parent_graph_fingerprint": str(metadata.get("graph_fingerprint", "")),
        "construction_code_sha256": construction_code_sha256,
        "edge_count": len(edges),
        "deterministic_edge_count": sum(
            edge.get("provenance") == "deterministic_io" for edge in edges
        ),
        "llm_validated_edge_count": sum(
            edge.get("provenance") == "llm_validated" for edge in edges
        ),
        "repair_diagnostic_only": True,
    }
    repaired = {
        "metadata": repaired_metadata,
        "skills": bundle.get("skills", []),
        "edges": edges,
    }
    old_edges = list(bundle.get("edges", []))
    report = {
        "label": "cached LLM quality-gate replay; not an independent graph run",
        "old_edge_count": len(old_edges),
        "new_edge_count": len(edges),
        "old_by_type": _counts(old_edges, "type"),
        "new_by_type": _counts(edges, "type"),
        "old_by_provenance": _counts(old_edges, "provenance"),
        "new_by_provenance": _counts(edges, "provenance"),
        "recomputed_deterministic_edge_count": len(deterministic),
        "cached_llm_edge_count": len(cached_llm),
        "accepted_llm_edge_count": len(accepted_llm),
        "rejected_llm_edge_count": len(rejected_llm),
        "rejected_llm_edges": sorted(rejected_llm, key=_edge_key),
        "deduplicated_edge_count": len(deterministic) + len(accepted_llm) - len(edges),
        "parent_graph_fingerprint": str(metadata.get("graph_fingerprint", "")),
        "repair_fingerprint": repair_fingerprint,
        "construction_code_sha256": construction_code_sha256,
    }
    return repaired, report


def _render_report(report: dict[str, Any]) -> str:
    lines = [
        "# Cached graph quality-gate replay",
        "",
        f"- Label: {report['label']}",
        f"- Edges: {report['old_edge_count']} -> {report['new_edge_count']}",
        f"- Recomputed deterministic edges: {report['recomputed_deterministic_edge_count']}",
        f"- Cached LLM edges accepted/rejected: {report['accepted_llm_edge_count']}/{report['rejected_llm_edge_count']}",
        f"- Deduplicated typed edges: {report['deduplicated_edge_count']}",
        "",
        "## Rejected cached LLM edges",
        "",
        "| Type | Source | Target |",
        "|---|---|---|",
    ]
    lines.extend(
        f"| {edge.get('type', '')} | {edge.get('source', '')} | {edge.get('target', '')} |"
        for edge in report["rejected_llm_edges"]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay current deterministic and LLM quality gates over a bundle."
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    repaired, report = revalidate_bundle(bundle)
    atomic_write_json(args.output, repaired)
    atomic_write_json(args.report, report)
    args.report.with_suffix(".md").write_text(_render_report(report), encoding="utf-8")


if __name__ == "__main__":
    main()
