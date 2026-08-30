#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any

import igraph as ig

from gos.core.parsing import parse_skill_document
from gos.core.schema import SkillEdge


GENERIC_EVIDENCE = {
    "api",
    "data",
    "file",
    "json",
    "object",
    "operation",
    "path",
    "payload",
    "request",
    "response",
    "result",
    "schema",
    "session",
    "tool",
    "value",
}
PREREQUISITE_MARKERS = (
    "depends on",
    "dependency",
    "prerequisite",
    "requires",
)


def _edge_from_raw(graph: ig.Graph, raw_edge: Any) -> SkillEdge:
    attrs = raw_edge.attributes()
    return SkillEdge(
        source=graph.vs[raw_edge.source]["name"],
        target=graph.vs[raw_edge.target]["name"],
        description=attrs.get("description", ""),
        type=attrs.get("type", ""),
        weight=attrs.get("weight", 0.0),
        confidence=attrs.get("confidence", -1.0),
        provenance=attrs.get("provenance", ""),
        evidence=attrs.get("evidence", ""),
        validator_model=attrs.get("validator_model", ""),
        chunks=attrs.get("chunks", []),
    )


def _generic_only_evidence(evidence: str) -> bool:
    tokens = set(re.findall(r"[a-z0-9]+", evidence.lower()))
    return bool(tokens) and not (tokens - GENERIC_EVIDENCE)


def _explicit_prerequisite_audit(
    skillset: Path,
    edge_pairs: set[tuple[str, str]],
    graph_node_names: set[str],
) -> dict[str, Any]:
    documents: dict[str, str] = {}
    for path in sorted(skillset.rglob("SKILL.md")):
        content = path.read_text(encoding="utf-8", errors="replace")
        parsed = parse_skill_document(content, source_path=str(path))
        if parsed is not None and parsed.name in graph_node_names:
            documents[parsed.name] = content

    mentions: list[dict[str, Any]] = []
    names = sorted(documents, key=len, reverse=True)
    for consumer, content in documents.items():
        lowered = content.lower().replace("_", "-")
        for producer in names:
            if producer == consumer:
                continue
            normalized = producer.lower().replace("_", "-")
            for match in re.finditer(re.escape(normalized), lowered):
                left = max(0, match.start() - 120)
                right = min(len(lowered), match.end() + 120)
                context = lowered[left:right]
                if not any(marker in context for marker in PREREQUISITE_MARKERS):
                    continue
                mentions.append(
                    {
                        "producer": producer,
                        "consumer": consumer,
                        "edge_present": (producer, consumer) in edge_pairs,
                    }
                )
                break

    unique = {(item["producer"], item["consumer"]): item for item in mentions}
    records = list(unique.values())
    hits = sum(bool(item["edge_present"]) for item in records)
    return {
        "method": "heuristic explicit-name mentions near prerequisite markers; not gold recall",
        "mentions": len(records),
        "edge_hits": hits,
        "coverage": hits / len(records) if records else None,
        "miss_examples": [item for item in records if not item["edge_present"]][:20],
    }


def analyze_workspace(
    workspace: Path,
    skillset: Path | None = None,
) -> dict[str, Any]:
    graph_path = workspace / "graph_igraph_data.pklz"
    if not graph_path.is_file():
        raise FileNotFoundError(f"Missing graph artifact: {graph_path}")

    graph = ig.Graph.Read_Picklez(str(graph_path))
    if not graph.is_directed():
        raise ValueError(
            f"GoS graph must be directed; rebuild the undirected artifact at {graph_path}."
        )

    edges = [_edge_from_raw(graph, raw_edge) for raw_edge in graph.es]
    identities = Counter((edge.source, edge.target, edge.type) for edge in edges)
    duplicates = sum(count - 1 for count in identities.values() if count > 1)
    if duplicates:
        raise ValueError(f"Graph contains {duplicates} duplicate typed directed edges.")

    by_type = Counter(edge.type for edge in edges)
    by_provenance = Counter(edge.provenance for edge in edges)
    degrees = graph.degree(mode="all") if graph.vcount() else []
    isolates = sum(degree == 0 for degree in degrees)
    components = graph.connected_components(mode="weak") if graph.vcount() else []
    giant_size = max((len(component) for component in components), default=0)
    top_hubs = sorted(
        (
            {"name": graph.vs[index]["name"], "degree": int(degree)}
            for index, degree in enumerate(degrees)
        ),
        key=lambda item: (-item["degree"], item["name"]),
    )[:20]
    suspicious = [
        {
            "source": edge.source,
            "target": edge.target,
            "type": edge.type,
            "evidence": edge.evidence,
        }
        for edge in edges
        if _generic_only_evidence(edge.evidence)
    ]

    report: dict[str, Any] = {
        "integrity": {"valid": True},
        "graph": {
            "directed": True,
            "nodes": graph.vcount(),
            "edges": graph.ecount(),
            "isolates": isolates,
            "isolate_fraction": isolates / graph.vcount() if graph.vcount() else 0.0,
            "weak_components": len(components),
            "giant_component_nodes": giant_size,
            "giant_component_fraction": giant_size / graph.vcount()
            if graph.vcount()
            else 0.0,
            "top_hubs": top_hubs,
        },
        "edges": {
            "by_type": dict(sorted(by_type.items())),
            "by_provenance": dict(sorted(by_provenance.items())),
            "duplicate_typed_directed": duplicates,
            "generic_only_evidence": len(suspicious),
            "generic_only_evidence_examples": suspicious[:20],
        },
    }

    construction_path = workspace / "construction_report.json"
    if construction_path.is_file():
        report["construction"] = json.loads(
            construction_path.read_text(encoding="utf-8")
        )

    if skillset is not None:
        edge_pairs = {(edge.source, edge.target) for edge in edges}
        report["explicit_prerequisites"] = _explicit_prerequisite_audit(
            skillset,
            edge_pairs,
            set(graph.vs["name"]),
        )

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a constructed GoS workspace."
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--skillset", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = analyze_workspace(args.workspace, args.skillset)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
