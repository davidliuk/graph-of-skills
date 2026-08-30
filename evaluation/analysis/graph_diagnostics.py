from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import igraph as ig

from .deterministic_edges import recompute_deterministic_edges
from .manifest import atomic_write_json
from .run_retrieval_ablation import (
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_MAX_SKILL_CHARS,
    DEFAULT_SEED_TOP_K,
    DEFAULT_TOP_N,
    run_ablation,
)


def graph_statistics(bundle: dict[str, Any]) -> dict[str, Any]:
    names = [str(skill["name"]) for skill in bundle.get("skills", [])]
    name_set = set(names)
    edges = bundle.get("edges", [])
    invalid_endpoints = [
        {
            "source": str(edge.get("source", "")),
            "target": str(edge.get("target", "")),
            "type": str(edge.get("type", "")),
        }
        for edge in edges
        if edge.get("source") not in name_set or edge.get("target") not in name_set
    ]
    keys = [
        (str(edge.get("source")), str(edge.get("target")), str(edge.get("type")))
        for edge in edges
    ]
    duplicate_count = len(keys) - len(set(keys))

    graph = ig.Graph(directed=True)
    graph.add_vertices(names)
    valid_edges = [
        (str(edge["source"]), str(edge["target"]))
        for edge in edges
        if edge.get("source") in name_set and edge.get("target") in name_set
    ]
    graph.add_edges(valid_edges)
    weak_components = graph.connected_components(mode="weak") if names else []
    giant_count = max((len(component) for component in weak_components), default=0)
    degrees = graph.degree(mode="all") if names else []
    isolates = sum(degree == 0 for degree in degrees)
    node_count = len(names)
    return {
        "directed": graph.is_directed(),
        "nodes": node_count,
        "edges": len(edges),
        "density": (
            len(edges) / (node_count * (node_count - 1)) if node_count > 1 else 0.0
        ),
        "average_total_degree": statistics_mean(degrees),
        "isolates": isolates,
        "isolate_fraction": isolates / node_count if node_count else 0.0,
        "weak_components": len(weak_components),
        "giant_component_nodes": giant_count,
        "giant_component_fraction": giant_count / node_count if node_count else 0.0,
        "duplicate_typed_directed": duplicate_count,
        "invalid_endpoint_count": len(invalid_endpoints),
        "invalid_endpoint_examples": invalid_endpoints[:10],
        "by_type": dict(
            sorted(Counter(str(edge.get("type", "")) for edge in edges).items())
        ),
        "by_provenance": dict(
            sorted(Counter(str(edge.get("provenance", "")) for edge in edges).items())
        ),
    }


def statistics_mean(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def stratified_edge_sample(
    edges: list[dict[str, Any]],
    *,
    per_stratum: int,
    salt: str,
) -> list[dict[str, Any]]:
    strata: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for edge in edges:
        stratum = f"{edge.get('provenance', '')}/{edge.get('type', '')}"
        identity = "\x1f".join(
            str(edge.get(field, ""))
            for field in ("source", "target", "type", "provenance", "description")
        )
        digest = hashlib.sha256(f"{salt}\x1f{identity}".encode("utf-8")).hexdigest()
        strata[stratum].append((digest, edge))

    selected: list[dict[str, Any]] = []
    for stratum, candidates in sorted(strata.items()):
        for digest, edge in sorted(candidates, key=lambda item: item[0])[:per_stratum]:
            selected.append(
                {
                    "sample_id": digest[:12],
                    "stratum": stratum,
                    **edge,
                }
            )
    return selected


def _edge_keys(edges: list[dict[str, Any]]) -> set[tuple[str, str, str]]:
    return {
        (str(edge["source"]), str(edge["target"]), str(edge["type"])) for edge in edges
    }


def _jaccard(left: set[Any], right: set[Any]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def threshold_analysis(
    bundle: dict[str, Any], thresholds: list[float]
) -> tuple[dict[str, Any], dict[float, list[dict[str, Any]]]]:
    persisted = [
        edge
        for edge in bundle.get("edges", [])
        if edge.get("provenance") == "deterministic_io"
    ]
    persisted_keys = _edge_keys(persisted)
    persisted_dependency_keys = _edge_keys(
        [edge for edge in bundle.get("edges", []) if edge.get("type") == "dependency"]
    )
    llm_dependency_keys = _edge_keys(
        [
            edge
            for edge in bundle.get("edges", [])
            if edge.get("type") == "dependency"
            and edge.get("provenance") == "llm_validated"
        ]
    )
    rows: dict[str, Any] = {}
    edges_by_threshold: dict[float, list[dict[str, Any]]] = {}
    previous_keys: set[tuple[str, str, str]] | None = None
    for threshold in thresholds:
        recomputed = recompute_deterministic_edges(
            bundle.get("skills", []), threshold=threshold
        )
        keys = _edge_keys(recomputed)
        rows[f"{threshold:.3f}"] = {
            "threshold": threshold,
            "deterministic_edge_count": len(recomputed),
            "persisted_edge_count": len(persisted),
            "jaccard_with_persisted": _jaccard(keys, persisted_keys),
            "added_vs_persisted": len(keys - persisted_keys),
            "removed_vs_persisted": len(persisted_keys - keys),
            "llm_shadowed_deterministic_count": len(keys & llm_dependency_keys),
            "missing_from_persisted_dependencies": len(
                keys - persisted_dependency_keys
            ),
            "coverage_by_any_persisted_dependency": (
                len(keys & persisted_dependency_keys) / len(keys) if keys else 1.0
            ),
            "jaccard_with_previous_threshold": (
                _jaccard(keys, previous_keys) if previous_keys is not None else None
            ),
        }
        edges_by_threshold[threshold] = recomputed
        previous_keys = keys
    return rows, edges_by_threshold


def write_annotation_packet(
    bundle: dict[str, Any],
    sample: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    by_name = {str(skill["name"]): skill for skill in bundle.get("skills", [])}
    blind_path = output_dir / "edge_annotation_blind.csv"
    fields = [
        "sample_id",
        "source",
        "target",
        "proposed_type",
        "description",
        "evidence",
        "source_inputs",
        "source_outputs",
        "target_inputs",
        "target_outputs",
        "valid_relation",
        "type_correct",
        "direction_correct",
        "corrected_type",
        "notes",
    ]
    with blind_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for edge in sample:
            source = by_name.get(str(edge["source"]), {})
            target = by_name.get(str(edge["target"]), {})
            writer.writerow(
                {
                    "sample_id": edge["sample_id"],
                    "source": edge["source"],
                    "target": edge["target"],
                    "proposed_type": edge["type"],
                    "description": edge.get("description", ""),
                    "evidence": edge.get("evidence", ""),
                    "source_inputs": " | ".join(source.get("inputs", [])),
                    "source_outputs": " | ".join(source.get("outputs", [])),
                    "target_inputs": " | ".join(target.get("inputs", [])),
                    "target_outputs": " | ".join(target.get("outputs", [])),
                    "valid_relation": "",
                    "type_correct": "",
                    "direction_correct": "",
                    "corrected_type": "",
                    "notes": "",
                }
            )
    atomic_write_json(
        output_dir / "edge_annotation_key.json",
        [
            {
                "sample_id": edge["sample_id"],
                "stratum": edge["stratum"],
                "provenance": edge.get("provenance", ""),
                "confidence": edge.get("confidence", 0.0),
                "validator_model": edge.get("validator_model", ""),
            }
            for edge in sample
        ],
    )
    (output_dir / "annotation_instructions.md").write_text(
        """# Blind Edge Annotation Instructions

Annotate `edge_annotation_blind.csv` without opening `edge_annotation_key.json`.
The CSV intentionally hides provenance, confidence, and validator model.

For each row, use `yes`, `no`, or `uncertain`:

- `valid_relation`: the two skills have a concrete operational relationship, not merely a shared broad container such as DataFrame, report, code, or analysis.
- `type_correct`: the proposed type is the best of dependency, workflow, semantic, or alternative.
- `direction_correct`: for dependency, source produces/provides a concrete prerequisite consumed by target; for workflow, source normally precedes target. For semantic/alternative, enter `n/a` unless the proposed wording asserts a direction.
- `corrected_type`: fill only when the relation is valid but the type is wrong.
- `notes`: briefly identify decisive evidence or the reason for rejection.

Do not infer a dependency solely because both skills use the same file/container type. A dependency requires a plausible producer-to-consumer artifact or prerequisite. A workflow requires a plausible ordered composition. Semantic means related capabilities without interchangeability. Alternative means substantially substitutable ways to achieve the same goal.

After all rows are frozen, join labels to `edge_annotation_key.json` by `sample_id` and aggregate validity/type/direction accuracy by hidden stratum. Preserve `uncertain` separately; do not silently count it as correct.

From the repository root, aggregate a completed packet with:

```bash
uv run python -m evaluation.analysis.aggregate_edge_annotations \
  --annotations edge_annotation_blind.csv \
  --key edge_annotation_key.json \
  --output-dir annotation_summary
```
""",
        encoding="utf-8",
    )


def render_markdown(result: dict[str, Any]) -> str:
    graph = result["graph"]
    lines = [
        "# Repaired Graph Diagnostics",
        "",
        f"Nodes: {graph['nodes']}; edges: {graph['edges']}; directed: {graph['directed']}.",
        f"Density: {graph['density']:.6f}; isolates: {graph['isolates']} "
        f"({graph['isolate_fraction']:.1%}); weak components: {graph['weak_components']}; "
        f"giant component: {graph['giant_component_nodes']}/{graph['nodes']}.",
        "",
        f"Edge types: {graph['by_type']}.",
        f"Edge provenance: {graph['by_provenance']}.",
        "",
        "## Deterministic dependency threshold sensitivity",
        "",
        "| zeta | Edges | Jaccard with deterministic provenance | LLM-shadowed | Missing dependency keys | Coverage by any dependency |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["thresholds"].values():
        lines.append(
            f"| {row['threshold']:.2f} | {row['deterministic_edge_count']} | "
            f"{row['jaccard_with_persisted']:.3f} | "
            f"{row['llm_shadowed_deterministic_count']} | "
            f"{row['missing_from_persisted_dependencies']} | "
            f"{row['coverage_by_any_persisted_dependency']:.3f} |"
        )
    if result.get("retrieval_by_threshold"):
        lines.extend(["", "## Retrieval sensitivity", ""])
        for threshold, aggregates in result["retrieval_by_threshold"].items():
            reverse = aggregates["reverse-ppr"]
            lines.append(
                f"- zeta={threshold}: reverse-PPR available oracle recall "
                f"{reverse['macro_available_oracle_recall']:.3f}, complete rate "
                f"{reverse['available_bundle_complete_rate']:.3f}."
            )
    lines.extend(
        [
            "",
            "The annotation CSV hides provenance, confidence, and validator model; "
            "the separate key is for post-label stratified aggregation.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose a repaired GoS graph bundle."
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--construction-report", type=Path, required=True)
    parser.add_argument("--coverage", type=Path)
    parser.add_argument("--skillset-name")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", action="append", type=float, default=[])
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--seed-top-k", type=int, default=DEFAULT_SEED_TOP_K)
    parser.add_argument("--max-skill-chars", type=int, default=DEFAULT_MAX_SKILL_CHARS)
    parser.add_argument(
        "--max-context-chars", type=int, default=DEFAULT_MAX_CONTEXT_CHARS
    )
    parser.add_argument("--sample-per-stratum", type=int, default=8)
    parser.add_argument("--sample-salt", default="gos-emnlp-rebuttal-edge-audit-v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    construction = json.loads(args.construction_report.read_text(encoding="utf-8"))
    thresholds = sorted(set(args.threshold or [0.4, 0.6, 0.8]))
    threshold_rows, edges_by_threshold = threshold_analysis(bundle, thresholds)
    result: dict[str, Any] = {
        "schema_version": 1,
        "graph_fingerprint": bundle.get("metadata", {}).get("graph_fingerprint", ""),
        "graph": graph_statistics(bundle),
        "construction": construction,
        "thresholds": threshold_rows,
    }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.coverage and args.skillset_name:
        coverage = json.loads(args.coverage.read_text(encoding="utf-8"))
        fixed_edges = [
            edge
            for edge in bundle.get("edges", [])
            if edge.get("provenance") != "deterministic_io"
        ]
        retrieval_by_threshold: dict[str, Any] = {}
        for threshold, deterministic_edges in edges_by_threshold.items():
            threshold_bundle = {**bundle, "edges": [*fixed_edges, *deterministic_edges]}
            retrieval = run_ablation(
                bundle=threshold_bundle,
                coverage=coverage,
                skillset_name=args.skillset_name,
                top_n=args.top_n,
                seed_top_k=args.seed_top_k,
                max_skill_chars=args.max_skill_chars,
                max_context_chars=args.max_context_chars,
            )
            retrieval_by_threshold[f"{threshold:.3f}"] = retrieval["aggregates"]
        result["retrieval_by_threshold"] = retrieval_by_threshold
        result["retrieval_configuration"] = {
            "top_n": args.top_n,
            "seed_top_k": args.seed_top_k,
            "max_skill_chars": args.max_skill_chars,
            "max_context_chars": args.max_context_chars,
        }

    sample = stratified_edge_sample(
        bundle.get("edges", []),
        per_stratum=args.sample_per_stratum,
        salt=args.sample_salt,
    )
    result["annotation_sample"] = {
        "count": len(sample),
        "per_stratum_requested": args.sample_per_stratum,
        "strata": dict(sorted(Counter(edge["stratum"] for edge in sample).items())),
        "salt": args.sample_salt,
    }
    write_annotation_packet(bundle, sample, output_dir)
    atomic_write_json(output_dir / "summary.json", result)
    (output_dir / "results.md").write_text(render_markdown(result), encoding="utf-8")
    print(render_markdown(result), end="")


if __name__ == "__main__":
    main()
