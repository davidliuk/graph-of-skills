from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path
from typing import Any

import igraph as ig

from .manifest import atomic_write_json


def _sum_usage(usage: dict[str, Any], metric: str) -> float:
    total = 0.0
    for value in usage.values():
        if isinstance(value, dict):
            if metric in value and isinstance(value[metric], (int, float)):
                total += float(value[metric])
            else:
                total += _sum_usage(value, metric)
    return total


def _nonempty(vertex: Any, attributes: set[str], field: str) -> bool:
    return field in attributes and bool(str(vertex[field] or "").strip())


def summarize_scale_workspace(name: str, workspace: Path) -> dict[str, Any]:
    workspace = workspace.resolve()
    with gzip.open(workspace / "graph_igraph_data.pklz", "rb") as stream:
        graph = pickle.load(stream)  # noqa: S301 - trusted local experiment artifact
    if not isinstance(graph, ig.Graph):
        raise TypeError(f"Unexpected graph payload under {workspace}: {type(graph)}")
    report = json.loads(
        (workspace / "construction_report.json").read_text(encoding="utf-8")
    )
    preparation_path = workspace / "scale_preparation_report.json"
    preparation = (
        json.loads(preparation_path.read_text(encoding="utf-8"))
        if preparation_path.is_file()
        else {}
    )
    attributes = set(graph.vs.attributes())
    node_count = graph.vcount()
    inputs_count = sum(_nonempty(v, attributes, "inputs") for v in graph.vs)
    outputs_count = sum(_nonempty(v, attributes, "outputs") for v in graph.vs)
    semantic_count = sum(
        any(
            _nonempty(v, attributes, field)
            for field in ("domain_tags", "tooling", "example_tasks")
        )
        for v in graph.vs
    )
    complete_count = sum(
        _nonempty(v, attributes, "inputs")
        and _nonempty(v, attributes, "outputs")
        and any(
            _nonempty(v, attributes, field)
            for field in ("domain_tags", "tooling", "example_tasks")
        )
        for v in graph.vs
    )

    prep_usage = preparation.get("embedding_usage", {})
    relink_usage = report.get("usage_totals", {})
    construction = report.get("construction", {})
    configuration = report.get("configuration", {})
    edges = report.get("edges", {})
    link_top_k = int(configuration.get("link_top_k", 0))
    submitted_candidates = int(construction.get("submitted_candidates", 0))
    max_submitted_candidates = node_count * link_top_k
    prep_calls = int(_sum_usage(prep_usage, "calls"))
    prep_input_tokens = int(_sum_usage(prep_usage, "input_tokens"))
    prep_cost = _sum_usage(prep_usage, "cost_usd")
    relink_calls = int(relink_usage.get("calls", 0))
    relink_input_tokens = int(relink_usage.get("input_tokens", 0))
    relink_cached_input_tokens = int(relink_usage.get("cached_input_tokens", 0))
    relink_output_tokens = int(relink_usage.get("output_tokens", 0))
    relink_reasoning_tokens = int(relink_usage.get("reasoning_tokens", 0))
    relink_cost = float(relink_usage.get("cost_usd", 0.0))
    edge_count = int(edges.get("total", graph.ecount()))
    return {
        "name": name,
        "workspace": str(workspace),
        "node_source": preparation.get("label", "rich normalized nodes"),
        "nodes": node_count,
        "inputs_present_nodes": inputs_count,
        "outputs_present_nodes": outputs_count,
        "semantic_present_nodes": semantic_count,
        "io_semantic_complete_nodes": complete_count,
        "io_semantic_complete_fraction": complete_count / node_count if node_count else 0.0,
        "edges": edge_count,
        "edge_density": edge_count / (node_count * (node_count - 1)) if node_count > 1 else 0.0,
        "edges_by_type": edges.get("by_type", {}),
        "edges_by_provenance": edges.get("by_provenance", {}),
        "link_top_k": link_top_k,
        "validator_requests": int(construction.get("validator_requests", 0)),
        "submitted_candidates": submitted_candidates,
        "submitted_candidate_upper_bound": max_submitted_candidates,
        "submitted_candidate_utilization": (
            submitted_candidates / max_submitted_candidates
            if max_submitted_candidates
            else 0.0
        ),
        "relation_preparation_seconds": float(
            report.get("timing", {}).get("preparation_seconds", 0.0)
        ),
        "relation_wall_seconds": float(
            report.get("timing", {}).get("wall_seconds", 0.0)
        ),
        "node_embedding_wall_seconds": float(preparation.get("wall_seconds", 0.0)),
        "total_wall_seconds": float(preparation.get("wall_seconds", 0.0))
        + float(report.get("timing", {}).get("wall_seconds", 0.0)),
        "node_embedding_calls": prep_calls,
        "relink_calls": relink_calls,
        "total_calls": prep_calls + relink_calls,
        "total_input_tokens": prep_input_tokens + relink_input_tokens,
        "total_cached_input_tokens": int(_sum_usage(prep_usage, "cached_input_tokens"))
        + relink_cached_input_tokens,
        "total_output_tokens": relink_output_tokens,
        "total_reasoning_tokens": relink_reasoning_tokens,
        "total_cost_usd": prep_cost + relink_cost,
        "attempt_failures": int(relink_usage.get("failures", 0)),
        "remaining_failed_focus": len(
            report.get("relink", {}).get("failed_focus", {})
        ),
        "cache_hits": int(relink_usage.get("cache_hits", 0))
        + int(_sum_usage(prep_usage, "cache_hits")),
    }


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Offline Construction Scale and Cost",
        "",
        "Scale workspaces above 200 reuse released frozen normalized nodes; metadata completeness is reported because the node populations are not homogeneous.",
        "",
        "| Run | N | Complete metadata | Edges | Density | Validators | Submitted / Nk | Prep | Total wall | Calls | Input / output tokens | Cached / reasoning | Cost | Failures / remain | Cache hits |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['nodes']} | {row['io_semantic_complete_nodes']}/{row['nodes']} "
            f"({row['io_semantic_complete_fraction']:.1%}) | {row['edges']} | "
            f"{row['edge_density']:.5f} | {row['validator_requests']} | "
            f"{row['submitted_candidates']}/{row['submitted_candidate_upper_bound']} | "
            f"{row['relation_preparation_seconds']:.1f}s | {row['total_wall_seconds']:.1f}s | "
            f"{row['total_calls']} | {row['total_input_tokens']} / {row['total_output_tokens']} | "
            f"{row['total_cached_input_tokens']} / {row['total_reasoning_tokens']} | "
            f"${row['total_cost_usd']:.4f} | {row['attempt_failures']} / "
            f"{row['remaining_failed_focus']} | {row['cache_hits']} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize construction scale runs.")
    parser.add_argument("--workspace", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for item in args.workspace:
        name, separator, raw_path = item.partition("=")
        if not separator or not name or not raw_path:
            raise SystemExit(f"Invalid --workspace {item!r}; expected NAME=PATH")
        rows.append(summarize_scale_workspace(name, Path(raw_path)))
    rows.sort(key=lambda row: (row["nodes"], row["name"]))
    result = {"schema_version": 1, "runs": rows}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / "summary.json", result)
    (args.output_dir / "results.md").write_text(render_markdown(rows), encoding="utf-8")
    print(render_markdown(rows), end="")


if __name__ == "__main__":
    main()
