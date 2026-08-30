from __future__ import annotations

import argparse
import itertools
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from .deterministic_edges import recompute_deterministic_edges
from .graph_diagnostics import graph_statistics
from .manifest import atomic_write_json
from .workspace_bundle import load_workspace_bundle


NORMALIZED_NODE_FIELDS = (
    "description",
    "inputs",
    "outputs",
    "domain_tags",
    "tooling",
    "example_tasks",
    "script_entrypoints",
)


def _typed_keys(
    bundle: dict[str, Any], provenance: str | None = None
) -> set[tuple[str, str, str]]:
    return {
        (str(edge["source"]), str(edge["target"]), str(edge["type"]))
        for edge in bundle.get("edges", [])
        if provenance is None or edge.get("provenance") == provenance
    }


def _jaccard(left: set[Any], right: set[Any]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _normalized_field_values(skill: dict[str, Any], field: str) -> set[str]:
    value = skill.get(field, [])
    values = value if isinstance(value, (list, tuple, set)) else [value]
    return {
        " ".join(str(item).lower().split())
        for item in values
        if str(item or "").strip()
    }


def compare_normalized_nodes(
    bundles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    indexed = {
        run: {str(skill["name"]): skill for skill in bundle.get("skills", [])}
        for run, bundle in bundles.items()
    }
    common_names = (
        set.intersection(*(set(skills) for skills in indexed.values()))
        if indexed
        else set()
    )
    run_pairs = list(itertools.combinations(sorted(indexed), 2))
    fields: dict[str, Any] = {}
    for field in NORMALIZED_NODE_FIELDS:
        jaccards: list[float] = []
        exact: list[bool] = []
        for left_name, right_name in run_pairs:
            for skill_name in sorted(common_names):
                left = _normalized_field_values(indexed[left_name][skill_name], field)
                right = _normalized_field_values(indexed[right_name][skill_name], field)
                jaccards.append(_jaccard(left, right))
                exact.append(left == right)
        all_runs_exact_count = 0
        for skill_name in sorted(common_names):
            values = [
                _normalized_field_values(indexed[run][skill_name], field)
                for run in sorted(indexed)
            ]
            all_runs_exact_count += int(all(value == values[0] for value in values[1:]))
        fields[field] = {
            "pairwise_mean_jaccard": statistics.fmean(jaccards) if jaccards else 1.0,
            "pairwise_exact_fraction": statistics.fmean(exact) if exact else 1.0,
            "all_runs_exact_count": all_runs_exact_count,
            "all_runs_exact_fraction": (
                all_runs_exact_count / len(common_names) if common_names else 1.0
            ),
        }
    return {
        "run_node_counts": {
            run: len(skills) for run, skills in sorted(indexed.items())
        },
        "common_node_count": len(common_names),
        "fields": fields,
    }


def compare_bundles(bundles: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pairwise: list[dict[str, Any]] = []
    for left_name, right_name in itertools.combinations(sorted(bundles), 2):
        left = bundles[left_name]
        right = bundles[right_name]
        left_all = _typed_keys(left)
        right_all = _typed_keys(right)
        left_det = _typed_keys(left, "deterministic_io")
        right_det = _typed_keys(right, "deterministic_io")
        left_llm = _typed_keys(left, "llm_validated")
        right_llm = _typed_keys(right, "llm_validated")
        left_deterministic_candidates = {
            (edge["source"], edge["target"], edge["type"])
            for edge in recompute_deterministic_edges(
                left.get("skills", []),
                threshold=float(
                    left.get("metadata", {}).get("dependency_match_threshold", 0.6)
                ),
            )
        }
        right_deterministic_candidates = {
            (edge["source"], edge["target"], edge["type"])
            for edge in recompute_deterministic_edges(
                right.get("skills", []),
                threshold=float(
                    right.get("metadata", {}).get("dependency_match_threshold", 0.6)
                ),
            )
        }
        reversed_same_type = {
            key for key in left_all - right_all if (key[1], key[0], key[2]) in right_all
        }
        pairwise.append(
            {
                "left": left_name,
                "right": right_name,
                "left_edges": len(left_all),
                "right_edges": len(right_all),
                "typed_directed_intersection": len(left_all & right_all),
                "typed_directed_union": len(left_all | right_all),
                "typed_directed_jaccard": _jaccard(left_all, right_all),
                "deterministic_typed_directed_jaccard": _jaccard(left_det, right_det),
                "deterministic_candidate_jaccard": _jaccard(
                    left_deterministic_candidates,
                    right_deterministic_candidates,
                ),
                "llm_typed_directed_jaccard": _jaccard(left_llm, right_llm),
                "reversed_same_type_count": len(reversed_same_type),
            }
        )

    frequency: Counter[tuple[str, str, str]] = Counter()
    for bundle in bundles.values():
        frequency.update(_typed_keys(bundle))
    run_count = len(bundles)
    consensus = {
        "unique_typed_directed_edges": len(frequency),
        "present_in_all_runs": sum(count == run_count for count in frequency.values()),
        "present_in_one_run": sum(count == 1 for count in frequency.values()),
        "frequency_distribution": dict(sorted(Counter(frequency.values()).items())),
    }
    return {
        "run_count": run_count,
        "node_normalization": compare_normalized_nodes(bundles),
        "runs": {
            name: graph_statistics(bundle) for name, bundle in sorted(bundles.items())
        },
        "pairwise": pairwise,
        "consensus": consensus,
    }


def _report_metadata(workspace: Path) -> dict[str, Any]:
    report = json.loads(
        (workspace / "construction_report.json").read_text(encoding="utf-8")
    )
    progress = json.loads(
        (workspace / "relink_progress.json").read_text(encoding="utf-8")
    )
    return {
        "workspace": str(workspace.resolve()),
        "fingerprint": progress.get("fingerprint", ""),
        "status": progress.get("status", ""),
        "cache_hits": report.get("usage_totals", {}).get("cache_hits", 0),
        "calls": report.get("usage_totals", {}).get("calls", 0),
        "input_tokens": report.get("usage_totals", {}).get("input_tokens", 0),
        "output_tokens": report.get("usage_totals", {}).get("output_tokens", 0),
        "reasoning_tokens": report.get("usage_totals", {}).get("reasoning_tokens", 0),
        "cost_usd": report.get("usage_totals", {}).get("cost_usd", 0.0),
        "wall_seconds": report.get("timing", {}).get("wall_seconds", 0.0),
        "failed_focus_count": len(report.get("relink", {}).get("failed_focus", {})),
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Independent Graph-Construction Stability",
        "",
        "All runs must share one construction fingerprint and report zero complete-response cache hits.",
        "",
        "| Pair | Typed directed Jaccard | Deterministic provenance Jaccard | Deterministic candidate Jaccard | LLM-validated Jaccard | Reversed same-type edges |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["comparison"]["pairwise"]:
        lines.append(
            f"| {row['left']} vs {row['right']} | {row['typed_directed_jaccard']:.3f} | "
            f"{row['deterministic_typed_directed_jaccard']:.3f} | "
            f"{row['deterministic_candidate_jaccard']:.3f} | "
            f"{row['llm_typed_directed_jaccard']:.3f} | "
            f"{row['reversed_same_type_count']} |"
        )
    lines.extend(
        [
            "",
            "| Run | Edges | Calls | Input tokens | Output tokens | Cost | Wall time | Cache hits | Failures |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, metadata in result["run_metadata"].items():
        stats = result["comparison"]["runs"][name]
        lines.append(
            f"| {name} | {stats['edges']} | {metadata['calls']} | "
            f"{metadata['input_tokens']} | {metadata['output_tokens']} | "
            f"${metadata['cost_usd']:.4f} | {metadata['wall_seconds']:.1f}s | "
            f"{metadata['cache_hits']} | {metadata['failed_focus_count']} |"
        )
    jaccards = [
        row["typed_directed_jaccard"] for row in result["comparison"]["pairwise"]
    ]
    if jaccards:
        lines.extend(
            [
                "",
                f"Mean typed-directed pairwise Jaccard: {statistics.fmean(jaccards):.3f}.",
            ]
        )
    lines.extend(
        [
            "",
            "## Normalized-node stability",
            "",
            "| Field | Pairwise set Jaccard | Pairwise exact fraction | Exact in all runs |",
            "|---|---:|---:|---:|",
        ]
    )
    node_stability = result["comparison"]["node_normalization"]
    for field, row in node_stability["fields"].items():
        lines.append(
            f"| {field} | {row['pairwise_mean_jaccard']:.3f} | "
            f"{row['pairwise_exact_fraction']:.3f} | "
            f"{row['all_runs_exact_count']}/{node_stability['common_node_count']} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare independent repaired graph runs."
    )
    parser.add_argument("--skills-root", type=Path, required=True)
    parser.add_argument(
        "--workspace",
        action="append",
        required=True,
        metavar="NAME=PATH",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspaces: dict[str, Path] = {}
    for item in args.workspace:
        name, separator, raw_path = item.partition("=")
        if not separator or not name or not raw_path:
            raise SystemExit(f"Invalid --workspace {item!r}; expected NAME=PATH")
        workspaces[name] = Path(raw_path)

    bundles = {
        name: load_workspace_bundle(workspace, args.skills_root)
        for name, workspace in sorted(workspaces.items())
    }
    result = {
        "schema_version": 1,
        "comparison": compare_bundles(bundles),
        "run_metadata": {
            name: _report_metadata(workspace)
            for name, workspace in sorted(workspaces.items())
        },
    }
    fingerprints = {
        metadata["fingerprint"] for metadata in result["run_metadata"].values()
    }
    if len(fingerprints) != 1:
        raise ValueError(
            f"Runs do not share one construction fingerprint: {fingerprints}"
        )
    cache_hits = sum(
        int(metadata["cache_hits"]) for metadata in result["run_metadata"].values()
    )
    if cache_hits:
        raise ValueError(f"Runs include {cache_hits} complete-response cache hits")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "summary.json", result)
    (output_dir / "results.md").write_text(render_markdown(result), encoding="utf-8")
    print(render_markdown(result), end="")


if __name__ == "__main__":
    main()
