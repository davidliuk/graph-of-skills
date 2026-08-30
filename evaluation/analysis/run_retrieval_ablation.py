from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

from evaluation.skillsbench.graphskills_assets import query as retrieval_runtime

from .deterministic_edges import recompute_deterministic_edges
from .manifest import ExperimentManifest, atomic_write_json


DEFAULT_TOP_N = 8
DEFAULT_SEED_TOP_K = 5
DEFAULT_MAX_SKILL_CHARS = 2400
DEFAULT_MAX_CONTEXT_CHARS = 12000


CONDITIONS = {
    "reverse-ppr": {"propagation_mode": "ppr", "reverse_mode": "full"},
    "forward-ppr": {"propagation_mode": "ppr", "reverse_mode": "none"},
    "no-graph": {"propagation_mode": "none", "reverse_mode": "none"},
    "one-hop": {"propagation_mode": "one-hop", "reverse_mode": "full"},
}


def apply_edge_view(bundle: dict[str, Any], edge_view: str) -> dict[str, Any]:
    if edge_view == "full":
        return bundle
    if edge_view != "deterministic-core":
        raise ValueError(f"Unsupported edge view: {edge_view}")
    threshold = float(bundle.get("metadata", {}).get("dependency_match_threshold", 0.6))
    return {
        **bundle,
        "metadata": {**bundle.get("metadata", {}), "edge_view": edge_view},
        "edges": recompute_deterministic_edges(
            bundle.get("skills", []), threshold=threshold
        ),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _git_metadata(repo_root: Path) -> tuple[str, list[str]]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return commit, sorted(line[3:] for line in status if len(line) > 3)


def _bundle_identity_maps(
    bundle: dict[str, Any],
) -> tuple[dict[str, str], dict[str, str]]:
    name_to_directory: dict[str, str] = {}
    source_to_directory: dict[str, str] = {}
    for skill in bundle.get("skills", []):
        directory = str(skill.get("skill_dir") or "")
        source = str(skill.get("source_path") or "")
        if not directory and source:
            directory = Path(source).parent.name
        name = str(skill.get("name") or "")
        if name:
            name_to_directory[name] = directory
        if source:
            source_to_directory[source] = directory
    return name_to_directory, source_to_directory


def score_retrieval_row(
    *,
    task_id: str,
    condition: str,
    coverage: dict[str, Any],
    retrieval: dict[str, Any],
    bundle: dict[str, Any],
    latency_ms: float,
) -> dict[str, Any]:
    name_to_directory, source_to_directory = _bundle_identity_maps(bundle)
    available_oracle_directories = set(coverage.get("exact", [])) | set(
        coverage.get("aliases", {}).values()
    )
    retrieved_directories = []
    for skill in retrieval.get("skills", []):
        source = str(skill.get("source_path") or "")
        directory = source_to_directory.get(source) or Path(source).parent.name
        if directory:
            retrieved_directories.append(directory)
    retrieved_set = set(retrieved_directories)
    oracle_hits = available_oracle_directories & retrieved_set

    dependency_pairs: set[tuple[str, str]] = set()
    for edge in bundle.get("edges", []):
        if edge.get("type") != "dependency":
            continue
        source = name_to_directory.get(str(edge.get("source") or ""), "")
        target = name_to_directory.get(str(edge.get("target") or ""), "")
        if (
            source in available_oracle_directories
            and target in available_oracle_directories
        ):
            dependency_pairs.add((source, target))
    recovered_pairs = {
        pair
        for pair in dependency_pairs
        if pair[0] in retrieved_set and pair[1] in retrieved_set
    }

    oracle_count = int(coverage.get("oracle_count", 0))
    available_count = int(
        coverage.get("available_count", len(available_oracle_directories))
    )
    retrieved_count = len(retrieved_directories)
    return {
        "task_id": task_id,
        "condition": condition,
        "oracle_count": oracle_count,
        "available_oracle_count": available_count,
        "absent_oracle_count": len(coverage.get("absent", [])),
        "retrieved_skill_count": retrieved_count,
        "retrieved_skill_directories": retrieved_directories,
        "oracle_hit_count": len(oracle_hits),
        "oracle_hit_directories": sorted(oracle_hits),
        "full_oracle_recall": len(oracle_hits) / oracle_count if oracle_count else 1.0,
        "available_oracle_recall": (
            len(oracle_hits) / available_count if available_count else 1.0
        ),
        "bundle_precision": len(oracle_hits) / retrieved_count
        if retrieved_count
        else 0.0,
        "full_bundle_complete": len(oracle_hits) == oracle_count,
        "available_bundle_complete": len(oracle_hits) == available_count,
        "oracle_dependency_pair_count": len(dependency_pairs),
        "recovered_oracle_dependency_pair_count": len(recovered_pairs),
        "rendered_context_chars": len(retrieval.get("rendered_context", "")),
        "latency_ms": float(latency_ms),
        "seed_directories": [
            source_to_directory.get(str(seed.get("source_path") or ""), "")
            for seed in retrieval.get("seeds", [])
        ],
    }


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return round(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction, 6)


AVAILABILITY_CONDITIONED_METRICS = frozenset(
    {
        "available_oracle_recall",
        "bundle_precision",
        "available_bundle_complete",
    }
)


def _has_available_oracle(row: dict[str, Any]) -> bool:
    return int(row.get("available_oracle_count", 1)) > 0


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_condition.setdefault(str(row["condition"]), []).append(row)

    aggregates: dict[str, dict[str, Any]] = {}
    for condition, condition_rows in sorted(by_condition.items()):
        available_rows = [row for row in condition_rows if _has_available_oracle(row)]
        pair_count = sum(
            int(row["oracle_dependency_pair_count"]) for row in condition_rows
        )
        recovered_pair_count = sum(
            int(row["recovered_oracle_dependency_pair_count"]) for row in condition_rows
        )
        aggregates[condition] = {
            "task_count": len(condition_rows),
            "available_task_count": len(available_rows),
            "macro_full_oracle_recall": statistics.fmean(
                float(row["full_oracle_recall"]) for row in condition_rows
            ),
            "macro_available_oracle_recall": statistics.fmean(
                float(row["available_oracle_recall"]) for row in available_rows
            )
            if available_rows
            else 0.0,
            "macro_bundle_precision": statistics.fmean(
                float(row["bundle_precision"]) for row in available_rows
            )
            if available_rows
            else 0.0,
            "full_bundle_complete_rate": statistics.fmean(
                bool(row["full_bundle_complete"]) for row in condition_rows
            ),
            "available_bundle_complete_rate": statistics.fmean(
                bool(row["available_bundle_complete"]) for row in available_rows
            )
            if available_rows
            else 0.0,
            "oracle_dependency_pair_count": pair_count,
            "recovered_oracle_dependency_pair_count": recovered_pair_count,
            "oracle_dependency_pair_recovery": (
                recovered_pair_count / pair_count if pair_count else 0.0
            ),
            "mean_retrieved_skill_count": statistics.fmean(
                int(row["retrieved_skill_count"]) for row in condition_rows
            ),
            "mean_rendered_context_chars": statistics.fmean(
                int(row["rendered_context_chars"]) for row in condition_rows
            ),
            "latency_ms_p50": _percentile(
                [float(row["latency_ms"]) for row in condition_rows], 0.5
            ),
            "latency_ms_p95": _percentile(
                [float(row["latency_ms"]) for row in condition_rows], 0.95
            ),
        }
    return aggregates


def paired_comparisons(
    rows: list[dict[str, Any]],
    *,
    reference: str = "reverse-ppr",
    metrics: list[str] | None = None,
    bootstrap_samples: int = 5000,
) -> dict[str, Any]:
    selected_metrics = metrics or [
        "full_oracle_recall",
        "available_oracle_recall",
        "bundle_precision",
        "full_bundle_complete",
        "available_bundle_complete",
    ]
    lookup = {(str(row["condition"]), str(row["task_id"])): row for row in rows}
    conditions = sorted({str(row["condition"]) for row in rows} - {reference})
    task_ids = sorted(
        str(row["task_id"]) for row in rows if row["condition"] == reference
    )
    comparisons: dict[str, Any] = {}
    for baseline in conditions:
        metric_results: dict[str, Any] = {}
        for metric in selected_metrics:
            deltas: list[float] = []
            for task_id in task_ids:
                reference_row = lookup.get((reference, task_id))
                baseline_row = lookup.get((baseline, task_id))
                if (
                    reference_row is None
                    or baseline_row is None
                    or metric not in reference_row
                    or metric not in baseline_row
                    or (
                        metric in AVAILABILITY_CONDITIONED_METRICS
                        and (
                            not _has_available_oracle(reference_row)
                            or not _has_available_oracle(baseline_row)
                        )
                    )
                ):
                    continue
                deltas.append(
                    float(reference_row[metric]) - float(baseline_row[metric])
                )
            seed = int.from_bytes(
                hashlib.sha256(
                    f"{reference}\x1f{baseline}\x1f{metric}".encode("utf-8")
                ).digest()[:8],
                "big",
            )
            rng = random.Random(seed)
            bootstrap_means = []
            if deltas:
                for _ in range(max(bootstrap_samples, 0)):
                    sample = [deltas[rng.randrange(len(deltas))] for _ in deltas]
                    bootstrap_means.append(statistics.fmean(sample))
            epsilon = 1e-12
            metric_results[metric] = {
                "task_count": len(deltas),
                "mean_delta": statistics.fmean(deltas) if deltas else 0.0,
                "wins": sum(delta > epsilon for delta in deltas),
                "ties": sum(abs(delta) <= epsilon for delta in deltas),
                "losses": sum(delta < -epsilon for delta in deltas),
                "bootstrap_ci95": [
                    _percentile(bootstrap_means, 0.025),
                    _percentile(bootstrap_means, 0.975),
                ],
                "bootstrap_samples": max(bootstrap_samples, 0),
            }
        comparisons[f"{reference}-minus-{baseline}"] = metric_results
    return comparisons


def render_results_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Matched Offline Retrieval Ablation",
        "",
        "These are lexical-seed diagnostics over benchmark-derived oracle skill sets; "
        "they are not downstream agent rewards and do not establish human-annotated dependency recall.",
        "",
        "| Condition | Full oracle recall | Available oracle recall | Precision | Available complete | Dependency-pair recovery | Bundle skills | Context chars | Latency p50 / p95 (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, row in result["aggregates"].items():
        lines.append(
            f"| {condition} | {row['macro_full_oracle_recall']:.3f} | "
            f"{row['macro_available_oracle_recall']:.3f} | "
            f"{row['macro_bundle_precision']:.3f} | "
            f"{row['available_bundle_complete_rate']:.3f} | "
            f"{row['oracle_dependency_pair_recovery']:.3f} "
            f"({row['recovered_oracle_dependency_pair_count']}/{row['oracle_dependency_pair_count']}) | "
            f"{row['mean_retrieved_skill_count']:.2f} | "
            f"{row['mean_rendered_context_chars']:.0f} | "
            f"{row['latency_ms_p50']:.2f} / {row['latency_ms_p95']:.2f} |"
        )
    comparisons = result.get("paired_comparisons", {})
    if comparisons:
        lines.extend(
            [
                "",
                "## Paired task-level differences",
                "",
                "| Comparison | Metric | Mean delta | Bootstrap 95% CI | W / T / L |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for comparison, metrics in comparisons.items():
            for metric, values in metrics.items():
                ci = values["bootstrap_ci95"]
                lines.append(
                    f"| {comparison} | {metric} | {values['mean_delta']:.3f} | "
                    f"[{ci[0]:.3f}, {ci[1]:.3f}] | {values['wins']} / "
                    f"{values['ties']} / {values['losses']} |"
                )
    return "\n".join(lines) + "\n"


def run_ablation(
    *,
    bundle: dict[str, Any],
    coverage: dict[str, Any],
    skillset_name: str,
    top_n: int,
    seed_top_k: int,
    max_skill_chars: int,
    max_context_chars: int,
    edge_view: str = "full",
) -> dict[str, Any]:
    bundle = apply_edge_view(bundle, edge_view)
    task_rows = coverage.get("tasks", [])
    rows: list[dict[str, Any]] = []
    if task_rows:
        warmup_query = str(task_rows[0].get("instruction") or "")
        for settings in CONDITIONS.values():
            retrieval_runtime.retrieve(
                bundle,
                warmup_query,
                top_n=top_n,
                seed_top_k=seed_top_k,
                max_skill_chars=max_skill_chars,
                max_context_chars=max_context_chars,
                seed_mode="lexical",
                **settings,
            )

    for condition, settings in CONDITIONS.items():
        for task in task_rows:
            task_coverage = task["skillsets"][skillset_name]
            started = time.perf_counter_ns()
            retrieval = retrieval_runtime.retrieve(
                bundle,
                str(task.get("instruction") or ""),
                top_n=top_n,
                seed_top_k=seed_top_k,
                max_skill_chars=max_skill_chars,
                max_context_chars=max_context_chars,
                seed_mode="lexical",
                **settings,
            )
            latency_ms = (time.perf_counter_ns() - started) / 1_000_000.0
            rows.append(
                score_retrieval_row(
                    task_id=str(task["task_id"]),
                    condition=condition,
                    coverage=task_coverage,
                    retrieval=retrieval,
                    bundle=bundle,
                    latency_ms=latency_ms,
                )
            )
    return {
        "schema_version": 1,
        "label": f"lexical-seed matched offline retrieval diagnostic ({edge_view})",
        "skillset_name": skillset_name,
        "configuration": {
            "seed_mode": "lexical",
            "top_n": top_n,
            "seed_top_k": seed_top_k,
            "max_skill_chars": max_skill_chars,
            "max_context_chars": max_context_chars,
            "conditions": CONDITIONS,
            "edge_view": edge_view,
        },
        "aggregates": aggregate_rows(rows),
        "paired_comparisons": paired_comparisons(rows),
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run matched offline retrieval ablations."
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--skillset-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--seed-top-k", type=int, default=DEFAULT_SEED_TOP_K)
    parser.add_argument("--max-skill-chars", type=int, default=DEFAULT_MAX_SKILL_CHARS)
    parser.add_argument(
        "--max-context-chars", type=int, default=DEFAULT_MAX_CONTEXT_CHARS
    )
    parser.add_argument(
        "--edge-view",
        choices=("full", "deterministic-core"),
        default="full",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    coverage = json.loads(args.coverage.read_text(encoding="utf-8"))
    result = run_ablation(
        bundle=bundle,
        coverage=coverage,
        skillset_name=args.skillset_name,
        top_n=args.top_n,
        seed_top_k=args.seed_top_k,
        max_skill_chars=args.max_skill_chars,
        max_context_chars=args.max_context_chars,
        edge_view=args.edge_view,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = result.pop("rows")
    atomic_write_json(output_dir / "summary.json", result)
    with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    (output_dir / "results.md").write_text(
        render_results_markdown(result), encoding="utf-8"
    )

    repo_root = Path(__file__).resolve().parents[2]
    git_commit, dirty_paths = _git_metadata(repo_root)
    skillset = coverage["skillsets"][args.skillset_name]
    manifest = ExperimentManifest(
        run_id=output_dir.name,
        experiment="matched_offline_retrieval",
        corpus_path=str(skillset["library_path"]),
        corpus_sha256=str(skillset["library_sha256"]),
        task_path=str(coverage["tasks_root"]),
        task_sha256=str(coverage["tasks_sha256"]),
        git_commit=git_commit,
        dirty_paths=dirty_paths,
        configuration={
            **result["configuration"],
            "bundle_path": str(args.bundle.resolve()),
            "bundle_sha256": _file_sha256(args.bundle),
            "graph_fingerprint": bundle.get("metadata", {}).get(
                "graph_fingerprint", ""
            ),
        },
    )
    manifest.write(output_dir / "manifest.json")
    print(render_results_markdown(result), end="")


if __name__ == "__main__":
    main()
