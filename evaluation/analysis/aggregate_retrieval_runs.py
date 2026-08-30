from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from .manifest import atomic_write_json
from .run_retrieval_ablation import (
    AVAILABILITY_CONDITIONED_METRICS,
    aggregate_rows,
    paired_comparisons,
)


SUMMARY_METRICS = (
    "macro_full_oracle_recall",
    "macro_available_oracle_recall",
    "macro_bundle_precision",
    "full_bundle_complete_rate",
    "available_bundle_complete_rate",
    "oracle_dependency_pair_recovery",
)
PAIRED_METRICS = (
    "full_oracle_recall",
    "available_oracle_recall",
    "bundle_precision",
    "full_bundle_complete",
    "available_bundle_complete",
)


def aggregate_retrieval_runs(
    run_rows: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    per_run = {name: aggregate_rows(rows) for name, rows in sorted(run_rows.items())}
    pooled_rows: list[dict[str, Any]] = []
    for run_name, rows in sorted(run_rows.items()):
        for row in rows:
            pooled_rows.append(
                {
                    **row,
                    "graph_run": run_name,
                    "task_id": f"{run_name}/{row['task_id']}",
                }
            )

    task_metric_values: dict[tuple[str, str], dict[str, list[float]]] = {}
    for rows in run_rows.values():
        for row in rows:
            key = (str(row["condition"]), str(row["task_id"]))
            metric_values = task_metric_values.setdefault(
                key, {metric: [] for metric in PAIRED_METRICS}
            )
            for metric in PAIRED_METRICS:
                if (
                    metric in AVAILABILITY_CONDITIONED_METRICS
                    and int(row.get("available_oracle_count", 1)) == 0
                ):
                    continue
                metric_values[metric].append(float(row[metric]))
    task_averaged_rows = [
        {
            "condition": condition,
            "task_id": task_id,
            **{
                metric: statistics.fmean(values)
                for metric, values in metric_values.items()
                if values
            },
        }
        for (condition, task_id), metric_values in sorted(task_metric_values.items())
    ]

    conditions = sorted(
        {condition for aggregates in per_run.values() for condition in aggregates}
    )
    across_runs: dict[str, Any] = {}
    for condition in conditions:
        across_runs[condition] = {}
        for metric in SUMMARY_METRICS:
            values = [
                float(aggregates[condition][metric])
                for aggregates in per_run.values()
                if condition in aggregates
            ]
            across_runs[condition][metric] = {
                "mean": statistics.fmean(values) if values else 0.0,
                "min": min(values, default=0.0),
                "max": max(values, default=0.0),
                "population_stddev": statistics.pstdev(values) if values else 0.0,
            }

    return {
        "run_count": len(run_rows),
        "per_run_aggregates": per_run,
        "across_run_summary": across_runs,
        "pooled_aggregates": aggregate_rows(pooled_rows),
        "pooled_paired_comparisons": paired_comparisons(pooled_rows),
        "task_averaged_paired_comparisons": paired_comparisons(
            task_averaged_rows,
            metrics=list(PAIRED_METRICS),
        ),
    }


def render_markdown(result: dict[str, Any]) -> str:
    available_task_counts = [
        aggregates["reverse-ppr"]["available_task_count"]
        for aggregates in result["per_run_aggregates"].values()
        if "reverse-ppr" in aggregates
    ]
    available_task_count = min(available_task_counts, default=0)
    lines = [
        "# Retrieval Diagnostics Across Independent Graph Builds",
        "",
        "Each task/build pair is retained as a paired unit. Run-level ranges expose graph-construction variance.",
        "",
        "| Condition | Available oracle recall mean [range] | Available completeness mean [range] | Dependency-pair recovery mean [range] |",
        "|---|---:|---:|---:|",
    ]
    for condition, metrics in result["across_run_summary"].items():
        recall = metrics["macro_available_oracle_recall"]
        complete = metrics["available_bundle_complete_rate"]
        dependency = metrics["oracle_dependency_pair_recovery"]
        lines.append(
            f"| {condition} | {recall['mean']:.3f} [{recall['min']:.3f}, {recall['max']:.3f}] | "
            f"{complete['mean']:.3f} [{complete['min']:.3f}, {complete['max']:.3f}] | "
            f"{dependency['mean']:.3f} [{dependency['min']:.3f}, {dependency['max']:.3f}] |"
        )
    lines.extend(
        [
            "",
            "## Per-run reverse PPR",
            "",
            "| Run | Available recall | Available complete | Dependency-pair recovery |",
            "|---|---:|---:|---:|",
        ]
    )
    for run_name, aggregates in result["per_run_aggregates"].items():
        reverse = aggregates["reverse-ppr"]
        lines.append(
            f"| {run_name} | {reverse['macro_available_oracle_recall']:.3f} | "
            f"{reverse['available_bundle_complete_rate']:.3f} | "
            f"{reverse['oracle_dependency_pair_recovery']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Task-averaged paired differences",
            "",
            "The three graph replicates are averaged within each task before "
            f"bootstrapping the {available_task_count} tasks with at least one available oracle skill.",
            "",
            "| Comparison | Available recall delta [95% CI] | Available completeness delta [95% CI] |",
            "|---|---:|---:|",
        ]
    )
    for comparison, metrics in result["task_averaged_paired_comparisons"].items():
        recall = metrics["available_oracle_recall"]
        complete = metrics["available_bundle_complete"]
        lines.append(
            f"| {comparison} | {recall['mean_delta']:.3f} "
            f"[{recall['bootstrap_ci95'][0]:.3f}, {recall['bootstrap_ci95'][1]:.3f}] | "
            f"{complete['mean_delta']:.3f} "
            f"[{complete['bootstrap_ci95'][0]:.3f}, {complete['bootstrap_ci95'][1]:.3f}] |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate retrieval results across graph builds."
    )
    parser.add_argument("--run", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_rows: dict[str, list[dict[str, Any]]] = {}
    input_dirs: dict[str, str] = {}
    for item in args.run:
        name, separator, raw_path = item.partition("=")
        if not separator or not name or not raw_path:
            raise SystemExit(f"Invalid --run {item!r}; expected NAME=DIR")
        run_dir = Path(raw_path)
        run_rows[name] = [
            json.loads(line)
            for line in (run_dir / "rows.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        input_dirs[name] = str(run_dir.resolve())

    result = aggregate_retrieval_runs(run_rows)
    result["input_dirs"] = input_dirs
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "summary.json", result)
    (output_dir / "results.md").write_text(render_markdown(result), encoding="utf-8")
    print(render_markdown(result), end="")


if __name__ == "__main__":
    main()
