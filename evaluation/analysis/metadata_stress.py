from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .manifest import atomic_write_json
from .run_retrieval_ablation import apply_edge_view, run_ablation


BROAD_DESCRIPTION = (
    "General-purpose data file processing, analysis, automation, and tool workflow."
)


def corrupt_bundle(
    bundle: dict[str, Any],
    *,
    mode: str,
    fraction: float,
    salt: str,
) -> tuple[dict[str, Any], list[str]]:
    if mode not in {"drop-io", "drop-description", "broad-description"}:
        raise ValueError(f"Unsupported corruption mode: {mode}")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("Corruption fraction must be between 0 and 1")
    skills = [dict(skill) for skill in bundle.get("skills", [])]
    ranked = sorted(
        (
            hashlib.sha256(
                f"{salt}\x1f{skill.get('name', '')}".encode("utf-8")
            ).hexdigest(),
            index,
        )
        for index, skill in enumerate(skills)
    )
    selected_count = int(len(skills) * fraction)
    if fraction > 0 and skills:
        selected_count = max(selected_count, 1)
    selected_indices = {index for _, index in ranked[:selected_count]}
    selected_names = sorted(str(skills[index].get("name", "")) for index in selected_indices)

    for index in selected_indices:
        skill = skills[index]
        if mode == "drop-io":
            skill["inputs"] = []
            skill["outputs"] = []
        elif mode == "drop-description":
            skill["description"] = ""
            skill["rendered_snippet"] = ""
        elif mode == "broad-description":
            skill["description"] = BROAD_DESCRIPTION
            skill["rendered_snippet"] = BROAD_DESCRIPTION

    return (
        {
            **bundle,
            "metadata": {
                **bundle.get("metadata", {}),
                "corruption": {"mode": mode, "fraction": fraction, "salt": salt},
            },
            "skills": skills,
        },
        selected_names,
    )


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Deterministic-Core Metadata Robustness",
        "",
        "This stress test recomputes deterministic I/O edges only; it does not simulate a fresh LLM validation pass.",
        "",
        "| Mode | Fraction | Deterministic edges | Reverse available recall | Delta | Complete rate | Delta | Dependency pairs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["conditions"]:
        lines.append(
            f"| {row['mode']} | {row['fraction']:.0%} | {row['deterministic_edges']} | "
            f"{row['reverse_available_oracle_recall']:.3f} | {row['recall_delta']:+.3f} | "
            f"{row['reverse_available_complete_rate']:.3f} | {row['complete_delta']:+.3f} | "
            f"{row['recovered_dependency_pairs']}/{row['eligible_dependency_pairs']} "
            f"({row['reverse_dependency_pair_recovery']:.3f}) |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic metadata stress tests.")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--skillset-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fraction", action="append", type=float, default=[])
    parser.add_argument("--salt", default="gos-emnlp-metadata-stress-v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    coverage = json.loads(args.coverage.read_text(encoding="utf-8"))
    baseline_result = run_ablation(
        bundle=bundle,
        coverage=coverage,
        skillset_name=args.skillset_name,
        top_n=5,
        seed_top_k=4,
        max_skill_chars=1800,
        max_context_chars=9000,
        edge_view="deterministic-core",
    )
    baseline = baseline_result["aggregates"]["reverse-ppr"]
    conditions = []
    for mode in ("drop-io", "drop-description", "broad-description"):
        for fraction in sorted(set(args.fraction or [0.1, 0.25, 0.5])):
            corrupted, selected = corrupt_bundle(
                bundle, mode=mode, fraction=fraction, salt=args.salt
            )
            deterministic_bundle = apply_edge_view(corrupted, "deterministic-core")
            run = run_ablation(
                bundle=corrupted,
                coverage=coverage,
                skillset_name=args.skillset_name,
                top_n=5,
                seed_top_k=4,
                max_skill_chars=1800,
                max_context_chars=9000,
                edge_view="deterministic-core",
            )
            reverse = run["aggregates"]["reverse-ppr"]
            conditions.append(
                {
                    "mode": mode,
                    "fraction": fraction,
                    "selected_node_count": len(selected),
                    "selected_node_names": selected,
                    "deterministic_edges": len(deterministic_bundle["edges"]),
                    "reverse_available_oracle_recall": reverse[
                        "macro_available_oracle_recall"
                    ],
                    "recall_delta": reverse["macro_available_oracle_recall"]
                    - baseline["macro_available_oracle_recall"],
                    "reverse_available_complete_rate": reverse[
                        "available_bundle_complete_rate"
                    ],
                    "complete_delta": reverse["available_bundle_complete_rate"]
                    - baseline["available_bundle_complete_rate"],
                    "reverse_dependency_pair_recovery": reverse[
                        "oracle_dependency_pair_recovery"
                    ],
                    "eligible_dependency_pairs": reverse[
                        "oracle_dependency_pair_count"
                    ],
                    "recovered_dependency_pairs": reverse[
                        "recovered_oracle_dependency_pair_count"
                    ],
                }
            )
    result = {
        "schema_version": 1,
        "label": "deterministic-core controlled metadata stress",
        "salt": args.salt,
        "baseline": baseline,
        "conditions": conditions,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / "summary.json", result)
    (args.output_dir / "results.md").write_text(render_markdown(result), encoding="utf-8")
    print(render_markdown(result), end="")


if __name__ == "__main__":
    main()
