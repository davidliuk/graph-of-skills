from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path
from typing import Any

from gos.core.engine import SkillGraphRAG
from gos.core.schema import SkillNode

from .deterministic_edges import _node
from .manifest import atomic_write_json
from .workspace_bundle import load_workspace_bundle


def _sample_indices(nodes: list[SkillNode], sample_size: int) -> list[int]:
    ranked = sorted(
        range(len(nodes)),
        key=lambda index: hashlib.sha256(nodes[index].name.encode("utf-8")).hexdigest(),
    )
    return ranked[: min(max(sample_size, 0), len(nodes))]


def benchmark_prefilter(
    nodes: list[SkillNode],
    *,
    sample_size: int,
    candidate_top_k: int,
) -> dict[str, Any]:
    engine = object.__new__(SkillGraphRAG)
    sample = _sample_indices(nodes, sample_size)

    started = time.perf_counter()
    brute = {
        index: engine._lexical_candidate_scores_for_node(
            nodes[index], nodes, index, candidate_top_k
        )
        for index in sample
    }
    brute_seconds = time.perf_counter() - started

    started = time.perf_counter()
    indexes = engine._build_pair_evidence_indexes(nodes)
    build_seconds = time.perf_counter() - started
    candidate_sets = {
        index: engine._evidence_candidate_indices_for_node(
            nodes[index], indexes, node_index=index
        )
        for index in sample
    }
    started = time.perf_counter()
    optimized = {
        index: engine._lexical_candidate_scores_for_node(
            nodes[index],
            nodes,
            index,
            candidate_top_k,
            candidate_sets[index],
        )
        for index in sample
    }
    optimized_scoring_seconds = time.perf_counter() - started

    mismatches = [index for index in sample if brute[index] != optimized[index]]
    brute_pair_count = len(sample) * max(len(nodes) - 1, 0)
    prefiltered_pair_count = sum(len(values) for values in candidate_sets.values())
    return {
        "schema_version": 1,
        "node_count": len(nodes),
        "sample_count": len(sample),
        "sample_names": [nodes[index].name for index in sample],
        "candidate_top_k": candidate_top_k,
        "exact_output_match": not mismatches,
        "mismatch_names": [nodes[index].name for index in mismatches],
        "brute_pair_scores": brute_pair_count,
        "prefiltered_pair_scores": prefiltered_pair_count,
        "pair_score_reduction": (
            1.0 - prefiltered_pair_count / brute_pair_count
            if brute_pair_count
            else 0.0
        ),
        "mean_prefilter_candidates": (
            prefiltered_pair_count / len(sample) if sample else 0.0
        ),
        "brute_scoring_seconds": brute_seconds,
        "index_build_seconds": build_seconds,
        "optimized_scoring_seconds": optimized_scoring_seconds,
        "optimized_total_seconds": build_seconds + optimized_scoring_seconds,
        "measured_speedup": (
            brute_seconds / (build_seconds + optimized_scoring_seconds)
            if build_seconds + optimized_scoring_seconds > 0
            else 0.0
        ),
    }


def render_markdown(result: dict[str, Any]) -> str:
    return (
        "# Exact Evidence-Prefilter Benchmark\n\n"
        f"Nodes: {result['node_count']}; fixed-hash sample: {result['sample_count']}; "
        f"exact ranked-output match: {result['exact_output_match']}.\n\n"
        f"- Pair scores: {result['brute_pair_scores']} brute force versus "
        f"{result['prefiltered_pair_scores']} after the exact evidence prefilter "
        f"({result['pair_score_reduction']:.1%} reduction).\n"
        f"- Mean prefiltered candidates: {result['mean_prefilter_candidates']:.1f}.\n"
        f"- Brute scoring: {result['brute_scoring_seconds']:.3f}s.\n"
        f"- Inverted-index build: {result['index_build_seconds']:.3f}s; optimized "
        f"scoring: {result['optimized_scoring_seconds']:.3f}s; total: "
        f"{result['optimized_total_seconds']:.3f}s.\n"
        f"- Measured speedup on this sample: {result['measured_speedup']:.2f}x.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark exact pair prefiltering")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--skills-root", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--candidate-top-k", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    bundle = load_workspace_bundle(args.workspace, args.skills_root)
    nodes = [_node(skill) for skill in bundle["skills"]]
    result = benchmark_prefilter(
        nodes,
        sample_size=args.sample_size,
        candidate_top_k=args.candidate_top_k,
    )
    if not result["exact_output_match"]:
        raise RuntimeError(f"Prefilter changed ranked candidates: {result['mismatch_names']}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / "summary.json", result)
    markdown = render_markdown(result)
    (args.output_dir / "results.md").write_text(markdown, encoding="utf-8")
    print(markdown, end="")


if __name__ == "__main__":
    main()
