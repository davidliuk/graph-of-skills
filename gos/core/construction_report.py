from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from .prompts import PROMPTS
from .relink import load_relink_progress, summarize_relink_usage


@dataclass
class ConstructionCounters:
    wall_time_seconds: float = 0.0
    focus_nodes: int = 0
    candidate_pairs: int = 0
    validator_requests: int = 0
    submitted_candidates: int = 0
    returned_relations: int = 0
    accepted_relations: int = 0
    rejected_relations: int = 0
    duplicate_edges_dropped: int = 0


def _git_revision() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
            cwd=Path(__file__).resolve().parents[2],
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip()


def _usage_dict(service: Any) -> dict[str, Any]:
    usage = getattr(service, "usage", None)
    if usage is None or not hasattr(usage, "to_dict"):
        return {}
    return usage.to_dict()


async def build_construction_report(engine: Any) -> dict[str, Any]:
    await engine.state_manager.query_start()
    try:
        nodes = await engine._load_all_nodes()
        edges = await engine._load_all_edges()
        raw_graph = engine.state_manager.graph_storage._graph
        directed = bool(raw_graph is not None and raw_graph.is_directed())
    finally:
        await engine.state_manager.query_done()

    by_type = Counter(edge.type for edge in edges)
    by_provenance = Counter(edge.provenance for edge in edges)
    identities = Counter((edge.source, edge.target, edge.type) for edge in edges)
    duplicates = sum(count - 1 for count in identities.values() if count > 1)
    relink_progress = load_relink_progress(
        Path(engine.working_dir) / "relink_progress.json"
    )
    construction = (
        dict(relink_progress.construction)
        if relink_progress is not None and relink_progress.construction
        else asdict(engine.construction_counters)
    )
    usage = (
        dict(relink_progress.usage)
        if relink_progress is not None and relink_progress.usage
        else {
            "llm": _usage_dict(engine.llm_service),
            "embedding": _usage_dict(engine.config.embedding_service),
        }
    )
    usage_totals = summarize_relink_usage(usage)
    wall_seconds = max(float(construction.get("wall_time_seconds", 0.0)), 0.0)
    preparation_seconds = (
        max(float(relink_progress.preparation_seconds), 0.0)
        if relink_progress is not None
        else 0.0
    )
    checkpoint_write_seconds = (
        max(float(relink_progress.validation_write_seconds), 0.0)
        if relink_progress is not None
        else 0.0
    )
    validation_and_wait_seconds = max(
        wall_seconds - preparation_seconds - checkpoint_write_seconds,
        0.0,
    )
    completed_focus_count = (
        len(relink_progress.completed_focus_names)
        if relink_progress is not None
        else int(construction.get("focus_nodes", 0))
    )
    validator_requests = int(construction.get("validator_requests", 0))

    return {
        "schema_version": 1,
        "git_commit": _git_revision(),
        "models": {
            "llm": str(getattr(engine.llm_service, "model", "")),
            "embedding": str(getattr(engine.config.embedding_service, "model", "")),
        },
        "configuration": {
            "llm_temperature": getattr(engine.llm_service, "temperature", None),
            "link_top_k": engine.config.link_top_k,
            "dependency_match_threshold": engine.config.dependency_match_threshold,
            "relation_min_confidence": engine.config.relation_min_confidence,
            "use_full_markdown": engine.config.use_full_markdown,
            "enable_semantic_linking": engine.config.enable_semantic_linking,
            "openrouter_response_cache": bool(
                getattr(engine.llm_service, "response_cache", False)
            ),
        },
        "prompt_sha256": {
            name: hashlib.sha256(str(prompt).encode("utf-8")).hexdigest()
            for name, prompt in sorted(PROMPTS.items())
            if name in {"skill_extraction_system", "search_and_link_system"}
        },
        "graph": {"directed": directed},
        "nodes": {"total": len(nodes)},
        "edges": {
            "total": len(edges),
            "by_type": dict(sorted(by_type.items())),
            "by_provenance": dict(sorted(by_provenance.items())),
            "duplicate_typed_directed": duplicates,
        },
        "construction": construction,
        "usage": usage,
        "usage_totals": usage_totals,
        "timing": {
            "wall_seconds": wall_seconds,
            "preparation_seconds": preparation_seconds,
            "checkpoint_write_seconds": checkpoint_write_seconds,
            "validation_and_wait_seconds": validation_and_wait_seconds,
            "provider_elapsed_seconds_sum": usage_totals["elapsed_seconds"],
        },
        "throughput": {
            "completed_focus_per_second": (
                completed_focus_count / wall_seconds if wall_seconds > 0 else 0.0
            ),
            "validator_requests_per_second": (
                validator_requests / wall_seconds if wall_seconds > 0 else 0.0
            ),
            "persisted_edges_per_second": (
                len(edges) / wall_seconds if wall_seconds > 0 else 0.0
            ),
            "provider_parallelism_factor": (
                float(usage_totals["elapsed_seconds"]) / wall_seconds
                if wall_seconds > 0
                else 0.0
            ),
        },
        "observability": (
            {
                "run_id": relink_progress.run_id,
                "attempt_count": relink_progress.attempt_count,
                "last_attempt_id": relink_progress.last_attempt_id,
                "event_count": relink_progress.event_count,
                "event_log": "relink_events.jsonl",
            }
            if relink_progress is not None
            else {}
        ),
        "relink": (
            {
                "status": relink_progress.status,
                "concurrency": relink_progress.concurrency,
                "checkpoint_every": relink_progress.checkpoint_every,
                "checkpoint_count": relink_progress.checkpoint_count,
                "resumed_focus_count": relink_progress.resumed_focus_count,
                "completed_focus_count": len(relink_progress.completed_focus_names),
                "failed_focus": dict(sorted(relink_progress.failed_focus.items())),
                "persisted_edge_count": relink_progress.persisted_edge_count,
                "preparation_seconds": relink_progress.preparation_seconds,
                "validation_write_seconds": relink_progress.validation_write_seconds,
            }
            if relink_progress is not None
            else {}
        ),
    }


async def write_construction_report(engine: Any, path: Path) -> dict[str, Any]:
    report = await build_construction_report(engine)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return report
