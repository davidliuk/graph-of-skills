from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from .schema import SkillEdge, SkillNode


RELINK_PROGRESS_SCHEMA_VERSION = 1
RELINK_EVENT_SCHEMA_VERSION = 1
USAGE_METRIC_FIELDS = (
    "calls",
    "failures",
    "cache_hits",
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cost_usd",
    "elapsed_seconds",
)
FORBIDDEN_EVENT_KEYS = {
    "api_key",
    "authorization",
    "headers",
    "prompt",
    "raw_content",
    "request_headers",
    "response",
    "system_prompt",
}
SECRET_VALUE_PATTERN = re.compile(
    r"(?i)(?:bearer\s+)?sk-(?:or-v1-)?[a-z0-9_-]{12,}"
)


class RelinkProgressError(RuntimeError):
    """Raised when a relink progress ledger is invalid or incompatible."""


class RelinkProgressMismatch(RelinkProgressError):
    """Raised when resume inputs do not match the persisted run fingerprint."""


@dataclass(frozen=True)
class FocusLinkJob:
    focus_name: str
    focus_index: int
    deterministic_edges: tuple[SkillEdge, ...]
    candidates: tuple[SkillNode, ...]
    candidate_pairs: int


@dataclass(frozen=True)
class FocusLinkResult:
    focus_name: str
    edges: tuple[SkillEdge, ...]
    error: str = ""
    candidate_count: int = 0
    deterministic_edge_count: int = 0
    validated_edge_count: int = 0
    validation_seconds: float = 0.0


@dataclass(frozen=True)
class RelinkResult:
    total_focus_count: int
    resumed_focus_count: int
    processed_focus_count: int
    completed_focus_count: int
    failed_focus: dict[str, str]
    checkpoint_count: int
    edge_count: int
    elapsed_seconds: float


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RelinkProgress:
    schema_version: int
    status: str
    fingerprint: str
    total_focus_nodes: int
    completed_focus_names: list[str] = field(default_factory=list)
    failed_focus: dict[str, str] = field(default_factory=dict)
    checkpoint_every: int = 10
    concurrency: int = 8
    construction: dict[str, int | float] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    persisted_edge_count: int = 0
    checkpoint_count: int = 0
    resumed_focus_count: int = 0
    preparation_seconds: float = 0.0
    validation_write_seconds: float = 0.0
    run_id: str = ""
    attempt_count: int = 0
    last_attempt_id: str = ""
    event_count: int = 0
    updated_at: str = field(default_factory=_utc_now)

    @classmethod
    def new(
        cls,
        *,
        fingerprint: str,
        total_focus_nodes: int,
        concurrency: int,
        checkpoint_every: int,
    ) -> "RelinkProgress":
        return cls(
            schema_version=RELINK_PROGRESS_SCHEMA_VERSION,
            status="running",
            fingerprint=fingerprint,
            total_focus_nodes=total_focus_nodes,
            concurrency=concurrency,
            checkpoint_every=checkpoint_every,
            run_id=str(uuid4()),
        )

    def touch(self) -> None:
        self.updated_at = _utc_now()


def build_relink_fingerprint(
    *,
    nodes: list[SkillNode],
    llm_model: str,
    embedding_model: str,
    prompt_sha256: dict[str, str],
    link_top_k: int,
    relation_min_confidence: float,
    dependency_match_threshold: float,
    type_weights: dict[str, float],
    construction_code_sha256: str,
) -> str:
    node_entries = [
        {
            "name": node.name,
            "raw_content_sha256": hashlib.sha256(
                (node.raw_content or "").encode("utf-8")
            ).hexdigest(),
        }
        for node in sorted(nodes, key=lambda item: item.name)
    ]
    payload = {
        "nodes": node_entries,
        "llm_model": llm_model,
        "embedding_model": embedding_model,
        "prompt_sha256": dict(sorted(prompt_sha256.items())),
        "link_top_k": int(link_top_k),
        "relation_min_confidence": float(relation_min_confidence),
        "dependency_match_threshold": float(dependency_match_threshold),
        "type_weights": dict(sorted(type_weights.items())),
        "construction_code_sha256": str(construction_code_sha256),
    }
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(rendered.encode('utf-8')).hexdigest()}"


def load_relink_progress(path: Path) -> RelinkProgress | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RelinkProgressError(
            f"Cannot read relink progress at `{path}`: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise RelinkProgressError("Relink progress must be a JSON object.")
    if payload.get("schema_version") != RELINK_PROGRESS_SCHEMA_VERSION:
        raise RelinkProgressError(
            "Unsupported relink progress schema; restart the relink operation."
        )
    try:
        return RelinkProgress(**payload)
    except TypeError as exc:
        raise RelinkProgressError(f"Invalid relink progress fields: {exc}") from exc


def write_relink_progress(path: Path, progress: RelinkProgress) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    progress.touch()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(asdict(progress), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def merge_relink_usage(
    prior: dict[str, Any],
    current: dict[str, Any],
) -> dict[str, Any]:
    """Merge cumulative usage from prior and current relink processes."""

    def merge(left: Any, right: Any) -> Any:
        if isinstance(left, dict) and isinstance(right, dict):
            return {
                key: merge(left.get(key), right.get(key))
                if key in left and key in right
                else json.loads(json.dumps(left.get(key, right.get(key))))
                for key in sorted(set(left) | set(right))
            }
        if (
            isinstance(left, (int, float))
            and not isinstance(left, bool)
            and isinstance(right, (int, float))
            and not isinstance(right, bool)
        ):
            return left + right
        return json.loads(json.dumps(right if right is not None else left))

    return merge(prior, current)


def diff_relink_usage(
    prior: dict[str, Any],
    current: dict[str, Any],
) -> dict[str, Any]:
    """Return non-negative numeric usage accumulated since ``prior``."""

    def diff(left: Any, right: Any) -> Any:
        if isinstance(right, dict):
            result = {
                key: diff(left.get(key) if isinstance(left, dict) else None, value)
                for key, value in right.items()
            }
            return {key: value for key, value in result.items() if value != {}}
        if isinstance(right, (int, float)) and not isinstance(right, bool):
            baseline = (
                left
                if isinstance(left, (int, float)) and not isinstance(left, bool)
                else 0
            )
            return max(right - baseline, 0)
        return {}

    return diff(prior, current)


def summarize_relink_usage(usage: dict[str, Any]) -> dict[str, int | float]:
    """Aggregate provider/stage usage into one stable metrics dictionary."""

    totals: dict[str, int | float] = {
        name: 0.0 if name in {"cost_usd", "elapsed_seconds"} else 0
        for name in USAGE_METRIC_FIELDS
    }

    def visit(value: Any) -> None:
        if not isinstance(value, dict):
            return
        for key, nested in value.items():
            if (
                key in totals
                and isinstance(nested, (int, float))
                and not isinstance(nested, bool)
            ):
                totals[key] += nested
            elif isinstance(nested, dict):
                visit(nested)

    visit(usage)
    return totals


def _sanitize_relink_event(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _sanitize_relink_event(nested)
            for key, nested in value.items()
            if str(key).lower() not in FORBIDDEN_EVENT_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_relink_event(item) for item in value]
    if isinstance(value, str):
        return SECRET_VALUE_PATTERN.sub("[REDACTED]", value)
    return value


def summarize_relink_error(exc: Exception) -> str:
    """Return a one-line, secret-free error without provider response bodies."""

    kind = type(exc).__name__
    if kind == "ValidationError":
        return "ValidationError: provider response schema validation failed"
    first_line = str(exc).splitlines()[0].strip() or "operation failed"
    first_line = SECRET_VALUE_PATTERN.sub("[REDACTED]", first_line)
    first_line = re.sub(r"\binput_value\s*=.*$", "", first_line).strip()
    return f"{kind}: {first_line[:240]}"


def append_relink_event(path: Path, event: dict[str, Any]) -> dict[str, Any]:
    """Durably append one secret-free structured relink event."""

    payload = _sanitize_relink_event(event)
    payload.setdefault("schema_version", RELINK_EVENT_SCHEMA_VERSION)
    payload.setdefault("timestamp", _utc_now())
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    return payload
