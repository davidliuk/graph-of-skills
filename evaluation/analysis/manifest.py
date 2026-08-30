from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA_VERSION = 1
SENSITIVE_KEY_PARTS = ("api_key", "apikey", "authorization", "password", "secret", "token")


def sha256_tree(root: Path) -> str:
    """Hash relative paths and contents for every regular file below *root*."""
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Hash root is not a directory: {root}")

    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _without_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _without_secrets(item)
            for key, item in value.items()
            if not any(part in str(key).lower() for part in SENSITIVE_KEY_PARTS)
        }
    if isinstance(value, (list, tuple)):
        return [_without_secrets(item) for item in value]
    return value


def validate_workspace_report(
    workspace: Path,
    *,
    expected_skill_count: int | None = None,
) -> dict[str, Any]:
    report_path = workspace / "construction_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Missing construction report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))

    if report.get("graph", {}).get("directed") is not True:
        raise ValueError(f"Workspace is not a repaired directed graph: {workspace}")

    node_section = report.get("nodes", {})
    node_count = node_section.get("total", node_section.get("count"))
    if expected_skill_count is not None and int(node_count or -1) != expected_skill_count:
        raise ValueError(
            "Workspace skill count does not match corpus: "
            f"expected={expected_skill_count}, report={node_count}"
        )
    return report


@dataclass(frozen=True)
class ExperimentManifest:
    run_id: str
    experiment: str
    corpus_path: str
    corpus_sha256: str
    task_path: str
    task_sha256: str
    git_commit: str
    dirty_paths: list[str] = field(default_factory=list)
    configuration: dict[str, Any] = field(default_factory=dict)
    workspace_path: str = ""
    graph_fingerprint: str = ""
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return _without_secrets(asdict(self))

    def write(self, path: Path) -> None:
        atomic_write_json(path, self.to_dict())

