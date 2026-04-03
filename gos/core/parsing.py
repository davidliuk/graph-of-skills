from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
import re
from typing import Any

import yaml


FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
SECTION_PATTERN_TEMPLATE = r"(?ims)^\s*#{{1,6}}\s*(?:{titles})\s*$\n(.*?)(?=^\s*#{{1,6}}\s+|\Z)"


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, set):
        return [_json_safe_value(item) for item in value]
    return value


@dataclass
class ParsedSkillDocument:
    name: str
    description: str
    one_line_capability: str = ""
    source_path: str = ""
    raw_content: str = ""
    body: str = ""
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    domain_tags: list[str] = field(default_factory=list)
    tooling: list[str] = field(default_factory=list)
    example_tasks: list[str] = field(default_factory=list)
    script_entrypoints: list[str] = field(default_factory=list)
    compatibility: list[str] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    rendered_snippet: str = ""
    skill_id: str = ""
    frontmatter: dict[str, Any] = field(default_factory=dict)


def extract_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    match = FRONTMATTER_PATTERN.search(content)
    if not match:
        return {}, content

    raw_frontmatter = match.group(1)
    body = content[match.end() :].strip()

    try:
        frontmatter = yaml.safe_load(raw_frontmatter) or {}
    except yaml.YAMLError:
        frontmatter = {}

    if not isinstance(frontmatter, dict):
        frontmatter = {}

    frontmatter = _json_safe_value(frontmatter)

    return frontmatter, body


def normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        parts = re.split(r"[\n,;]+", value)
        return [part.strip() for part in parts if part.strip()]

    if isinstance(value, dict):
        items: list[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list, tuple, set)):
                items.append(f"{key}: {item}")
            else:
                items.append(f"{key}: {str(item).strip()}")
        return [item for item in items if item]

    if isinstance(value, (list, tuple, set)):
        normalized: list[str] = []
        for item in value:
            normalized.extend(normalize_string_list(item))
        return normalized

    return [str(value).strip()]


def compact_text(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def clip_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text

    clipped = text[: max_chars - 3].rstrip()
    return f"{clipped}..."


def build_rendered_snippet(
    name: str,
    description: str,
    body: str,
    compatibility: list[str],
    allowed_tools: list[str],
    max_chars: int,
) -> str:
    parts = [f"# {name}", description.strip()]

    if compatibility:
        parts.append(f"Compatibility: {', '.join(compatibility)}")
    if allowed_tools:
        parts.append(f"Allowed tools: {', '.join(allowed_tools)}")
    if body.strip():
        parts.append(body.strip())

    return clip_text(compact_text("\n\n".join(part for part in parts if part)), max_chars)


def extract_markdown_section(body: str, titles: list[str]) -> list[str]:
    if not body.strip():
        return []

    title_pattern = "|".join(re.escape(title) for title in titles)
    pattern = re.compile(SECTION_PATTERN_TEMPLATE.format(titles=title_pattern))
    match = pattern.search(body)
    if not match:
        return []

    section_body = match.group(1).strip()
    if not section_body:
        return []

    bullet_items = re.findall(r"(?m)^\s*[-*+]\s+(.*)$", section_body)
    if bullet_items:
        return [item.strip() for item in bullet_items if item.strip()]

    return [line.strip() for line in section_body.splitlines() if line.strip()]


def _first_sentence(text: str) -> str:
    stripped = compact_text(text)
    if not stripped:
        return ""

    match = re.search(r"(?<=[.!?])\s+", stripped)
    if match:
        return stripped[: match.start()].strip()
    return stripped


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = compact_text(str(value))
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _script_entrypoints_for_source(source_path: str) -> list[str]:
    if not source_path:
        return []

    skill_path = source_path.strip()
    if not skill_path:
        return []

    base_dir = re.sub(r"[/\\]+SKILL\.md$", "", skill_path, flags=re.IGNORECASE)
    scripts_dir = base_dir + "/scripts"

    try:
        from pathlib import Path

        scripts_path = Path(scripts_dir)
        if not scripts_path.exists() or not scripts_path.is_dir():
            return []

        entrypoints: list[str] = []
        for child in sorted(scripts_path.rglob("*")):
            if not child.is_file():
                continue
            if child.suffix.lower() not in {".py", ".sh", ".js", ".ts"}:
                continue
            try:
                entrypoints.append(str(child.relative_to(scripts_path.parent)))
            except ValueError:
                entrypoints.append(child.name)
        return entrypoints
    except OSError:
        return []


def parse_skill_document(
    content: str,
    *,
    source_path: str = "",
    snippet_chars: int = 800,
) -> ParsedSkillDocument | None:
    frontmatter, body = extract_frontmatter(content)
    if not frontmatter:
        return None

    name = str(frontmatter.get("name", "")).strip()
    description = str(frontmatter.get("description", "")).strip()
    if not description:
        description = compact_text(extract_markdown_section(body, ["Overview", "Summary", "Description"]))
    if not description:
        description = _first_sentence(compact_text(body))

    if not name or not description:
        return None

    metadata = frontmatter.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {"value": metadata}
    metadata = _json_safe_value(metadata)

    inputs = normalize_string_list(frontmatter.get("inputs") or frontmatter.get("input"))
    outputs = normalize_string_list(frontmatter.get("outputs") or frontmatter.get("output"))

    if not inputs:
        inputs = extract_markdown_section(body, ["Inputs", "Input", "Requirements", "Arguments"])
    if not outputs:
        outputs = extract_markdown_section(body, ["Outputs", "Output", "Returns", "Results"])

    compatibility = normalize_string_list(frontmatter.get("compatibility"))
    allowed_tools = normalize_string_list(
        frontmatter.get("allowed-tools") or frontmatter.get("allowed_tools")
    )

    one_line_capability = compact_text(
        str(
            frontmatter.get("one_line_capability")
            or frontmatter.get("capability")
            or frontmatter.get("summary")
            or metadata.get("summary")
            or _first_sentence(description)
        )
    )

    domain_tags = _dedupe_preserve_order(
        normalize_string_list(frontmatter.get("domain"))
        + normalize_string_list(frontmatter.get("domains"))
        + normalize_string_list(frontmatter.get("tags"))
        + normalize_string_list(frontmatter.get("category"))
        + normalize_string_list(frontmatter.get("categories"))
        + normalize_string_list(metadata.get("domain"))
        + normalize_string_list(metadata.get("domains"))
        + normalize_string_list(metadata.get("tags"))
        + extract_markdown_section(body, ["Domain", "Domains", "Tags", "Category"])
    )

    tooling = _dedupe_preserve_order(
        normalize_string_list(frontmatter.get("tooling"))
        + normalize_string_list(frontmatter.get("libraries"))
        + normalize_string_list(frontmatter.get("dependencies"))
        + normalize_string_list(metadata.get("tooling"))
        + normalize_string_list(metadata.get("libraries"))
        + allowed_tools
        + extract_markdown_section(body, ["Tools", "Tooling", "Libraries", "Dependencies"])
    )

    example_tasks = _dedupe_preserve_order(
        normalize_string_list(frontmatter.get("examples"))
        + normalize_string_list(frontmatter.get("example_tasks"))
        + normalize_string_list(metadata.get("examples"))
        + normalize_string_list(metadata.get("example_tasks"))
        + extract_markdown_section(body, ["Examples", "Example Tasks", "Use When", "Use Cases"])
    )

    script_entrypoints = _script_entrypoints_for_source(source_path)

    rendered_snippet = build_rendered_snippet(
        name=name,
        description=description,
        body=body,
        compatibility=compatibility,
        allowed_tools=allowed_tools,
        max_chars=snippet_chars,
    )

    skill_id = source_path or name

    return ParsedSkillDocument(
        name=name,
        description=description,
        one_line_capability=one_line_capability,
        source_path=source_path,
        raw_content=content,
        body=body,
        inputs=inputs,
        outputs=outputs,
        domain_tags=domain_tags,
        tooling=tooling,
        example_tasks=example_tasks,
        script_entrypoints=script_entrypoints,
        compatibility=compatibility,
        allowed_tools=allowed_tools,
        metadata=metadata,
        rendered_snippet=rendered_snippet,
        skill_id=skill_id,
        frontmatter=frontmatter,
    )


def build_extraction_input(document: ParsedSkillDocument) -> str:
    parts = [
        f"Skill Name: {document.name}",
        f"Description: {document.description}",
    ]

    if document.script_entrypoints:
        parts.append(f"Known Script Entrypoints: {document.script_entrypoints}")
    if document.body:
        parts.append(f"Body:\n{document.body}")

    return "\n".join(parts)
