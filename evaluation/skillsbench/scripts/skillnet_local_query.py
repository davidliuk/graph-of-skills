#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-large"
DEFAULT_TOP_N = 8
DEFAULT_MAX_SKILL_CHARS = 2400
DEFAULT_MAX_CONTEXT_CHARS = 12000
DEFAULT_INDEX_PATH = "/opt/graphskills/shared/skillnet_local_index.json"


def resolve_api_key() -> str:
    base_url = resolve_base_url()
    if "openrouter.ai" in base_url:
        key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    else:
        key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY or OPENROUTER_API_KEY is required for skillnet-query")
    return key


def resolve_base_url() -> str:
    configured = (os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL).strip().rstrip("/")
    if "openrouter.ai/api" in configured and not configured.endswith("/v1"):
        return configured + "/v1"
    return configured


def embedding_request(texts: list[str], model: str) -> list[list[float]]:
    api_key = resolve_api_key()
    endpoint = resolve_base_url() + "/embeddings"
    payload = json.dumps({"model": model, "input": texts, "encoding_format": "float"}).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Embedding request failed: {exc.code} {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Embedding request failed: {exc}") from exc

    data = body.get("data") or []
    if not data:
        raise RuntimeError(f"Embedding response missing data: {body}")
    return [item["embedding"] for item in data]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for l, r in zip(left, right):
        dot += l * r
        left_norm += l * l
        right_norm += r * r
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / (math.sqrt(left_norm) * math.sqrt(right_norm))


def clip_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def load_index(index_path: Path) -> dict[str, Any]:
    with index_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def render_no_hit(query: str) -> str:
    return "\n".join(
        [
            f"# Skill bundle for query: {query}",
            "## Retrieval Status",
            "Retrieval Status: NO_SKILL_HIT",
            "No relevant skill bundle was found.",
            "Do not claim that you used a retrieved skill.",
            "Proceed on a no-skill path.",
            "Before implementing, inspect the task tests/verifier and satisfy the minimum acceptance requirements.",
        ]
    )


def render_hit(query: str, skills: list[dict[str, Any]], max_skill_chars: int, max_context_chars: int) -> str:
    sections = [
        f"# Skill bundle for query: {query}",
        "## Retrieval Status",
        "Retrieval Status: SKILL_HIT",
        "Use retrieved skills to narrow the solution space and take the shortest path to verifier pass.",
        "Before implementing, inspect the task tests/verifier and identify the minimum acceptance requirements.",
        "Satisfy only the minimum requirements first.",
        "Treat retrieved skills as constraints and reusable implementations, not permission to branch out.",
        "Use the exact Source paths already provided. Do not reconstruct paths from the skill name or scan the whole library if a Source path is already available.",
        "Prefer adapting retrieved scripts/interfaces over building a more general replacement.",
    ]
    for skill in skills:
        source_path = skill["source_path"]
        try:
            content = Path(source_path).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            content = skill.get("rendered_snippet", "") or skill.get("description", "")
        content = clip_text(content, max_skill_chars)
        block = "\n".join(
            [
                f"## Skill: {skill['name']}",
                f"Source: {source_path}",
                f"SkillNet Score: {skill['score']:.4f}",
                content,
            ]
        )
        candidate = "\n\n".join(sections + [block])
        if len(candidate) > max_context_chars:
            break
        sections.append(block)
    return clip_text("\n\n".join(sections), max_context_chars)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local SkillNet-style retrieval over a fixed skill catalog")
    parser.add_argument("prompt")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--max-skill-chars", type=int, default=DEFAULT_MAX_SKILL_CHARS)
    parser.add_argument("--max-context-chars", type=int, default=DEFAULT_MAX_CONTEXT_CHARS)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--index", default=os.getenv("SKILLNET_LOCAL_INDEX", DEFAULT_INDEX_PATH))
    args = parser.parse_args()

    query = args.prompt.strip()
    if not query:
        print(render_no_hit(query))
        return

    index = load_index(Path(args.index))
    skills = index.get("skills") or []
    if not skills:
        print(render_no_hit(query))
        return

    model = index.get("metadata", {}).get("embedding_model") or os.getenv("GOS_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
    query_embedding = embedding_request([query], model)[0]

    ranked: list[dict[str, Any]] = []
    for skill in skills:
        score = cosine_similarity(query_embedding, skill["embedding"])
        ranked.append(
            {
                "name": skill["name"],
                "description": skill.get("description", ""),
                "source_path": skill["source_path"],
                "rendered_snippet": skill.get("rendered_snippet", ""),
                "score": score,
            }
        )
    ranked.sort(key=lambda item: item["score"], reverse=True)
    selected = [item for item in ranked[: args.top_n] if item["score"] > 0.05]

    if args.json:
        payload = {
            "query": query,
            "retrieval_status": "SKILL_HIT" if selected else "NO_SKILL_HIT",
            "skills": selected,
        }
        print(json.dumps(payload, indent=2))
        return

    if not selected:
        print(render_no_hit(query))
        return

    print(render_hit(query, selected, args.max_skill_chars, args.max_context_chars))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"skillnet-query failed: {exc}", file=sys.stderr)
        raise
