#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


MODEL_ALIASES = {
    'openai/text-embedding-3-large': 'openai/text-embedding-3-large',
}

EMBEDDING_RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


def build_rank_distribution(count: int) -> list[float]:
    if count <= 0:
        return []
    weights = [1.0 / float(index + 1) for index in range(count)]
    total = sum(weights)
    return [weight / total for weight in weights]


def clip_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    clipped = text[: max_chars - 3].rstrip()
    return f"{clipped}..."


def split_lines(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    return [str(value).strip()]


def rewrite_source_path(source_path: str, skills_dir: str | None) -> str:
    if not source_path or not skills_dir:
        return source_path
    normalized = source_path.replace('\\', '/').rstrip('/')
    if normalized.endswith('/SKILL.md'):
        skill_name = normalized.split('/')[-2]
        return f"{skills_dir.rstrip('/')}/{skill_name}/SKILL.md"
    return source_path


def render_skill_payload(skill: dict[str, Any], max_chars: int, skills_dir: str | None) -> str:
    content = (
        skill.get('raw_content')
        or skill.get('rendered_snippet')
        or f"{skill['name']}\n\n{skill.get('description', '')}"
    )
    content = clip_text(content, max_chars)
    header = [
        f"## Skill: {skill['name']}",
        f"Source: {rewrite_source_path(skill.get('source_path') or 'inline', skills_dir)}",
    ]
    return "\n".join(header + [content])


def fit_skills_to_budget(
    query: str,
    ranked_skills: list[dict[str, Any]],
    max_context_chars: int,
) -> list[dict[str, Any]]:
    if not ranked_skills or max_context_chars <= 0:
        return []

    total_chars = len(f"# Skill bundle for query: {query}")
    selected: list[dict[str, Any]] = []
    for skill in ranked_skills:
        remaining_context = max_context_chars - total_chars - 2
        if remaining_context <= 0:
            break
        payload = skill['payload']
        if len(payload) > remaining_context:
            payload = clip_text(payload, remaining_context)
        if not payload:
            break
        updated_skill = dict(skill)
        updated_skill['payload'] = payload
        selected.append(updated_skill)
        total_chars += 2 + len(payload)
        if len(payload) < len(skill['payload']):
            break
    return selected


def render_context(query: str, skills: list[dict[str, Any]], *, max_chars: int) -> str:
    if not skills:
        return clip_text(
            "\n\n".join(
                [
                    f"# Skill bundle for query: {query}",
                    "## Retrieval Status",
                    "Retrieval Status: NO_SKILL_HIT",
                    "No relevant skill bundle was found.",
                    "Do not claim that you used a retrieved skill.",
                    "Proceed on a no-skill path.",
                    "Before implementing, inspect the task tests/verifier and satisfy the minimum acceptance requirements.",
                ]
            ),
            max_chars,
        )

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
        sections.append(skill['payload'])
    return clip_text("\n\n".join(sections), max_chars)


def render_summary(
    query: str,
    skills: list[dict[str, Any]],
    seeds: list[dict[str, Any]],
    skills_dir: str | None,
) -> str:
    if not skills:
        return "\n".join(
            [
                "### Retrieval Status",
                "- Retrieval Status: NO_SKILL_HIT",
                "- No relevant skill bundle was found for this query.",
                "- Do not claim that you used a retrieved skill.",
                "- Proceed on a no-skill path and inspect the task verifier/tests for the minimum requirements.",
                f"\nQuery: {query}",
            ]
        )

    lines = [
        "### Retrieval Status",
        "- Retrieval Status: SKILL_HIT",
        "- This is vector-only retrieval. No graph propagation or lexical expansion was used.",
        "- Use retrieved skills to narrow the solution space and take the shortest path to verifier pass.",
        "- Before coding, inspect the task verifier/tests and identify the minimum acceptance requirements.",
        "- Satisfy only those minimum requirements first.",
        "- Treat retrieved skills as constraints and reusable implementations, not permission to open extra branches.",
        "- Use the exact `Source:` path returned below. Do not reconstruct paths from the skill name or scan the whole library if a Source path is already available.",
        "- Prefer adapting the retrieved skill's scripts/interfaces over writing a broader replacement.",
        "\n### Retrieved Skills",
    ]
    for skill in skills:
        semantic_rank = (
            f", seed rank {skill['semantic_rank']}"
            if skill.get('semantic_rank') is not None
            else ""
        )
        lines.append(
            f"- {skill['name']}: {skill.get('description', '')} (vector score={skill['score']:.4f}{semantic_rank})"
        )
        source_path = rewrite_source_path(skill.get('source_path', ''), skills_dir)
        if source_path:
            lines.append(f"  Source: {source_path}")
        script_entrypoints = split_lines(skill.get('script_entrypoints'))
        if script_entrypoints:
            lines.append(f"  Scripts: {', '.join(script_entrypoints[:3])}")

    if seeds:
        lines.append("\n### Vector Seeds")
        for seed in seeds:
            source_path = rewrite_source_path(seed.get('source_path', ''), skills_dir)
            suffix = f"; Source: {source_path}" if source_path else ""
            lines.append(
                f"- {seed['name']} (seed weight={seed['seed_weight']:.4f}, rank={seed['semantic_rank']}){suffix}"
            )

    return "\n".join(lines)


def normalize_embedding_model(model: str) -> str:
    normalized = (model or '').strip()
    return MODEL_ALIASES.get(normalized, normalized)


def fetch_embedding(query: str, model: str, api_key: str, base_url: str) -> list[float]:
    normalized_model = normalize_embedding_model(model)
    endpoint = f"{base_url.rstrip('/')}/embeddings"
    max_attempts = max(1, int(os.getenv('VECTORSKILLS_EMBED_RETRY_ATTEMPTS', '4')))
    last_error: str | None = None
    payload = json.dumps({'model': normalized_model, 'input': [query]}).encode('utf-8')
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }

    for attempt in range(1, max_attempts + 1):
        request = urllib.request.Request(endpoint, data=payload, headers=headers, method='POST')
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = response.read().decode('utf-8', errors='replace')
                response_json = json.loads(body)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode('utf-8', errors='replace')
            last_error = (
                f"Embedding request failed for model={normalized_model!r} via {endpoint}: "
                f"status={exc.code}, body={body[:400]}"
            )
            if exc.code not in EMBEDDING_RETRYABLE_STATUS_CODES:
                raise RuntimeError(last_error) from exc
        except Exception as exc:
            last_error = (
                f"Embedding request failed for model={normalized_model!r} via {endpoint}: "
                f"request_error={exc}"
            )
        else:
            data = response_json.get('data') or []
            if not data:
                raise RuntimeError(f"Embedding response missing data: {response_json}")
            embedding = data[0].get('embedding') or []
            if not embedding:
                raise RuntimeError(f"Embedding response contained an empty vector: {response_json}")
            return [float(x) for x in embedding]

        if attempt < max_attempts:
            time.sleep(min(8.0, 0.75 * attempt))

    raise RuntimeError(last_error or f"Embedding request failed for model={normalized_model!r} via {endpoint}")


def load_vector_store(store_path: Path) -> dict[str, Any]:
    with store_path.open('rb') as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid vector store payload: {type(payload)}")
    ids = [int(item) for item in (payload.get('ids') or [])]
    dim = int(payload.get('dim') or 0)
    blob = payload.get('vectors_f32_le') or b''
    if dim <= 0:
        raise RuntimeError(f"Invalid vector store dim: {dim}")
    if len(blob) != len(ids) * dim * 4:
        raise RuntimeError(
            f"Vector blob size mismatch: ids={len(ids)}, dim={dim}, bytes={len(blob)}"
        )
    return {'ids': ids, 'dim': dim, 'blob': blob}


def knn_query_vectors(query_vector: list[float], store: dict[str, Any], *, top_k: int) -> tuple[list[int], list[float]]:
    ids: list[int] = store['ids']
    dim: int = store['dim']
    blob: bytes = store['blob']
    if not ids or top_k <= 0:
        return [], []
    if len(query_vector) != dim:
        raise RuntimeError(f"Query embedding dim mismatch: expected {dim}, got {len(query_vector)}")

    q_norm = math.sqrt(sum(value * value for value in query_vector))
    if q_norm == 0.0:
        raise RuntimeError('Query embedding had zero norm')
    normalized_query = [value / q_norm for value in query_vector]
    matrix = memoryview(blob).cast('f')

    scored: list[tuple[float, int]] = []
    for row_idx, skill_id in enumerate(ids):
        start = row_idx * dim
        row = matrix[start : start + dim]
        dot = 0.0
        norm_sq = 0.0
        for idx in range(dim):
            value = float(row[idx])
            dot += value * normalized_query[idx]
            norm_sq += value * value
        distance = 1.0 if norm_sq == 0.0 else 1.0 - (dot / math.sqrt(norm_sq))
        scored.append((distance, skill_id))

    scored.sort(key=lambda item: item[0])
    top = scored[: min(top_k, len(scored))]
    return [skill_id for distance, skill_id in top], [distance for distance, skill_id in top]


def retrieve(
    metadata: dict[str, Any],
    query: str,
    labels: list[int],
    distances: list[float],
    *,
    max_skill_chars: int,
    max_context_chars: int,
    skills_dir: str | None,
) -> dict[str, Any]:
    skills = metadata.get('skills', [])
    ranked_indices: list[int] = []
    seen: set[int] = set()
    for raw_index in labels:
        index = int(raw_index)
        if index < 0 or index >= len(skills) or index in seen:
            continue
        seen.add(index)
        ranked_indices.append(index)

    if not ranked_indices:
        return {
            'query': query,
            'summary': render_summary(query, [], [], skills_dir),
            'rendered_context': render_context(query, [], max_chars=max_context_chars),
            'skills': [],
            'seeds': [],
            'budget': {
                'seed_top_k': 0,
                'top_n': 0,
                'max_skill_chars': max_skill_chars,
                'max_context_chars': max_context_chars,
                'ppr_damping': 0.0,
            },
        }

    weights = build_rank_distribution(len(ranked_indices))
    selected_skills: list[dict[str, Any]] = []
    seeds: list[dict[str, Any]] = []
    for rank, index in enumerate(ranked_indices, start=1):
        skill = skills[index]
        distance = float(distances[rank - 1]) if rank - 1 < len(distances) else 1.0
        score = max(0.0, 1.0 - distance)
        selected_skills.append(
            {
                'name': skill['name'],
                'description': skill.get('description', ''),
                'source_path': skill.get('source_path', ''),
                'score': score,
                'semantic_rank': rank,
                'script_entrypoints': skill.get('script_entrypoints', []),
                'payload': render_skill_payload(skill, max_skill_chars, skills_dir),
            }
        )
        seeds.append(
            {
                'name': skill['name'],
                'source_path': skill.get('source_path', ''),
                'seed_weight': float(weights[rank - 1]),
                'semantic_rank': rank,
            }
        )

    budgeted_skills = fit_skills_to_budget(query, selected_skills, max_context_chars)
    rendered_context = render_context(query, budgeted_skills, max_chars=max_context_chars)
    summary = render_summary(query, budgeted_skills, seeds, skills_dir)
    return {
        'query': query,
        'summary': summary,
        'rendered_context': rendered_context,
        'skills': budgeted_skills,
        'seeds': seeds,
        'budget': {
            'seed_top_k': len(seeds),
            'top_n': len(budgeted_skills),
            'max_skill_chars': max_skill_chars,
            'max_context_chars': max_context_chars,
            'ppr_damping': 0.0,
        },
    }


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Query the local vector-skills benchmark bundle.')
    parser.add_argument('query', nargs='?', default='', help='Task or subproblem description.')
    parser.add_argument(
        '--metadata',
        type=Path,
        default=script_dir / 'metadata.json',
        help='Path to exported skill metadata JSON.',
    )
    parser.add_argument(
        '--vector-store',
        type=Path,
        default=script_dir / 'vectors.pkl',
        help='Path to exported vector store.',
    )
    parser.add_argument('--top-n', type=int, default=5, help='Maximum number of retrieved skills.')
    parser.add_argument(
        '--max-skill-chars',
        type=int,
        default=1800,
        help='Maximum characters to render per skill payload.',
    )
    parser.add_argument(
        '--max-context-chars',
        type=int,
        default=9000,
        help='Hard cap on the rendered skill bundle.',
    )
    parser.add_argument('--json', action='store_true', help='Print the full retrieval result as JSON.')
    parser.add_argument(
        '--raw',
        action='store_true',
        help='Print the rendered skill bundle instead of the summary.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = json.loads(args.metadata.read_text(encoding='utf-8'))
    query = args.query.strip()
    if not query:
        result = {
            'query': query,
            'summary': 'Empty query.',
            'rendered_context': '',
            'skills': [],
            'seeds': [],
            'budget': {
                'seed_top_k': 0,
                'top_n': 0,
                'max_skill_chars': args.max_skill_chars,
                'max_context_chars': args.max_context_chars,
                'ppr_damping': 0.0,
            },
        }
    else:
        api_key = os.getenv('OPENROUTER_API_KEY') or os.getenv('OPENAI_API_KEY') or ''
        if not api_key:
            raise SystemExit('Missing OPENROUTER_API_KEY or OPENAI_API_KEY for vectorskills-query')
        model = os.getenv('GOS_EMBEDDING_MODEL', 'openai/text-embedding-3-large')
        base_url = os.getenv('OPENAI_BASE_URL', 'https://openrouter.ai/api/v1')
        query_vector = fetch_embedding(query, model=model, api_key=api_key, base_url=base_url)
        store = load_vector_store(args.vector_store)
        labels, distances = knn_query_vectors(query_vector, store, top_k=args.top_n)
        result = retrieve(
            metadata,
            query,
            labels,
            distances,
            max_skill_chars=args.max_skill_chars,
            max_context_chars=args.max_context_chars,
            skills_dir=os.getenv('GOS_SKILLS_DIR', '/opt/graphskills/skills'),
        )

    if args.json:
        print(json.dumps(result, indent=2))
        return
    if args.raw:
        print(result['rendered_context'])
        return
    print(result['summary'])


if __name__ == '__main__':
    main()
