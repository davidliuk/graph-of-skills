import json
import logging
import re

from fastmcp import FastMCP

from gos.core.engine import SkillGraphRAG
from gos.utils.config import settings

logging.getLogger("fast_graphrag").setLevel(logging.WARNING)

mcp = FastMCP("Graph of Skills")
_engine = None


async def get_engine():
    global _engine
    if _engine is None:
        _engine = SkillGraphRAG()
    return _engine


@mcp.tool()
async def search_skills(
    query: str,
    max_skills: int = settings.RETRIEVAL_TOP_N,
) -> str:
    """
    Search the graph and return a concise summary of the most relevant skills.
    Use this when you want a quick overview before deciding which skills to load.
    """
    engine = await get_engine()
    bundle = await engine.async_retrieve(query, top_n=max_skills)
    return bundle.summary


@mcp.tool()
async def retrieve_skill_bundle(
    query: str,
    max_skills: int = settings.RETRIEVAL_TOP_N,
    seed_top_k: int = settings.SEED_TOP_K,
    seed_candidate_top_k_semantic: int = settings.SEED_CANDIDATE_TOP_K_SEMANTIC,
    seed_candidate_top_k_lexical: int = settings.SEED_CANDIDATE_TOP_K_LEXICAL,
    max_chars_per_skill: int = settings.MAX_SKILL_CHARS,
    max_context_chars: int = settings.MAX_CONTEXT_CHARS,
) -> str:
    """
    Retrieve and hydrate the skills needed for a task.
    The response includes `rendered_context`, which contains the actual skill content
    to inject into the agent's working context instead of loading the full skill library.
    """
    engine = await get_engine()
    engine.config.seed_candidate_top_k_semantic = seed_candidate_top_k_semantic
    engine.config.seed_candidate_top_k_lexical = seed_candidate_top_k_lexical
    bundle = await engine.async_retrieve(
        query,
        top_n=max_skills,
        seed_top_k=seed_top_k,
        max_chars_per_skill=max_chars_per_skill,
        max_context_chars=max_context_chars,
    )
    return bundle.model_dump_json(indent=2)


@mcp.tool()
async def hydrate_skills(
    skill_names: str,
    max_chars_per_skill: int = settings.MAX_SKILL_CHARS,
) -> str:
    """
    Load the actual content for a known list of skills.
    Provide skill names as a comma-separated or newline-separated string.
    """
    engine = await get_engine()
    names = [name.strip() for name in re_split_skill_names(skill_names) if name.strip()]
    skills = await engine.async_hydrate_skills(
        names,
        max_chars_per_skill=max_chars_per_skill,
    )
    rendered_context = "\n\n".join(skill.payload for skill in skills)
    return json.dumps(
        {
            "skills": [skill.model_dump() for skill in skills],
            "rendered_context": rendered_context,
        },
        indent=2,
        ensure_ascii=False,
    )


def re_split_skill_names(skill_names: str) -> list[str]:
    return [part for part in re.split(r"[\n,]+", skill_names) if part]


@mcp.tool()
async def get_graph_info() -> str:
    """Return graph size and retrieval defaults."""
    engine = await get_engine()
    await engine.state_manager.query_start()
    try:
        node_count = await engine.state_manager.graph_storage.node_count()
        edge_count = await engine.state_manager.graph_storage.edge_count()
    finally:
        await engine.state_manager.query_done()

    return json.dumps(
        {
            "skills": node_count,
            "edges": edge_count,
            "seed_top_k": settings.SEED_TOP_K,
            "seed_candidate_top_k_semantic": settings.SEED_CANDIDATE_TOP_K_SEMANTIC,
            "seed_candidate_top_k_lexical": settings.SEED_CANDIDATE_TOP_K_LEXICAL,
            "retrieval_top_n": settings.RETRIEVAL_TOP_N,
            "max_skill_chars": settings.MAX_SKILL_CHARS,
            "max_context_chars": settings.MAX_CONTEXT_CHARS,
        },
        indent=2,
    )


if __name__ == "__main__":
    mcp.run()


def main() -> None:
    mcp.run()
