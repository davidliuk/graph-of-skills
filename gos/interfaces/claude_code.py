"""Graph of Skills — MCP server for Claude Code.

Provides dependency-aware skill retrieval, graph management, and exploration
tools that Claude Code can use to find, load, and manage agent skills.

Launch:
    gos-claude                       # uses GOS_WORKING_DIR from .env
    gos-claude --workspace ./my_ws   # explicit workspace
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Annotated

from fastmcp import FastMCP

from gos.core.engine import SkillGraphRAG, build_default_embedding_service, build_default_llm_service
from gos.core.parsing import parse_skill_document
from gos.utils.config import settings

logging.getLogger("fast_graphrag").setLevel(logging.WARNING)

mcp = FastMCP(
    "Graph of Skills for Claude Code",
    instructions=(
        "Graph of Skills (GoS) retrieves the most relevant agent skills from a "
        "large skill library using dependency-aware structural retrieval with "
        "Personalized PageRank over a skill graph. Use `search_skills` for a "
        "quick overview, `retrieve_skill_bundle` to get injectable skill content, "
        "and the management tools to build or extend the graph."
    ),
)

_engines: dict[str, SkillGraphRAG] = {}


def _resolve_workspace(workspace: str | None = None) -> str:
    return str(Path(workspace).expanduser()) if workspace else settings.WORKING_DIR


def _get_engine(workspace: str | None = None) -> SkillGraphRAG:
    ws = _resolve_workspace(workspace)
    if ws not in _engines:
        config = SkillGraphRAG.Config(
            llm_service=build_default_llm_service(),
            embedding_service=build_default_embedding_service(),
            working_dir=ws,
            prebuilt_working_dir=settings.PREBUILT_WORKING_DIR,
            domain=settings.DOMAIN,
            use_full_markdown=settings.USE_FULL_MARKDOWN,
            link_top_k=settings.LINK_TOP_K,
            seed_top_k=settings.SEED_TOP_K,
            seed_candidate_top_k_semantic=settings.SEED_CANDIDATE_TOP_K_SEMANTIC,
            seed_candidate_top_k_lexical=settings.SEED_CANDIDATE_TOP_K_LEXICAL,
            retrieval_top_n=settings.RETRIEVAL_TOP_N,
            enable_semantic_linking=settings.ENABLE_SEMANTIC_LINKING,
            dependency_match_threshold=settings.DEPENDENCY_MATCH_THRESHOLD,
            ppr_damping=settings.PPR_DAMPING,
            ppr_max_iter=settings.PPR_MAX_ITER,
            ppr_tolerance=settings.PPR_TOLERANCE,
            max_skill_chars=settings.MAX_SKILL_CHARS,
            max_context_chars=settings.MAX_CONTEXT_CHARS,
            snippet_chars=settings.SNIPPET_CHARS,
            enable_query_rewrite=settings.ENABLE_QUERY_REWRITE,
        )
        _engines[ws] = SkillGraphRAG(config=config)
    return _engines[ws]


# ---------------------------------------------------------------------------
# Retrieval tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def search_skills(
    query: Annotated[str, "Natural-language description of the task you need skills for"],
    max_skills: Annotated[int, "Maximum number of skills to return"] = 8,
    workspace: Annotated[str | None, "Workspace directory (defaults to GOS_WORKING_DIR)"] = None,
) -> str:
    """Search the skill graph and return a ranked summary of the most relevant skills.

    Use this for a quick overview before deciding which skills to load in full.
    Returns skill names, descriptions, scores, and graph relations — but not
    the full skill content. Follow up with `retrieve_skill_bundle` or
    `hydrate_skills` to get the actual content.
    """
    engine = _get_engine(workspace)
    bundle = await engine.async_retrieve(query, top_n=max_skills)
    return bundle.summary


@mcp.tool()
async def retrieve_skill_bundle(
    query: Annotated[str, "Natural-language description of the task you need skills for"],
    max_skills: Annotated[int, "Maximum number of skills to return"] = 8,
    max_chars_per_skill: Annotated[int, "Character budget per skill"] = 2400,
    max_context_chars: Annotated[int, "Total character budget for the bundle"] = 12000,
    workspace: Annotated[str | None, "Workspace directory (defaults to GOS_WORKING_DIR)"] = None,
) -> str:
    """Retrieve a full, agent-ready skill bundle for a task.

    Returns `rendered_context` — the actual skill content (SKILL.md bodies,
    source paths, graph evidence) ready to inject into your working context.
    Also returns structured metadata: seeds, scores, rewritten query, and
    graph relations between the retrieved skills.

    This is the primary tool for getting skills to execute a task.
    """
    engine = _get_engine(workspace)
    bundle = await engine.async_retrieve(
        query,
        top_n=max_skills,
        max_chars_per_skill=max_chars_per_skill,
        max_context_chars=max_context_chars,
    )
    return bundle.model_dump_json(indent=2)


@mcp.tool()
async def hydrate_skills(
    skill_names: Annotated[str, "Comma-separated or newline-separated list of skill names to load"],
    max_chars_per_skill: Annotated[int, "Character budget per skill"] = 2400,
    workspace: Annotated[str | None, "Workspace directory (defaults to GOS_WORKING_DIR)"] = None,
) -> str:
    """Load the full content for specific skills by name.

    Use this when you already know which skills you need (e.g. from a
    previous `search_skills` call) and want their actual SKILL.md content.
    """
    engine = _get_engine(workspace)
    names = [n.strip() for n in re.split(r"[\n,]+", skill_names) if n.strip()]
    skills = await engine.async_hydrate_skills(names, max_chars_per_skill=max_chars_per_skill)
    rendered_context = "\n\n".join(skill.payload for skill in skills)
    return json.dumps(
        {
            "skills": [skill.model_dump() for skill in skills],
            "rendered_context": rendered_context,
        },
        indent=2,
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# Graph exploration tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_status(
    workspace: Annotated[str | None, "Workspace directory (defaults to GOS_WORKING_DIR)"] = None,
) -> str:
    """Return workspace status: number of skills, edges, and retrieval defaults."""
    engine = _get_engine(workspace)
    await engine.state_manager.query_start()
    try:
        node_count = await engine.state_manager.graph_storage.node_count()
        edge_count = await engine.state_manager.graph_storage.edge_count()
    finally:
        await engine.state_manager.query_done()

    ws = _resolve_workspace(workspace)
    return json.dumps(
        {
            "workspace": ws,
            "skills": node_count,
            "edges": edge_count,
            "config": {
                "embedding_model": settings.EMBEDDING_MODEL,
                "llm_model": settings.LLM_MODEL,
                "seed_top_k": settings.SEED_TOP_K,
                "retrieval_top_n": settings.RETRIEVAL_TOP_N,
                "max_skill_chars": settings.MAX_SKILL_CHARS,
                "max_context_chars": settings.MAX_CONTEXT_CHARS,
            },
        },
        indent=2,
    )


@mcp.tool()
async def list_skills(
    workspace: Annotated[str | None, "Workspace directory (defaults to GOS_WORKING_DIR)"] = None,
) -> str:
    """List all indexed skills with their names, descriptions, and source paths.

    Returns a concise table of every skill in the graph, useful for browsing
    the library or verifying that indexing captured all expected skills.
    """
    engine = _get_engine(workspace)
    await engine.state_manager.query_start()
    try:
        nodes = await engine._load_all_nodes()
    finally:
        await engine.state_manager.query_done()

    if not nodes:
        return json.dumps({"skills": [], "count": 0}, indent=2)

    skills = []
    for node in sorted(nodes, key=lambda n: n.name):
        skills.append({
            "name": node.name,
            "description": node.description[:120] if node.description else "",
            "capability": node.one_line_capability[:120] if node.one_line_capability else "",
            "source_path": node.source_path,
            "domain_tags": node.domain_tags_list,
        })
    return json.dumps({"skills": skills, "count": len(skills)}, indent=2)


@mcp.tool()
async def get_skill_detail(
    skill_name: Annotated[str, "Exact name of the skill to inspect"],
    workspace: Annotated[str | None, "Workspace directory (defaults to GOS_WORKING_DIR)"] = None,
) -> str:
    """Get the full detail of a single skill including all metadata and content.

    Returns the complete SKILL.md content, I/O schema, domain tags, tooling,
    example tasks, script entrypoints, compatibility, and graph neighbors.
    """
    engine = _get_engine(workspace)
    await engine.state_manager.query_start()
    try:
        nodes = await engine._load_all_nodes()
        edges = await engine._load_all_edges()
    finally:
        await engine.state_manager.query_done()

    target = None
    for node in nodes:
        if node.name == skill_name:
            target = node
            break

    if target is None:
        return json.dumps({"error": f"Skill '{skill_name}' not found in the graph."})

    neighbors = []
    for edge in edges:
        if edge.source == skill_name:
            neighbors.append({
                "direction": "outgoing",
                "neighbor": edge.target,
                "type": edge.type,
                "description": edge.description,
                "weight": edge.weight,
            })
        elif edge.target == skill_name:
            neighbors.append({
                "direction": "incoming",
                "neighbor": edge.source,
                "type": edge.type,
                "description": edge.description,
                "weight": edge.weight,
            })

    return json.dumps(
        {
            "name": target.name,
            "description": target.description,
            "one_line_capability": target.one_line_capability,
            "inputs": target.input_types,
            "outputs": target.output_types,
            "domain_tags": target.domain_tags_list,
            "tooling": target.tooling_list,
            "example_tasks": target.example_tasks_list,
            "script_entrypoints": target.script_entrypoints_list,
            "compatibility": target.compatibility_list,
            "allowed_tools": target.allowed_tools_list,
            "source_path": target.source_path,
            "rendered_snippet": target.rendered_snippet[:500] if target.rendered_snippet else "",
            "raw_content_length": len(target.raw_content),
            "neighbors": neighbors,
        },
        indent=2,
        ensure_ascii=False,
    )


@mcp.tool()
async def get_skill_neighbors(
    skill_name: Annotated[str, "Exact name of the skill to find neighbors for"],
    workspace: Annotated[str | None, "Workspace directory (defaults to GOS_WORKING_DIR)"] = None,
) -> str:
    """Get all graph neighbors (related skills) for a given skill.

    Returns incoming and outgoing edges with their types (dependency,
    workflow, semantic, alternative), descriptions, and weights.
    Useful for understanding skill relationships and pipelines.
    """
    engine = _get_engine(workspace)
    await engine.state_manager.query_start()
    try:
        edges = await engine._load_all_edges()
    finally:
        await engine.state_manager.query_done()

    incoming = []
    outgoing = []
    for edge in edges:
        entry = {
            "neighbor": "",
            "type": edge.type,
            "description": edge.description,
            "weight": edge.weight,
            "confidence": edge.confidence,
        }
        if edge.source == skill_name:
            entry["neighbor"] = edge.target
            outgoing.append(entry)
        elif edge.target == skill_name:
            entry["neighbor"] = edge.source
            incoming.append(entry)

    return json.dumps(
        {
            "skill": skill_name,
            "incoming": sorted(incoming, key=lambda e: e["weight"], reverse=True),
            "outgoing": sorted(outgoing, key=lambda e: e["weight"], reverse=True),
            "total_edges": len(incoming) + len(outgoing),
        },
        indent=2,
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# Graph management tools
# ---------------------------------------------------------------------------


def _discover_skill_files(path: Path) -> list[tuple[Path, str, dict[str, str]]]:
    discovered = []
    for file_path in sorted(path.rglob(settings.SKILL_FILENAME)):
        content = file_path.read_text(encoding="utf-8")
        parsed = parse_skill_document(
            content,
            source_path=str(file_path),
            snippet_chars=settings.SNIPPET_CHARS,
        )
        if parsed is not None:
            discovered.append((
                file_path,
                content,
                {
                    "source_path": str(file_path),
                    "raw_content": content,
                    "snippet_chars": str(settings.SNIPPET_CHARS),
                },
            ))
    return discovered


@mcp.tool()
async def index_skills(
    path: Annotated[str, "Path to a directory containing SKILL.md files to index"],
    workspace: Annotated[str | None, "Workspace directory (defaults to GOS_WORKING_DIR)"] = None,
    clear: Annotated[bool, "Clear the workspace before indexing"] = False,
) -> str:
    """Index a directory of SKILL.md files and build the skill graph.

    Scans the given path recursively for SKILL.md files, extracts structured
    metadata (I/O schema, domain tags, tooling, etc.), builds embedding
    vectors, and links skills via dependency/workflow/semantic edges.

    Set `clear=true` to rebuild from scratch, or `false` to incrementally
    add new skills to an existing workspace.
    """
    source = Path(path).expanduser()
    if not source.exists() or not source.is_dir():
        return json.dumps({"error": f"Path does not exist or is not a directory: {path}"})

    ws = _resolve_workspace(workspace)
    ws_path = Path(ws)

    if clear and ws_path.exists():
        shutil.rmtree(ws_path)
        _engines.pop(ws, None)

    documents = _discover_skill_files(source)
    if not documents:
        return json.dumps({"error": f"No valid SKILL.md documents found in {path}"})

    _engines.pop(ws, None)
    engine = _get_engine(workspace)

    contents = [content for _, content, _ in documents]
    metadatas = [metadata for _, _, metadata in documents]
    sync_result = await engine.async_ensure_skills(contents, metadatas)

    return json.dumps(
        {
            "workspace": ws,
            "source_path": str(source),
            "documents_found": len(documents),
            "reused": sync_result.reused_count,
            "inserted": sync_result.inserted_count,
            "updated": sync_result.updated_count,
            "final_skill_count": sync_result.final_skill_count,
            "inserted_names": sync_result.inserted_skill_names[:20],
        },
        indent=2,
    )


@mcp.tool()
async def add_skill(
    path: Annotated[str, "Path to a SKILL.md file or directory of skills to add"],
    workspace: Annotated[str | None, "Workspace directory (defaults to GOS_WORKING_DIR)"] = None,
) -> str:
    """Add one or more SKILL.md documents to an existing skill graph.

    Incrementally inserts and links new skills without rebuilding the
    entire graph. Accepts either a single SKILL.md file path or a
    directory that will be scanned recursively.
    """
    source = Path(path).expanduser()
    if not source.exists():
        return json.dumps({"error": f"Path does not exist: {path}"})

    ws = _resolve_workspace(workspace)

    if source.is_dir():
        documents = _discover_skill_files(source)
    elif source.is_file():
        content = source.read_text(encoding="utf-8")
        parsed = parse_skill_document(
            content,
            source_path=str(source),
            snippet_chars=settings.SNIPPET_CHARS,
        )
        if parsed is None:
            return json.dumps({"error": f"Not a valid SKILL.md document: {path}"})
        documents = [(
            source,
            content,
            {
                "source_path": str(source),
                "raw_content": content,
                "snippet_chars": str(settings.SNIPPET_CHARS),
            },
        )]
    else:
        return json.dumps({"error": f"Path is not a file or directory: {path}"})

    if not documents:
        return json.dumps({"error": f"No valid SKILL.md documents found at {path}"})

    _engines.pop(ws, None)
    engine = _get_engine(workspace)

    contents = [content for _, content, _ in documents]
    metadatas = [metadata for _, _, metadata in documents]
    sync_result = await engine.async_ensure_skills(contents, metadatas)

    return json.dumps(
        {
            "workspace": ws,
            "documents_added": len(documents),
            "reused": sync_result.reused_count,
            "inserted": sync_result.inserted_count,
            "updated": sync_result.updated_count,
            "final_skill_count": sync_result.final_skill_count,
            "inserted_names": sync_result.inserted_skill_names[:20],
        },
        indent=2,
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


@mcp.prompt()
def skill_retrieval_workflow() -> str:
    """Step-by-step workflow for finding and using skills from the GoS graph."""
    return (
        "To find and use skills from the Graph of Skills library:\n\n"
        "1. **Check status**: Call `get_status` to verify the workspace has indexed skills.\n"
        "2. **Search**: Call `search_skills` with a natural-language task description.\n"
        "   Review the ranked summary to understand which skills are relevant.\n"
        "3. **Retrieve**: Call `retrieve_skill_bundle` with the same query to get\n"
        "   the full skill content. The `rendered_context` field contains the actual\n"
        "   SKILL.md bodies ready to follow.\n"
        "4. **Execute**: Follow the instructions in the retrieved skills. Use the\n"
        "   `Source:` paths to locate skill scripts and entrypoints.\n"
        "5. **Explore**: Use `get_skill_detail` or `get_skill_neighbors` to\n"
        "   understand how skills relate to each other via the dependency graph.\n\n"
        "Tips:\n"
        "- Be specific in your query — include the goal, artifact types, and tools.\n"
        "- If no skills match, try rephrasing with different keywords.\n"
        "- Use `hydrate_skills` when you already know exact skill names.\n"
        "- Use `list_skills` to browse the full library."
    )


@mcp.prompt()
def index_new_skills() -> str:
    """How to build or extend the skill graph with new SKILL.md documents."""
    return (
        "To index skills into the Graph of Skills:\n\n"
        "1. **Prepare**: Ensure your skills are SKILL.md files with YAML frontmatter:\n"
        "   ```yaml\n"
        "   ---\n"
        "   name: my-skill\n"
        "   description: What this skill does\n"
        "   inputs:\n"
        "     - input type\n"
        "   outputs:\n"
        "     - output type\n"
        "   domain:\n"
        "     - domain tag\n"
        "   ---\n"
        "   # Markdown body with instructions\n"
        "   ```\n\n"
        "2. **Index**: Call `index_skills` with the path to your skill directory.\n"
        "   Set `clear=true` to rebuild from scratch.\n\n"
        "3. **Add incrementally**: Call `add_skill` to add new skills to an\n"
        "   existing workspace without rebuilding.\n\n"
        "4. **Verify**: Call `get_status` to confirm the skill count.\n"
        "   Call `list_skills` to see all indexed skills."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="GoS MCP server for Claude Code")
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Override GOS_WORKING_DIR for the session",
    )
    args = parser.parse_args()

    if args.workspace:
        import os
        os.environ["GOS_WORKING_DIR"] = args.workspace

    mcp.run()


if __name__ == "__main__":
    main()
