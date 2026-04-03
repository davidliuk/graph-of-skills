import asyncio
import json
import logging
from pathlib import Path
import re
import shutil
from typing import Any

import typer
from loguru import logger
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from gos.core.engine import (
    SkillGraphRAG,
    build_default_embedding_service,
    build_default_llm_service,
)
from gos.core.parsing import parse_skill_document
from gos.experiments import (
    available_experiment_presets,
    format_experiment_report,
    get_experiment_preset,
    run_preset_experiment,
)
from gos.utils.config import settings

# Suppress noisy "No data file found" INFO/WARNING logs from fast_graphrag
# storage backends — these are normal on a fresh workspace and add no signal.
logger.disable("fast_graphrag")
logging.getLogger("fast_graphrag").setLevel(logging.ERROR)

app = typer.Typer(help="GoS: Graph of Skills Management CLI")
graphskills_query_app = typer.Typer(
    add_completion=False,
    help="Retrieve a bounded skill bundle from a prebuilt Graph of Skills workspace.",
)
vectorskills_query_app = typer.Typer(
    add_completion=False,
    help="Retrieve a bounded skill bundle from a prebuilt workspace using vector similarity only.",
)


def _load_skill_document(
    file_path: Path,
    *,
    enable_query_rewrite: bool | None = None,
) -> tuple[Path, str, dict[str, str]] | None:
    content = file_path.read_text(encoding="utf-8")
    parsed = parse_skill_document(
        content,
        source_path=str(file_path),
        snippet_chars=settings.SNIPPET_CHARS,
    )

    is_named_skill_file = file_path.name == settings.SKILL_FILENAME
    if is_named_skill_file:
        if parsed is None:
            return None
    elif not settings.ALLOW_FRONTMATTER_DOCS or parsed is None:
        return None

    return (
        file_path,
        content,
        {
            "source_path": str(file_path),
            "raw_content": content,
            "snippet_chars": str(settings.SNIPPET_CHARS),
        },
    )


def _discover_skill_documents(
    path: Path,
    *,
    enable_query_rewrite: bool | None = None,
) -> list[tuple[Path, str, dict[str, str]]]:
    discovered: list[tuple[Path, str, dict[str, str]]] = []

    # Indexing should only consider canonical skill entrypoints.
    # Broad *.md discovery accidentally pulls in references/, rules/, and
    # workflow docs that also contain frontmatter, inflating the workspace.
    for file_path in sorted(path.rglob(settings.SKILL_FILENAME)):
        document = _load_skill_document(
            file_path,
            enable_query_rewrite=enable_query_rewrite,
        )
        if document is not None:
            discovered.append(document)

    return discovered


def _resolve_skill_documents(
    path: Path,
    *,
    enable_query_rewrite: bool | None = None,
) -> list[tuple[Path, str, dict[str, str]]]:
    if not path.exists():
        raise typer.BadParameter(f"Path does not exist: {path}")
    if path.is_dir():
        return _discover_skill_documents(
            path,
            enable_query_rewrite=enable_query_rewrite,
        )
    if not path.is_file():
        raise typer.BadParameter(f"Path is not a readable file or directory: {path}")

    document = _load_skill_document(
        path,
        enable_query_rewrite=enable_query_rewrite,
    )
    if document is None:
        raise typer.BadParameter(f"Path is not a valid skill markdown document: {path}")
    return [document]


def _build_engine(
    *,
    workspace: Path | None = None,
    retrieval_top_n: int | None = None,
    seed_top_k: int | None = None,
    seed_candidate_top_k_semantic: int | None = None,
    seed_candidate_top_k_lexical: int | None = None,
    max_skill_chars: int | None = None,
    max_context_chars: int | None = None,
    enable_semantic_linking: bool | None = None,
    enable_query_rewrite: bool | None = None,
) -> SkillGraphRAG:
    config = SkillGraphRAG.Config(
        llm_service=build_default_llm_service(),
        embedding_service=build_default_embedding_service(),
        working_dir=str(workspace or Path(settings.WORKING_DIR)),
        prebuilt_working_dir=settings.PREBUILT_WORKING_DIR,
        domain=settings.DOMAIN,
        use_full_markdown=settings.USE_FULL_MARKDOWN,
        link_top_k=settings.LINK_TOP_K,
        seed_top_k=seed_top_k if seed_top_k is not None else settings.SEED_TOP_K,
        seed_candidate_top_k_semantic=(
            seed_candidate_top_k_semantic
            if seed_candidate_top_k_semantic is not None
            else settings.SEED_CANDIDATE_TOP_K_SEMANTIC
        ),
        seed_candidate_top_k_lexical=(
            seed_candidate_top_k_lexical
            if seed_candidate_top_k_lexical is not None
            else settings.SEED_CANDIDATE_TOP_K_LEXICAL
        ),
        retrieval_top_n=(
            retrieval_top_n if retrieval_top_n is not None else settings.RETRIEVAL_TOP_N
        ),
        enable_semantic_linking=(
            settings.ENABLE_SEMANTIC_LINKING
            if enable_semantic_linking is None
            else enable_semantic_linking
        ),
        dependency_match_threshold=settings.DEPENDENCY_MATCH_THRESHOLD,
        ppr_damping=settings.PPR_DAMPING,
        ppr_max_iter=settings.PPR_MAX_ITER,
        ppr_tolerance=settings.PPR_TOLERANCE,
        max_skill_chars=(
            max_skill_chars if max_skill_chars is not None else settings.MAX_SKILL_CHARS
        ),
        max_context_chars=(
            max_context_chars
            if max_context_chars is not None
            else settings.MAX_CONTEXT_CHARS
        ),
        snippet_chars=settings.SNIPPET_CHARS,
        enable_query_rewrite=(
            settings.ENABLE_QUERY_REWRITE
            if enable_query_rewrite is None
            else enable_query_rewrite
        ),
    )
    return SkillGraphRAG(config=config)


async def _retrieve_bundle(
    *,
    prompt: str,
    workspace: Path,
    max_skills: int,
    seed_top_k: int,
    seed_candidate_top_k_semantic: int,
    seed_candidate_top_k_lexical: int,
    max_skill_chars: int,
    max_context_chars: int,
):
    engine = _build_engine(
        workspace=workspace,
        retrieval_top_n=max_skills,
        seed_top_k=seed_top_k,
        seed_candidate_top_k_semantic=seed_candidate_top_k_semantic,
        seed_candidate_top_k_lexical=seed_candidate_top_k_lexical,
        max_skill_chars=max_skill_chars,
        max_context_chars=max_context_chars,
    )
    return await engine.async_retrieve(
        prompt,
        top_n=max_skills,
        seed_top_k=seed_top_k,
        max_chars_per_skill=max_skill_chars,
        max_context_chars=max_context_chars,
    )


async def _retrieve_vector_bundle(
    *,
    prompt: str,
    workspace: Path,
    max_skills: int,
    max_skill_chars: int,
    max_context_chars: int,
):
    engine = _build_engine(
        workspace=workspace,
        retrieval_top_n=max_skills,
        max_skill_chars=max_skill_chars,
        max_context_chars=max_context_chars,
    )
    return await engine.async_retrieve_vector(
        prompt,
        top_n=max_skills,
        max_chars_per_skill=max_skill_chars,
        max_context_chars=max_context_chars,
    )


def _rewrite_source_paths(text: str, skills_dir: str) -> str:
    """Rewrite `Source: {any_path}/{skill_name}/SKILL.md` to `Source: {skills_dir}/{skill_name}/SKILL.md`.

    This allows agents in containerised environments to resolve skill scripts
    even though the graph was indexed from a different (host) filesystem path.
    """
    skill_filename = settings.SKILL_FILENAME
    pattern = re.compile(
        r"^(Source:\s*).*?([^/\\]+)[/\\]" + re.escape(skill_filename) + r"\s*$",
        re.MULTILINE,
    )
    base = skills_dir.rstrip("/")
    return pattern.sub(lambda m: f"{m.group(1)}{base}/{m.group(2)}/{skill_filename}", text)


def _render_bundle_output(bundle: Any, *, raw: bool, as_json: bool) -> str:
    if as_json:
        return bundle.model_dump_json(indent=2)
    text = bundle.rendered_context if raw else bundle.summary
    if settings.SKILLS_DIR and not as_json:
        text = _rewrite_source_paths(text, settings.SKILLS_DIR)
    return text


async def _sync_skill_documents(
    *,
    skill_documents: list[tuple[Path, str, dict[str, str]]],
    workspace: Path,
) -> None:
    if not skill_documents:
        typer.echo("No valid skill documents found.")
        return

    workspace.parent.mkdir(parents=True, exist_ok=True)
    engine = _build_engine(workspace=workspace)
    if engine.bootstrapped_from:
        typer.echo(f"Bootstrapped workspace from existing graph: {engine.bootstrapped_from}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        collect_task = progress.add_task("[cyan]Preparing documents...", total=len(skill_documents))
        contents: list[str] = []
        metadatas: list[dict[str, str]] = []
        for _, content, metadata in skill_documents:
            contents.append(content)
            metadatas.append(metadata)
            progress.update(collect_task, advance=1)

        link_task = progress.add_task("[magenta]Indexing and linking...", total=None)
        sync_result = await engine.async_ensure_skills(contents, metadatas)
        progress.update(link_task, completed=True)

    typer.echo(
        "Skill sync complete: "
        f"reused={sync_result.reused_count}, "
        f"inserted={sync_result.inserted_count}, "
        f"updated={sync_result.updated_count}, "
        f"final_skills={sync_result.final_skill_count}"
    )


@app.command()
def index(
    path: Path = typer.Argument(..., help="Path to a directory of skill markdown files"),
    clear: bool = typer.Option(False, "--clear", help="Clear the existing workspace first"),
    workspace: Path = typer.Option(
        Path(settings.WORKING_DIR),
        "--workspace",
        help="Workspace directory used to store the indexed graph.",
    ),
):
    """Index valid skill documents and build the skill graph."""

    async def run_indexing():
        if clear and workspace.exists():
            shutil.rmtree(workspace)
            typer.echo(f"Cleared workspace: {workspace}")

        typer.echo(f"Found {len(skill_documents)} valid skill documents. Starting indexing...")
        await _sync_skill_documents(skill_documents=skill_documents, workspace=workspace)

    skill_documents = _discover_skill_documents(path)
    asyncio.run(run_indexing())


@app.command()
def add(
    path: Path = typer.Argument(
        ...,
        help="Path to a skill markdown file or directory to add into the workspace graph.",
    ),
    workspace: Path = typer.Option(
        Path(settings.WORKING_DIR),
        "--workspace",
        help="Workspace directory used to store and update the indexed graph.",
    ),
):
    """Add skills incrementally and persist the updated graph workspace."""

    skill_documents = _resolve_skill_documents(path)

    async def run_add():
        typer.echo(f"Adding {len(skill_documents)} skill document(s) into {workspace}...")
        await _sync_skill_documents(skill_documents=skill_documents, workspace=workspace)

    asyncio.run(run_add())


@app.command()
def query(
    prompt: str = typer.Argument(..., help="Query to run against the skill graph"),
    max_skills: int = typer.Option(settings.RETRIEVAL_TOP_N, "--max-skills"),
    seed_top_k: int = typer.Option(settings.SEED_TOP_K, "--seed-top-k"),
    seed_candidate_top_k_semantic: int = typer.Option(
        settings.SEED_CANDIDATE_TOP_K_SEMANTIC,
        "--seed-candidate-top-k-semantic",
    ),
    seed_candidate_top_k_lexical: int = typer.Option(
        settings.SEED_CANDIDATE_TOP_K_LEXICAL,
        "--seed-candidate-top-k-lexical",
    ),
    max_skill_chars: int = typer.Option(settings.MAX_SKILL_CHARS, "--max-skill-chars"),
    max_context_chars: int = typer.Option(settings.MAX_CONTEXT_CHARS, "--max-context-chars"),
    raw: bool = typer.Option(False, "--raw", help="Print the rendered skill bundle instead of the summary"),
    workspace: Path = typer.Option(
        Path(settings.WORKING_DIR),
        "--workspace",
        help="Workspace directory containing the indexed graph.",
    ),
):
    """Retrieve the skill bundle for a task query."""

    async def run_query():
        bundle = await _retrieve_bundle(
            prompt=prompt,
            workspace=workspace,
            max_skills=max_skills,
            seed_top_k=seed_top_k,
            seed_candidate_top_k_semantic=seed_candidate_top_k_semantic,
            seed_candidate_top_k_lexical=seed_candidate_top_k_lexical,
            max_skill_chars=max_skill_chars,
            max_context_chars=max_context_chars,
        )
        typer.echo(_render_bundle_output(bundle, raw=raw, as_json=False))

    asyncio.run(run_query())


@app.command()
def retrieve(
    prompt: str = typer.Argument(..., help="Task or subproblem description to retrieve skills for"),
    max_skills: int = typer.Option(settings.RETRIEVAL_TOP_N, "--max-skills"),
    seed_top_k: int = typer.Option(settings.SEED_TOP_K, "--seed-top-k"),
    seed_candidate_top_k_semantic: int = typer.Option(
        settings.SEED_CANDIDATE_TOP_K_SEMANTIC,
        "--seed-candidate-top-k-semantic",
    ),
    seed_candidate_top_k_lexical: int = typer.Option(
        settings.SEED_CANDIDATE_TOP_K_LEXICAL,
        "--seed-candidate-top-k-lexical",
    ),
    max_skill_chars: int = typer.Option(settings.MAX_SKILL_CHARS, "--max-skill-chars"),
    max_context_chars: int = typer.Option(settings.MAX_CONTEXT_CHARS, "--max-context-chars"),
    as_json: bool = typer.Option(False, "--json", help="Print the full retrieval bundle as JSON."),
    workspace: Path = typer.Option(
        Path(settings.WORKING_DIR),
        "--workspace",
        help="Workspace directory containing the indexed graph.",
    ),
):
    """Retrieve the agent-ready skill bundle content."""

    async def run_retrieve():
        bundle = await _retrieve_bundle(
            prompt=prompt,
            workspace=workspace,
            max_skills=max_skills,
            seed_top_k=seed_top_k,
            seed_candidate_top_k_semantic=seed_candidate_top_k_semantic,
            seed_candidate_top_k_lexical=seed_candidate_top_k_lexical,
            max_skill_chars=max_skill_chars,
            max_context_chars=max_context_chars,
        )
        typer.echo(_render_bundle_output(bundle, raw=True, as_json=as_json))

    asyncio.run(run_retrieve())


@app.command()
def status(
    workspace: Path = typer.Option(
        Path(settings.WORKING_DIR),
        "--workspace",
        help="Workspace directory containing the indexed graph.",
    ),
):
    """Show graph size and workspace."""

    async def show_status():
        engine = _build_engine(workspace=workspace)
        await engine.state_manager.query_start()
        try:
            node_count = await engine.state_manager.graph_storage.node_count()
            edge_count = await engine.state_manager.graph_storage.edge_count()
        finally:
            await engine.state_manager.query_done()

        typer.echo(f"Skills (nodes): {node_count}")
        typer.echo(f"Edges: {edge_count}")
        typer.echo(f"Workspace: {workspace}")
        typer.echo(f"Retrieval top-N: {settings.RETRIEVAL_TOP_N}")
        typer.echo(f"Seed top-K: {settings.SEED_TOP_K}")

    asyncio.run(show_status())


@app.command()
def experiment_presets():
    """List the built-in experiment presets."""

    for preset in available_experiment_presets():
        typer.echo(f"- {preset.name}: {preset.description}")


@app.command()
def experiment(
    preset: str = typer.Option(
        "research-subset",
        "--preset",
        help="Built-in experiment preset to run. Use `gos experiment-presets` to list options.",
    ),
    workspace: Path = typer.Option(
        Path("./gos_experiment_workspace"),
        "--workspace",
        help="Dedicated workspace directory for the experiment run.",
    ),
    clear: bool = typer.Option(
        True,
        "--clear/--no-clear",
        help="Clear the target workspace before running the experiment.",
    ),
    max_skills: int = typer.Option(settings.RETRIEVAL_TOP_N, "--max-skills"),
    seed_top_k: int = typer.Option(settings.SEED_TOP_K, "--seed-top-k"),
    seed_candidate_top_k_semantic: int = typer.Option(
        settings.SEED_CANDIDATE_TOP_K_SEMANTIC,
        "--seed-candidate-top-k-semantic",
    ),
    seed_candidate_top_k_lexical: int = typer.Option(
        settings.SEED_CANDIDATE_TOP_K_LEXICAL,
        "--seed-candidate-top-k-lexical",
    ),
    max_skill_chars: int = typer.Option(settings.MAX_SKILL_CHARS, "--max-skill-chars"),
    max_context_chars: int = typer.Option(settings.MAX_CONTEXT_CHARS, "--max-context-chars"),
    semantic_linking: bool = typer.Option(
        settings.ENABLE_SEMANTIC_LINKING,
        "--semantic-linking/--no-semantic-linking",
        help="Enable sparse LLM-validated semantic/workflow edges during linking.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Optional JSON file path for the full experiment report.",
    ),
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Print the full experiment report as JSON instead of the text summary.",
    ),
):
    """Run a reproducible built-in experiment against the local skill graph."""

    async def run_experiment():
        try:
            selected_preset = get_experiment_preset(preset)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

        if clear and workspace.exists():
            shutil.rmtree(workspace)
            typer.echo(f"Cleared workspace: {workspace}")

        workspace.parent.mkdir(parents=True, exist_ok=True)
        engine = _build_engine(
            workspace=workspace,
            retrieval_top_n=max_skills,
            seed_top_k=seed_top_k,
            seed_candidate_top_k_semantic=seed_candidate_top_k_semantic,
            seed_candidate_top_k_lexical=seed_candidate_top_k_lexical,
            max_skill_chars=max_skill_chars,
            max_context_chars=max_context_chars,
            enable_semantic_linking=semantic_linking,
        )
        if engine.bootstrapped_from:
            typer.echo(f"Bootstrapped workspace from existing graph: {engine.bootstrapped_from}")

        report = await run_preset_experiment(
            engine,
            selected_preset,
            top_n=max_skills,
            seed_top_k=seed_top_k,
            max_chars_per_skill=max_skill_chars,
            max_context_chars=max_context_chars,
        )

        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

        if as_json:
            typer.echo(json.dumps(report.to_dict(), indent=2))
        else:
            typer.echo(format_experiment_report(report))

        if output is not None:
            typer.echo(f"Wrote report: {output}")

    asyncio.run(run_experiment())


if __name__ == "__main__":
    app()


def main() -> None:
    app()


@graphskills_query_app.command()
def graphskills_query(
    prompt: str = typer.Argument(..., help="Task or subproblem description to retrieve skills for"),
    top_n: int = typer.Option(settings.RETRIEVAL_TOP_N, "--top-n"),
    seed_top_k: int = typer.Option(settings.SEED_TOP_K, "--seed-top-k"),
    seed_candidate_top_k_semantic: int = typer.Option(
        settings.SEED_CANDIDATE_TOP_K_SEMANTIC,
        "--seed-candidate-top-k-semantic",
    ),
    seed_candidate_top_k_lexical: int = typer.Option(
        settings.SEED_CANDIDATE_TOP_K_LEXICAL,
        "--seed-candidate-top-k-lexical",
    ),
    max_skill_chars: int = typer.Option(settings.MAX_SKILL_CHARS, "--max-skill-chars"),
    max_context_chars: int = typer.Option(settings.MAX_CONTEXT_CHARS, "--max-context-chars"),
    as_json: bool = typer.Option(False, "--json", help="Print the full retrieval bundle as JSON."),
    workspace: Path = typer.Option(
        Path(settings.WORKING_DIR),
        "--workspace",
        help="Workspace directory containing the indexed graph.",
    ),
) -> None:
    """Purpose-built wrapper for agent skills and container environments."""

    async def run_graphskills_query():
        bundle = await _retrieve_bundle(
            prompt=prompt,
            workspace=workspace,
            max_skills=top_n,
            seed_top_k=seed_top_k,
            seed_candidate_top_k_semantic=seed_candidate_top_k_semantic,
            seed_candidate_top_k_lexical=seed_candidate_top_k_lexical,
            max_skill_chars=max_skill_chars,
            max_context_chars=max_context_chars,
        )
        typer.echo(_render_bundle_output(bundle, raw=True, as_json=as_json))

    asyncio.run(run_graphskills_query())


def graphskills_query_main() -> None:
    graphskills_query_app()


@vectorskills_query_app.command()
def vectorskills_query(
    prompt: str = typer.Argument(..., help="Task or subproblem description to retrieve skills for"),
    top_n: int = typer.Option(settings.RETRIEVAL_TOP_N, "--top-n"),
    max_skill_chars: int = typer.Option(settings.MAX_SKILL_CHARS, "--max-skill-chars"),
    max_context_chars: int = typer.Option(settings.MAX_CONTEXT_CHARS, "--max-context-chars"),
    as_json: bool = typer.Option(False, "--json", help="Print the full retrieval bundle as JSON."),
    workspace: Path = typer.Option(
        Path(settings.WORKING_DIR),
        "--workspace",
        help="Workspace directory containing the indexed graph.",
    ),
) -> None:
    """Purpose-built wrapper for vector-only retrieval in container environments."""

    async def run_vectorskills_query():
        bundle = await _retrieve_vector_bundle(
            prompt=prompt,
            workspace=workspace,
            max_skills=top_n,
            max_skill_chars=max_skill_chars,
            max_context_chars=max_context_chars,
        )
        typer.echo(_render_bundle_output(bundle, raw=True, as_json=as_json))

    asyncio.run(run_vectorskills_query())


def vectorskills_query_main() -> None:
    vectorskills_query_app()
