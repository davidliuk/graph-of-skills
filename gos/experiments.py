from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from gos.core.engine import SkillGraphRAG
from gos.core.parsing import ParsedSkillDocument, parse_skill_document
from gos.utils.config import settings


@dataclass(frozen=True)
class ExperimentPreset:
    name: str
    description: str
    skill_paths: tuple[str, ...]
    queries: tuple[str, ...]


RESEARCH_SUBSET_PRESET = ExperimentPreset(
    name="research-subset",
    description=(
        "A 12-skill research workflow slice covering discovery, orchestration, "
        "review, writing, and citation audit."
    ),
    skill_paths=(
        "skills/academic-researcher/SKILL.md",
        "skills/biorxiv-database/SKILL.md",
        "skills/dataset-discovery/SKILL.md",
        "skills/inno-deep-research/SKILL.md",
        "skills/inno-paper-reviewer/SKILL.md",
        "skills/inno-paper-writing/SKILL.md",
        "skills/inno-prepare-resources/SKILL.md",
        "skills/inno-reference-audit/SKILL.md",
        "skills/inno-research-orchestrator/SKILL.md",
        "skills/ml-paper-writing/SKILL.md",
        "skills/research-grants/SKILL.md",
        "skills/scientific-writing/SKILL.md",
    ),
    queries=(
        "I need a workflow to search papers, prepare resources, and orchestrate a deep research pass for a new scientific project.",
        "Help me review a draft paper, audit its references, and strengthen the scientific writing quality.",
        "I want to go from literature search to producing an ML paper draft with suitable datasets and citations.",
    ),
)


EXPERIMENT_PRESETS = {
    RESEARCH_SUBSET_PRESET.name: RESEARCH_SUBSET_PRESET,
}


@dataclass
class ExperimentQueryReport:
    query: str
    seed_count: int
    seeds: list[dict[str, Any]]
    skill_count: int
    skills: list[dict[str, Any]]
    relation_count: int
    relations: list[dict[str, Any]]
    context_chars: int
    summary: str


@dataclass
class ExperimentReport:
    preset: str
    description: str
    workspace: str
    llm_model: str
    embedding_model: str
    max_skills: int
    seed_top_k: int
    max_skill_chars: int
    max_context_chars: int
    skill_count: int
    skill_paths: list[str]
    parsed_skills: list[str]
    node_count: int
    edge_count: int
    queries: list[ExperimentQueryReport]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def available_experiment_presets() -> list[ExperimentPreset]:
    return list(EXPERIMENT_PRESETS.values())


def get_experiment_preset(name: str) -> ExperimentPreset:
    key = name.strip().lower()
    preset = EXPERIMENT_PRESETS.get(key)
    if preset is None:
        available = ", ".join(sorted(EXPERIMENT_PRESETS))
        raise ValueError(f"Unknown preset `{name}`. Available presets: {available}")
    return preset


def repository_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_preset_documents(
    preset: ExperimentPreset,
    *,
    base_dir: Path | None = None,
) -> list[tuple[Path, str, dict[str, str], ParsedSkillDocument]]:
    root = base_dir or repository_root()
    documents: list[tuple[Path, str, dict[str, str], ParsedSkillDocument]] = []

    for relative_path in preset.skill_paths:
        path = root / relative_path
        if not path.exists():
            raise FileNotFoundError(f"Preset `{preset.name}` is missing skill file: {path}")

        content = path.read_text(encoding="utf-8")
        parsed = parse_skill_document(
            content,
            source_path=str(path),
            snippet_chars=settings.SNIPPET_CHARS,
        )
        if parsed is None:
            raise ValueError(f"Preset `{preset.name}` contains an invalid skill document: {path}")

        documents.append(
            (
                path,
                content,
                {
                    "source_path": str(path),
                    "raw_content": content,
                    "snippet_chars": str(settings.SNIPPET_CHARS),
                },
                parsed,
            )
        )

    return documents


async def run_preset_experiment(
    engine: SkillGraphRAG,
    preset: ExperimentPreset,
    *,
    base_dir: Path | None = None,
    top_n: int | None = None,
    seed_top_k: int | None = None,
    max_chars_per_skill: int | None = None,
    max_context_chars: int | None = None,
) -> ExperimentReport:
    requested_top_n = top_n or engine.config.retrieval_top_n
    requested_seed_top_k = seed_top_k or engine.config.seed_top_k
    requested_skill_chars = max_chars_per_skill or engine.config.max_skill_chars
    requested_context_chars = max_context_chars or engine.config.max_context_chars

    documents = resolve_preset_documents(preset, base_dir=base_dir)
    contents = [content for _, content, _, _ in documents]
    metadatas = [metadata for _, _, metadata, _ in documents]
    parsed_skills = [parsed.name for _, _, _, parsed in documents]
    skill_paths = [str(path) for path, _, _, _ in documents]

    await engine.async_ensure_skills(contents, metadatas)

    await engine.state_manager.query_start()
    try:
        node_count = await engine.state_manager.graph_storage.node_count()
        edge_count = await engine.state_manager.graph_storage.edge_count()
    finally:
        await engine.state_manager.query_done()

    query_reports: list[ExperimentQueryReport] = []
    for query in preset.queries:
        result = await engine.async_retrieve(
            query,
            top_n=requested_top_n,
            seed_top_k=requested_seed_top_k,
            max_chars_per_skill=requested_skill_chars,
            max_context_chars=requested_context_chars,
        )
        query_reports.append(
            ExperimentQueryReport(
                query=query,
                seed_count=len(result.seeds),
                seeds=[
                    {
                        "name": seed.name,
                        "source_path": seed.source_path,
                        "seed_weight": seed.seed_weight,
                        "semantic_rank": seed.semantic_rank,
                    }
                    for seed in result.seeds
                ],
                skill_count=len(result.skills),
                skills=[
                    {
                        "name": skill.name,
                        "source_path": skill.source_path,
                        "score": skill.score,
                        "semantic_rank": skill.semantic_rank,
                    }
                    for skill in result.skills
                ],
                relation_count=len(result.relations),
                relations=[
                    {
                        "source": relation.source,
                        "target": relation.target,
                        "type": relation.type,
                        "weight": relation.weight,
                        "confidence": relation.confidence,
                    }
                    for relation in result.relations
                ],
                context_chars=len(result.rendered_context),
                summary=result.summary,
            )
        )

    return ExperimentReport(
        preset=preset.name,
        description=preset.description,
        workspace=engine.config.working_dir,
        llm_model=getattr(engine.llm_service, "model", "unknown"),
        embedding_model=getattr(engine.config.embedding_service, "model", "unknown"),
        max_skills=requested_top_n,
        seed_top_k=requested_seed_top_k,
        max_skill_chars=requested_skill_chars,
        max_context_chars=requested_context_chars,
        skill_count=len(skill_paths),
        skill_paths=skill_paths,
        parsed_skills=parsed_skills,
        node_count=node_count,
        edge_count=edge_count,
        queries=query_reports,
    )


def format_experiment_report(report: ExperimentReport) -> str:
    lines = [
        f"Preset: {report.preset}",
        f"Description: {report.description}",
        f"Workspace: {report.workspace}",
        f"LLM model: {report.llm_model}",
        f"Embedding model: {report.embedding_model}",
        f"Skill docs: {report.skill_count}",
        f"Graph: {report.node_count} nodes, {report.edge_count} edges",
        (
            "Retrieval budget: "
            f"top_n={report.max_skills}, seed_top_k={report.seed_top_k}, "
            f"max_skill_chars={report.max_skill_chars}, "
            f"max_context_chars={report.max_context_chars}"
        ),
    ]

    for index, query_report in enumerate(report.queries, start=1):
        lines.append("")
        lines.append(f"Query {index}: {query_report.query}")
        lines.append(
            "Seeds: "
            + ", ".join(
                f"{seed['name']}:{seed['seed_weight']:.4f}"
                for seed in query_report.seeds
            )
        )
        lines.append(
            "Top skills: "
            + ", ".join(
                f"{skill['name']}:{skill['score']:.4f}"
                for skill in query_report.skills
            )
        )
        lines.append(
            f"Relations: {query_report.relation_count}, context chars: {query_report.context_chars}"
        )

    return "\n".join(lines)
