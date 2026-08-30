from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import re
import shutil
import time
from typing import Any, Callable, cast
from uuid import uuid4

import numpy as np
from loguru import logger

from fast_graphrag._graphrag import BaseGraphRAG, QueryParam
from fast_graphrag._llm import (
    BaseLLMService,
)
from fast_graphrag._services._chunk_extraction import (
    BaseChunkingService,
    DefaultChunkingService,
)
from fast_graphrag._services._state_manager import DefaultStateManagerService
from fast_graphrag._storage._gdb_igraph import IGraphStorageConfig
from fast_graphrag._storage._ikv_pickle import PickleIndexedKeyValueStorage
from fast_graphrag._storage._namespace import Workspace
from fast_graphrag._storage._vdb_hnswlib import (
    HNSWVectorStorage,
    HNSWVectorStorageConfig,
)
from fast_graphrag._types import (
    GTChunk,
    GTEmbedding,
    GTHash,
    TContext,
    TId,
    TQueryResponse,
)

from gos.utils.config import settings

from .parsing import parse_skill_document
from .policies import (
    SkillEdgeUpsertPolicy,
    SkillGraphUpsertPolicy,
    SkillNodeUpsertPolicy,
)
from .prompts import PROMPTS
from .retrieval import (
    build_personalization,
    build_rank_distribution,
    build_transition_matrix,
    personalized_pagerank,
)
from .relink import (
    FocusLinkJob,
    FocusLinkResult,
    RelinkProgress,
    RelinkProgressMismatch,
    RelinkResult,
    append_relink_event,
    build_relink_fingerprint,
    diff_relink_usage,
    load_relink_progress,
    merge_relink_usage,
    summarize_relink_error,
    summarize_relink_usage,
    write_relink_progress,
)
from .schema import (
    GOSRelationList,
    QuerySchema,
    RetrievedRelation,
    RetrievedSkill,
    RetrievalBudget,
    SkillEdge,
    SkillNode,
    SkillRetrievalResult,
    SkillSeed,
    SkillSyncResult,
    VALID_RELATION_TYPES,
)
from .litellm_services import LiteLLMEmbeddingService, LiteLLMService
from .services import SkillInformationExtractionService
from .storage import DirectedIGraphStorage
from .construction_report import ConstructionCounters


TYPE_WEIGHTS = {
    "dependency": 1.0,
    "workflow": 0.7,
    "semantic": 0.4,
    "alternative": 0.3,
}

DEFAULT_OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

HNSW_INDEX_FILENAME_PATTERN = re.compile(r"^entities_hnsw_index_(\d+)\.bin$")

TOKEN_STOPWORDS = {
    "a",
    "an",
    "and",
    "any",
    "arg",
    "args",
    "array",
    "bool",
    "boolean",
    "data",
    "dataframe",
    "dict",
    "file",
    "float",
    "for",
    "from",
    "in",
    "input",
    "int",
    "json",
    "list",
    "object",
    "of",
    "on",
    "or",
    "output",
    "path",
    "record",
    "result",
    "set",
    "str",
    "string",
    "text",
    "the",
    "that",
    "this",
    "to",
    "value",
    "which",
    "with",
    "without",
}

GENERIC_SCHEMA_TOKENS = TOKEN_STOPWORDS | {
    "analysis",
    "api",
    "automate",
    "artifact",
    "code",
    "configuration",
    "content",
    "context",
    "dataframe",
    "documentation",
    "document",
    "entry",
    "event",
    "item",
    "message",
    "model",
    "operation",
    "format",
    "payload",
    "project",
    "request",
    "report",
    "repository",
    "response",
    "schema",
    "script",
    "session",
    "source",
    "structure",
    "structured",
    "table",
    "tool",
}

CONCRETE_ARTIFACT_FORMATS = {
    "avi",
    "bibtex",
    "csv",
    "docx",
    "excel",
    "flv",
    "hdf5",
    "jpeg",
    "jpg",
    "jsonl",
    "markdown",
    "mov",
    "mp3",
    "mp4",
    "mpeg",
    "mpg",
    "netcdf",
    "npy",
    "npz",
    "parquet",
    "pcap",
    "pcapng",
    "pdf",
    "png",
    "pptx",
    "quakeml",
    "sqlite",
    "stl",
    "tsv",
    "wav",
    "webm",
    "webp",
    "wmv",
    "xls",
    "xlsx",
    "xml",
    "yaml",
    "yml",
}

# These values can describe a transport/container without identifying the
# artifact's semantics.  A weak-only match is deterministic only when the two
# skills also share explicit domain evidence; otherwise it is left to the
# bounded relation validator.
WEAK_ARTIFACT_EVIDENCE = {
    "application",
    "audio",
    "city",
    "classification",
    "command",
    "constraint",
    "coordinate",
    "count",
    "csv",
    "description",
    "directory",
    "execution",
    "excel",
    "feature",
    "function",
    "html",
    "id",
    "ids",
    "image",
    "java",
    "jsonl",
    "label",
    "local",
    "media",
    "metadata",
    "name",
    "pdf",
    "problem",
    "python",
    "reference",
    "search",
    "security",
    "segment",
    "serie",
    "series",
    "specification",
    "spreadsheet",
    "sqlite",
    "station",
    "stream",
    "test",
    "time",
    "training",
    "tsv",
    "vulnerability",
    "video",
    "xls",
    "xlsx",
}

# These terminal nouns describe a result/state rather than the artifact that can
# be handed to another skill.  In phrases such as ``security configuration`` or
# ``image description text``, removing the generic suffix must not promote the
# preceding domain modifier into an artifact head.
NON_ARTIFACT_RESULT_HEADS = {
    "analysis",
    "classification",
    "code",
    "configuration",
    "description",
    "documentation",
    "metadata",
    "payload",
    "project",
    "report",
    "request",
    "response",
    "result",
    "schema",
    "status",
    "structure",
    "summary",
    "value",
}

ARTIFACT_CONTAINER_HEADS = {
    "array",
    "data",
    "document",
    "entry",
    "file",
    "item",
    "json",
    "list",
    "object",
    "record",
    "source",
    "string",
    "table",
    "text",
}

PROGRAMMING_LANGUAGE_TOKENS = {
    "csharp",
    "erlang",
    "java",
    "javascript",
    "kotlin",
    "python",
    "ruby",
    "rust",
    "scala",
    "swift",
    "typescript",
}

# A shared language or broad operational/domain word is not, by itself, an
# interface contract.  The bounded LLM validator may still propose a workflow
# or semantic relation when the pair is useful for another reason.
NON_ARTIFACT_SINGLETON_EVIDENCE = PROGRAMMING_LANGUAGE_TOKENS | {
    "control",
    "environment",
    "execution",
    "package",
    "security",
    "test",
}

ALTERNATIVE_GENERIC_TOKENS = {
    "api",
    "automate",
    "automation",
    "composio",
    "execute",
    "execution",
    "integration",
    "mcp",
    "platform",
    "rube",
    "service",
    "skill",
    "tool",
    "tooling",
    "through",
    "workflow",
    "wrapper",
}

SEMANTIC_GENERIC_TOKENS = ALTERNATIVE_GENERIC_TOKENS | {
    "alway",
    "always",
    "analysis",
    "current",
    "data",
    "engineering",
    "first",
    "mathematics",
    "research",
    "schema",
    "search",
    "software",
    "task",
    "via",
}


class UnconfiguredLLMService:
    def __init__(self, model: str, error: Exception | None = None) -> None:
        self.model = model
        self.error = error

    async def send_message(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, str]] | None = None,
        response_model=None,
        **kwargs,
    ):
        details = f" Original error: {self.error}" if self.error else ""
        raise RuntimeError(
            "LLM service is not configured. Set the appropriate model credentials "
            f"for `{self.model}` or pass a custom llm_service in SkillGraphRAG.Config.{details}"
        )


class UnconfiguredEmbeddingService:
    def __init__(
        self,
        model: str,
        embedding_dim: int,
        error: Exception | None = None,
    ) -> None:
        self.model = model
        self.embedding_dim = embedding_dim
        self.error = error

    async def encode(self, texts: list[str], model: str | None = None):
        details = f" Original error: {self.error}" if self.error else ""
        raise RuntimeError(
            "Embedding service is not configured. Set the appropriate model credentials "
            f"for `{self.model}` or pass a custom embedding_service in SkillGraphRAG.Config.{details}"
        )


def parse_model_spec(model_name: str) -> tuple[str | None, str]:
    if "/" not in model_name:
        if model_name.startswith("gemini"):
            return "gemini", model_name
        return None, model_name

    provider, actual_model = model_name.split("/", 1)
    return provider, actual_model


def _secret_value(secret: Any) -> str | None:
    if secret is None:
        return None
    return str(secret.get_secret_value()).strip() or None


def _normalize_openai_compat_base_url(configured: str) -> str:
    configured = configured.strip().rstrip("/")
    if "openrouter.ai/api" in configured and not configured.endswith("/v1"):
        return f"{configured}/v1"
    return configured


def _optional_openai_compat_base_url() -> str | None:
    """When unset, callers should use the vendor default (e.g. api.openai.com), not OpenRouter."""
    raw = str(settings.OPENAI_BASE_URL or "").strip()
    if not raw:
        return None
    return _normalize_openai_compat_base_url(raw)


def _resolve_openrouter_api_key() -> str | None:
    """API key for OpenAI-compatible HTTP APIs (OpenRouter, Azure AI Foundry, local proxies).

    Non-OpenRouter bases always use OPENAI_API_KEY so a global OPENROUTER_API_KEY does not
    override Azure or other keys.
    """
    base = str(settings.OPENAI_BASE_URL or "").strip().lower()
    if base and "openrouter.ai" not in base:
        return _secret_value(settings.OPENAI_API_KEY)
    return _secret_value(settings.OPENROUTER_API_KEY) or _secret_value(
        settings.OPENAI_API_KEY
    )


def _resolve_openrouter_base_url() -> str:
    """Base URL for the explicit ``openrouter/...`` model prefix (defaults to OpenRouter if unset)."""
    configured = str(settings.OPENAI_BASE_URL or "").strip()
    if not configured:
        return DEFAULT_OPENROUTER_API_BASE
    return _normalize_openai_compat_base_url(configured)


def build_default_llm_service() -> BaseLLMService | UnconfiguredLLMService:
    provider, model_name = parse_model_spec(settings.LLM_MODEL)
    try:
        if provider == "gemini":
            api_key = _secret_value(settings.GEMINI_API_KEY)
            return LiteLLMService(model=settings.LLM_MODEL, api_key=api_key)

        if provider == "openrouter":
            return LiteLLMService(
                model=settings.LLM_MODEL,
                api_key=_resolve_openrouter_api_key(),
                base_url=_resolve_openrouter_base_url(),
                response_cache=settings.OPENROUTER_RESPONSE_CACHE,
            )

        if provider == "openai":
            # LiteLLM: ``openai/<deployment>`` + optional OPENAI_BASE_URL (Azure, proxies, or OpenRouter).
            optional_base = _optional_openai_compat_base_url()
            api_key = (
                _secret_value(settings.OPENAI_API_KEY)
                if optional_base is None
                else _resolve_openrouter_api_key()
            )
            return LiteLLMService(
                model=settings.LLM_MODEL,
                api_key=api_key,
                base_url=optional_base,
            )

        api_key = _secret_value(settings.OPENAI_API_KEY)
        return LiteLLMService(model=model_name, api_key=api_key)
    except Exception as exc:
        logger.warning(f"Falling back to unconfigured LLM placeholder: {exc}")
        return UnconfiguredLLMService(settings.LLM_MODEL, exc)


def build_default_embedding_service() -> Any:
    # Read once: pydantic-settings + tests may refresh env-backed fields between accesses.
    embedding_model = settings.EMBEDDING_MODEL
    provider, model_name = parse_model_spec(embedding_model)
    try:
        if provider == "gemini":
            api_key = _secret_value(settings.GEMINI_API_KEY)
            return LiteLLMEmbeddingService(
                model=embedding_model,
                embedding_dim=settings.EMBEDDING_DIM,
                embedding_concurrency=settings.EMBEDDING_CONCURRENCY,
                api_key=api_key,
            )

        if provider == "openrouter":
            # Legacy: ``openrouter/openai/<model>`` — second segment is passed to LiteLLM.
            return LiteLLMEmbeddingService(
                model=model_name,
                embedding_dim=settings.EMBEDDING_DIM,
                embedding_concurrency=settings.EMBEDDING_CONCURRENCY,
                api_key=_resolve_openrouter_api_key(),
                base_url=_resolve_openrouter_base_url(),
                response_cache=settings.OPENROUTER_RESPONSE_CACHE,
            )

        if provider == "openai":
            # ``openai/<deployment>`` with OPENAI_BASE_URL for Azure / proxies; omit URL for api.openai.com.
            optional_base = _optional_openai_compat_base_url()
            api_key = (
                _secret_value(settings.OPENAI_API_KEY)
                if optional_base is None
                else _resolve_openrouter_api_key()
            )
            return LiteLLMEmbeddingService(
                model=embedding_model,
                embedding_dim=settings.EMBEDDING_DIM,
                embedding_concurrency=settings.EMBEDDING_CONCURRENCY,
                api_key=api_key,
                base_url=optional_base,
            )

        api_key = _secret_value(settings.OPENAI_API_KEY)
        return LiteLLMEmbeddingService(
            model=model_name,
            embedding_dim=settings.EMBEDDING_DIM,
            embedding_concurrency=settings.EMBEDDING_CONCURRENCY,
            api_key=api_key,
        )
    except Exception as exc:
        logger.warning(f"Falling back to unconfigured embedding placeholder: {exc}")
        return UnconfiguredEmbeddingService(
            embedding_model,
            settings.EMBEDDING_DIM,
            exc,
        )


@dataclass
class SkillGraphRAG(
    BaseGraphRAG[GTEmbedding, GTHash, GTChunk, SkillNode, SkillEdge, TId]
):
    """Graph-backed skill retrieval with explicit offline linking and online PPR."""

    working_dir: str = field(default=settings.WORKING_DIR)
    domain: str = field(default=settings.DOMAIN)
    example_queries: str = field(default="")
    entity_types: list[str] = field(default_factory=lambda: ["Skill"])
    n_checkpoints: int = field(default=0)
    config: "SkillGraphRAG.Config" = field(
        default_factory=lambda: SkillGraphRAG.Config()
    )
    bootstrapped_from: str = field(default="", init=False)
    construction_counters: ConstructionCounters = field(
        default_factory=ConstructionCounters,
        init=False,
    )

    @dataclass
    class Config:
        llm_service: Any = field(default_factory=build_default_llm_service)
        embedding_service: Any = field(default_factory=build_default_embedding_service)
        working_dir: str = field(default=settings.WORKING_DIR)
        prebuilt_working_dir: str | None = field(default=settings.PREBUILT_WORKING_DIR)
        domain: str = field(default=settings.DOMAIN)
        use_full_markdown: bool = field(default=settings.USE_FULL_MARKDOWN)
        link_top_k: int = field(default=settings.LINK_TOP_K)
        seed_top_k: int = field(default=settings.SEED_TOP_K)
        seed_candidate_top_k_semantic: int = field(
            default=settings.SEED_CANDIDATE_TOP_K_SEMANTIC
        )
        seed_candidate_top_k_lexical: int = field(
            default=settings.SEED_CANDIDATE_TOP_K_LEXICAL
        )
        retrieval_top_n: int = field(default=settings.RETRIEVAL_TOP_N)
        enable_semantic_linking: bool = field(default=settings.ENABLE_SEMANTIC_LINKING)
        dependency_match_threshold: float = field(
            default=settings.DEPENDENCY_MATCH_THRESHOLD
        )
        relation_min_confidence: float = field(default=settings.RELATION_MIN_CONFIDENCE)
        relink_concurrency: int = field(default=settings.RELINK_CONCURRENCY)
        relink_checkpoint_every: int = field(default=settings.RELINK_CHECKPOINT_EVERY)
        extraction_concurrency: int = field(default=settings.EXTRACTION_CONCURRENCY)
        ppr_damping: float = field(default=settings.PPR_DAMPING)
        ppr_max_iter: int = field(default=settings.PPR_MAX_ITER)
        ppr_tolerance: float = field(default=settings.PPR_TOLERANCE)
        max_skill_chars: int = field(default=settings.MAX_SKILL_CHARS)
        max_context_chars: int = field(default=settings.MAX_CONTEXT_CHARS)
        snippet_chars: int = field(default=settings.SNIPPET_CHARS)
        rerank_candidate_multiplier: int = field(
            default=settings.RERANK_CANDIDATE_MULTIPLIER
        )
        enable_query_rewrite: bool = field(default=settings.ENABLE_QUERY_REWRITE)

    def _detect_workspace_embedding_dim(self) -> int | None:
        workspace = Path(self.working_dir).expanduser()
        if not workspace.exists() or not workspace.is_dir():
            return None

        for file_path in sorted(workspace.iterdir()):
            match = HNSW_INDEX_FILENAME_PATTERN.match(file_path.name)
            if match is not None:
                return int(match.group(1))

        return None

    def _resolve_entity_storage_embedding_dim(self) -> int:
        configured_dim = int(
            getattr(
                self.config.embedding_service,
                "embedding_dim",
                settings.EMBEDDING_DIM,
            )
        )
        workspace_dim = self._detect_workspace_embedding_dim()
        if workspace_dim is None:
            return configured_dim

        if workspace_dim != configured_dim:
            logger.info(
                "GoS: detected workspace embedding dim "
                f"{workspace_dim} in `{self.working_dir}`; "
                f"overriding configured dim {configured_dim} for workspace loading."
            )
        return workspace_dim

    def __post_init__(self):
        self.working_dir = self.config.working_dir
        self.domain = self.config.domain
        self.llm_service = self.config.llm_service
        self.bootstrapped_from = self._bootstrap_prebuilt_workspace()
        self.chunking_service = cast(
            BaseChunkingService[GTChunk],
            DefaultChunkingService(),
        )

        self.information_extraction_service = SkillInformationExtractionService(
            use_full_markdown=self.config.use_full_markdown,
            snippet_chars=self.config.snippet_chars,
            extraction_concurrency=self.config.extraction_concurrency,
            graph_upsert=SkillGraphUpsertPolicy(
                config=None,
                nodes_upsert_cls=SkillNodeUpsertPolicy,
                edges_upsert_cls=SkillEdgeUpsertPolicy,
            ),
        )

        entity_storage = HNSWVectorStorage[TId, GTEmbedding](
            config=HNSWVectorStorageConfig()
        )
        entity_storage.embedding_dim = self._resolve_entity_storage_embedding_dim()

        self.state_manager = DefaultStateManagerService(
            workspace=Workspace(self.working_dir),
            graph_storage=DirectedIGraphStorage(
                config=IGraphStorageConfig(SkillNode, SkillEdge)
            ),
            entity_storage=entity_storage,
            chunk_storage=PickleIndexedKeyValueStorage[GTHash, GTChunk](config=None),
            embedding_service=self.config.embedding_service,
            node_upsert_policy=SkillNodeUpsertPolicy(config=None),
            edge_upsert_policy=SkillEdgeUpsertPolicy(config=None),
            # fast-graphrag otherwise injects untyped embedding-similarity
            # `description="is"` edges. GoS semantic edges are created only by
            # the scoped typed relation validator.
            insert_similarity_score_threshold=2.0,
        )
        self.llm_service = self.config.llm_service

    def _bootstrap_prebuilt_workspace(self) -> str:
        configured_source = str(self.config.prebuilt_working_dir or "").strip()
        if not configured_source:
            return ""

        source = Path(configured_source).expanduser()
        target = Path(self.working_dir).expanduser()

        try:
            if source.resolve() == target.resolve():
                return ""
        except OSError:
            pass

        if not source.exists() or not source.is_dir():
            logger.warning(
                f"GoS: prebuilt workspace `{source}` does not exist or is not a directory."
            )
            return ""

        if target.exists() and any(target.iterdir()):
            logger.info(
                f"GoS: workspace `{target}` already has content, skipping bootstrap from `{source}`."
            )
            return ""

        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target, dirs_exist_ok=True)
        logger.info(
            f"GoS: bootstrapped workspace `{target}` from prebuilt graph `{source}`."
        )
        return str(source)

    def _prepare_metadata(
        self, skill_text: str, metadata: dict[str, Any] | None
    ) -> dict[str, Any]:
        prepared = dict(metadata or {})
        prepared.setdefault("raw_content", skill_text)
        prepared.setdefault("snippet_chars", self.config.snippet_chars)
        return prepared

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    async def _load_all_nodes(self) -> list[SkillNode]:
        target = self.state_manager.graph_storage
        node_count = await target.node_count()
        nodes: list[SkillNode] = []
        for index in range(node_count):
            node = await target.get_node_by_index(index)
            if node is not None:
                nodes.append(node)
        return nodes

    async def _load_all_edges(self) -> list[SkillEdge]:
        target = self.state_manager.graph_storage
        edge_count = await target.edge_count()

        for getter_name in ("get_edge_by_index", "get_relation_by_index"):
            getter = getattr(target, getter_name, None)
            if getter is None:
                continue

            edges: list[SkillEdge] = []
            for index in range(edge_count):
                edge = await self._maybe_await(getter(index))
                if edge is not None:
                    edges.append(edge)
            return edges

        raw_graph = (
            getattr(target, "_graph", None)
            or getattr(target, "graph", None)
            or getattr(target, "g", None)
        )
        if raw_graph is None:
            logger.warning(
                "Graph storage does not expose edge iteration; retrieval will use nodes only."
            )
            return []

        edges = []
        vertices = raw_graph.vs
        for raw_edge in raw_graph.es:
            attrs = raw_edge.attributes()
            edges.append(
                SkillEdge(
                    source=vertices[raw_edge.source]["name"],
                    target=vertices[raw_edge.target]["name"],
                    description=attrs.get("description", ""),
                    type=attrs.get("type", "dependency"),
                    weight=float(attrs.get("weight", 1.0)),
                    confidence=float(attrs.get("confidence", 1.0)),
                )
            )
        return edges

    @staticmethod
    def _node_lookup_maps(
        nodes: list[SkillNode],
    ) -> tuple[dict[str, SkillNode], dict[str, SkillNode], dict[str, SkillNode]]:
        by_skill_id: dict[str, SkillNode] = {}
        by_source_path: dict[str, SkillNode] = {}
        by_name: dict[str, SkillNode] = {}

        for node in nodes:
            if node.skill_id:
                by_skill_id.setdefault(node.skill_id, node)
            if node.source_path:
                by_source_path.setdefault(node.source_path, node)
            if node.name:
                by_name.setdefault(node.name, node)

        return by_skill_id, by_source_path, by_name

    @staticmethod
    def _find_existing_node(
        *,
        name: str,
        skill_id: str,
        source_path: str,
        by_skill_id: dict[str, SkillNode],
        by_source_path: dict[str, SkillNode],
        by_name: dict[str, SkillNode],
    ) -> SkillNode | None:
        if skill_id and skill_id in by_skill_id:
            return by_skill_id[skill_id]
        if source_path and source_path in by_source_path:
            return by_source_path[source_path]
        if name and name in by_name:
            return by_name[name]
        return None

    async def _graph_counts(self) -> tuple[int, int]:
        await self.state_manager.query_start()
        try:
            node_count = await self.state_manager.graph_storage.node_count()
            edge_count = await self.state_manager.graph_storage.edge_count()
        finally:
            await self.state_manager.query_done()
        return node_count, edge_count

    def _signature_tokens(self, values: list[str]) -> set[str]:
        tokens: set[str] = set()
        for value in values:
            lowered = value.lower()
            normalized = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
            if normalized:
                tokens.add(normalized)
            for token in re.findall(r"[a-z0-9]+", lowered):
                token = token.rstrip("s") if len(token) > 3 else token
                if len(token) < 3 or token in TOKEN_STOPWORDS:
                    continue
                tokens.add(token)
        return tokens

    def _schema_overlap_score(
        self,
        producer_values: list[str],
        consumer_values: list[str],
    ) -> tuple[float, list[str]]:
        best_score = 0.0
        best_evidence: set[str] = set()

        for producer in producer_values:
            producer_tokens, producer_heads = self._schema_artifact_signature(producer)
            if not producer_tokens:
                continue
            for consumer in consumer_values:
                consumer_tokens, consumer_heads = self._schema_artifact_signature(
                    consumer
                )
                if not consumer_tokens:
                    continue

                overlap = producer_tokens & consumer_tokens
                if not overlap:
                    continue
                compatible_types = (producer_heads & consumer_heads) | (
                    overlap & CONCRETE_ARTIFACT_FORMATS
                )
                # Schema generators disagree about neutral container heads.
                # Preserve a multi-token signature here; the deterministic
                # gate below still requires a concrete format or domain match.
                if len(overlap) >= 2 and producer_heads and consumer_heads:
                    compatible_types |= overlap - NON_ARTIFACT_RESULT_HEADS
                if not compatible_types:
                    continue

                score = len(overlap) / max(
                    min(len(producer_tokens), len(consumer_tokens)),
                    1,
                )
                if score > best_score:
                    best_score = score
                    best_evidence = overlap

        return best_score, sorted(best_evidence)

    @staticmethod
    def _schema_artifact_tokens(value: str) -> set[str]:
        tokens, _ = SkillGraphRAG._schema_artifact_signature(value)
        return tokens

    @staticmethod
    def _schema_artifact_signature(value: str) -> tuple[set[str], set[str]]:
        rendered = str(value or "").strip()
        if rendered.startswith("```"):
            return set(), set()

        # Completion models sometimes describe an artifact by where it came from
        # (for example, "count data from video footage").  The source context is
        # not part of the produced artifact type and must not induce an edge.
        primary = re.split(
            r"\b(?:based\s+on|derived\s+from|for|from|using|via)\b",
            rendered,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]

        def meaningful_tokens(part: str) -> list[str]:
            result: list[str] = []
            for raw_token in re.findall(r"[a-z0-9]+", part.lower()):
                if raw_token in GENERIC_SCHEMA_TOKENS:
                    continue
                token = raw_token.rstrip("s") if len(raw_token) > 3 else raw_token
                if len(token) < 3 or token in GENERIC_SCHEMA_TOKENS:
                    continue
                result.append(token)
            return result

        ordered_tokens = meaningful_tokens(primary)
        tokens: set[str] = set()
        tokens.update(ordered_tokens)

        heads: set[str] = set()
        for part in re.split(r"\b(?:and|or)\b|[,;/|]+", primary, flags=re.I):
            # Parenthetical text normally contains encoding/shape qualifiers,
            # not the main artifact noun.  Preserve known concrete formats but
            # determine the semantic head from the text before the qualifier.
            qualifier_tokens = {
                raw_token.rstrip("s") if len(raw_token) > 3 else raw_token
                for raw_token in re.findall(r"[a-z0-9]+", part.lower())
            }
            heads.update(qualifier_tokens & CONCRETE_ARTIFACT_FORMATS)

            main = re.split(r"[([{]", part, maxsplit=1)[0]
            raw_tokens = [
                raw_token.rstrip("s") if len(raw_token) > 3 else raw_token
                for raw_token in re.findall(r"[a-z0-9]+", main.lower())
            ]
            raw_tokens = [token for token in raw_tokens if len(token) >= 3]
            if not raw_tokens:
                continue

            terminal_index = len(raw_tokens) - 1
            while (
                terminal_index >= 0
                and raw_tokens[terminal_index] in ARTIFACT_CONTAINER_HEADS
            ):
                terminal_index -= 1
            if terminal_index < 0:
                continue

            terminal = raw_tokens[terminal_index]
            if terminal in NON_ARTIFACT_RESULT_HEADS:
                continue
            if terminal not in GENERIC_SCHEMA_TOKENS:
                heads.add(terminal)

        return tokens, heads

    def _extract_task_name(self, query: str) -> str:
        tokens = [token for token in re.split(r"[^a-zA-Z0-9]+", query.strip()) if token]
        if not tokens:
            return ""

        slug = "-".join(token.lower() for token in tokens[:8])
        return slug[:80]

    @staticmethod
    def _dedupe_text(values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            normalized = re.sub(r"\s+", " ", str(value or "").strip())
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(normalized)
        return result

    def _extract_artifacts(self, query: str) -> list[str]:
        artifacts = re.findall(
            r"[A-Za-z0-9_./-]+\.(?:py|md|json|csv|stl|dot|txt|yaml|yml|bib|pptx|xlsx|docx)",
            query,
        )
        return self._dedupe_text(
            [artifact.strip() for artifact in artifacts if artifact.strip()]
        )

    def _fallback_query_schema(self, query: str) -> QuerySchema:
        normalized_query = re.sub(r"\s+", " ", query.strip())
        artifacts = self._extract_artifacts(normalized_query)
        keywords = sorted(self._signature_tokens([normalized_query, *artifacts]))
        return QuerySchema(
            goal=normalized_query,
            task_name=self._extract_task_name(normalized_query),
            artifacts=artifacts,
            keywords=keywords,
        )

    def _normalize_query_schema(
        self, query: str, schema: QuerySchema | None
    ) -> QuerySchema:
        fallback = self._fallback_query_schema(query)
        if schema is None:
            return fallback

        task_name = schema.task_name.strip() or fallback.task_name
        goal = schema.goal.strip() or fallback.goal
        artifacts = self._dedupe_text(schema.artifacts + fallback.artifacts)
        domain = self._dedupe_text(schema.domain)
        operations = self._dedupe_text(schema.operations)
        constraints = self._dedupe_text(schema.constraints)
        keyword_seed = [
            goal,
            task_name,
            *domain,
            *operations,
            *artifacts,
            *constraints,
            *schema.keywords,
            *fallback.keywords,
        ]
        keywords = self._dedupe_text(sorted(self._signature_tokens(keyword_seed)))

        return QuerySchema(
            goal=goal,
            task_name=task_name,
            domain=domain,
            operations=operations,
            artifacts=artifacts,
            constraints=constraints,
            keywords=keywords,
        )

    async def _rewrite_query_schema_with_llm(self, query: str) -> QuerySchema | None:
        try:
            schema, _ = await self.llm_service.send_message(
                system_prompt=PROMPTS["query_rewrite_system"],
                prompt=PROMPTS["query_rewrite_prompt"].format(query=query.strip()),
                response_model=QuerySchema,
            )
        except Exception as exc:
            logger.debug(f"Query rewrite fell back to lexical normalization: {exc}")
            return None

        return self._normalize_query_schema(query, schema)

    async def _rewrite_query_schema_async(self, query: str) -> QuerySchema:
        if not self.config.enable_query_rewrite:
            return self._fallback_query_schema(query)

        inferred = await self._rewrite_query_schema_with_llm(query)
        return self._normalize_query_schema(query, inferred)

    def _query_schema_values(self, query_schema: QuerySchema) -> list[str]:
        values = [query_schema.goal, query_schema.task_name]
        values.extend(query_schema.domain)
        values.extend(query_schema.operations)
        values.extend(query_schema.artifacts)
        values.extend(query_schema.constraints)
        values.extend(query_schema.keywords)
        return [value for value in values if value]

    def _token_overlap_score(self, query_tokens: set[str], values: list[str]) -> float:
        candidate_tokens = self._signature_tokens(values)
        if not query_tokens or not candidate_tokens:
            return 0.0
        overlap = query_tokens & candidate_tokens
        if not overlap:
            return 0.0
        return len(overlap) / max(len(query_tokens), 1)

    def _field_bonus(
        self, query_tokens: set[str], values: list[str], weight: float
    ) -> float:
        return weight * self._token_overlap_score(query_tokens, values)

    def _rerank_skill_score(
        self,
        query_schema: QuerySchema,
        node: SkillNode,
        graph_score: float,
        semantic_rank: int | None,
    ) -> float:
        query_tokens = self._signature_tokens(self._query_schema_values(query_schema))
        score = graph_score
        score += (
            self._field_bonus(query_tokens, [query_schema.task_name], 0.35)
            if query_schema.task_name
            else 0.0
        )
        score += self._field_bonus(query_tokens, [node.name], 1.25)
        score += self._field_bonus(
            query_tokens, [node.one_line_capability, node.description], 0.9
        )
        score += self._field_bonus(query_tokens, node.domain_tags_list, 1.15)
        score += self._field_bonus(query_tokens, node.tooling_list, 0.95)
        score += self._field_bonus(
            query_tokens, node.input_types + node.output_types, 0.75
        )
        score += self._field_bonus(query_tokens, node.example_tasks_list, 0.8)
        score += self._field_bonus(query_tokens, node.script_entrypoints_list, 0.6)

        normalized_query_text = "\n".join(
            self._query_schema_values(query_schema)
        ).lower()
        normalized_node_name = re.sub(r"[^a-z0-9]+", " ", node.name.lower()).strip()
        if normalized_node_name and normalized_node_name in normalized_query_text:
            score += 1.2

        artifact_overlap = self._shared_field_score(
            query_schema.artifacts, node.script_entrypoints_list
        )
        if artifact_overlap:
            score += 0.9 * artifact_overlap

        if node.script_entrypoints_list:
            score += 0.08
        if semantic_rank is not None:
            score += 0.2 / float(semantic_rank)
        if query_schema.domain and node.domain_tags_list:
            overlap = self._signature_tokens(
                query_schema.domain
            ) & self._signature_tokens(node.domain_tags_list)
            if overlap:
                score += 0.35
        return score

    def _node_text_values(self, node: SkillNode) -> list[str]:
        return [
            node.name,
            node.description,
            node.one_line_capability,
            *node.input_types,
            *node.output_types,
            *node.domain_tags_list,
            *node.tooling_list,
            *node.example_tasks_list,
            *node.script_entrypoints_list,
            node.rendered_snippet,
        ]

    def _rewrite_node_query_schema(self, node: SkillNode) -> QuerySchema:
        text_values = self._node_text_values(node)
        goal = node.one_line_capability or node.description or node.name
        task_name = re.sub(r"[^a-z0-9]+", "-", node.name.lower()).strip("-")
        domain = self._dedupe_text(node.domain_tags_list)
        operations = self._dedupe_text(node.tooling_list + node.example_tasks_list)
        artifacts = self._dedupe_text(
            self._extract_artifacts("\n".join(text_values))
            + node.script_entrypoints_list
        )
        constraints = self._dedupe_text(
            node.compatibility_list + node.allowed_tools_list
        )
        keywords = sorted(self._signature_tokens(text_values))
        return QuerySchema(
            goal=goal,
            task_name=task_name,
            domain=domain,
            operations=operations,
            artifacts=artifacts,
            constraints=constraints,
            keywords=keywords,
        )

    def _shared_field_score(
        self, left_values: list[str], right_values: list[str]
    ) -> float:
        left_tokens = self._signature_tokens(left_values)
        right_tokens = self._signature_tokens(right_values)
        if not left_tokens or not right_tokens:
            return 0.0
        overlap = left_tokens & right_tokens
        if not overlap:
            return 0.0
        return len(overlap) / max(min(len(left_tokens), len(right_tokens)), 1)

    def _link_pair_feature_score(
        self,
        source_node: SkillNode,
        candidate_node: SkillNode,
    ) -> tuple[float, bool]:
        score = 0.0
        evidence = False

        shared_domain = self._shared_field_score(
            source_node.domain_tags_list,
            candidate_node.domain_tags_list,
        )
        if shared_domain >= 0.5:
            score += 1.0 * shared_domain
            evidence = True

        shared_tooling = self._shared_field_score(
            source_node.tooling_list,
            candidate_node.tooling_list,
        )
        if shared_tooling >= 0.5:
            score += 0.75 * shared_tooling
            evidence = True

        shared_examples = self._shared_field_score(
            source_node.example_tasks_list,
            candidate_node.example_tasks_list,
        )
        if shared_examples >= 0.5:
            score += 0.6 * shared_examples
            evidence = True

        shared_scripts = self._shared_field_score(
            source_node.script_entrypoints_list,
            candidate_node.script_entrypoints_list,
        )
        if shared_scripts >= 0.5:
            score += 0.35 * shared_scripts
            evidence = True

        schema_forward, _ = self._schema_overlap_score(
            source_node.output_types,
            candidate_node.input_types,
        )
        schema_reverse, _ = self._schema_overlap_score(
            candidate_node.output_types,
            source_node.input_types,
        )
        schema_score = max(schema_forward, schema_reverse)
        if schema_score:
            score += 0.85 * schema_score
            evidence = True

        shared_io = self._shared_field_score(
            source_node.input_types + source_node.output_types,
            candidate_node.input_types + candidate_node.output_types,
        )
        if shared_io >= 0.5:
            score += 0.4 * shared_io
            evidence = True

        return score, evidence

    def _dependency_evidence_supported(
        self,
        producer: SkillNode,
        consumer: SkillNode,
        evidence_values: list[str],
    ) -> bool:
        """Gate weak container/format matches with explicit shared domain evidence."""
        evidence = set(evidence_values)
        if not evidence:
            return False
        if len(evidence) == 1 and evidence <= NON_ARTIFACT_SINGLETON_EVIDENCE:
            return False
        # A format plus a semantic artifact descriptor is a concrete contract
        # even across domains (for example, extracted image + PNG).  A bare
        # ubiquitous format such as CSV/PDF still needs domain agreement to
        # avoid dense generic-format hubs.
        if len(evidence) > 1 and evidence & CONCRETE_ARTIFACT_FORMATS:
            return True
        if evidence <= WEAK_ARTIFACT_EVIDENCE:
            return (
                self._shared_field_score(
                    producer.domain_tags_list,
                    consumer.domain_tags_list,
                )
                >= 0.5
            )
        # Natural-language type labels such as ``network topology data`` are
        # ambiguous across domains.  Without a concrete serialization format,
        # deterministic linking additionally requires explicit domain-token
        # agreement; otherwise the bounded validator decides the relation.
        producer_domain = (
            self._signature_tokens(producer.domain_tags_list)
            - SEMANTIC_GENERIC_TOKENS
        )
        consumer_domain = (
            self._signature_tokens(consumer.domain_tags_list)
            - SEMANTIC_GENERIC_TOKENS
        )
        shared_domain = producer_domain & consumer_domain
        if producer_domain and consumer_domain:
            # Explicit, disjoint domain metadata is contradictory evidence even
            # when the natural-language schema labels happen to be identical
            # (for example, packet versus electric-grid network topology).
            return bool(shared_domain)

        # Sparse skill front matter often omits domain tags.  Do not turn that
        # absence into negative evidence: a multi-token concrete contract such
        # as ``seismic catalog`` remains a useful deterministic handoff.  One
        # broad token is intentionally insufficient and stays validator-only.
        strong_evidence = evidence - WEAK_ARTIFACT_EVIDENCE
        return len(strong_evidence) >= 2

    def _alternative_relation_supported(
        self,
        left: SkillNode,
        right: SkillNode,
    ) -> bool:
        """Require alternatives to share interface shape and concrete capability."""

        def interface_overlap(left_values: list[str], right_values: list[str]) -> float:
            def canonical(value: str) -> str:
                return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

            left_exact = {canonical(value) for value in left_values if canonical(value)}
            right_exact = {
                canonical(value) for value in right_values if canonical(value)
            }
            if left_exact & right_exact:
                return 1.0
            interface_stopwords = {
                "data",
                "file",
                "files",
                "input",
                "object",
                "output",
                "result",
                "results",
                "string",
            }

            def raw_tokens(values: list[str]) -> set[str]:
                return {
                    token
                    for value in values
                    for token in re.findall(r"[a-z0-9]+", value.lower())
                    if len(token) >= 3 and token not in interface_stopwords
                }

            left_raw = raw_tokens(left_values)
            right_raw = raw_tokens(right_values)
            raw_overlap = left_raw & right_raw
            if raw_overlap:
                return len(raw_overlap) / max(min(len(left_raw), len(right_raw)), 1)
            return self._shared_field_score(left_values, right_values)

        input_overlap = interface_overlap(left.input_types, right.input_types)
        output_overlap = interface_overlap(left.output_types, right.output_types)
        if input_overlap < 0.25 or output_overlap < 0.25:
            return False

        left_tokens = (
            self._signature_tokens(
                [left.name, left.one_line_capability, left.description]
            )
            - ALTERNATIVE_GENERIC_TOKENS
        )
        right_tokens = (
            self._signature_tokens(
                [right.name, right.one_line_capability, right.description]
            )
            - ALTERNATIVE_GENERIC_TOKENS
        )
        if not left_tokens or not right_tokens:
            return False
        overlap = left_tokens & right_tokens
        return len(overlap) / max(min(len(left_tokens), len(right_tokens)), 1) >= 0.25

    def _alternative_dominates_dependency(
        self,
        left: SkillNode,
        right: SkillNode,
    ) -> bool:
        """Return true only for near-substitutes, not merely related algorithms."""
        if not self._alternative_relation_supported(left, right):
            return False
        return (
            self._shared_field_score(left.input_types, right.input_types) >= 0.5
            and self._shared_field_score(left.output_types, right.output_types) >= 0.5
        )

    def _semantic_relation_supported(
        self,
        left: SkillNode,
        right: SkillNode,
    ) -> bool:
        """Require a concrete shared capability/domain, not a common wrapper."""

        def concrete_tokens(node: SkillNode) -> set[str]:
            return (
                self._signature_tokens(
                    [
                        node.name,
                        node.one_line_capability,
                        node.description,
                    ]
                )
                - SEMANTIC_GENERIC_TOKENS
            )

        left_tokens = concrete_tokens(left)
        right_tokens = concrete_tokens(right)

        def generic_automation_wrapper(node: SkillNode) -> bool:
            name = node.name.lower().replace("_", "-")
            wrapper_tokens = self._signature_tokens(
                [
                    node.name,
                    node.description,
                    *node.domain_tags_list,
                    *node.tooling_list,
                ]
            )
            return name.endswith("-automation") and bool(
                wrapper_tokens & {"composio", "mcp", "rube", "wrapper"}
            )

        # Large tool catalogs contain many mechanically generated automation
        # wrappers.  A shared broad domain (for example, messaging) does not
        # make wrappers for two unrelated products semantically equivalent.
        if generic_automation_wrapper(left) and generic_automation_wrapper(right):
            overlap = left_tokens & right_tokens
            return len(overlap) >= 2 or (
                bool(overlap)
                and len(overlap)
                / max(min(len(left_tokens), len(right_tokens)), 1)
                >= 0.2
            )

        left_domain = (
            self._signature_tokens(left.domain_tags_list) - SEMANTIC_GENERIC_TOKENS
        )
        right_domain = (
            self._signature_tokens(right.domain_tags_list) - SEMANTIC_GENERIC_TOKENS
        )
        if left_domain & right_domain:
            return True

        if not left_tokens or not right_tokens:
            return False
        overlap = left_tokens & right_tokens
        return len(overlap) >= 2 or (
            len(overlap) / max(min(len(left_tokens), len(right_tokens)), 1) >= 0.2
        )

    def _pair_evidence_tokens_for_node(
        self,
        node: SkillNode,
    ) -> dict[str, set[str]]:
        schema_tokens: set[str] = set()
        for value in node.input_types + node.output_types:
            schema_tokens.update(self._schema_artifact_tokens(value))
        return {
            "domain": self._signature_tokens(node.domain_tags_list),
            "tooling": self._signature_tokens(node.tooling_list),
            "examples": self._signature_tokens(node.example_tasks_list),
            "scripts": self._signature_tokens(node.script_entrypoints_list),
            "io": self._signature_tokens(node.input_types + node.output_types)
            | schema_tokens,
        }

    def _build_pair_evidence_indexes(
        self,
        nodes: list[SkillNode],
    ) -> dict[str, dict[str, set[int]]]:
        indexes: dict[str, dict[str, set[int]]] = {
            category: {}
            for category in ("domain", "tooling", "examples", "scripts", "io")
        }
        for index, node in enumerate(nodes):
            for category, tokens in self._pair_evidence_tokens_for_node(node).items():
                postings = indexes[category]
                for token in tokens:
                    postings.setdefault(token, set()).add(index)
        return indexes

    def _evidence_candidate_indices_for_node(
        self,
        node: SkillNode,
        indexes: dict[str, dict[str, set[int]]],
        *,
        node_index: int | None = None,
    ) -> set[int]:
        candidates: set[int] = set()
        for category, tokens in self._pair_evidence_tokens_for_node(node).items():
            postings = indexes.get(category, {})
            for token in tokens:
                candidates.update(postings.get(token, set()))
        if node_index is not None:
            candidates.discard(node_index)
        return candidates

    def _link_candidate_score(
        self,
        source_schema: QuerySchema,
        source_node: SkillNode,
        candidate_node: SkillNode,
        graph_score: float,
        semantic_rank: int | None,
    ) -> tuple[float, bool]:
        query_tokens = self._signature_tokens(self._query_schema_values(source_schema))
        score = graph_score
        score += self._field_bonus(query_tokens, [candidate_node.name], 1.05)
        score += self._field_bonus(
            query_tokens,
            [candidate_node.one_line_capability, candidate_node.description],
            0.8,
        )
        score += self._field_bonus(query_tokens, candidate_node.domain_tags_list, 1.0)
        score += self._field_bonus(query_tokens, candidate_node.tooling_list, 0.85)
        score += self._field_bonus(
            query_tokens,
            candidate_node.input_types + candidate_node.output_types,
            0.55,
        )
        score += self._field_bonus(query_tokens, candidate_node.example_tasks_list, 0.7)
        score += self._field_bonus(
            query_tokens, candidate_node.script_entrypoints_list, 0.45
        )

        pair_score, pair_evidence = self._link_pair_feature_score(
            source_node, candidate_node
        )
        score += pair_score
        if semantic_rank is not None:
            score += 0.2 / float(semantic_rank)

        lexical_overlap = self._token_overlap_score(
            query_tokens, self._node_text_values(candidate_node)
        )
        if not pair_evidence:
            score -= max(0.4, lexical_overlap)
        has_evidence = pair_evidence
        return score, has_evidence

    def _lexical_candidate_scores_for_node(
        self,
        node: SkillNode,
        nodes: list[SkillNode],
        node_index: int,
        candidate_top_k: int,
        candidate_indices: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        source_schema = self._rewrite_node_query_schema(node)
        scored: list[tuple[int, float]] = []
        indices = (
            range(len(nodes))
            if candidate_indices is None
            else sorted(candidate_indices)
        )
        for index in indices:
            if index == node_index:
                continue
            candidate = nodes[index]
            score, has_evidence = self._link_candidate_score(
                source_schema,
                node,
                candidate,
                0.0,
                None,
            )
            if has_evidence:
                scored.append((index, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:candidate_top_k]

    async def _semantic_candidate_scores_for_node(
        self,
        node: SkillNode,
        nodes: list[SkillNode],
        node_index: int,
        candidate_top_k: int,
    ) -> list[tuple[int, float]]:
        source_schema = self._rewrite_node_query_schema(node)
        query_text = source_schema.to_query_text() or node.to_str()
        try:
            node_embedding = await self.config.embedding_service.encode([query_text])
            indices, _ = await self.state_manager.entity_storage.get_knn(
                node_embedding,
                top_k=candidate_top_k,
            )
        except Exception as exc:
            logger.warning(f"Skipping semantic candidate search for {node.name}: {exc}")
            return []

        candidates: list[tuple[int, float]] = []
        seen: set[int] = set()
        for rank, raw_index in enumerate(indices[0], start=1):
            index = int(raw_index)
            if index == node_index or index < 0 or index >= len(nodes) or index in seen:
                continue
            seen.add(index)
            score, has_evidence = self._link_candidate_score(
                source_schema,
                node,
                nodes[index],
                1.0 / float(rank),
                rank,
            )
            if has_evidence:
                candidates.append((index, score))

        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[:candidate_top_k]

    async def _rank_link_candidates_for_node(
        self,
        node: SkillNode,
        nodes: list[SkillNode],
        node_index: int,
    ) -> list[int]:
        candidate_top_k = max(
            self.config.link_top_k,
            self.config.link_top_k * max(self.config.rerank_candidate_multiplier, 1),
        )
        semantic_candidates = await self._semantic_candidate_scores_for_node(
            node,
            nodes,
            node_index,
            candidate_top_k,
        )
        lexical_candidates = self._lexical_candidate_scores_for_node(
            node,
            nodes,
            node_index,
            candidate_top_k,
        )

        combined_scores: dict[int, float] = {}
        for index, score in semantic_candidates + lexical_candidates:
            combined_scores[index] = max(
                score, combined_scores.get(index, float("-inf"))
            )

        ranked = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
        return [index for index, _ in ranked[:candidate_top_k]]

    def _build_io_indexes(
        self,
        nodes: list[SkillNode],
    ) -> tuple[dict[str, set[int]], dict[str, set[int]]]:
        output_index: dict[str, set[int]] = {}
        input_index: dict[str, set[int]] = {}

        for index, node in enumerate(nodes):
            for value in node.output_types:
                for token in self._schema_artifact_tokens(value):
                    output_index.setdefault(token, set()).add(index)
            for value in node.input_types:
                for token in self._schema_artifact_tokens(value):
                    input_index.setdefault(token, set()).add(index)

        return output_index, input_index

    async def _prepare_focus_link_jobs(
        self,
        nodes: list[SkillNode],
        focus_names: set[str],
    ) -> list[FocusLinkJob]:
        """Prepare immutable relink jobs with one batched embedding request."""
        pending = [
            (index, node)
            for index, node in enumerate(nodes)
            if node.name in focus_names
        ]
        if not pending:
            return []

        candidate_top_k = min(
            len(nodes),
            max(
                self.config.link_top_k,
                self.config.link_top_k
                * max(self.config.rerank_candidate_multiplier, 1),
            ),
        )
        source_schemas = [self._rewrite_node_query_schema(node) for _, node in pending]
        query_texts = [
            schema.to_query_text() or node.to_str()
            for schema, (_, node) in zip(source_schemas, pending)
        ]

        knn_rows: list[list[int]] = [[] for _ in pending]
        try:
            embeddings = await self.config.embedding_service.encode(query_texts)
            raw_indices, _ = await self.state_manager.entity_storage.get_knn(
                embeddings,
                top_k=candidate_top_k,
            )
            knn_rows = [[int(raw_index) for raw_index in row] for row in raw_indices]
        except Exception as exc:
            logger.warning(
                f"Skipping batched semantic candidate search during relink: {exc}"
            )

        output_index, input_index = self._build_io_indexes(nodes)
        pair_evidence_indexes = self._build_pair_evidence_indexes(nodes)
        jobs: list[FocusLinkJob] = []

        for row, ((node_index, node), source_schema) in enumerate(
            zip(pending, source_schemas)
        ):
            semantic_candidates: list[tuple[int, float]] = []
            seen: set[int] = set()
            for rank, candidate_index in enumerate(knn_rows[row], start=1):
                if (
                    candidate_index == node_index
                    or candidate_index < 0
                    or candidate_index >= len(nodes)
                    or candidate_index in seen
                ):
                    continue
                seen.add(candidate_index)
                score, has_evidence = self._link_candidate_score(
                    source_schema,
                    node,
                    nodes[candidate_index],
                    1.0 / float(rank),
                    rank,
                )
                if has_evidence:
                    semantic_candidates.append((candidate_index, score))

            lexical_candidates = self._lexical_candidate_scores_for_node(
                node,
                nodes,
                node_index,
                candidate_top_k,
                self._evidence_candidate_indices_for_node(
                    node,
                    pair_evidence_indexes,
                    node_index=node_index,
                ),
            )
            combined_scores: dict[int, float] = {}
            for candidate_index, score in semantic_candidates + lexical_candidates:
                combined_scores[candidate_index] = max(
                    score,
                    combined_scores.get(candidate_index, float("-inf")),
                )
            ranked_candidate_indices = [
                candidate_index
                for candidate_index, _ in sorted(
                    combined_scores.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:candidate_top_k]
            ]
            ranked_lookup = {
                candidate_index: rank
                for rank, candidate_index in enumerate(
                    ranked_candidate_indices,
                    start=1,
                )
            }
            candidate_indices = set(ranked_candidate_indices)
            for value in node.input_types:
                for token in self._schema_artifact_tokens(value):
                    candidate_indices.update(output_index.get(token, set()))
            for value in node.output_types:
                for token in self._schema_artifact_tokens(value):
                    candidate_indices.update(input_index.get(token, set()))
            candidate_indices.discard(node_index)

            eligible_indices = [
                candidate_index
                for candidate_index in candidate_indices
                if candidate_index >= node_index
            ]
            deterministic_edges: list[SkillEdge] = []
            llm_candidates: list[tuple[int, SkillNode]] = []
            for candidate_index in sorted(eligible_indices):
                candidate = nodes[candidate_index]
                deterministic_edges.extend(
                    self._dependency_edges_for_pair(node, candidate)
                )
                candidate_rank = ranked_lookup.get(
                    candidate_index,
                    len(ranked_lookup) + candidate_index + 1,
                )
                llm_candidates.append((candidate_rank, candidate))
            llm_candidates.sort(key=lambda item: item[0])

            jobs.append(
                FocusLinkJob(
                    focus_name=node.name,
                    focus_index=node_index,
                    deterministic_edges=tuple(deterministic_edges),
                    candidates=tuple(
                        candidate
                        for _, candidate in llm_candidates[: self.config.link_top_k]
                    ),
                    candidate_pairs=len(eligible_indices),
                )
            )

        return jobs

    def _lexical_seed_scores(
        self,
        query: str,
        nodes: list[SkillNode],
        seed_top_k: int,
        query_schema: QuerySchema | None = None,
    ) -> list[tuple[int, float, int]]:
        effective_schema = query_schema or self._fallback_query_schema(query)
        query_tokens = self._signature_tokens(
            self._query_schema_values(effective_schema)
        )
        if not query_tokens:
            return []

        scored: list[tuple[int, float]] = []
        for index, node in enumerate(nodes):
            node_tokens = self._signature_tokens(
                [
                    node.name,
                    node.description,
                    node.one_line_capability,
                    node.inputs,
                    node.outputs,
                    node.domain_tags,
                    node.tooling,
                    node.example_tasks,
                    node.script_entrypoints,
                    node.rendered_snippet,
                ]
            )
            overlap = query_tokens & node_tokens
            if overlap:
                score = len(overlap) / max(len(query_tokens), 1)
                score += self._rerank_skill_score(effective_schema, node, 0.0, None)
                scored.append((index, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        selected = scored[:seed_top_k]
        if not selected:
            return []

        weights = build_rank_distribution(len(selected))
        return [
            (index, float(weights[rank]), rank + 1)
            for rank, (index, _) in enumerate(selected)
        ]

    async def _semantic_seed_scores(
        self,
        query: str,
        nodes: list[SkillNode],
        seed_top_k: int,
        query_schema: QuerySchema | None = None,
    ) -> list[tuple[int, float, int]]:
        effective_schema = query_schema or self._fallback_query_schema(query)
        query_text = effective_schema.to_query_text() or query
        semantic_candidate_top_k = max(
            seed_top_k,
            self.config.seed_candidate_top_k_semantic,
            seed_top_k * max(self.config.rerank_candidate_multiplier, 1),
        )
        lexical_candidate_top_k = max(
            seed_top_k,
            self.config.seed_candidate_top_k_lexical,
        )
        lexical_seed_entries = self._lexical_seed_scores(
            query,
            nodes,
            lexical_candidate_top_k,
            effective_schema,
        )
        lexical_rank_lookup = {index: rank for index, _, rank in lexical_seed_entries}
        try:
            query_embedding = await self.config.embedding_service.encode([query_text])
            indices, _ = await self.state_manager.entity_storage.get_knn(
                query_embedding,
                top_k=semantic_candidate_top_k,
            )
        except Exception as exc:
            logger.warning(
                f"Vector seeding failed, falling back to lexical seeding: {exc}"
            )
            return lexical_seed_entries[:seed_top_k]

        semantic_rank_lookup: dict[int, int] = {}
        semantic_graph_scores: dict[int, float] = {}
        for rank, raw_index in enumerate(indices[0], start=1):
            index = int(raw_index)
            if index < 0 or index >= len(nodes) or index in semantic_rank_lookup:
                continue
            semantic_rank_lookup[index] = rank
            semantic_graph_scores[index] = 1.0 / float(rank)

        combined_indices = set(semantic_rank_lookup) | set(lexical_rank_lookup)
        if not combined_indices:
            return []

        candidates: list[tuple[int, float, int | None, int | None]] = []
        for index in combined_indices:
            semantic_rank = semantic_rank_lookup.get(index)
            lexical_rank = lexical_rank_lookup.get(index)
            graph_score = semantic_graph_scores.get(index, 0.0)
            rerank_score = self._rerank_skill_score(
                effective_schema,
                nodes[index],
                graph_score,
                semantic_rank,
            )

            if lexical_rank is not None:
                rerank_score += 0.15 / float(lexical_rank)
            if semantic_rank is not None and lexical_rank is not None:
                rerank_score += 0.1

            candidates.append((index, rerank_score, semantic_rank, lexical_rank))

        candidates.sort(
            key=lambda item: (
                item[1],
                item[2] is not None,
                -(item[2] or 10**9),
                -(item[3] or 10**9),
            ),
            reverse=True,
        )
        ranked_indices = [index for index, _, _, _ in candidates[:seed_top_k]]
        weights = build_rank_distribution(len(ranked_indices))
        return [
            (index, float(weights[rank]), rank + 1)
            for rank, index in enumerate(ranked_indices)
        ]

    async def _vector_seed_scores(
        self,
        query: str,
        nodes: list[SkillNode],
        top_k: int,
    ) -> list[tuple[int, float, int]]:
        if top_k <= 0:
            return []

        query_text = query.strip()
        if not query_text:
            return []

        query_embedding = await self.config.embedding_service.encode([query_text])
        indices, _ = await self.state_manager.entity_storage.get_knn(
            query_embedding,
            top_k=top_k,
        )

        ranked_indices: list[int] = []
        seen: set[int] = set()
        for raw_index in indices[0]:
            index = int(raw_index)
            if index < 0 or index >= len(nodes) or index in seen:
                continue
            seen.add(index)
            ranked_indices.append(index)
            if len(ranked_indices) >= top_k:
                break

        weights = build_rank_distribution(len(ranked_indices))
        return [
            (index, float(weights[rank]), rank + 1)
            for rank, index in enumerate(ranked_indices)
        ]

    def _format_skill_for_linking(self, node: SkillNode) -> str:
        lines = [
            f"{node.name}: {node.description or node.one_line_capability or 'n/a'}"
        ]
        if node.one_line_capability and node.one_line_capability != node.description:
            lines.append(f"Capability: {node.one_line_capability}")
        if node.inputs:
            lines.append(f"Inputs: {node.inputs}")
        if node.outputs:
            lines.append(f"Outputs: {node.outputs}")
        if node.domain_tags:
            lines.append(f"Domain Tags: {node.domain_tags}")
        if node.tooling:
            lines.append(f"Tooling: {node.tooling}")
        if node.example_tasks:
            lines.append(f"Example Tasks: {node.example_tasks}")
        if node.script_entrypoints:
            lines.append(f"Script Entrypoints: {node.script_entrypoints}")
        if node.compatibility:
            lines.append(f"Compatibility: {node.compatibility}")
        return "; ".join(lines)

    def _record_edge(
        self,
        edge_map: dict[tuple[str, str, str], SkillEdge],
        edge: SkillEdge,
    ) -> None:
        key = (edge.source, edge.target, edge.type)
        existing = edge_map.get(key)
        if existing is None:
            edge_map[key] = edge
            return

        self.construction_counters.duplicate_edges_dropped += 1

        if (edge.confidence, edge.weight) > (existing.confidence, existing.weight):
            edge_map[key] = edge

    def _dependency_edges_for_pair(
        self,
        node: SkillNode,
        candidate: SkillNode,
    ) -> list[SkillEdge]:
        # Near-substitutable skills often share manifests and outputs.  Treating
        # one alternative as a producer for the other creates arbitrary cycles;
        # retain the alternative relation instead of inferring a dependency.
        if self._alternative_dominates_dependency(node, candidate):
            return []
        edges: list[SkillEdge] = []

        forward_score, forward_evidence = self._schema_overlap_score(
            node.output_types,
            candidate.input_types,
        )
        if (
            forward_score >= self.config.dependency_match_threshold
            and self._dependency_evidence_supported(
                node,
                candidate,
                forward_evidence,
            )
        ):
            evidence = ", ".join(forward_evidence) or "compatible I/O"
            edges.append(
                SkillEdge(
                    source=node.name,
                    target=candidate.name,
                    description=f"{node.name} produces data that {candidate.name} consumes: {evidence}.",
                    type="dependency",
                    weight=forward_score,
                    confidence=forward_score,
                    provenance="deterministic_io",
                    evidence=", ".join(forward_evidence),
                )
            )

        reverse_score, reverse_evidence = self._schema_overlap_score(
            candidate.output_types,
            node.input_types,
        )
        if (
            reverse_score >= self.config.dependency_match_threshold
            and self._dependency_evidence_supported(
                candidate,
                node,
                reverse_evidence,
            )
        ):
            evidence = ", ".join(reverse_evidence) or "compatible I/O"
            edges.append(
                SkillEdge(
                    source=candidate.name,
                    target=node.name,
                    description=f"{candidate.name} produces data that {node.name} consumes: {evidence}.",
                    type="dependency",
                    weight=reverse_score,
                    confidence=reverse_score,
                    provenance="deterministic_io",
                    evidence=", ".join(reverse_evidence),
                )
            )

        return edges

    async def _validate_candidate_relations(
        self,
        node: SkillNode,
        candidates: list[SkillNode],
        *,
        raise_on_failure: bool = False,
    ) -> list[SkillEdge]:
        if not self.config.enable_semantic_linking or not candidates:
            return []

        self.construction_counters.validator_requests += 1
        self.construction_counters.submitted_candidates += len(candidates)

        candidate_lines = [
            f"- {self._format_skill_for_linking(candidate)}" for candidate in candidates
        ]

        try:
            relations_list, _ = await self.llm_service.send_message(
                system_prompt=PROMPTS["search_and_link_system"],
                prompt=PROMPTS["search_and_link_prompt"].format(
                    new_skill=self._format_skill_for_linking(node),
                    candidate_skills="\n".join(candidate_lines),
                ),
                response_model=GOSRelationList,
                gos_stage="relation_validation",
            )
        except Exception as exc:
            logger.warning(f"LLM relation validation failed for {node.name}: {exc}")
            if raise_on_failure:
                raise
            return []

        self.construction_counters.returned_relations += len(relations_list.relations)

        candidate_by_name = {candidate.name: candidate for candidate in candidates}
        nodes_by_name = {node.name: node, **candidate_by_name}
        allowed_names = {node.name, *candidate_by_name}
        validated_edges: list[SkillEdge] = []
        for relation in relations_list.relations:
            relation_type = relation.type.strip().lower()
            source = relation.source.strip()
            target = relation.target.strip()
            confidence = float(relation.confidence)
            evidence = self._dedupe_text(relation.evidence)

            if relation_type not in VALID_RELATION_TYPES:
                continue
            if source not in allowed_names or target not in allowed_names:
                continue
            if source == target or node.name not in {source, target}:
                continue
            other_name = target if source == node.name else source
            if other_name not in candidate_by_name:
                continue
            if confidence < self.config.relation_min_confidence or not evidence:
                continue

            if relation_type == "dependency":
                if not self._llm_dependency_direction_supported(
                    source,
                    target,
                    relation.description,
                    nodes_by_name,
                    evidence,
                ):
                    continue
            elif relation_type == "workflow":
                if not self._workflow_direction_supported(
                    source,
                    target,
                    relation.description,
                    nodes_by_name,
                    evidence,
                ):
                    continue
            elif relation_type == "semantic":
                if not self._semantic_relation_supported(
                    nodes_by_name[source], nodes_by_name[target]
                ):
                    continue
            elif relation_type == "alternative":
                if not self._alternative_relation_supported(
                    nodes_by_name[source], nodes_by_name[target]
                ):
                    continue
            if relation_type in {"semantic", "alternative"}:
                source, target = sorted((source, target))

            validated_edges.append(
                SkillEdge(
                    source=source,
                    target=target,
                    description=relation.description,
                    type=relation_type,
                    weight=TYPE_WEIGHTS[relation_type] * confidence,
                    confidence=confidence,
                    provenance="llm_validated",
                    evidence="; ".join(evidence),
                    validator_model=str(getattr(self.llm_service, "model", "")),
                )
            )
        self.construction_counters.accepted_relations += len(validated_edges)
        self.construction_counters.rejected_relations += len(
            relations_list.relations
        ) - len(validated_edges)
        return validated_edges

    def _llm_dependency_direction_supported(
        self,
        source: str,
        target: str,
        description: str,
        nodes_by_name: dict[str, SkillNode],
        evidence_values: list[str] | None = None,
    ) -> bool:
        source_node = nodes_by_name[source]
        target_node = nodes_by_name[target]

        if self._alternative_dominates_dependency(source_node, target_node):
            return False

        deterministic = self._dependency_edges_for_pair(source_node, target_node)
        if any(
            edge.source == source and edge.target == target for edge in deterministic
        ):
            return True

        directional_evidence = [description, *(evidence_values or [])]
        evidence_tokens = self._signature_tokens(directional_evidence)
        forward_score, forward_schema_evidence = self._schema_overlap_score(
            source_node.output_types,
            target_node.input_types,
        )
        reverse_score, reverse_schema_evidence = self._schema_overlap_score(
            target_node.output_types,
            source_node.input_types,
        )

        def supported(
            producer: SkillNode,
            consumer: SkillNode,
            score: float,
            schema_evidence: list[str],
        ) -> bool:
            evidence = set(schema_evidence)
            return (
                score > 0
                and bool(evidence & evidence_tokens)
                and not (
                    len(evidence) == 1
                    and evidence <= NON_ARTIFACT_SINGLETON_EVIDENCE
                )
            )

        forward_supported = supported(
            source_node,
            target_node,
            forward_score,
            forward_schema_evidence,
        )
        reverse_supported = supported(
            target_node,
            source_node,
            reverse_score,
            reverse_schema_evidence,
        )
        if forward_supported != reverse_supported:
            return forward_supported
        if forward_supported and forward_score != reverse_score:
            return forward_score > reverse_score

        # Some producers expose individual metadata fields while consumers
        # summarize them under a container (for example frame rate + resolution
        # -> video metadata).  Allow this only when at least two concrete,
        # validator-cited interface tokens agree after removing shared domain
        # vocabulary.  This excludes a lone word such as ``control`` or ``java``.
        forward_fallback = self._llm_dependency_fallback_score(
            source_node,
            target_node,
            directional_evidence,
        )
        reverse_fallback = self._llm_dependency_fallback_score(
            target_node,
            source_node,
            directional_evidence,
        )
        forward_fallback_supported = forward_fallback >= 2.0
        reverse_fallback_supported = reverse_fallback >= 2.0
        if forward_fallback_supported != reverse_fallback_supported:
            return forward_fallback_supported
        if forward_fallback_supported and forward_fallback != reverse_fallback:
            return forward_fallback > reverse_fallback

        source_name = source.lower().replace("_", "-")
        target_text = (
            "\n".join(
                [target_node.raw_content, target_node.description, target_node.inputs]
            )
            .lower()
            .replace("_", "-")
        )
        prerequisite_markers = (
            "prerequisite",
            "requires",
            "depends on",
            "dependency",
        )
        if source_name in target_text and any(
            marker in target_text for marker in prerequisite_markers
        ):
            return True

        # Free-form prose is insufficient direction evidence. The previous check
        # accepted any text containing both endpoint names plus producer/consumer
        # verbs, even when the text described target -> source.
        return False

    def _llm_dependency_fallback_score(
        self,
        producer: SkillNode,
        consumer: SkillNode,
        evidence_values: list[str],
    ) -> float:
        """Score multi-field handoffs unsupported by an artifact-head match."""
        producer_tokens = self._signature_tokens(producer.output_types)
        consumer_tokens = self._signature_tokens(consumer.input_types)
        evidence_tokens = self._signature_tokens(evidence_values)
        shared_domain_tokens = self._signature_tokens(producer.domain_tags_list) & (
            self._signature_tokens(consumer.domain_tags_list)
        )
        overlap = (
            producer_tokens & consumer_tokens & evidence_tokens
        ) - GENERIC_SCHEMA_TOKENS
        overlap -= NON_ARTIFACT_RESULT_HEADS
        overlap -= NON_ARTIFACT_SINGLETON_EVIDENCE
        overlap -= shared_domain_tokens
        return sum(
            0.25 if token in WEAK_ARTIFACT_EVIDENCE else 1.0 for token in overlap
        )

    def _directional_interface_evidence_score(
        self,
        producer: SkillNode,
        consumer: SkillNode,
        evidence_values: list[str],
    ) -> float:
        """Count concrete handoff tokens supported by both interfaces and evidence."""
        producer_tokens = self._signature_tokens(producer.output_types)
        consumer_tokens = self._signature_tokens(consumer.input_types)
        evidence_tokens = self._signature_tokens(evidence_values)
        overlap = (
            producer_tokens & consumer_tokens & evidence_tokens
        ) - GENERIC_SCHEMA_TOKENS
        return sum(
            0.25 if token in WEAK_ARTIFACT_EVIDENCE else 1.0 for token in overlap
        )

    def _workflow_direction_supported(
        self,
        source: str,
        target: str,
        description: str,
        nodes_by_name: dict[str, SkillNode],
        evidence_values: list[str] | None = None,
    ) -> bool:
        """Accept workflow order only when interfaces favor source -> target."""
        source_node = nodes_by_name[source]
        target_node = nodes_by_name[target]

        deterministic = self._dependency_edges_for_pair(source_node, target_node)
        forward = any(
            edge.source == source and edge.target == target for edge in deterministic
        )
        reverse = any(
            edge.source == target and edge.target == source for edge in deterministic
        )
        if forward != reverse:
            return forward

        directional_evidence = [description, *(evidence_values or [])]
        forward_score = self._directional_interface_evidence_score(
            source_node,
            target_node,
            directional_evidence,
        )
        reverse_score = self._directional_interface_evidence_score(
            target_node,
            source_node,
            directional_evidence,
        )
        return forward_score > reverse_score

    async def async_insert_skill(
        self,
        skill_text: str,
        metadata: dict[str, Any] | None = None,
    ):
        prepared_metadata = self._prepare_metadata(skill_text, metadata)
        source_path = str(prepared_metadata.get("source_path") or "")
        parsed = parse_skill_document(skill_text, source_path=source_path)
        skill_names = {parsed.name} if parsed and parsed.name else set()
        updated_names = await self._existing_skill_names(skill_names)

        result = await self.async_insert(
            content=[skill_text], metadata=[prepared_metadata]
        )
        if updated_names:
            await self._delete_incident_edges(updated_names)
        if skill_names:
            await self._link_skills_incremental(skill_names)
        else:
            await self._link_all_skills()
        return result

    async def async_insert_skills(
        self,
        skill_texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ):
        prepared_metadatas: list[dict[str, Any]] = []
        provided_metadatas = metadatas or []
        for index, skill_text in enumerate(skill_texts):
            metadata = (
                provided_metadatas[index] if index < len(provided_metadatas) else None
            )
            prepared_metadatas.append(self._prepare_metadata(skill_text, metadata))

        new_names: set[str] = set()
        for index, skill_text in enumerate(skill_texts):
            source_path = str(prepared_metadatas[index].get("source_path") or "")
            parsed = parse_skill_document(skill_text, source_path=source_path)
            if parsed and parsed.name:
                new_names.add(parsed.name)

        updated_names = await self._existing_skill_names(new_names)
        existing_node_count, _ = await self._graph_counts()
        result = await self.async_insert(
            content=skill_texts, metadata=prepared_metadatas
        )
        if updated_names:
            await self._delete_incident_edges(updated_names)
        if existing_node_count == 0 and len(new_names) > 1:
            await self.async_relink_all(
                concurrency=self.config.relink_concurrency,
                checkpoint_every=self.config.relink_checkpoint_every,
                resume=False,
            )
        elif new_names:
            await self._link_skills_incremental(new_names)
        else:
            await self._link_all_skills()
        return result

    async def _existing_skill_names(self, skill_names: set[str]) -> set[str]:
        """Return requested names that are already present in the graph."""
        if not skill_names:
            return set()

        await self.state_manager.query_start()
        try:
            return {
                node.name
                for node in await self._load_all_nodes()
                if node.name in skill_names
            }
        finally:
            await self.state_manager.query_done()

    async def _delete_incident_edges(self, skill_names: set[str]) -> None:
        """Remove stale edges before relinking updated skill definitions."""
        if not skill_names:
            return

        await self.state_manager.insert_start()
        try:
            target = self.state_manager.graph_storage
            stale_indices: list[int] = []
            for index in range(await target.edge_count()):
                edge = await target.get_edge_by_index(index)
                if edge is not None and (
                    edge.source in skill_names or edge.target in skill_names
                ):
                    stale_indices.append(index)

            if stale_indices:
                logger.info(
                    f"GoS: removing {len(stale_indices)} stale edge(s) incident "
                    f"to updated skills {sorted(skill_names)}."
                )
                await target.delete_edges_by_index(stale_indices)
        finally:
            await self.state_manager.insert_done()

    async def async_ensure_skills(
        self,
        skill_texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> SkillSyncResult:
        prepared_metadatas: list[dict[str, Any]] = []
        provided_metadatas = metadatas or []
        for index, skill_text in enumerate(skill_texts):
            metadata = (
                provided_metadatas[index] if index < len(provided_metadatas) else None
            )
            prepared_metadatas.append(self._prepare_metadata(skill_text, metadata))

        await self.state_manager.query_start()
        try:
            existing_nodes = await self._load_all_nodes()
        finally:
            await self.state_manager.query_done()

        existing_count = len(existing_nodes)
        by_skill_id, by_source_path, by_name = self._node_lookup_maps(existing_nodes)

        pending_skill_ids: set[str] = set()
        pending_source_paths: set[str] = set()
        pending_names: set[str] = set()

        missing_texts: list[str] = []
        missing_metadatas: list[dict[str, Any]] = []
        inserted_skill_names: list[str] = []
        updated_skill_names: list[str] = []
        reused_count = 0

        for index, skill_text in enumerate(skill_texts):
            metadata = prepared_metadatas[index]
            source_path = str(metadata.get("source_path") or "")
            snippet_chars = int(
                metadata.get("snippet_chars", self.config.snippet_chars)
            )
            parsed = parse_skill_document(
                skill_text,
                source_path=source_path,
                snippet_chars=snippet_chars,
            )
            name = (
                parsed.name if parsed is not None else source_path or f"skill_{index}"
            )
            skill_id = (
                parsed.skill_id
                if parsed is not None and parsed.skill_id
                else str(metadata.get("skill_id") or source_path or name)
            )

            if (
                (skill_id and skill_id in pending_skill_ids)
                or (source_path and source_path in pending_source_paths)
                or (name and name in pending_names)
            ):
                reused_count += 1
                continue

            existing_node = self._find_existing_node(
                name=name,
                skill_id=skill_id,
                source_path=source_path,
                by_skill_id=by_skill_id,
                by_source_path=by_source_path,
                by_name=by_name,
            )

            if existing_node is None:
                missing_texts.append(skill_text)
                missing_metadatas.append(metadata)
                inserted_skill_names.append(name)
            else:
                existing_raw_content = existing_node.raw_content or ""
                existing_source_path = existing_node.source_path or ""
                existing_skill_id = existing_node.skill_id or ""
                if (
                    existing_raw_content == skill_text
                    and existing_source_path == source_path
                    and existing_skill_id == skill_id
                ):
                    reused_count += 1
                    continue

                missing_texts.append(skill_text)
                missing_metadatas.append(metadata)
                updated_skill_names.append(name)

            if skill_id:
                pending_skill_ids.add(skill_id)
            if source_path:
                pending_source_paths.add(source_path)
            if name:
                pending_names.add(name)

        if missing_texts:
            await self.async_insert_skills(missing_texts, missing_metadatas)

        final_skill_count, _ = await self._graph_counts()
        return SkillSyncResult(
            requested_skill_count=len(skill_texts),
            existing_skill_count=existing_count,
            final_skill_count=final_skill_count,
            reused_count=reused_count,
            inserted_count=len(inserted_skill_names),
            updated_count=len(updated_skill_names),
            inserted_skill_names=inserted_skill_names,
            updated_skill_names=updated_skill_names,
            prebuilt_working_dir=self.bootstrapped_from,
        )

    def _relink_fingerprint(self, nodes: list[SkillNode]) -> str:
        prompt_sha256 = {
            name: hashlib.sha256(str(prompt).encode("utf-8")).hexdigest()
            for name, prompt in PROMPTS.items()
            if name in {"skill_extraction_system", "search_and_link_system"}
        }
        return build_relink_fingerprint(
            nodes=nodes,
            llm_model=str(getattr(self.llm_service, "model", "")),
            embedding_model=str(getattr(self.config.embedding_service, "model", "")),
            prompt_sha256=prompt_sha256,
            link_top_k=self.config.link_top_k,
            relation_min_confidence=self.config.relation_min_confidence,
            dependency_match_threshold=self.config.dependency_match_threshold,
            type_weights=TYPE_WEIGHTS,
            construction_code_sha256=self._construction_code_sha256(),
        )

    def _construction_code_sha256(self) -> str:
        """Hash construction behavior that makes durable focus results reusable."""
        methods = (
            self.__class__._schema_artifact_signature,
            self.__class__._schema_overlap_score,
            self.__class__._dependency_evidence_supported,
            self.__class__._alternative_relation_supported,
            self.__class__._alternative_dominates_dependency,
            self.__class__._semantic_relation_supported,
            self.__class__._link_candidate_score,
            self.__class__._pair_evidence_tokens_for_node,
            self.__class__._build_pair_evidence_indexes,
            self.__class__._evidence_candidate_indices_for_node,
            self.__class__._lexical_candidate_scores_for_node,
            self.__class__._prepare_focus_link_jobs,
            self.__class__._dependency_edges_for_pair,
            self.__class__._directional_interface_evidence_score,
            self.__class__._llm_dependency_fallback_score,
            self.__class__._llm_dependency_direction_supported,
            self.__class__._workflow_direction_supported,
            self.__class__._validate_candidate_relations,
        )
        policy_constants = {
            "token_stopwords": sorted(TOKEN_STOPWORDS),
            "generic_schema_tokens": sorted(GENERIC_SCHEMA_TOKENS),
            "concrete_artifact_formats": sorted(CONCRETE_ARTIFACT_FORMATS),
            "weak_artifact_evidence": sorted(WEAK_ARTIFACT_EVIDENCE),
            "non_artifact_result_heads": sorted(NON_ARTIFACT_RESULT_HEADS),
            "artifact_container_heads": sorted(ARTIFACT_CONTAINER_HEADS),
            "programming_language_tokens": sorted(PROGRAMMING_LANGUAGE_TOKENS),
            "non_artifact_singleton_evidence": sorted(
                NON_ARTIFACT_SINGLETON_EVIDENCE
            ),
            "alternative_generic_tokens": sorted(ALTERNATIVE_GENERIC_TOKENS),
            "semantic_generic_tokens": sorted(SEMANTIC_GENERIC_TOKENS),
            "type_weights": TYPE_WEIGHTS,
        }
        rendered = "\n\n".join(inspect.getsource(method) for method in methods)
        rendered += "\n\n" + json.dumps(policy_constants, sort_keys=True)
        return f"sha256:{hashlib.sha256(rendered.encode('utf-8')).hexdigest()}"

    def _relink_usage_snapshot(self) -> dict[str, Any]:
        def usage_for(service: Any) -> dict[str, Any]:
            usage = getattr(service, "usage", None)
            if usage is None or not hasattr(usage, "to_dict"):
                return {}
            return usage.to_dict()

        return {
            "llm": usage_for(self.llm_service),
            "embedding": usage_for(self.config.embedding_service),
        }

    def _restore_construction_counters(
        self,
        values: dict[str, int | float],
    ) -> None:
        valid_fields = asdict(self.construction_counters)
        for name, value in values.items():
            if name in valid_fields and isinstance(value, (int, float)):
                setattr(self.construction_counters, name, value)

    async def _delete_all_edges(self) -> None:
        await self.state_manager.insert_start()
        try:
            target = self.state_manager.graph_storage
            edge_count = await target.edge_count()
            if edge_count:
                await target.delete_edges_by_index(range(edge_count))
        finally:
            await self.state_manager.insert_done()

    async def _checkpoint_relink_results(
        self,
        *,
        results: list[FocusLinkResult],
        progress: RelinkProgress,
        progress_path: Path,
        event_path: Path,
        attempt_id: str,
        attempt_started: float,
        attempt_checkpointed_focus_count: int,
        prior_usage: dict[str, Any],
        progress_callback: Callable[[RelinkProgress], None] | None,
    ) -> None:
        if not results:
            return

        started = time.perf_counter()
        await self.state_manager.insert_start()
        try:
            target = self.state_manager.graph_storage
            edge_map: dict[tuple[str, str, str], SkillEdge] = {}
            for result in results:
                for edge in result.edges:
                    self._record_edge(edge_map, edge)
            if edge_map:
                await self.state_manager.edge_upsert_policy(
                    self.llm_service,
                    target,
                    list(edge_map.values()),
                )
            persisted_edge_count = await target.edge_count()
        finally:
            await self.state_manager.insert_done()

        completed = set(progress.completed_focus_names)
        for result in results:
            if result.focus_name not in completed:
                progress.completed_focus_names.append(result.focus_name)
                completed.add(result.focus_name)
            if result.error:
                progress.failed_focus[result.focus_name] = result.error
            else:
                progress.failed_focus.pop(result.focus_name, None)

        checkpoint_write_seconds = time.perf_counter() - started
        previous_usage = dict(progress.usage)
        cumulative_usage = merge_relink_usage(
            prior_usage,
            self._relink_usage_snapshot(),
        )
        progress.persisted_edge_count = persisted_edge_count
        progress.checkpoint_count += 1
        progress.validation_write_seconds += checkpoint_write_seconds
        progress.construction = asdict(self.construction_counters)
        progress.usage = cumulative_usage
        progress.event_count += 1
        write_relink_progress(progress_path, progress)
        attempt_elapsed = time.perf_counter() - attempt_started
        focus_results = [
            {
                "focus_name": result.focus_name,
                "candidate_count": result.candidate_count,
                "deterministic_edge_count": result.deterministic_edge_count,
                "validated_edge_count": result.validated_edge_count,
                "persisted_candidate_edge_count": len(result.edges),
                "validation_seconds": result.validation_seconds,
                "error": result.error,
            }
            for result in sorted(results, key=lambda item: item.focus_name)
        ]
        append_relink_event(
            event_path,
            {
                "event": "checkpoint",
                "sequence": progress.event_count,
                "run_id": progress.run_id,
                "attempt_id": attempt_id,
                "checkpoint": progress.checkpoint_count,
                "batch": {
                    "focus_count": len(results),
                    "candidate_count": sum(
                        result.candidate_count for result in results
                    ),
                    "deterministic_edge_count": sum(
                        result.deterministic_edge_count for result in results
                    ),
                    "validated_edge_count": sum(
                        result.validated_edge_count for result in results
                    ),
                    "failed_focus_count": sum(bool(result.error) for result in results),
                    "validation_seconds_sum": sum(
                        result.validation_seconds for result in results
                    ),
                    "validation_seconds_max": max(
                        (result.validation_seconds for result in results),
                        default=0.0,
                    ),
                    "focus_results": focus_results,
                },
                "totals": {
                    "total_focus_count": progress.total_focus_nodes,
                    "attempt_checkpointed_focus_count": (
                        attempt_checkpointed_focus_count
                    ),
                    "completed_focus_count": len(progress.completed_focus_names),
                    "failed_focus_count": len(progress.failed_focus),
                    "persisted_edge_count": progress.persisted_edge_count,
                    "checkpoint_count": progress.checkpoint_count,
                },
                "timing": {
                    "attempt_elapsed_seconds": attempt_elapsed,
                    "checkpoint_write_seconds": checkpoint_write_seconds,
                    "cumulative_preparation_seconds": progress.preparation_seconds,
                    "cumulative_checkpoint_write_seconds": (
                        progress.validation_write_seconds
                    ),
                },
                "throughput": {
                    "attempt_focus_per_second": (
                        attempt_checkpointed_focus_count / attempt_elapsed
                        if attempt_elapsed > 0
                        else 0.0
                    ),
                },
                "usage_delta": diff_relink_usage(
                    previous_usage,
                    cumulative_usage,
                ),
                "usage_totals": summarize_relink_usage(cumulative_usage),
            },
        )
        if progress_callback is not None:
            progress_callback(progress)

    async def async_relink_all(
        self,
        *,
        concurrency: int = 8,
        checkpoint_every: int = 10,
        resume: bool = True,
        restart: bool = False,
        progress_callback: Callable[[RelinkProgress], None] | None = None,
    ) -> RelinkResult:
        if concurrency < 1:
            raise ValueError("Relink concurrency must be at least 1.")
        if checkpoint_every < 1:
            raise ValueError("Relink checkpoint size must be at least 1.")

        started_at = datetime.now(timezone.utc).isoformat()
        started = time.perf_counter()
        progress_path = Path(self.working_dir) / "relink_progress.json"
        event_path = Path(self.working_dir) / "relink_events.jsonl"
        if restart:
            await self._delete_all_edges()
            progress_path.unlink(missing_ok=True)

        preparation_started = time.perf_counter()
        await self.state_manager.query_start()
        try:
            nodes = await self._load_all_nodes()
            if len(nodes) < 2:
                raise ValueError("Relink requires at least two persisted skill nodes.")

            fingerprint = self._relink_fingerprint(nodes)
            existing = load_relink_progress(progress_path) if resume else None
            if existing is not None and existing.fingerprint != fingerprint:
                raise RelinkProgressMismatch(
                    "Persisted relink progress does not match the current nodes or "
                    "configuration; rerun with --restart."
                )

            if existing is None:
                progress = RelinkProgress.new(
                    fingerprint=fingerprint,
                    total_focus_nodes=len(nodes),
                    concurrency=concurrency,
                    checkpoint_every=checkpoint_every,
                )
            else:
                progress = existing
                progress.status = "running"
                progress.concurrency = concurrency
                progress.checkpoint_every = checkpoint_every
                self._restore_construction_counters(progress.construction)

            if not progress.run_id:
                progress.run_id = str(uuid4())
            attempt_id = str(uuid4())
            progress.attempt_count += 1
            progress.last_attempt_id = attempt_id

            resumed_names = set(progress.completed_focus_names) - set(
                progress.failed_focus
            )
            prior_usage = dict(progress.usage)
            progress.resumed_focus_count = len(resumed_names)
            pending_names = {node.name for node in nodes} - resumed_names
            jobs = await self._prepare_focus_link_jobs(nodes, pending_names)
            nodes_by_name = {node.name: node for node in nodes}
        finally:
            await self.state_manager.query_done()

        preparation_seconds = time.perf_counter() - preparation_started
        progress.preparation_seconds += preparation_seconds
        progress.usage = merge_relink_usage(
            prior_usage,
            self._relink_usage_snapshot(),
        )
        progress.construction = asdict(self.construction_counters)
        progress.event_count += 1
        write_relink_progress(progress_path, progress)
        append_relink_event(
            event_path,
            {
                "event": "attempt_started",
                "timestamp": started_at,
                "sequence": progress.event_count,
                "run_id": progress.run_id,
                "attempt_id": attempt_id,
                "attempt_number": progress.attempt_count,
                "fingerprint": progress.fingerprint,
                "models": {
                    "llm": str(getattr(self.llm_service, "model", "")),
                    "embedding": str(
                        getattr(self.config.embedding_service, "model", "")
                    ),
                },
                "configuration": {
                    "concurrency": concurrency,
                    "checkpoint_every": checkpoint_every,
                    "resume": resume,
                    "restart": restart,
                    "link_top_k": self.config.link_top_k,
                    "dependency_match_threshold": (
                        self.config.dependency_match_threshold
                    ),
                    "relation_min_confidence": (self.config.relation_min_confidence),
                    "response_cache": bool(
                        getattr(self.llm_service, "response_cache", False)
                    ),
                },
                "totals": {
                    "total_focus_count": len(nodes),
                    "resumed_focus_count": len(resumed_names),
                    "pending_focus_count": len(jobs),
                    "completed_focus_count": len(progress.completed_focus_names),
                    "failed_focus_count": len(progress.failed_focus),
                    "persisted_edge_count": progress.persisted_edge_count,
                },
                "timing": {
                    "preparation_seconds": preparation_seconds,
                    "cumulative_preparation_seconds": progress.preparation_seconds,
                },
                "usage_delta": diff_relink_usage(prior_usage, progress.usage),
                "usage_totals": summarize_relink_usage(progress.usage),
            },
        )
        semaphore = asyncio.Semaphore(concurrency)

        async def validate(job: FocusLinkJob) -> FocusLinkResult:
            async with semaphore:
                validation_started = time.perf_counter()
                self.construction_counters.focus_nodes += 1
                self.construction_counters.candidate_pairs += job.candidate_pairs
                try:
                    validated = await self._validate_candidate_relations(
                        nodes_by_name[job.focus_name],
                        list(job.candidates),
                        raise_on_failure=True,
                    )
                    return FocusLinkResult(
                        focus_name=job.focus_name,
                        edges=job.deterministic_edges + tuple(validated),
                        candidate_count=len(job.candidates),
                        deterministic_edge_count=len(job.deterministic_edges),
                        validated_edge_count=len(validated),
                        validation_seconds=time.perf_counter() - validation_started,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    return FocusLinkResult(
                        focus_name=job.focus_name,
                        edges=job.deterministic_edges,
                        error=summarize_relink_error(exc),
                        candidate_count=len(job.candidates),
                        deterministic_edge_count=len(job.deterministic_edges),
                        validated_edge_count=0,
                        validation_seconds=time.perf_counter() - validation_started,
                    )

        tasks = [asyncio.create_task(validate(job)) for job in jobs]
        batch: list[FocusLinkResult] = []
        attempt_checkpointed_focus_count = 0
        try:
            for completed_task in asyncio.as_completed(tasks):
                batch.append(await completed_task)
                if len(batch) >= checkpoint_every:
                    next_checkpointed_count = attempt_checkpointed_focus_count + len(
                        batch
                    )
                    await self._checkpoint_relink_results(
                        results=batch,
                        progress=progress,
                        progress_path=progress_path,
                        event_path=event_path,
                        attempt_id=attempt_id,
                        attempt_started=started,
                        attempt_checkpointed_focus_count=next_checkpointed_count,
                        prior_usage=prior_usage,
                        progress_callback=progress_callback,
                    )
                    attempt_checkpointed_focus_count = next_checkpointed_count
                    batch = []
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            known_focus_names = set(progress.completed_focus_names) | {
                result.focus_name for result in batch
            }
            for task in tasks:
                if task.cancelled() or not task.done():
                    continue
                try:
                    result = task.result()
                except asyncio.CancelledError:
                    continue
                except Exception:
                    continue
                if result.focus_name not in known_focus_names:
                    batch.append(result)
                    known_focus_names.add(result.focus_name)
            if batch:
                next_checkpointed_count = attempt_checkpointed_focus_count + len(batch)
                await self._checkpoint_relink_results(
                    results=batch,
                    progress=progress,
                    progress_path=progress_path,
                    event_path=event_path,
                    attempt_id=attempt_id,
                    attempt_started=started,
                    attempt_checkpointed_focus_count=next_checkpointed_count,
                    prior_usage=prior_usage,
                    progress_callback=progress_callback,
                )
                attempt_checkpointed_focus_count = next_checkpointed_count
            cancelled_elapsed = time.perf_counter() - started
            previous_usage = dict(progress.usage)
            self.construction_counters.wall_time_seconds += cancelled_elapsed
            progress.status = "cancelled"
            progress.construction = asdict(self.construction_counters)
            progress.usage = merge_relink_usage(
                prior_usage,
                self._relink_usage_snapshot(),
            )
            progress.event_count += 1
            write_relink_progress(progress_path, progress)
            append_relink_event(
                event_path,
                {
                    "event": "attempt_cancelled",
                    "sequence": progress.event_count,
                    "run_id": progress.run_id,
                    "attempt_id": attempt_id,
                    "attempt_number": progress.attempt_count,
                    "totals": {
                        "total_focus_count": len(nodes),
                        "resumed_focus_count": len(resumed_names),
                        "processed_focus_count": attempt_checkpointed_focus_count,
                        "completed_focus_count": len(progress.completed_focus_names),
                        "failed_focus_count": len(progress.failed_focus),
                        "persisted_edge_count": progress.persisted_edge_count,
                        "checkpoint_count": progress.checkpoint_count,
                    },
                    "timing": {
                        "attempt_elapsed_seconds": cancelled_elapsed,
                        "cumulative_wall_seconds": (
                            self.construction_counters.wall_time_seconds
                        ),
                        "cumulative_preparation_seconds": (
                            progress.preparation_seconds
                        ),
                        "cumulative_checkpoint_write_seconds": (
                            progress.validation_write_seconds
                        ),
                    },
                    "usage_delta": diff_relink_usage(
                        previous_usage,
                        progress.usage,
                    ),
                    "usage_totals": summarize_relink_usage(progress.usage),
                },
            )
            raise

        if batch:
            next_checkpointed_count = attempt_checkpointed_focus_count + len(batch)
            await self._checkpoint_relink_results(
                results=batch,
                progress=progress,
                progress_path=progress_path,
                event_path=event_path,
                attempt_id=attempt_id,
                attempt_started=started,
                attempt_checkpointed_focus_count=next_checkpointed_count,
                prior_usage=prior_usage,
                progress_callback=progress_callback,
            )
            attempt_checkpointed_focus_count = next_checkpointed_count

        elapsed = time.perf_counter() - started
        self.construction_counters.wall_time_seconds += elapsed
        previous_usage = dict(progress.usage)
        progress.status = "complete"
        progress.construction = asdict(self.construction_counters)
        progress.usage = merge_relink_usage(
            prior_usage,
            self._relink_usage_snapshot(),
        )
        progress.event_count += 1
        write_relink_progress(progress_path, progress)
        append_relink_event(
            event_path,
            {
                "event": "attempt_completed",
                "sequence": progress.event_count,
                "run_id": progress.run_id,
                "attempt_id": attempt_id,
                "attempt_number": progress.attempt_count,
                "totals": {
                    "total_focus_count": len(nodes),
                    "resumed_focus_count": len(resumed_names),
                    "processed_focus_count": len(jobs),
                    "completed_focus_count": len(progress.completed_focus_names),
                    "failed_focus_count": len(progress.failed_focus),
                    "failed_focus_names": sorted(progress.failed_focus),
                    "persisted_edge_count": progress.persisted_edge_count,
                    "checkpoint_count": progress.checkpoint_count,
                },
                "timing": {
                    "attempt_elapsed_seconds": elapsed,
                    "cumulative_wall_seconds": (
                        self.construction_counters.wall_time_seconds
                    ),
                    "cumulative_preparation_seconds": progress.preparation_seconds,
                    "cumulative_checkpoint_write_seconds": (
                        progress.validation_write_seconds
                    ),
                },
                "throughput": {
                    "processed_focus_per_second": (
                        len(jobs) / elapsed if elapsed > 0 else 0.0
                    ),
                    "persisted_edges_per_second": (
                        progress.persisted_edge_count
                        / self.construction_counters.wall_time_seconds
                        if self.construction_counters.wall_time_seconds > 0
                        else 0.0
                    ),
                },
                "usage_delta": diff_relink_usage(
                    previous_usage,
                    progress.usage,
                ),
                "usage_totals": summarize_relink_usage(progress.usage),
            },
        )

        return RelinkResult(
            total_focus_count=len(nodes),
            resumed_focus_count=len(resumed_names),
            processed_focus_count=len(jobs),
            completed_focus_count=len(progress.completed_focus_names),
            failed_focus=dict(progress.failed_focus),
            checkpoint_count=progress.checkpoint_count,
            edge_count=progress.persisted_edge_count,
            elapsed_seconds=elapsed,
        )

    async def _link_all_skills(self):
        await self.async_relink_all(
            concurrency=self.config.relink_concurrency,
            checkpoint_every=self.config.relink_checkpoint_every,
            resume=False,
        )

    async def _link_skills_incremental(self, new_node_names: set[str]) -> None:
        """Incrementally link newly inserted/updated skills against the full graph.

        Only processes pairs where at least one node is in new_node_names,
        reducing cost from O(|all|²) to O(|new| × |all|).

        For new-new pairs, _dependency_edges_for_pair already emits edges in
        both directions, so we skip pairs where the candidate is also new but
        has a lower index — avoiding duplicate LLM calls on the same pair.
        """
        if not new_node_names:
            return

        await self.state_manager.insert_start()
        try:
            target = self.state_manager.graph_storage
            all_nodes = await self._load_all_nodes()
            if len(all_nodes) < 2:
                return

            new_nodes = [n for n in all_nodes if n.name in new_node_names]
            if not new_nodes:
                logger.warning(
                    f"GoS: incremental link: none of {new_node_names} found after insert."
                )
                return

            logger.info(
                f"GoS: incremental linking {len(new_nodes)} skill(s) against {len(all_nodes)} total."
            )

            node_index_by_name: dict[str, int] = {
                n.name: i for i, n in enumerate(all_nodes)
            }
            new_node_indices: set[int] = {node_index_by_name[n.name] for n in new_nodes}

            output_index, input_index = self._build_io_indexes(all_nodes)
            node_names = {node.name for node in all_nodes}
            edge_map: dict[tuple[str, str, str], SkillEdge] = {}

            for node in new_nodes:
                self.construction_counters.focus_nodes += 1
                node_index = node_index_by_name[node.name]

                ranked_candidate_indices = await self._rank_link_candidates_for_node(
                    node,
                    all_nodes,
                    node_index,
                )
                ranked_candidate_lookup = {
                    candidate_index: rank
                    for rank, candidate_index in enumerate(
                        ranked_candidate_indices, start=1
                    )
                }
                candidate_indices = set(ranked_candidate_indices)

                for value in node.input_types:
                    for token in self._schema_artifact_tokens(value):
                        candidate_indices.update(output_index.get(token, set()))
                for value in node.output_types:
                    for token in self._schema_artifact_tokens(value):
                        candidate_indices.update(input_index.get(token, set()))

                candidate_indices.discard(node_index)
                self.construction_counters.candidate_pairs += len(candidate_indices)
                llm_candidates: list[tuple[int, SkillNode]] = []

                for candidate_index in sorted(candidate_indices):
                    # For new-new pairs, only process when current node has the
                    # lower index to avoid emitting the same edges twice.
                    if (
                        candidate_index in new_node_indices
                        and candidate_index < node_index
                    ):
                        continue

                    candidate = all_nodes[candidate_index]
                    deterministic_edges = self._dependency_edges_for_pair(
                        node, candidate
                    )
                    if deterministic_edges:
                        for edge in deterministic_edges:
                            self._record_edge(edge_map, edge)

                    candidate_rank = ranked_candidate_lookup.get(
                        candidate_index,
                        len(ranked_candidate_lookup) + candidate_index + 1,
                    )
                    llm_candidates.append((candidate_rank, candidate))

                if llm_candidates:
                    llm_candidates.sort(key=lambda item: item[0])
                    validated_edges = await self._validate_candidate_relations(
                        node,
                        [
                            candidate
                            for _, candidate in llm_candidates[: self.config.link_top_k]
                        ],
                    )
                    for edge in validated_edges:
                        if edge.source in node_names and edge.target in node_names:
                            self._record_edge(edge_map, edge)

            if edge_map:
                logger.info(f"GoS: committing {len(edge_map)} incremental edges.")
                await self.state_manager.edge_upsert_policy(
                    self.llm_service,
                    target,
                    list(edge_map.values()),
                )
        finally:
            await self.state_manager.insert_done()

    def _render_summary(
        self,
        query: str,
        query_schema: QuerySchema,
        skills: list[RetrievedSkill],
        relations: list[RetrievedRelation],
        seeds: list[SkillSeed],
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
                f", seed rank {skill.semantic_rank}" if skill.semantic_rank else ""
            )
            lines.append(
                f"- {skill.name}: {skill.description} "
                f"(score={skill.score:.4f}, rerank={skill.rerank_score:.4f}{semantic_rank})"
            )
            if skill.source_path:
                lines.append(f"  Source: {skill.source_path}")
            if skill.script_entrypoints:
                preview = ", ".join(skill.script_entrypoints[:3])
                lines.append(f"  Scripts: {preview}")

        if any(self._query_schema_values(query_schema)):
            lines.append("\n### Query Schema")
            lines.append(f"- goal: {query_schema.goal}")
            if query_schema.task_name:
                lines.append(f"- task_name: {query_schema.task_name}")
            if query_schema.domain:
                lines.append(f"- domain: {', '.join(query_schema.domain)}")
            if query_schema.operations:
                lines.append(f"- operations: {', '.join(query_schema.operations)}")
            if query_schema.artifacts:
                lines.append(f"- artifacts: {', '.join(query_schema.artifacts)}")
            if query_schema.constraints:
                lines.append(f"- constraints: {', '.join(query_schema.constraints)}")

        if seeds:
            lines.append("\n### Semantic Seeds")
            for seed in seeds:
                lines.append(
                    f"- {seed.name} (seed weight={seed.seed_weight:.4f}, rank={seed.semantic_rank})"
                )

        if relations:
            lines.append("\n### Graph Edges")
            for relation in relations:
                lines.append(
                    f"- {relation.source} --({relation.type})--> {relation.target}: "
                    f"{relation.description} (weight={relation.weight:.3f})"
                )

        return "\n".join(lines)

    def _render_context(
        self,
        query: str,
        skills: list[RetrievedSkill],
        relations: list[RetrievedRelation],
        *,
        max_chars: int | None = None,
    ) -> str:
        if not skills:
            context = "\n\n".join(
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
            if max_chars is not None and len(context) > max_chars:
                return self._clip_text(context, max_chars)
            return context

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
            sections.append(skill.payload)

        if relations:
            relation_lines = ["## Graph evidence"]
            for relation in relations:
                candidate_lines = relation_lines + [
                    f"- {relation.source} --({relation.type})--> {relation.target}: {relation.description}"
                ]
                candidate_context = "\n\n".join(sections + ["\n".join(candidate_lines)])
                if max_chars is not None and len(candidate_context) > max_chars:
                    break
                relation_lines = candidate_lines

            if len(relation_lines) > 1:
                sections.append("\n".join(relation_lines))

        context = "\n\n".join(sections)
        if max_chars is not None and len(context) > max_chars:
            return self._clip_text(context, max_chars)
        return context

    @staticmethod
    def _clip_text(text: str, max_chars: int) -> str:
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        clipped = text[: max_chars - 3].rstrip()
        return f"{clipped}..."

    def _fit_skills_to_context_budget(
        self,
        query: str,
        skills: list[RetrievedSkill],
        max_context_chars: int,
    ) -> list[RetrievedSkill]:
        if not skills or max_context_chars <= 0:
            return []

        header = f"# Skill bundle for query: {query}"
        total_chars = len(header)
        fitted_skills: list[RetrievedSkill] = []

        for skill in skills:
            remaining_context = max_context_chars - total_chars - 2
            if remaining_context <= 0:
                break

            payload = skill.payload
            if len(payload) > remaining_context:
                payload = self._clip_text(payload, remaining_context)

            if not payload:
                break

            fitted_skills.append(skill.model_copy(update={"payload": payload}))
            total_chars += 2 + len(payload)

            if len(payload) < len(skill.payload):
                break

        return fitted_skills

    async def async_hydrate_skills(
        self,
        skill_names: list[str],
        *,
        max_chars_per_skill: int | None = None,
    ) -> list[RetrievedSkill]:
        names = {name.strip() for name in skill_names if name.strip()}
        if not names:
            return []

        await self.state_manager.query_start()
        try:
            nodes = await self._load_all_nodes()
            selected_nodes = [node for node in nodes if node.name in names]
            return [
                RetrievedSkill(
                    name=node.name,
                    description=node.description,
                    source_path=node.source_path,
                    one_line_capability=node.one_line_capability,
                    score=0.0,
                    rerank_score=0.0,
                    inputs=node.input_types,
                    outputs=node.output_types,
                    domain_tags=node.domain_tags_list,
                    tooling=node.tooling_list,
                    example_tasks=node.example_tasks_list,
                    script_entrypoints=node.script_entrypoints_list,
                    compatibility=node.compatibility_list,
                    allowed_tools=node.allowed_tools_list,
                    rendered_snippet=node.rendered_snippet,
                    payload=node.render_for_agent(
                        max_chars_per_skill or self.config.max_skill_chars
                    ),
                )
                for node in selected_nodes
            ]
        finally:
            await self.state_manager.query_done()

    async def async_retrieve(
        self,
        query: str,
        *,
        top_n: int | None = None,
        seed_top_k: int | None = None,
        max_chars_per_skill: int | None = None,
        max_context_chars: int | None = None,
    ) -> SkillRetrievalResult:
        requested_top_n = top_n or self.config.retrieval_top_n
        requested_seed_top_k = seed_top_k or self.config.seed_top_k
        requested_skill_chars = max_chars_per_skill or self.config.max_skill_chars
        requested_context_chars = max_context_chars or self.config.max_context_chars

        budget = RetrievalBudget(
            seed_top_k=requested_seed_top_k,
            seed_candidate_top_k_semantic=max(
                requested_seed_top_k,
                self.config.seed_candidate_top_k_semantic,
                requested_seed_top_k * max(self.config.rerank_candidate_multiplier, 1),
            ),
            seed_candidate_top_k_lexical=max(
                requested_seed_top_k,
                self.config.seed_candidate_top_k_lexical,
            ),
            top_n=requested_top_n,
            max_chars_per_skill=requested_skill_chars,
            max_context_chars=requested_context_chars,
            ppr_damping=self.config.ppr_damping,
        )

        if not query.strip():
            return SkillRetrievalResult(
                query=query, budget=budget, summary="Empty query."
            )

        await self.state_manager.query_start()
        try:
            nodes = await self._load_all_nodes()
            if not nodes:
                return SkillRetrievalResult(
                    query=query,
                    budget=budget,
                    summary="No indexed skills available.",
                    rendered_context="No indexed skills available. Proceed without retrieved skills and inspect the task verifier/tests for minimum requirements.",
                )

            edges = await self._load_all_edges()
            rewritten_query = await self._rewrite_query_schema_async(query)
            seed_entries = await self._semantic_seed_scores(
                query,
                nodes,
                requested_seed_top_k,
                rewritten_query,
            )
            if not seed_entries:
                return SkillRetrievalResult(
                    query=query,
                    rewritten_query=rewritten_query,
                    budget=budget,
                    summary=self._render_summary(
                        query,
                        rewritten_query,
                        [],
                        [],
                        [],
                    ),
                    rendered_context=self._render_context(
                        query,
                        [],
                        [],
                        max_chars=requested_context_chars,
                    ),
                )

            transition, _ = build_transition_matrix(nodes, edges)
            personalization = build_personalization(
                len(nodes),
                [index for index, _, _ in seed_entries],
                [weight for _, weight, _ in seed_entries],
            )
            scores = personalized_pagerank(
                transition,
                personalization,
                damping=self.config.ppr_damping,
                max_iter=self.config.ppr_max_iter,
                tol=self.config.ppr_tolerance,
            )

            rank_lookup = {index: rank for index, _, rank in seed_entries}
            selected_skills: list[RetrievedSkill] = []
            total_chars = 0

            for raw_index in np.argsort(scores)[::-1]:
                index = int(raw_index)
                if len(selected_skills) >= requested_top_n:
                    break

                remaining_context = requested_context_chars - total_chars
                if remaining_context <= 0:
                    break

                node = nodes[index]
                payload_budget = min(requested_skill_chars, remaining_context)
                payload = node.render_for_agent(payload_budget)
                if not payload:
                    continue

                rerank_score = self._rerank_skill_score(
                    rewritten_query,
                    node,
                    float(scores[index]),
                    rank_lookup.get(index),
                )
                selected_skills.append(
                    RetrievedSkill(
                        name=node.name,
                        description=node.description,
                        source_path=node.source_path,
                        one_line_capability=node.one_line_capability,
                        score=float(scores[index]),
                        rerank_score=rerank_score,
                        semantic_rank=rank_lookup.get(index),
                        inputs=node.input_types,
                        outputs=node.output_types,
                        domain_tags=node.domain_tags_list,
                        tooling=node.tooling_list,
                        example_tasks=node.example_tasks_list,
                        script_entrypoints=node.script_entrypoints_list,
                        compatibility=node.compatibility_list,
                        allowed_tools=node.allowed_tools_list,
                        rendered_snippet=node.rendered_snippet,
                        payload=payload,
                    )
                )
                total_chars += len(payload)

            selected_skills.sort(
                key=lambda skill: (skill.rerank_score, skill.score), reverse=True
            )
            selected_names = {skill.name for skill in selected_skills}
            retrieved_relations = [
                RetrievedRelation(
                    source=edge.source,
                    target=edge.target,
                    description=edge.description,
                    type=edge.type,
                    weight=edge.weight,
                    confidence=edge.confidence,
                )
                for edge in edges
                if edge.source in selected_names
                and edge.target in selected_names
                and edge.description != "is"
            ]
            retrieved_relations.sort(key=lambda edge: edge.weight, reverse=True)

            seeds = [
                SkillSeed(
                    name=nodes[index].name,
                    source_path=nodes[index].source_path,
                    seed_weight=weight,
                    semantic_rank=rank,
                )
                for index, weight, rank in seed_entries
            ]

            budgeted_skills = self._fit_skills_to_context_budget(
                query,
                selected_skills,
                requested_context_chars,
            )
            budgeted_names = {skill.name for skill in budgeted_skills}
            budgeted_relations = [
                relation
                for relation in retrieved_relations
                if relation.source in budgeted_names
                and relation.target in budgeted_names
            ]

            rendered_context = self._render_context(
                query,
                budgeted_skills,
                budgeted_relations,
                max_chars=requested_context_chars,
            )
            summary = self._render_summary(
                query,
                rewritten_query,
                budgeted_skills,
                budgeted_relations,
                seeds,
            )

            return SkillRetrievalResult(
                query=query,
                rewritten_query=rewritten_query,
                budget=budget,
                seeds=seeds,
                skills=budgeted_skills,
                relations=budgeted_relations,
                rendered_context=rendered_context,
                summary=summary,
            )
        finally:
            await self.state_manager.query_done()

    async def async_retrieve_vector(
        self,
        query: str,
        *,
        top_n: int | None = None,
        max_chars_per_skill: int | None = None,
        max_context_chars: int | None = None,
    ) -> SkillRetrievalResult:
        requested_top_n = top_n or self.config.retrieval_top_n
        requested_skill_chars = max_chars_per_skill or self.config.max_skill_chars
        requested_context_chars = max_context_chars or self.config.max_context_chars

        budget = RetrievalBudget(
            seed_top_k=requested_top_n,
            seed_candidate_top_k_semantic=requested_top_n,
            seed_candidate_top_k_lexical=0,
            top_n=requested_top_n,
            max_chars_per_skill=requested_skill_chars,
            max_context_chars=requested_context_chars,
            ppr_damping=0.0,
        )

        if not query.strip():
            return SkillRetrievalResult(
                query=query, budget=budget, summary="Empty query."
            )

        await self.state_manager.query_start()
        try:
            nodes = await self._load_all_nodes()
            if not nodes:
                return SkillRetrievalResult(
                    query=query,
                    budget=budget,
                    summary="No indexed skills available.",
                    rendered_context="No indexed skills available. Proceed without retrieved skills and inspect the task verifier/tests for minimum requirements.",
                )

            query_schema = self._fallback_query_schema(query)
            seed_entries = await self._vector_seed_scores(
                query,
                nodes,
                requested_top_n,
            )
            if not seed_entries:
                return SkillRetrievalResult(
                    query=query,
                    rewritten_query=query_schema,
                    budget=budget,
                    summary=self._render_summary(
                        query,
                        query_schema,
                        [],
                        [],
                        [],
                    ),
                    rendered_context=self._render_context(
                        query,
                        [],
                        [],
                        max_chars=requested_context_chars,
                    ),
                )

            selected_skills = [
                RetrievedSkill(
                    name=nodes[index].name,
                    description=nodes[index].description,
                    source_path=nodes[index].source_path,
                    one_line_capability=nodes[index].one_line_capability,
                    score=score,
                    rerank_score=score,
                    semantic_rank=rank,
                    inputs=nodes[index].input_types,
                    outputs=nodes[index].output_types,
                    domain_tags=nodes[index].domain_tags_list,
                    tooling=nodes[index].tooling_list,
                    example_tasks=nodes[index].example_tasks_list,
                    script_entrypoints=nodes[index].script_entrypoints_list,
                    compatibility=nodes[index].compatibility_list,
                    allowed_tools=nodes[index].allowed_tools_list,
                    rendered_snippet=nodes[index].rendered_snippet,
                    payload=nodes[index].render_for_agent(requested_skill_chars),
                )
                for index, score, rank in seed_entries
            ]

            budgeted_skills = self._fit_skills_to_context_budget(
                query,
                selected_skills,
                requested_context_chars,
            )
            seeds = [
                SkillSeed(
                    name=nodes[index].name,
                    source_path=nodes[index].source_path,
                    seed_weight=weight,
                    semantic_rank=rank,
                )
                for index, weight, rank in seed_entries
            ]

            rendered_context = self._render_context(
                query,
                budgeted_skills,
                [],
                max_chars=requested_context_chars,
            )
            summary = self._render_summary(
                query,
                query_schema,
                budgeted_skills,
                [],
                seeds,
            )

            return SkillRetrievalResult(
                query=query,
                rewritten_query=query_schema,
                budget=budget,
                seeds=seeds,
                skills=budgeted_skills,
                relations=[],
                rendered_context=rendered_context,
                summary=summary,
            )
        finally:
            await self.state_manager.query_done()

    async def async_query(
        self,
        query: str,
        params: QueryParam | None = None,
        response_model=None,
    ) -> TQueryResponse[SkillNode, SkillEdge, GTHash, GTChunk]:
        result = await self.async_retrieve(query)

        context = TContext(
            entities=[
                (
                    SkillNode.from_lists(
                        name=skill.name,
                        description=skill.description,
                        one_line_capability=skill.one_line_capability,
                        inputs=skill.inputs,
                        outputs=skill.outputs,
                        domain_tags=skill.domain_tags,
                        tooling=skill.tooling,
                        example_tasks=skill.example_tasks,
                        script_entrypoints=skill.script_entrypoints,
                        compatibility=skill.compatibility,
                        allowed_tools=skill.allowed_tools,
                        source_path=skill.source_path,
                        rendered_snippet=skill.rendered_snippet,
                        raw_content=skill.payload,
                    ),
                    skill.score,
                )
                for skill in result.skills
            ],
            relations=[
                (
                    SkillEdge(
                        source=relation.source,
                        target=relation.target,
                        description=relation.description,
                        type=relation.type,
                        weight=relation.weight,
                        confidence=relation.confidence,
                    ),
                    relation.weight,
                )
                for relation in result.relations
            ],
            chunks=[],
        )

        return TQueryResponse(response=result.summary, context=context)

    def insert_skill(self, skill_text: str, metadata: dict[str, Any] | None = None):
        from fast_graphrag._utils import get_event_loop

        return get_event_loop().run_until_complete(
            self.async_insert_skill(skill_text, metadata)
        )
