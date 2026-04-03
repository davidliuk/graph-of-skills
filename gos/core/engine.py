from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from pathlib import Path
import re
import shutil
from typing import Any, cast

import numpy as np
from loguru import logger

from fast_graphrag._graphrag import BaseGraphRAG, QueryParam
from fast_graphrag._llm import (
    BaseLLMService,
    DefaultEmbeddingService,
    DefaultLLMService,
)
from fast_graphrag._services._chunk_extraction import BaseChunkingService, DefaultChunkingService
from fast_graphrag._services._state_manager import DefaultStateManagerService
from fast_graphrag._storage._gdb_igraph import IGraphStorage, IGraphStorageConfig
from fast_graphrag._storage._ikv_pickle import PickleIndexedKeyValueStorage
from fast_graphrag._storage._namespace import Workspace
from fast_graphrag._storage._vdb_hnswlib import HNSWVectorStorage, HNSWVectorStorageConfig
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
from .policies import SkillEdgeUpsertPolicy, SkillGraphUpsertPolicy, SkillNodeUpsertPolicy
from .prompts import PROMPTS
from .retrieval import build_personalization, build_rank_distribution, build_transition_matrix, personalized_pagerank
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
)
from .litellm_services import LiteLLMEmbeddingService, LiteLLMService
from .services import SkillInformationExtractionService


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
    "to",
    "value",
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


def _resolve_openrouter_api_key() -> str | None:
    return _secret_value(settings.OPENROUTER_API_KEY) or _secret_value(settings.OPENAI_API_KEY)


def _resolve_openrouter_base_url() -> str:
    configured = str(settings.OPENAI_BASE_URL or "").strip().rstrip("/")
    if not configured:
        return DEFAULT_OPENROUTER_API_BASE

    if "openrouter.ai/api" in configured and not configured.endswith("/v1"):
        return f"{configured}/v1"

    return configured


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
            )

        api_key = _secret_value(settings.OPENAI_API_KEY)
        return LiteLLMService(model=model_name, api_key=api_key)
    except Exception as exc:
        logger.warning(f"Falling back to unconfigured LLM placeholder: {exc}")
        return UnconfiguredLLMService(settings.LLM_MODEL, exc)


def build_default_embedding_service() -> Any:
    provider, model_name = parse_model_spec(settings.EMBEDDING_MODEL)
    try:
        if provider == "gemini":
            api_key = _secret_value(settings.GEMINI_API_KEY)
            return LiteLLMEmbeddingService(
                model=settings.EMBEDDING_MODEL,
                embedding_dim=settings.EMBEDDING_DIM,
                api_key=api_key,
            )

        if provider == "openrouter":
            # Embeddings go through OpenRouter's OpenAI-compatible endpoint.
            return LiteLLMEmbeddingService(
                model=model_name,  # plain name; api_base routes to OpenRouter
                embedding_dim=settings.EMBEDDING_DIM,
                api_key=_resolve_openrouter_api_key(),
                base_url=_resolve_openrouter_base_url(),
            )

        api_key = _secret_value(settings.OPENAI_API_KEY)
        return LiteLLMEmbeddingService(
            model=model_name,
            embedding_dim=settings.EMBEDDING_DIM,
            api_key=api_key,
        )
    except Exception as exc:
        logger.warning(f"Falling back to unconfigured embedding placeholder: {exc}")
        return UnconfiguredEmbeddingService(
            settings.EMBEDDING_MODEL,
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
        seed_candidate_top_k_semantic: int = field(default=settings.SEED_CANDIDATE_TOP_K_SEMANTIC)
        seed_candidate_top_k_lexical: int = field(default=settings.SEED_CANDIDATE_TOP_K_LEXICAL)
        retrieval_top_n: int = field(default=settings.RETRIEVAL_TOP_N)
        enable_semantic_linking: bool = field(default=settings.ENABLE_SEMANTIC_LINKING)
        dependency_match_threshold: float = field(
            default=settings.DEPENDENCY_MATCH_THRESHOLD
        )
        ppr_damping: float = field(default=settings.PPR_DAMPING)
        ppr_max_iter: int = field(default=settings.PPR_MAX_ITER)
        ppr_tolerance: float = field(default=settings.PPR_TOLERANCE)
        max_skill_chars: int = field(default=settings.MAX_SKILL_CHARS)
        max_context_chars: int = field(default=settings.MAX_CONTEXT_CHARS)
        snippet_chars: int = field(default=settings.SNIPPET_CHARS)
        rerank_candidate_multiplier: int = field(default=settings.RERANK_CANDIDATE_MULTIPLIER)
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
            graph_storage=IGraphStorage[SkillNode, SkillEdge, TId](
                config=IGraphStorageConfig(SkillNode, SkillEdge)
            ),
            entity_storage=entity_storage,
            chunk_storage=PickleIndexedKeyValueStorage[GTHash, GTChunk](config=None),
            embedding_service=self.config.embedding_service,
            node_upsert_policy=SkillNodeUpsertPolicy(config=None),
            edge_upsert_policy=SkillEdgeUpsertPolicy(config=None),
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
            logger.warning(f"GoS: prebuilt workspace `{source}` does not exist or is not a directory.")
            return ""

        if target.exists() and any(target.iterdir()):
            logger.info(
                f"GoS: workspace `{target}` already has content, skipping bootstrap from `{source}`."
            )
            return ""

        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target, dirs_exist_ok=True)
        logger.info(f"GoS: bootstrapped workspace `{target}` from prebuilt graph `{source}`.")
        return str(source)

    def _prepare_metadata(self, skill_text: str, metadata: dict[str, Any] | None) -> dict[str, Any]:
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

        raw_graph = getattr(target, "_graph", None) or getattr(target, "graph", None) or getattr(target, "g", None)
        if raw_graph is None:
            logger.warning("Graph storage does not expose edge iteration; retrieval will use nodes only.")
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
                token = token.rstrip("s")
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
            producer_norm = re.sub(r"[^a-z0-9]+", "_", producer.lower()).strip("_")
            producer_tokens = self._signature_tokens([producer])
            for consumer in consumer_values:
                consumer_norm = re.sub(r"[^a-z0-9]+", "_", consumer.lower()).strip("_")
                consumer_tokens = self._signature_tokens([consumer])

                if producer_norm and producer_norm == consumer_norm:
                    return 1.0, [producer_norm]
                if producer_norm and consumer_norm and (
                    producer_norm in consumer_norm or consumer_norm in producer_norm
                ):
                    candidate = {producer_norm, consumer_norm}
                    return 0.85, sorted(candidate)

                overlap = producer_tokens & consumer_tokens
                if not overlap:
                    continue

                union = producer_tokens | consumer_tokens
                score = max(0.5, len(overlap) / max(len(union), 1))
                if score > best_score:
                    best_score = score
                    best_evidence = overlap

        return best_score, sorted(best_evidence)


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
        artifacts = re.findall(r"[A-Za-z0-9_./-]+\.(?:py|md|json|csv|stl|dot|txt|yaml|yml|bib|pptx|xlsx|docx)", query)
        return self._dedupe_text([artifact.strip() for artifact in artifacts if artifact.strip()])

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

    def _normalize_query_schema(self, query: str, schema: QuerySchema | None) -> QuerySchema:
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

    def _field_bonus(self, query_tokens: set[str], values: list[str], weight: float) -> float:
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
        score += self._field_bonus(query_tokens, [query_schema.task_name], 0.35) if query_schema.task_name else 0.0
        score += self._field_bonus(query_tokens, [node.name], 1.25)
        score += self._field_bonus(query_tokens, [node.one_line_capability, node.description], 0.9)
        score += self._field_bonus(query_tokens, node.domain_tags_list, 1.15)
        score += self._field_bonus(query_tokens, node.tooling_list, 0.95)
        score += self._field_bonus(query_tokens, node.input_types + node.output_types, 0.75)
        score += self._field_bonus(query_tokens, node.example_tasks_list, 0.8)
        score += self._field_bonus(query_tokens, node.script_entrypoints_list, 0.6)

        normalized_query_text = "\n".join(self._query_schema_values(query_schema)).lower()
        normalized_node_name = re.sub(r"[^a-z0-9]+", " ", node.name.lower()).strip()
        if normalized_node_name and normalized_node_name in normalized_query_text:
            score += 1.2

        artifact_overlap = self._shared_field_score(query_schema.artifacts, node.script_entrypoints_list)
        if artifact_overlap:
            score += 0.9 * artifact_overlap

        if node.script_entrypoints_list:
            score += 0.08
        if semantic_rank is not None:
            score += 0.2 / float(semantic_rank)
        if query_schema.domain and node.domain_tags_list:
            overlap = self._signature_tokens(query_schema.domain) & self._signature_tokens(node.domain_tags_list)
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
        artifacts = self._dedupe_text(self._extract_artifacts("\n".join(text_values)) + node.script_entrypoints_list)
        constraints = self._dedupe_text(node.compatibility_list + node.allowed_tools_list)
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

    def _shared_field_score(self, left_values: list[str], right_values: list[str]) -> float:
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
        score += self._field_bonus(query_tokens, candidate_node.script_entrypoints_list, 0.45)

        pair_score, pair_evidence = self._link_pair_feature_score(source_node, candidate_node)
        score += pair_score
        if semantic_rank is not None:
            score += 0.2 / float(semantic_rank)

        lexical_overlap = self._token_overlap_score(query_tokens, self._node_text_values(candidate_node))
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
    ) -> list[tuple[int, float]]:
        source_schema = self._rewrite_node_query_schema(node)
        scored: list[tuple[int, float]] = []
        for index, candidate in enumerate(nodes):
            if index == node_index:
                continue
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
            combined_scores[index] = max(score, combined_scores.get(index, float('-inf')))

        ranked = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
        return [index for index, _ in ranked[:candidate_top_k]]

    def _build_io_indexes(
        self,
        nodes: list[SkillNode],
    ) -> tuple[dict[str, set[int]], dict[str, set[int]]]:
        output_index: dict[str, set[int]] = {}
        input_index: dict[str, set[int]] = {}

        for index, node in enumerate(nodes):
            for token in self._signature_tokens(node.output_types):
                output_index.setdefault(token, set()).add(index)
            for token in self._signature_tokens(node.input_types):
                input_index.setdefault(token, set()).add(index)

        return output_index, input_index

    def _lexical_seed_scores(
        self,
        query: str,
        nodes: list[SkillNode],
        seed_top_k: int,
        query_schema: QuerySchema | None = None,
    ) -> list[tuple[int, float, int]]:
        effective_schema = query_schema or self._fallback_query_schema(query)
        query_tokens = self._signature_tokens(self._query_schema_values(effective_schema))
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
        lexical_rank_lookup = {
            index: rank for index, _, rank in lexical_seed_entries
        }
        try:
            query_embedding = await self.config.embedding_service.encode([query_text])
            indices, _ = await self.state_manager.entity_storage.get_knn(
                query_embedding,
                top_k=semantic_candidate_top_k,
            )
        except Exception as exc:
            logger.warning(f"Vector seeding failed, falling back to lexical seeding: {exc}")
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
        lines = [f"{node.name}: {node.description or node.one_line_capability or 'n/a'}"]
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

        if (edge.confidence, edge.weight) > (existing.confidence, existing.weight):
            edge_map[key] = edge

    def _dependency_edges_for_pair(
        self,
        node: SkillNode,
        candidate: SkillNode,
    ) -> list[SkillEdge]:
        edges: list[SkillEdge] = []

        forward_score, forward_evidence = self._schema_overlap_score(
            node.output_types,
            candidate.input_types,
        )
        if forward_score >= self.config.dependency_match_threshold:
            evidence = ", ".join(forward_evidence) or "compatible I/O"
            edges.append(
                SkillEdge(
                    source=node.name,
                    target=candidate.name,
                    description=f"{node.name} produces data that {candidate.name} consumes: {evidence}.",
                    type="dependency",
                    weight=forward_score,
                    confidence=forward_score,
                )
            )

        reverse_score, reverse_evidence = self._schema_overlap_score(
            candidate.output_types,
            node.input_types,
        )
        if reverse_score >= self.config.dependency_match_threshold:
            evidence = ", ".join(reverse_evidence) or "compatible I/O"
            edges.append(
                SkillEdge(
                    source=candidate.name,
                    target=node.name,
                    description=f"{candidate.name} produces data that {node.name} consumes: {evidence}.",
                    type="dependency",
                    weight=reverse_score,
                    confidence=reverse_score,
                )
            )

        return edges

    async def _validate_candidate_relations(
        self,
        node: SkillNode,
        candidates: list[SkillNode],
    ) -> list[SkillEdge]:
        if not self.config.enable_semantic_linking or not candidates:
            return []

        candidate_lines = [f"- {self._format_skill_for_linking(candidate)}" for candidate in candidates]

        try:
            relations_list, _ = await self.llm_service.send_message(
                system_prompt=PROMPTS["search_and_link_system"],
                prompt=PROMPTS["search_and_link_prompt"].format(
                    new_skill=self._format_skill_for_linking(node),
                    candidate_skills="\n".join(candidate_lines),
                ),
                response_model=GOSRelationList,
            )
        except Exception as exc:
            logger.warning(f"LLM relation validation failed for {node.name}: {exc}")
            return []

        validated_edges: list[SkillEdge] = []
        for relation in relations_list.relations:
            relation_type = relation.type.lower()
            validated_edges.append(
                SkillEdge(
                    source=relation.source,
                    target=relation.target,
                    description=relation.description,
                    type=relation_type,
                    weight=TYPE_WEIGHTS.get(relation_type, 0.3) * max(relation.confidence, 0.1),
                    confidence=relation.confidence,
                )
            )
        return validated_edges

    async def async_insert_skill(
        self,
        skill_text: str,
        metadata: dict[str, Any] | None = None,
    ):
        prepared_metadata = self._prepare_metadata(skill_text, metadata)
        result = await self.async_insert(content=[skill_text], metadata=[prepared_metadata])
        source_path = str(prepared_metadata.get("source_path") or "")
        parsed = parse_skill_document(skill_text, source_path=source_path)
        if parsed and parsed.name:
            await self._link_skills_incremental({parsed.name})
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
            metadata = provided_metadatas[index] if index < len(provided_metadatas) else None
            prepared_metadatas.append(self._prepare_metadata(skill_text, metadata))

        new_names: set[str] = set()
        for index, skill_text in enumerate(skill_texts):
            source_path = str(prepared_metadatas[index].get("source_path") or "")
            parsed = parse_skill_document(skill_text, source_path=source_path)
            if parsed and parsed.name:
                new_names.add(parsed.name)

        result = await self.async_insert(content=skill_texts, metadata=prepared_metadatas)
        if new_names:
            await self._link_skills_incremental(new_names)
        else:
            await self._link_all_skills()
        return result

    async def async_ensure_skills(
        self,
        skill_texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> SkillSyncResult:
        prepared_metadatas: list[dict[str, Any]] = []
        provided_metadatas = metadatas or []
        for index, skill_text in enumerate(skill_texts):
            metadata = provided_metadatas[index] if index < len(provided_metadatas) else None
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
            snippet_chars = int(metadata.get("snippet_chars", self.config.snippet_chars))
            parsed = parse_skill_document(
                skill_text,
                source_path=source_path,
                snippet_chars=snippet_chars,
            )
            name = parsed.name if parsed is not None else source_path or f"skill_{index}"
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

    async def _link_all_skills(self):
        await self.state_manager.insert_start()
        try:
            target = self.state_manager.graph_storage
            nodes = await self._load_all_nodes()
            if len(nodes) < 2:
                return

            logger.info(f"GoS: linking {len(nodes)} indexed skills.")

            output_index, input_index = self._build_io_indexes(nodes)
            node_names = {node.name for node in nodes}
            edge_map: dict[tuple[str, str, str], SkillEdge] = {}

            for index, node in enumerate(nodes):
                ranked_candidate_indices = await self._rank_link_candidates_for_node(
                    node,
                    nodes,
                    index,
                )
                ranked_candidate_lookup = {
                    candidate_index: rank
                    for rank, candidate_index in enumerate(ranked_candidate_indices, start=1)
                }
                candidate_indices = set(ranked_candidate_indices)

                for token in self._signature_tokens(node.input_types):
                    candidate_indices.update(output_index.get(token, set()))
                for token in self._signature_tokens(node.output_types):
                    candidate_indices.update(input_index.get(token, set()))

                candidate_indices.discard(index)
                llm_candidates: list[tuple[int, SkillNode]] = []

                for candidate_index in sorted(candidate_indices):
                    candidate = nodes[candidate_index]

                    deterministic_edges = self._dependency_edges_for_pair(node, candidate)
                    if deterministic_edges:
                        for edge in deterministic_edges:
                            self._record_edge(edge_map, edge)
                        continue

                    if candidate_index in ranked_candidate_lookup:
                        llm_candidates.append((ranked_candidate_lookup[candidate_index], candidate))

                if llm_candidates:
                    llm_candidates.sort(key=lambda item: item[0])
                    validated_edges = await self._validate_candidate_relations(
                        node,
                        [candidate for _, candidate in llm_candidates[: self.config.link_top_k]],
                    )
                    for edge in validated_edges:
                        if edge.source in node_names and edge.target in node_names:
                            self._record_edge(edge_map, edge)

            if edge_map:
                logger.info(f"GoS: committing {len(edge_map)} edges.")
                await self.state_manager.edge_upsert_policy(
                    self.llm_service,
                    target,
                    list(edge_map.values()),
                )
        finally:
            await self.state_manager.insert_done()

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
                logger.warning(f"GoS: incremental link: none of {new_node_names} found after insert.")
                return

            logger.info(
                f"GoS: incremental linking {len(new_nodes)} skill(s) against {len(all_nodes)} total."
            )

            node_index_by_name: dict[str, int] = {n.name: i for i, n in enumerate(all_nodes)}
            new_node_indices: set[int] = {node_index_by_name[n.name] for n in new_nodes}

            output_index, input_index = self._build_io_indexes(all_nodes)
            node_names = {node.name for node in all_nodes}
            edge_map: dict[tuple[str, str, str], SkillEdge] = {}

            for node in new_nodes:
                node_index = node_index_by_name[node.name]

                ranked_candidate_indices = await self._rank_link_candidates_for_node(
                    node,
                    all_nodes,
                    node_index,
                )
                ranked_candidate_lookup = {
                    candidate_index: rank
                    for rank, candidate_index in enumerate(ranked_candidate_indices, start=1)
                }
                candidate_indices = set(ranked_candidate_indices)

                for token in self._signature_tokens(node.input_types):
                    candidate_indices.update(output_index.get(token, set()))
                for token in self._signature_tokens(node.output_types):
                    candidate_indices.update(input_index.get(token, set()))

                candidate_indices.discard(node_index)
                llm_candidates: list[tuple[int, SkillNode]] = []

                for candidate_index in sorted(candidate_indices):
                    # For new-new pairs, only process when current node has the
                    # lower index to avoid emitting the same edges twice.
                    if candidate_index in new_node_indices and candidate_index < node_index:
                        continue

                    candidate = all_nodes[candidate_index]
                    deterministic_edges = self._dependency_edges_for_pair(node, candidate)
                    if deterministic_edges:
                        for edge in deterministic_edges:
                            self._record_edge(edge_map, edge)
                        continue

                    if candidate_index in ranked_candidate_lookup:
                        llm_candidates.append((ranked_candidate_lookup[candidate_index], candidate))

                if llm_candidates:
                    llm_candidates.sort(key=lambda item: item[0])
                    validated_edges = await self._validate_candidate_relations(
                        node,
                        [candidate for _, candidate in llm_candidates[: self.config.link_top_k]],
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
            semantic_rank = f", seed rank {skill.semantic_rank}" if skill.semantic_rank else ""
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
            return SkillRetrievalResult(query=query, budget=budget, summary="Empty query.")

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

            selected_skills.sort(key=lambda skill: (skill.rerank_score, skill.score), reverse=True)
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
                if relation.source in budgeted_names and relation.target in budgeted_names
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
            return SkillRetrievalResult(query=query, budget=budget, summary="Empty query.")

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
