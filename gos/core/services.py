from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Iterable

from fast_graphrag._llm import BaseLLMService
from fast_graphrag._services._information_extraction import (
    DefaultInformationExtractionService,
)
from fast_graphrag._storage._base import BaseGraphStorage
from fast_graphrag._types import TChunk, TId

from .parsing import build_extraction_input, parse_skill_document
from .prompts import PROMPTS
from .schema import GOSGraph, GOSSkill, SkillEdge, SkillNode
from .storage import DirectedIGraphStorage


@dataclass
class SkillInformationExtractionService(DefaultInformationExtractionService):
    use_full_markdown: bool = field(default=False)
    snippet_chars: int = field(default=800)
    extraction_concurrency: int = field(default=6)

    def extract(
        self,
        llm: BaseLLMService,
        documents: Iterable[Iterable[TChunk]],
        prompt_kwargs: dict[str, str],
        entity_types: list[str],
    ) -> list[asyncio.Future[Any]]:
        semaphore = asyncio.Semaphore(max(int(self.extraction_concurrency), 1))

        async def extract_bounded(document: Iterable[TChunk]) -> Any:
            async with semaphore:
                return await self._extract(
                    llm,
                    document,
                    dict(prompt_kwargs),
                    entity_types,
                )

        return [
            asyncio.create_task(extract_bounded(document)) for document in documents
        ]

    def _chunk_metadata(self, chunk: TChunk) -> dict[str, Any]:
        metadata = getattr(chunk, "metadata", None)
        if metadata is None:
            return {}
        if isinstance(metadata, dict):
            return metadata
        if hasattr(metadata, "model_dump"):
            return metadata.model_dump()
        if hasattr(metadata, "dict"):
            return metadata.dict()
        return {}

    @staticmethod
    def _dedupe(values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            normalized = str(value).strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(normalized)
        return result

    def _merge_field_lists(
        self, inferred: list[str], parsed: list[str], *, llm_primary: bool = False
    ) -> list[str]:
        if llm_primary:
            if inferred:
                return self._dedupe(inferred)
            return self._dedupe(parsed)
        if self.use_full_markdown and inferred:
            return self._dedupe(inferred + parsed)
        return self._dedupe(parsed + inferred)

    def _merge_metadata(
        self, inferred: dict[str, Any], parsed: dict[str, Any]
    ) -> dict[str, Any]:
        merged = dict(parsed)
        for key, value in inferred.items():
            if key not in merged:
                merged[key] = value
                continue
            existing = merged[key]
            if isinstance(existing, list) and isinstance(value, list):
                merged[key] = self._dedupe(
                    [*(str(item) for item in existing), *(str(item) for item in value)]
                )
            elif isinstance(existing, dict) and isinstance(value, dict):
                nested = dict(existing)
                nested.update(value)
                merged[key] = nested
        return merged

    def _normalize_inferred_skill(
        self,
        inferred: GOSSkill | None,
        document_name: str,
        document_description: str,
    ) -> GOSSkill | None:
        if inferred is None:
            return None

        name = inferred.name.strip() or document_name
        description = inferred.description.strip() or document_description
        if not name or not description:
            return None

        metadata = inferred.metadata if isinstance(inferred.metadata, dict) else {}
        return GOSSkill(
            name=name,
            description=description,
            one_line_capability=inferred.one_line_capability,
            inputs=self._dedupe(inferred.inputs),
            outputs=self._dedupe(inferred.outputs),
            domain_tags=self._dedupe(inferred.domain_tags),
            tooling=self._dedupe(inferred.tooling),
            example_tasks=self._dedupe(inferred.example_tasks),
            script_entrypoints=self._dedupe(inferred.script_entrypoints),
            compatibility=self._dedupe(inferred.compatibility),
            allowed_tools=self._dedupe(inferred.allowed_tools),
            source_path=inferred.source_path,
            rendered_snippet=inferred.rendered_snippet,
            raw_content=inferred.raw_content,
            metadata=metadata,
            skill_id=inferred.skill_id,
        )

    async def _infer_missing_fields(
        self,
        llm: BaseLLMService,
        document_input: str,
    ) -> GOSSkill | None:
        try:
            graph, _ = await llm.send_message(
                system_prompt=PROMPTS["skill_extraction_system"].format(
                    domain="Agent Skills"
                ),
                prompt=PROMPTS["skill_extraction_prompt"].format(
                    input_text=document_input
                ),
                response_model=GOSGraph,
                gos_stage="semantic_completion",
            )
        except Exception:
            return None

        if not graph.nodes:
            return None

        return graph.nodes[0]

    @staticmethod
    def _needs_semantic_completion(document: Any) -> bool:
        has_interface = bool(document.inputs and document.outputs)
        has_semantic_context = bool(
            document.domain_tags or document.tooling or document.example_tasks
        )
        return not (
            document.one_line_capability and has_interface and has_semantic_context
        )

    async def _extract_from_chunk(
        self,
        llm: BaseLLMService,
        chunk: TChunk,
        prompt_kwargs: dict[str, str],
        entity_types: list[str],
    ) -> GOSGraph:
        metadata = self._chunk_metadata(chunk)
        full_content = str(metadata.get("raw_content") or chunk.content or "")
        source_path = str(metadata.get("source_path") or "")

        document = parse_skill_document(
            full_content,
            source_path=source_path,
            snippet_chars=int(metadata.get("snippet_chars", self.snippet_chars)),
        )
        if document is None:
            return GOSGraph(nodes=[], edges=[])

        inferred = None
        if self.use_full_markdown and self._needs_semantic_completion(document):
            prompt_input = build_extraction_input(document)
            inferred = self._normalize_inferred_skill(
                await self._infer_missing_fields(llm, prompt_input),
                document.name,
                document.description,
            )

        inferred_inputs = inferred.inputs if inferred else []
        inferred_outputs = inferred.outputs if inferred else []
        inputs = self._dedupe(document.inputs or inferred_inputs)
        outputs = self._dedupe(document.outputs or inferred_outputs)
        domain_tags = self._merge_field_lists(
            inferred.domain_tags if inferred else [],
            document.domain_tags,
        )
        tooling = self._merge_field_lists(
            inferred.tooling if inferred else [],
            document.tooling,
        )
        example_tasks = self._merge_field_lists(
            inferred.example_tasks if inferred else [],
            document.example_tasks,
        )
        compatibility = self._merge_field_lists(
            inferred.compatibility if inferred else [], document.compatibility
        )
        allowed_tools = self._merge_field_lists(
            inferred.allowed_tools if inferred else [], document.allowed_tools
        )
        script_entrypoints = self._merge_field_lists(
            document.script_entrypoints,
            inferred.script_entrypoints if inferred else [],
        )
        one_line_capability = document.one_line_capability or (
            inferred.one_line_capability.strip()
            if inferred and inferred.one_line_capability
            else ""
        )
        metadata = self._merge_metadata(
            inferred.metadata if inferred else {}, document.metadata
        )
        metadata.setdefault("extraction_source", "llm+parser" if inferred else "parser")

        node = GOSSkill(
            name=document.name,
            description=document.description,
            one_line_capability=one_line_capability,
            inputs=inputs,
            outputs=outputs,
            domain_tags=domain_tags,
            tooling=tooling,
            example_tasks=example_tasks,
            script_entrypoints=script_entrypoints,
            compatibility=compatibility,
            allowed_tools=allowed_tools,
            source_path=document.source_path,
            rendered_snippet=document.rendered_snippet,
            raw_content=document.raw_content,
            metadata=metadata,
            skill_id=document.skill_id,
        )
        return GOSGraph(nodes=[node], edges=[])

    async def _merge(
        self,
        llm: BaseLLMService,
        graphs: list[GOSGraph],
    ) -> BaseGraphStorage[SkillNode, SkillEdge, TId]:
        from fast_graphrag._storage._gdb_igraph import IGraphStorageConfig

        graph_storage = DirectedIGraphStorage(
            config=IGraphStorageConfig(SkillNode, SkillEdge)
        )

        await graph_storage.insert_start()
        try:
            for graph in graphs:
                nodes = [
                    SkillNode.from_lists(
                        name=node.name,
                        description=node.description,
                        one_line_capability=node.one_line_capability,
                        inputs=node.inputs,
                        outputs=node.outputs,
                        domain_tags=node.domain_tags,
                        tooling=node.tooling,
                        example_tasks=node.example_tasks,
                        script_entrypoints=node.script_entrypoints,
                        compatibility=node.compatibility,
                        allowed_tools=node.allowed_tools,
                        source_path=node.source_path,
                        rendered_snippet=node.rendered_snippet,
                        raw_content=node.raw_content,
                        metadata=node.metadata,
                        skill_id=node.skill_id,
                    )
                    for node in graph.nodes
                ]
                edges = [
                    SkillEdge(
                        source=edge.source,
                        target=edge.target,
                        description=edge.description,
                        type=edge.type,
                        confidence=edge.confidence,
                    )
                    for edge in graph.edges
                ]
                await self.graph_upsert(llm, graph_storage, nodes, edges)
        finally:
            await graph_storage.insert_done()

        return graph_storage
