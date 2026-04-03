from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

from fast_graphrag._llm import BaseLLMService
from fast_graphrag._services._information_extraction import DefaultInformationExtractionService
from fast_graphrag._storage._base import BaseGraphStorage
from fast_graphrag._types import TChunk, TId

from .parsing import build_extraction_input, parse_skill_document
from .prompts import PROMPTS
from .schema import GOSGraph, GOSSkill, SkillEdge, SkillNode


@dataclass
class SkillInformationExtractionService(DefaultInformationExtractionService):
    use_full_markdown: bool = field(default=False)
    snippet_chars: int = field(default=800)

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

    def _merge_field_lists(self, inferred: list[str], parsed: list[str], *, llm_primary: bool = False) -> list[str]:
        if llm_primary:
            if inferred:
                return self._dedupe(inferred)
            return self._dedupe(parsed)
        if self.use_full_markdown and inferred:
            return self._dedupe(inferred + parsed)
        return self._dedupe(parsed + inferred)

    def _merge_metadata(self, inferred: dict[str, Any], parsed: dict[str, Any]) -> dict[str, Any]:
        merged = dict(parsed)
        for key, value in inferred.items():
            if key not in merged:
                merged[key] = value
                continue
            existing = merged[key]
            if isinstance(existing, list) and isinstance(value, list):
                merged[key] = self._dedupe([*(str(item) for item in existing), *(str(item) for item in value)])
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
                system_prompt=PROMPTS["skill_extraction_system"].format(domain="Agent Skills"),
                prompt=PROMPTS["skill_extraction_prompt"].format(input_text=document_input),
                response_model=GOSGraph,
            )
        except Exception:
            return None

        if not graph.nodes:
            return None

        return graph.nodes[0]

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
        if self.use_full_markdown:
            prompt_input = build_extraction_input(document)
            inferred = self._normalize_inferred_skill(
                await self._infer_missing_fields(llm, prompt_input),
                document.name,
                document.description,
            )

        llm_primary = self.use_full_markdown
        inputs = self._merge_field_lists(inferred.inputs if inferred else [], document.inputs, llm_primary=llm_primary)
        outputs = self._merge_field_lists(inferred.outputs if inferred else [], document.outputs, llm_primary=llm_primary)
        domain_tags = self._merge_field_lists(inferred.domain_tags if inferred else [], document.domain_tags, llm_primary=llm_primary)
        tooling = self._merge_field_lists(inferred.tooling if inferred else [], document.tooling, llm_primary=llm_primary)
        example_tasks = self._merge_field_lists(inferred.example_tasks if inferred else [], document.example_tasks, llm_primary=llm_primary)
        compatibility = self._merge_field_lists(inferred.compatibility if inferred else [], document.compatibility)
        allowed_tools = self._merge_field_lists(inferred.allowed_tools if inferred else [], document.allowed_tools)
        script_entrypoints = self._merge_field_lists(
            document.script_entrypoints,
            inferred.script_entrypoints if inferred else [],
        )
        one_line_capability = (
            (inferred.one_line_capability.strip() if inferred and inferred.one_line_capability else "")
            or document.one_line_capability
        )
        metadata = self._merge_metadata(inferred.metadata if inferred else {}, document.metadata)
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
        from fast_graphrag._storage._gdb_igraph import IGraphStorage, IGraphStorageConfig

        graph_storage = IGraphStorage[SkillNode, SkillEdge, TId](
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
