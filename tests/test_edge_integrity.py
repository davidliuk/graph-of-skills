import asyncio

import numpy as np
import pytest

from fast_graphrag._storage._gdb_igraph import IGraphStorageConfig
from fast_graphrag._storage._namespace import Workspace

from gos.core.engine import SkillGraphRAG
from gos.core.policies import SkillEdgeUpsertPolicy
from gos.core.schema import SkillEdge, SkillNode
from gos.core.storage import DirectedIGraphStorage


class ConstantEmbeddingService:
    model = "constant-embedding"
    embedding_dim = 4

    async def encode(self, texts, model=None):
        return np.ones((len(texts), self.embedding_dim), dtype=np.float32)


class NoopLLMService:
    model = "noop-llm"

    async def send_message(self, prompt, **kwargs):
        response_model = kwargs.get("response_model")
        return response_model(), []


def test_skill_edge_rejects_legacy_is_relation():
    with pytest.raises(ValueError, match="legacy|relation"):
        SkillEdge(source="a", target="b", description="is", type="")


def test_typed_edge_upsert_is_idempotent_and_keeps_stronger_record(tmp_path):
    async def scenario():
        storage = DirectedIGraphStorage(
            config=IGraphStorageConfig(SkillNode, SkillEdge),
            namespace=Workspace(str(tmp_path)).make_for("graph"),
        )
        await storage.insert_start()
        await storage.upsert_node(SkillNode(name="a"), None)
        await storage.upsert_node(SkillNode(name="b"), None)

        policy = SkillEdgeUpsertPolicy(config=None)
        weak = SkillEdge(
            source="a",
            target="b",
            description="weak workflow evidence",
            type="workflow",
            weight=0.3,
            confidence=0.6,
            provenance="llm_validated",
        )
        strong = SkillEdge(
            source="a",
            target="b",
            description="strong workflow evidence",
            type="workflow",
            weight=0.6,
            confidence=0.9,
            provenance="llm_validated",
        )

        await policy(NoopLLMService(), storage, [weak])
        await policy(NoopLLMService(), storage, [strong])

        assert await storage.edge_count() == 1
        edge = await storage.get_edge_by_index(0)
        assert edge is not None
        assert edge.description == "strong workflow evidence"
        assert edge.confidence == pytest.approx(0.9)

    asyncio.run(scenario())


def test_dependency_dominates_same_direction_workflow(tmp_path):
    async def scenario():
        storage = DirectedIGraphStorage(
            config=IGraphStorageConfig(SkillNode, SkillEdge),
            namespace=Workspace(str(tmp_path)).make_for("graph"),
        )
        await storage.insert_start()
        await storage.upsert_node(SkillNode(name="producer"), None)
        await storage.upsert_node(SkillNode(name="consumer"), None)

        policy = SkillEdgeUpsertPolicy(config=None)
        workflow = SkillEdge(
            source="producer",
            target="consumer",
            description="producer runs before consumer",
            type="workflow",
            weight=0.6,
            confidence=0.9,
            provenance="llm_validated",
        )
        dependency = SkillEdge(
            source="producer",
            target="consumer",
            description="producer emits audio consumed by consumer",
            type="dependency",
            weight=1.0,
            confidence=1.0,
            provenance="deterministic_io",
        )

        await policy(NoopLLMService(), storage, [workflow, dependency, workflow])

        assert await storage.edge_count() == 1
        edge = await storage.get_edge_by_index(0)
        assert edge is not None
        assert edge.type == "dependency"

    asyncio.run(scenario())


def test_indexing_does_not_inject_untyped_embedding_identity_edges(tmp_path):
    skills = [
        """---
name: alpha
description: Alpha standalone capability.
inputs: [alpha_input]
outputs: [alpha_output]
---
""",
        """---
name: beta
description: Beta standalone capability.
inputs: [beta_input]
outputs: [beta_output]
---
""",
    ]

    async def scenario():
        engine = SkillGraphRAG(
            config=SkillGraphRAG.Config(
                llm_service=NoopLLMService(),
                embedding_service=ConstantEmbeddingService(),
                working_dir=str(tmp_path),
                use_full_markdown=False,
                enable_semantic_linking=False,
            )
        )
        metadatas = [
            {
                "source_path": str(tmp_path / name / "SKILL.md"),
                "raw_content": content,
            }
            for name, content in zip(("alpha", "beta"), skills)
        ]
        await engine.async_insert_skills(skills, metadatas)

        await engine.state_manager.query_start()
        try:
            edges = await engine._load_all_edges()
        finally:
            await engine.state_manager.query_done()

        assert edges == []

    asyncio.run(scenario())


def test_updating_skill_removes_stale_incident_dependency(tmp_path):
    producer = """---
name: producer
description: Produce a normalized catalog.
inputs: [raw_catalog]
outputs: [normalized_catalog]
---
"""
    consumer_v1 = """---
name: consumer
description: Consume a normalized catalog.
inputs: [normalized_catalog]
outputs: [analysis_report]
---
"""
    consumer_v2 = """---
name: consumer
description: Consume an unrelated image tensor.
inputs: [image_tensor]
outputs: [analysis_report]
---
"""

    async def scenario():
        engine = SkillGraphRAG(
            config=SkillGraphRAG.Config(
                llm_service=NoopLLMService(),
                embedding_service=ConstantEmbeddingService(),
                working_dir=str(tmp_path),
                use_full_markdown=False,
                enable_semantic_linking=False,
            )
        )
        producer_path = str(tmp_path / "producer" / "SKILL.md")
        consumer_path = str(tmp_path / "consumer" / "SKILL.md")
        await engine.async_insert_skills(
            [producer, consumer_v1],
            [
                {"source_path": producer_path, "raw_content": producer},
                {"source_path": consumer_path, "raw_content": consumer_v1},
            ],
        )

        await engine.async_insert_skills(
            [consumer_v2],
            [{"source_path": consumer_path, "raw_content": consumer_v2}],
        )

        await engine.state_manager.query_start()
        try:
            edges = await engine._load_all_edges()
        finally:
            await engine.state_manager.query_done()

        assert not any(
            edge.source == "producer"
            and edge.target == "consumer"
            and edge.type == "dependency"
            for edge in edges
        )

    asyncio.run(scenario())
