import asyncio

import numpy as np

from gos.core.engine import SkillGraphRAG
from gos.core.schema import GOSRelationList, SkillNode


class FakeEmbeddingService:
    model = "fake-embedding"
    embedding_dim = 4

    async def encode(self, texts, model=None):
        return np.ones((len(texts), self.embedding_dim), dtype=np.float32)


class RelationLLMService:
    model = "relation-validator"

    def __init__(self, relations):
        self.relations = relations

    async def send_message(self, prompt, response_model=None, **kwargs):
        return GOSRelationList(relations=self.relations), []


def _node(name, *, raw_content=""):
    return SkillNode.from_lists(
        name=name,
        description=f"Description for {name}.",
        raw_content=raw_content,
    )


def _engine(tmp_path, relations):
    return SkillGraphRAG(
        config=SkillGraphRAG.Config(
            llm_service=RelationLLMService(relations),
            embedding_service=FakeEmbeddingService(),
            working_dir=str(tmp_path),
            use_full_markdown=False,
            enable_semantic_linking=True,
            relation_min_confidence=0.75,
        )
    )


def test_validator_rejects_out_of_scope_self_unknown_and_low_confidence_relations(
    tmp_path,
):
    relations = [
        {
            "source": "other",
            "target": "candidate",
            "description": "Out-of-scope workflow.",
            "type": "workflow",
            "confidence": 0.95,
            "evidence": ["not in the submitted focus pair"],
        },
        {
            "source": "focus",
            "target": "focus",
            "description": "Self relation.",
            "type": "semantic",
            "confidence": 0.95,
            "evidence": ["self"],
        },
        {
            "source": "focus",
            "target": "candidate",
            "description": "Unsupported relation type.",
            "type": "causal",
            "confidence": 0.95,
            "evidence": ["unsupported"],
        },
        {
            "source": "focus",
            "target": "candidate",
            "description": "Low-confidence semantic relation.",
            "type": "semantic",
            "confidence": 0.5,
            "evidence": ["weak topical overlap"],
        },
    ]

    async def scenario():
        engine = _engine(tmp_path, relations)
        edges = await engine._validate_candidate_relations(
            _node("focus"),
            [_node("candidate")],
        )
        assert edges == []

    asyncio.run(scenario())


def test_validator_keeps_scoped_llm_dependency_with_provenance(tmp_path):
    relations = [
        {
            "source": "producer",
            "target": "focus",
            "description": "producer provides a normalized catalog consumed by focus.",
            "type": "dependency",
            "confidence": 0.9,
            "evidence": ["normalized catalog", "focus requires producer"],
        }
    ]

    async def scenario():
        engine = _engine(tmp_path, relations)
        edges = await engine._validate_candidate_relations(
            _node("focus", raw_content="Prerequisites: producer"),
            [_node("producer")],
        )
        assert len(edges) == 1
        edge = edges[0]
        assert (edge.source, edge.target, edge.type) == (
            "producer",
            "focus",
            "dependency",
        )
        assert edge.provenance == "llm_validated"
        assert edge.validator_model == "relation-validator"
        assert "normalized catalog" in edge.evidence

    asyncio.run(scenario())


def test_validator_rejects_llm_dependency_with_wrong_prerequisite_direction(tmp_path):
    relations = [
        {
            "source": "focus",
            "target": "producer",
            "description": "focus depends on producer.",
            "type": "dependency",
            "confidence": 0.95,
            "evidence": ["focus requires producer"],
        }
    ]

    async def scenario():
        engine = _engine(tmp_path, relations)
        edges = await engine._validate_candidate_relations(
            _node("focus", raw_content="Prerequisites: producer"),
            [_node("producer")],
        )
        assert edges == []

    asyncio.run(scenario())


def test_semantic_relation_uses_canonical_endpoint_order(tmp_path):
    relations = [
        {
            "source": "zeta",
            "target": "alpha",
            "description": "Both implement narrow catalog normalization.",
            "type": "semantic",
            "confidence": 0.9,
            "evidence": ["catalog normalization"],
        }
    ]

    async def scenario():
        engine = _engine(tmp_path, relations)
        edges = await engine._validate_candidate_relations(
            _node("zeta"),
            [_node("alpha")],
        )
        assert len(edges) == 1
        assert (edges[0].source, edges[0].target) == ("alpha", "zeta")

    asyncio.run(scenario())
