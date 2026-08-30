import asyncio

import numpy as np

from gos.core.engine import SkillGraphRAG
from gos.core.relink import FocusLinkJob
from gos.core.schema import GOSRelationList


class RecordingEmbeddingService:
    model = "recording-embedding"
    embedding_dim = 8

    def __init__(self):
        self.batch_sizes = []

    async def encode(self, texts, model=None):
        self.batch_sizes.append(len(texts))
        vectors = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        for index, text in enumerate(texts):
            vectors[index, hash(text) % self.embedding_dim] = 1.0
        return vectors


class EmptyRelationLLM:
    model = "empty-relations"

    async def send_message(self, prompt, response_model=None, **kwargs):
        if response_model is GOSRelationList:
            return GOSRelationList(relations=[]), []
        return response_model(), []


SKILLS = [
    """---
name: producer
description: Produce a normalized seismic catalog.
inputs: [raw catalog]
outputs: [normalized seismic catalog]
---
""",
    """---
name: consumer
description: Consume a normalized seismic catalog.
inputs: [normalized seismic catalog]
outputs: [phase associations]
---
""",
    """---
name: distractor
description: Render unrelated travel itineraries.
inputs: [destinations]
outputs: [travel itinerary]
---
""",
]


def test_prepare_jobs_batches_embeddings_and_preserves_io_dependency(tmp_path):
    async def scenario():
        embedding = RecordingEmbeddingService()
        engine = SkillGraphRAG(
            config=SkillGraphRAG.Config(
                llm_service=EmptyRelationLLM(),
                embedding_service=embedding,
                working_dir=str(tmp_path),
                use_full_markdown=False,
                enable_semantic_linking=True,
                link_top_k=2,
            )
        )
        metadatas = [
            {
                "source_path": str(tmp_path / f"skill-{index}" / "SKILL.md"),
                "raw_content": content,
            }
            for index, content in enumerate(SKILLS)
        ]
        await engine.async_insert_skills(SKILLS, metadatas)

        await engine.state_manager.query_start()
        try:
            nodes = await engine._load_all_nodes()
            jobs = await engine._prepare_focus_link_jobs(
                nodes,
                {"producer", "consumer"},
            )
        finally:
            await engine.state_manager.query_done()

        assert all(isinstance(job, FocusLinkJob) for job in jobs)
        assert embedding.batch_sizes[-1] == 2
        by_name = {job.focus_name: job for job in jobs}
        producer = by_name["producer"]
        assert "consumer" in {candidate.name for candidate in producer.candidates}
        assert any(
            edge.source == "producer"
            and edge.target == "consumer"
            and edge.type == "dependency"
            for edge in producer.deterministic_edges
        )

    asyncio.run(scenario())
