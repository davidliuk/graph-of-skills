import tempfile
import unittest
from pathlib import Path

import numpy as np

from gos.core.engine import SkillGraphRAG
from gos.core.schema import GOSGraph, GOSRelationList


class FakeEmbeddingService:
    def __init__(self, embedding_dim: int = 16) -> None:
        self.embedding_dim = embedding_dim
        self.model = "fake-embedding"

    async def encode(self, texts, model=None):
        vectors = []
        for text in texts:
            vector = np.zeros(self.embedding_dim, dtype=np.float32)
            for token in text.lower().split():
                vector[hash(token) % self.embedding_dim] += 1.0
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            vectors.append(vector)
        return np.vstack(vectors)


class FakeLLMService:
    model = "fake-llm"

    async def send_message(
        self,
        prompt,
        system_prompt=None,
        history_messages=None,
        response_model=None,
        **kwargs,
    ):
        if response_model is GOSRelationList:
            return GOSRelationList(relations=[]), []
        if response_model is GOSGraph:
            return GOSGraph(nodes=[], edges=[]), []
        if response_model is None:
            return "", []
        return response_model(), []


class RecordingLLMService(FakeLLMService):
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def send_message(
        self,
        prompt,
        system_prompt=None,
        history_messages=None,
        response_model=None,
        **kwargs,
    ):
        if response_model is GOSRelationList:
            self.prompts.append(prompt)
        return await super().send_message(
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            response_model=response_model,
            **kwargs,
        )


SKILLS = [
    (
        """---
name: read_csv
description: Read a CSV file into a dataset.
one_line_capability: Load structured CSV data into tabular datasets.
domain:
  - data ingestion
  - tabular analytics
inputs:
  - csv_path: str
outputs:
  - dataset
compatibility:
  - codex
allowed-tools:
  - python
metadata:
  tags:
    - pandas
    - csv
---
# Tools
- pandas
# Usage
Load a CSV file and return a dataset object.
""",
        "read_csv/SKILL.md",
    ),
    (
        """---
name: analyze_trend
description: Analyze a dataset and produce a trend report.
one_line_capability: Summarize metrics and detect directional trends in tabular data.
domain:
  - analytics
  - trend analysis
inputs:
  - dataset
outputs:
  - trend_report
compatibility:
  - codex
allowed-tools:
  - python
metadata:
  example_tasks:
    - analyze sales csv trends
---
# Usage
Compute summary statistics over a dataset.
""",
        "analyze_trend/SKILL.md",
    ),
    (
        """---
name: render_chart
description: Render a chart from a trend report.
inputs:
  - trend_report
outputs:
  - chart
compatibility:
  - codex
allowed-tools:
  - python
---
# Usage
Render a chart image from a trend report.
""",
        "render_chart/SKILL.md",
    ),
]


class EngineSmokeTest(unittest.IsolatedAsyncioTestCase):
    async def test_insert_link_and_retrieve(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = SkillGraphRAG(
                config=SkillGraphRAG.Config(
                    llm_service=FakeLLMService(),
                    embedding_service=FakeEmbeddingService(),
                    working_dir=tmpdir,
                    use_full_markdown=False,
                    enable_semantic_linking=False,
                )
            )

            contents = [content for content, _ in SKILLS]
            metadatas = [
                {"source_path": str(Path(tmpdir) / relative_path), "raw_content": content}
                for content, relative_path in SKILLS
            ]

            await engine.async_insert_skills(contents, metadatas)
            result = await engine.async_retrieve(
                "analyze sales csv trends and make a chart",
                top_n=3,
                seed_top_k=2,
            )

            retrieved_names = [skill.name for skill in result.skills]
            self.assertIn("read_csv", retrieved_names)
            self.assertIn("analyze_trend", retrieved_names)
            self.assertTrue(
                any(
                    edge.source == "read_csv" and edge.target == "analyze_trend"
                    for edge in result.relations
                )
            )
            self.assertGreater(len(result.rendered_context), 0)

    async def test_retrieve_respects_context_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = SkillGraphRAG(
                config=SkillGraphRAG.Config(
                    llm_service=FakeLLMService(),
                    embedding_service=FakeEmbeddingService(),
                    working_dir=tmpdir,
                    use_full_markdown=False,
                    enable_semantic_linking=False,
                    max_context_chars=250,
                    max_skill_chars=180,
                )
            )

            contents = [content for content, _ in SKILLS]
            metadatas = [
                {"source_path": str(Path(tmpdir) / relative_path), "raw_content": content}
                for content, relative_path in SKILLS
            ]

            await engine.async_insert_skills(contents, metadatas)
            result = await engine.async_retrieve(
                "analyze sales csv trends and make a chart",
                top_n=3,
                seed_top_k=2,
                max_context_chars=250,
                max_chars_per_skill=180,
            )

            self.assertLessEqual(len(result.rendered_context), 250)
            self.assertGreater(len(result.skills), 0)

    async def test_prebuilt_workspace_reuses_existing_skills(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_workspace = Path(tmpdir) / "source_workspace"
            target_workspace = Path(tmpdir) / "target_workspace"

            source_engine = SkillGraphRAG(
                config=SkillGraphRAG.Config(
                    llm_service=FakeLLMService(),
                    embedding_service=FakeEmbeddingService(),
                    working_dir=str(source_workspace),
                    use_full_markdown=False,
                    enable_semantic_linking=False,
                )
            )

            contents = [content for content, _ in SKILLS[:2]]
            metadatas = [
                {
                    "source_path": str(Path(tmpdir) / relative_path),
                    "raw_content": content,
                }
                for content, relative_path in SKILLS[:2]
            ]

            await source_engine.async_insert_skills(contents, metadatas)

            target_engine = SkillGraphRAG(
                config=SkillGraphRAG.Config(
                    llm_service=FakeLLMService(),
                    embedding_service=FakeEmbeddingService(),
                    working_dir=str(target_workspace),
                    prebuilt_working_dir=str(source_workspace),
                    use_full_markdown=False,
                    enable_semantic_linking=False,
                )
            )

            sync_result = await target_engine.async_ensure_skills(contents, metadatas)

            await target_engine.state_manager.query_start()
            try:
                node_count = await target_engine.state_manager.graph_storage.node_count()
            finally:
                await target_engine.state_manager.query_done()

            self.assertEqual(sync_result.prebuilt_working_dir, str(source_workspace))
            self.assertEqual(sync_result.inserted_count, 0)
            self.assertEqual(sync_result.updated_count, 0)
            self.assertEqual(sync_result.reused_count, 2)
            self.assertEqual(node_count, 2)

    async def test_prebuilt_workspace_backfills_missing_skills(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_workspace = Path(tmpdir) / "source_workspace"
            target_workspace = Path(tmpdir) / "target_workspace"

            source_engine = SkillGraphRAG(
                config=SkillGraphRAG.Config(
                    llm_service=FakeLLMService(),
                    embedding_service=FakeEmbeddingService(),
                    working_dir=str(source_workspace),
                    use_full_markdown=False,
                    enable_semantic_linking=False,
                )
            )

            seeded_contents = [content for content, _ in SKILLS[:2]]
            seeded_metadatas = [
                {
                    "source_path": str(Path(tmpdir) / relative_path),
                    "raw_content": content,
                }
                for content, relative_path in SKILLS[:2]
            ]

            await source_engine.async_insert_skills(seeded_contents, seeded_metadatas)

            all_contents = [content for content, _ in SKILLS]
            all_metadatas = [
                {
                    "source_path": str(Path(tmpdir) / relative_path),
                    "raw_content": content,
                }
                for content, relative_path in SKILLS
            ]

            target_engine = SkillGraphRAG(
                config=SkillGraphRAG.Config(
                    llm_service=FakeLLMService(),
                    embedding_service=FakeEmbeddingService(),
                    working_dir=str(target_workspace),
                    prebuilt_working_dir=str(source_workspace),
                    use_full_markdown=False,
                    enable_semantic_linking=False,
                )
            )

            sync_result = await target_engine.async_ensure_skills(all_contents, all_metadatas)

            await target_engine.state_manager.query_start()
            try:
                node_count = await target_engine.state_manager.graph_storage.node_count()
            finally:
                await target_engine.state_manager.query_done()

            self.assertEqual(sync_result.inserted_count, 1)
            self.assertEqual(sync_result.updated_count, 0)
            self.assertEqual(sync_result.reused_count, 2)
            self.assertEqual(sync_result.inserted_skill_names, ["render_chart"])
            self.assertEqual(node_count, 3)


    async def test_retrieval_reranker_prefers_domain_specific_skill(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = SkillGraphRAG(
                config=SkillGraphRAG.Config(
                    llm_service=FakeLLMService(),
                    embedding_service=FakeEmbeddingService(),
                    working_dir=tmpdir,
                    use_full_markdown=False,
                    enable_semantic_linking=False,
                    retrieval_top_n=2,
                    seed_top_k=2,
                )
            )

            domain_skill = """---
name: tfidf-parallel-search
description: Parallel TF-IDF indexing and ranked batch search for Python corpora.
one_line_capability: Build deterministic TF-IDF inverted indexes with multiprocessing.
domain:
  - information retrieval
  - python parallelism
tags:
  - tfidf
  - processpoolexecutor
  - ranking
inputs:
  - document corpus
outputs:
  - ranked search results
allowed-tools:
  - python
metadata:
  example_tasks:
    - parallelize TF-IDF index building and search
---
# Tools
- ProcessPoolExecutor
- multiprocessing
# Examples
- parallelize TF-IDF indexing while preserving deterministic ranking
"""
            distractor_skill = """---
name: travel-search-planner
description: Search for flights, hotels, and attractions for multi-city travel.
one_line_capability: Plan travel itineraries and compare search results.
domain:
  - travel planning
tags:
  - search
  - ranking
inputs:
  - destinations
outputs:
  - itinerary
allowed-tools:
  - browser
---
# Examples
- search flights and rank accommodations for a vacation
"""

            contents = [domain_skill, distractor_skill]
            metadatas = [
                {"source_path": str(Path(tmpdir) / "tfidf-parallel-search/SKILL.md"), "raw_content": domain_skill},
                {"source_path": str(Path(tmpdir) / "travel-search-planner/SKILL.md"), "raw_content": distractor_skill},
            ]

            await engine.async_insert_skills(contents, metadatas)
            result = await engine.async_retrieve(
                "parallelize TF-IDF index building and search in Python using ProcessPoolExecutor while preserving deterministic ranking",
                top_n=2,
                seed_top_k=2,
            )

            self.assertEqual(result.skills[0].name, "tfidf-parallel-search")
            self.assertGreater(result.skills[0].rerank_score, result.skills[1].rerank_score)
            self.assertIn("information retrieval", result.rewritten_query.domain)
            self.assertIn("python parallelism", result.rewritten_query.domain)
            self.assertIn("tfidf", result.rewritten_query.operations)


    async def test_linking_reranker_prefers_domain_specific_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            llm = RecordingLLMService()
            engine = SkillGraphRAG(
                config=SkillGraphRAG.Config(
                    llm_service=llm,
                    embedding_service=FakeEmbeddingService(),
                    working_dir=tmpdir,
                    use_full_markdown=False,
                    enable_semantic_linking=True,
                    link_top_k=1,
                )
            )

            source_skill = """---
name: tfidf_parallel_indexer
description: Build TF-IDF indexes and batch search pipelines for Python corpora.
one_line_capability: Parallelize deterministic TF-IDF indexing and ranked search.
domain:
  - information retrieval
  - python parallelism
tags:
  - tfidf
  - processpoolexecutor
inputs:
  - document corpus
outputs:
  - index shards
allowed-tools:
  - python
metadata:
  example_tasks:
    - parallelize tfidf index building
---
# Tools
- ProcessPoolExecutor
- multiprocessing
"""
            matching_skill = """---
name: tfidf_rank_fusion
description: Merge ranked TF-IDF search results from parallel workers.
one_line_capability: Combine deterministic TF-IDF worker outputs into final rankings.
domain:
  - information retrieval
tags:
  - tfidf
  - ranking
inputs:
  - worker rankings
outputs:
  - fused ranking
allowed-tools:
  - python
metadata:
  example_tasks:
    - deterministic tfidf rank fusion
---
# Tools
- ProcessPoolExecutor
"""
            distractor_skill = """---
name: travel_search_planner
description: Rank flights and hotels for travel itineraries.
one_line_capability: Search vacation options and compare travel prices.
domain:
  - travel planning
tags:
  - search
  - ranking
inputs:
  - destinations
outputs:
  - itinerary
allowed-tools:
  - browser
---
# Examples
- rank hotels for a vacation
"""

            contents = [source_skill, matching_skill, distractor_skill]
            metadatas = [
                {"source_path": str(Path(tmpdir) / "tfidf_parallel_indexer/SKILL.md"), "raw_content": source_skill},
                {"source_path": str(Path(tmpdir) / "tfidf_rank_fusion/SKILL.md"), "raw_content": matching_skill},
                {"source_path": str(Path(tmpdir) / "travel_search_planner/SKILL.md"), "raw_content": distractor_skill},
            ]

            await engine.async_insert_skills(contents, metadatas)

            self.assertTrue(llm.prompts)
            combined_prompt = "\n".join(llm.prompts)
            self.assertIn("tfidf_rank_fusion", combined_prompt)
            self.assertNotIn("travel_search_planner", combined_prompt)
            self.assertIn("Domain Tags:", combined_prompt)
            self.assertIn("Tooling:", combined_prompt)


    async def test_full_markdown_extraction_prefers_llm_semantic_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            llm = ExtractionLLMService()
            engine = SkillGraphRAG(
                config=SkillGraphRAG.Config(
                    llm_service=llm,
                    embedding_service=FakeEmbeddingService(),
                    working_dir=tmpdir,
                    use_full_markdown=True,
                    enable_semantic_linking=False,
                )
            )

            skill = """---
name: mesh_volume_analyzer
description: Analyze STL meshes and compute component volumes.
---
# Usage
Load an STL mesh, find connected components, and compute each component volume.

# Tools
- trimesh
- numpy

# Examples
- compute connected component volumes for an STL
"""
            metadata = {
                "source_path": str(Path(tmpdir) / "mesh_volume_analyzer/SKILL.md"),
                "raw_content": skill,
            }

            await engine.async_insert_skills([skill], [metadata])
            hydrated = await engine.async_hydrate_skills(["mesh_volume_analyzer"])

            self.assertTrue(llm.extraction_prompts)
            self.assertEqual(len(hydrated), 1)
            self.assertEqual(hydrated[0].inputs, ["stl mesh"])
            self.assertEqual(hydrated[0].outputs, ["volume report"])
            self.assertEqual(hydrated[0].domain_tags, ["mesh geometry"])
            self.assertEqual(hydrated[0].tooling, ["numpy", "trimesh"])
            self.assertEqual(hydrated[0].example_tasks, ["compute connected component volumes for an STL"])


class ExtractionLLMService(FakeLLMService):
    def __init__(self) -> None:
        self.extraction_prompts: list[str] = []

    async def send_message(
        self,
        prompt,
        system_prompt=None,
        history_messages=None,
        response_model=None,
        **kwargs,
    ):
        if response_model is GOSGraph:
            self.extraction_prompts.append(prompt)
            return GOSGraph(
                nodes=[
                    {
                        "name": "mesh_volume_analyzer",
                        "description": "Analyze STL meshes and compute component volumes.",
                        "one_line_capability": "Compute connected components and mesh volume summaries.",
                        "inputs": ["stl mesh"],
                        "outputs": ["volume report"],
                        "domain_tags": ["mesh geometry"],
                        "tooling": ["numpy", "trimesh"],
                        "example_tasks": ["compute connected component volumes for an STL"],
                        "script_entrypoints": [],
                        "compatibility": [],
                        "allowed_tools": [],
                        "source_path": "",
                        "rendered_snippet": "",
                        "raw_content": "",
                        "metadata": {"llm": True},
                        "skill_id": "",
                    }
                ],
                edges=[],
            ), []
        return await super().send_message(
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            response_model=response_model,
            **kwargs,
        )


if __name__ == "__main__":
    unittest.main()
