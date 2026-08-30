import asyncio
import json

import numpy as np

from gos.core.construction_report import (
    build_construction_report,
    write_construction_report,
)
from gos.core.engine import SkillGraphRAG
from gos.core.relink import RelinkProgress, write_relink_progress
from gos.core.schema import GOSGraph, GOSRelationList


class FakeEmbeddingService:
    model = "fake-embedding"
    embedding_dim = 8

    async def encode(self, texts, model=None):
        vectors = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        for index in range(len(texts)):
            vectors[index, index % self.embedding_dim] = 1.0
        return vectors


class FakeLLMService:
    model = "fake-llm"
    temperature = 0.17
    api_key = "must-not-appear"

    async def send_message(self, prompt, response_model=None, **kwargs):
        if response_model is GOSRelationList:
            return GOSRelationList(relations=[]), []
        if response_model is GOSGraph:
            return GOSGraph(nodes=[], edges=[]), []
        return "", []


def test_construction_report_matches_directed_persisted_graph_and_has_no_secret(
    tmp_path,
):
    skills = [
        """---
name: catalog_reader
description: Read a seismic catalog.
inputs: [catalog_path]
outputs: [normalized_catalog]
domain: [seismology]
---
""",
        """---
name: phase_associator
description: Associate phases from a normalized catalog.
inputs: [normalized_catalog]
outputs: [phase_associations]
domain: [seismology]
---
""",
    ]

    async def scenario():
        engine = SkillGraphRAG(
            config=SkillGraphRAG.Config(
                llm_service=FakeLLMService(),
                embedding_service=FakeEmbeddingService(),
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
            for name, content in zip(("catalog_reader", "phase_associator"), skills)
        ]
        await engine.async_insert_skills(skills, metadatas)

        progress = RelinkProgress.new(
            fingerprint="sha256:test",
            total_focus_nodes=2,
            concurrency=4,
            checkpoint_every=1,
        )
        progress.status = "complete"
        progress.completed_focus_names = ["catalog_reader", "phase_associator"]
        progress.persisted_edge_count = 1
        progress.checkpoint_count = 2
        progress.resumed_focus_count = 1
        progress.preparation_seconds = 0.2
        progress.validation_write_seconds = 0.4
        progress.run_id = "run-test"
        progress.attempt_count = 2
        progress.last_attempt_id = "attempt-test"
        progress.event_count = 7
        progress.construction = {
            "wall_time_seconds": 2.0,
            "focus_nodes": 3,
            "validator_requests": 2,
        }
        progress.usage = {
            "llm": {
                "relation_validation": {
                    "calls": 2,
                    "failures": 1,
                    "input_tokens": 100,
                    "cached_input_tokens": 20,
                    "output_tokens": 30,
                    "reasoning_tokens": 10,
                    "cost_usd": 0.02,
                    "elapsed_seconds": 3.0,
                }
            },
            "embedding": {
                "embedding": {
                    "calls": 1,
                    "input_tokens": 50,
                    "cost_usd": 0.001,
                    "elapsed_seconds": 0.2,
                }
            },
        }
        write_relink_progress(tmp_path / "relink_progress.json", progress)

        report = await build_construction_report(engine)
        output = tmp_path / "construction_report.json"
        await write_construction_report(engine, output)
        persisted = json.loads(output.read_text(encoding="utf-8"))

        assert report["graph"]["directed"] is True
        assert report["nodes"]["total"] == 2
        assert report["edges"]["total"] == 1
        assert sum(report["edges"]["by_type"].values()) == 1
        assert report["edges"]["by_provenance"]["deterministic_io"] == 1
        assert report["construction"]["wall_time_seconds"] >= 0
        assert report["configuration"]["llm_temperature"] == 0.17
        assert report["relink"]["concurrency"] == 4
        assert report["relink"]["checkpoint_count"] == 2
        assert report["relink"]["resumed_focus_count"] == 1
        assert report["usage"]["llm"]["relation_validation"]["calls"] == 2
        assert report["usage_totals"]["calls"] == 3
        assert report["usage_totals"]["input_tokens"] == 150
        assert report["usage_totals"]["output_tokens"] == 30
        assert report["usage_totals"]["cost_usd"] == 0.021
        assert report["timing"]["wall_seconds"] == 2.0
        assert report["timing"]["preparation_seconds"] == 0.2
        assert report["timing"]["checkpoint_write_seconds"] == 0.4
        assert report["timing"]["validation_and_wait_seconds"] == 1.4
        assert report["throughput"]["completed_focus_per_second"] == 1.0
        assert report["throughput"]["validator_requests_per_second"] == 1.0
        assert report["observability"] == {
            "run_id": "run-test",
            "attempt_count": 2,
            "last_attempt_id": "attempt-test",
            "event_count": 7,
            "event_log": "relink_events.jsonl",
        }
        assert persisted["graph"]["directed"] is True
        assert "must-not-appear" not in json.dumps(persisted)

    asyncio.run(scenario())
