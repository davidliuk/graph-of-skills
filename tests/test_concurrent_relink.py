import asyncio
import json
from types import MethodType

import numpy as np
import pytest

from gos.core.engine import SkillGraphRAG
from gos.core.relink import (
    FocusLinkJob,
    RelinkProgress,
    RelinkProgressMismatch,
    RelinkResult,
    load_relink_progress,
    write_relink_progress,
)
from gos.core.schema import SkillEdge, SkillNode


class ConstantEmbeddingService:
    model = "constant-embedding"
    embedding_dim = 4

    async def encode(self, texts, model=None):
        return np.ones((len(texts), self.embedding_dim), dtype=np.float32)


class NoopLLMService:
    model = "noop-llm"

    async def send_message(self, prompt, response_model=None, **kwargs):
        return response_model(), []


async def _node_only_engine(tmp_path, node_count=8):
    engine = SkillGraphRAG(
        config=SkillGraphRAG.Config(
            llm_service=NoopLLMService(),
            embedding_service=ConstantEmbeddingService(),
            working_dir=str(tmp_path),
            use_full_markdown=False,
            enable_semantic_linking=True,
        )
    )
    await engine.state_manager.insert_start()
    try:
        for index in range(node_count):
            await engine.state_manager.graph_storage.upsert_node(
                SkillNode.from_lists(
                    name=f"skill-{index}",
                    description=f"Skill {index}",
                    raw_content=f"content-{index}",
                ),
                None,
            )
    finally:
        await engine.state_manager.insert_done()
    return engine


def _jobs(nodes, focus_names):
    by_name = {node.name: node for node in nodes}
    ordered_names = sorted(by_name)
    jobs = []
    for focus_name in ordered_names:
        if focus_name not in focus_names:
            continue
        index = ordered_names.index(focus_name)
        candidate = by_name[ordered_names[(index + 1) % len(ordered_names)]]
        jobs.append(
            FocusLinkJob(
                focus_name=focus_name,
                focus_index=index,
                deterministic_edges=(),
                candidates=(candidate,),
                candidate_pairs=1,
            )
        )
    return jobs


def _install_prepared_jobs(engine):
    async def prepare(self, nodes, focus_names):
        return _jobs(nodes, focus_names)

    engine._prepare_focus_link_jobs = MethodType(prepare, engine)


class TrackingValidator:
    def __init__(self, *, delay=0.02):
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self.completed = 0
        self.completed_event = asyncio.Event()
        self.focus_names = []

    async def __call__(self, node, candidates, *, raise_on_failure=False):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.focus_names.append(node.name)
        try:
            await asyncio.sleep(self.delay)
            candidate = candidates[0]
            return [
                SkillEdge(
                    source=node.name,
                    target=candidate.name,
                    description=f"{node.name} runs before {candidate.name}",
                    type="workflow",
                    weight=0.6,
                    confidence=0.9,
                    provenance="llm_validated",
                    evidence="test workflow",
                )
            ]
        finally:
            self.active -= 1
            self.completed += 1
            self.completed_event.set()


def _install_validator(engine, validator):
    engine._validate_candidate_relations = validator


async def _counts(engine):
    await engine.state_manager.query_start()
    try:
        return (
            await engine.state_manager.graph_storage.node_count(),
            await engine.state_manager.graph_storage.edge_count(),
        )
    finally:
        await engine.state_manager.query_done()


def test_relink_never_exceeds_configured_validator_concurrency(tmp_path):
    async def scenario():
        engine = await _node_only_engine(tmp_path, node_count=12)
        _install_prepared_jobs(engine)
        validator = TrackingValidator()
        _install_validator(engine, validator)

        result = await engine.async_relink_all(concurrency=3, checkpoint_every=4)

        assert validator.max_active == 3
        assert result.completed_focus_count == 12
        assert result.checkpoint_count == 3
        assert (await _counts(engine))[1] == 12

        events = [
            json.loads(line)
            for line in (tmp_path / "relink_events.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert [event["event"] for event in events] == [
            "attempt_started",
            "checkpoint",
            "checkpoint",
            "checkpoint",
            "attempt_completed",
        ]
        assert len({event["run_id"] for event in events}) == 1
        assert len({event["attempt_id"] for event in events}) == 1
        assert events[1]["batch"]["focus_count"] == 4
        assert events[1]["batch"]["candidate_count"] == 4
        assert events[1]["batch"]["validated_edge_count"] == 4
        assert len(events[1]["batch"]["focus_results"]) == 4
        assert events[-1]["totals"]["completed_focus_count"] == 12
        progress = load_relink_progress(tmp_path / "relink_progress.json")
        assert progress is not None
        assert progress.event_count == len(events)

    asyncio.run(scenario())


def test_cancellation_checkpoints_edges_and_resume_skips_completed(tmp_path):
    async def scenario():
        first = await _node_only_engine(tmp_path, node_count=8)
        _install_prepared_jobs(first)
        first_validator = TrackingValidator(delay=0.05)
        _install_validator(first, first_validator)

        task = asyncio.create_task(
            first.async_relink_all(concurrency=2, checkpoint_every=2)
        )
        while first_validator.completed < 4:
            first_validator.completed_event.clear()
            await first_validator.completed_event.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        progress = load_relink_progress(tmp_path / "relink_progress.json")
        assert progress is not None
        assert progress.status == "cancelled"
        completed_before_resume = set(progress.completed_focus_names)
        assert len(completed_before_resume) >= 4
        _, persisted_edges = await _counts(first)
        assert persisted_edges == len(completed_before_resume)
        cancelled_events = [
            json.loads(line)
            for line in (tmp_path / "relink_events.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert cancelled_events[-1]["event"] == "attempt_cancelled"
        assert cancelled_events[-1]["totals"]["completed_focus_count"] == len(
            completed_before_resume
        )
        assert progress.event_count == len(cancelled_events)

        second = SkillGraphRAG(
            config=first.config,
        )
        _install_prepared_jobs(second)
        second_validator = TrackingValidator(delay=0.001)
        _install_validator(second, second_validator)

        result = await second.async_relink_all(
            concurrency=2,
            checkpoint_every=2,
            resume=True,
        )
        assert completed_before_resume.isdisjoint(second_validator.focus_names)
        assert result.completed_focus_count == 8
        assert (await _counts(second))[1] == 8

    asyncio.run(scenario())


def test_resume_rejects_incompatible_progress(tmp_path):
    async def scenario():
        engine = await _node_only_engine(tmp_path, node_count=2)
        _install_prepared_jobs(engine)
        write_relink_progress(
            tmp_path / "relink_progress.json",
            RelinkProgress.new(
                fingerprint="sha256:not-current",
                total_focus_nodes=2,
                concurrency=1,
                checkpoint_every=1,
            ),
        )

        with pytest.raises(RelinkProgressMismatch, match="restart"):
            await engine.async_relink_all(
                concurrency=1,
                checkpoint_every=1,
                resume=True,
            )

    asyncio.run(scenario())


def test_one_focus_failure_does_not_abort_other_results(tmp_path):
    async def scenario():
        engine = await _node_only_engine(tmp_path, node_count=4)
        _install_prepared_jobs(engine)
        successful = TrackingValidator(delay=0.001)

        async def validate(node, candidates, *, raise_on_failure=False):
            if node.name == "skill-1":
                raise TimeoutError("provider timeout")
            return await successful(
                node,
                candidates,
                raise_on_failure=raise_on_failure,
            )

        engine._validate_candidate_relations = validate
        result = await engine.async_relink_all(concurrency=2, checkpoint_every=2)

        assert result.completed_focus_count == 4
        assert "skill-1" in result.failed_focus
        assert (await _counts(engine))[1] == 3

    asyncio.run(scenario())


def test_failure_ledger_and_events_omit_secret_and_response_body(tmp_path):
    async def scenario():
        engine = await _node_only_engine(tmp_path, node_count=2)
        _install_prepared_jobs(engine)

        async def invalid_response(node, candidates, *, raise_on_failure=False):
            raise ValueError(
                "provider rejected sk-or-v1-supersecretvalue\n"
                "input_value={'relations': [{'evidence': 'private response'}]}"
            )

        engine._validate_candidate_relations = invalid_response
        await engine.async_relink_all(concurrency=1, checkpoint_every=1)

        progress_text = (tmp_path / "relink_progress.json").read_text(
            encoding="utf-8"
        )
        event_text = (tmp_path / "relink_events.jsonl").read_text(encoding="utf-8")
        for rendered in (progress_text, event_text):
            assert "supersecretvalue" not in rendered
            assert "private response" not in rendered
            assert "input_value" not in rendered

    asyncio.run(scenario())


def test_resume_retries_only_failed_focus(tmp_path):
    async def scenario():
        first = await _node_only_engine(tmp_path, node_count=4)
        _install_prepared_jobs(first)
        successful = TrackingValidator(delay=0.001)

        async def fail_one(node, candidates, *, raise_on_failure=False):
            if node.name == "skill-1":
                raise TimeoutError("provider timeout")
            return await successful(
                node,
                candidates,
                raise_on_failure=raise_on_failure,
            )

        first._validate_candidate_relations = fail_one
        failed = await first.async_relink_all(concurrency=2, checkpoint_every=2)
        assert failed.failed_focus == {"skill-1": "TimeoutError: provider timeout"}

        second = SkillGraphRAG(config=first.config)
        _install_prepared_jobs(second)
        retry = TrackingValidator(delay=0.001)
        _install_validator(second, retry)

        resumed = await second.async_relink_all(
            concurrency=2,
            checkpoint_every=1,
            resume=True,
        )

        assert retry.focus_names == ["skill-1"]
        assert resumed.resumed_focus_count == 3
        assert resumed.failed_focus == {}
        assert (await _counts(second))[1] == 4

        events = [
            json.loads(line)
            for line in (tmp_path / "relink_events.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        starts = [event for event in events if event["event"] == "attempt_started"]
        assert len(starts) == 2
        assert starts[0]["run_id"] == starts[1]["run_id"]
        assert starts[0]["attempt_id"] != starts[1]["attempt_id"]
        assert starts[1]["totals"]["resumed_focus_count"] == 3
        assert starts[1]["totals"]["pending_focus_count"] == 1

    asyncio.run(scenario())


def test_restart_clears_edges_and_progress_but_retains_nodes(tmp_path):
    async def scenario():
        first = await _node_only_engine(tmp_path, node_count=3)
        _install_prepared_jobs(first)
        _install_validator(first, TrackingValidator(delay=0.001))
        await first.async_relink_all(concurrency=2, checkpoint_every=1)
        assert await _counts(first) == (3, 3)

        second = SkillGraphRAG(config=first.config)

        async def no_jobs(self, nodes, focus_names):
            return []

        second._prepare_focus_link_jobs = MethodType(no_jobs, second)
        await second.async_relink_all(
            concurrency=2,
            checkpoint_every=1,
            restart=True,
        )

        assert await _counts(second) == (3, 0)
        progress = load_relink_progress(tmp_path / "relink_progress.json")
        assert progress is not None
        assert progress.status == "complete"

    asyncio.run(scenario())


def test_resume_preserves_prior_process_usage(tmp_path):
    async def scenario():
        first = await _node_only_engine(tmp_path, node_count=2)
        _install_prepared_jobs(first)
        _install_validator(first, TrackingValidator(delay=0.001))
        await first.async_relink_all(concurrency=1, checkpoint_every=1)

        path = tmp_path / "relink_progress.json"
        progress = load_relink_progress(path)
        assert progress is not None
        progress.usage = {"llm": {"relation_validation": {"calls": 3, "cost_usd": 0.2}}}
        write_relink_progress(path, progress)

        second = SkillGraphRAG(config=first.config)
        _install_prepared_jobs(second)
        await second.async_relink_all(
            concurrency=1,
            checkpoint_every=1,
            resume=True,
        )

        resumed = load_relink_progress(path)
        assert resumed is not None
        assert resumed.usage["llm"]["relation_validation"]["calls"] == 3
        assert resumed.usage["llm"]["relation_validation"]["cost_usd"] == 0.2

    asyncio.run(scenario())


def test_clean_multi_skill_insert_uses_concurrent_full_relink(tmp_path):
    async def scenario():
        engine = SkillGraphRAG(
            config=SkillGraphRAG.Config(
                llm_service=NoopLLMService(),
                embedding_service=ConstantEmbeddingService(),
                working_dir=str(tmp_path),
                use_full_markdown=False,
                enable_semantic_linking=False,
                relink_concurrency=3,
                relink_checkpoint_every=2,
            )
        )
        calls = []

        async def relink(self, **kwargs):
            calls.append(kwargs)
            return RelinkResult(
                total_focus_count=2,
                resumed_focus_count=0,
                processed_focus_count=2,
                completed_focus_count=2,
                failed_focus={},
                checkpoint_count=1,
                edge_count=0,
                elapsed_seconds=0.0,
            )

        engine.async_relink_all = MethodType(relink, engine)
        skills = [
            "---\nname: one\ndescription: First skill.\n---\n",
            "---\nname: two\ndescription: Second skill.\n---\n",
        ]
        await engine.async_insert_skills(skills)

        assert len(calls) == 1
        assert calls[0]["concurrency"] == 3
        assert calls[0]["checkpoint_every"] == 2
        assert calls[0]["resume"] is False

    asyncio.run(scenario())
