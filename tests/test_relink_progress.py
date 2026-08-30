from dataclasses import asdict
import json

import pytest

from gos.core.relink import (
    RelinkProgress,
    RelinkProgressError,
    append_relink_event,
    build_relink_fingerprint,
    diff_relink_usage,
    load_relink_progress,
    merge_relink_usage,
    summarize_relink_usage,
    write_relink_progress,
)
from gos.core.schema import SkillNode


def _node(name: str, content: str) -> SkillNode:
    return SkillNode.from_lists(
        name=name,
        description=f"Description for {name}",
        raw_content=content,
    )


def _fingerprint(nodes, *, link_top_k=8, construction_code_sha256="code-v1"):
    return build_relink_fingerprint(
        nodes=nodes,
        llm_model="openrouter/minimax/minimax-m2.7",
        embedding_model="qwen/qwen3-embedding-8b",
        prompt_sha256={"relation": "abc", "completion": "def"},
        link_top_k=link_top_k,
        relation_min_confidence=0.75,
        dependency_match_threshold=0.6,
        type_weights={"dependency": 1.0, "workflow": 0.7},
        construction_code_sha256=construction_code_sha256,
    )


def test_relink_progress_round_trip_is_atomic_and_secret_free(tmp_path):
    path = tmp_path / "relink_progress.json"
    progress = RelinkProgress.new(
        fingerprint="sha256:abc",
        total_focus_nodes=2,
        concurrency=8,
        checkpoint_every=10,
    )
    progress.completed_focus_names = ["a"]
    progress.failed_focus = {"b": "Timeout"}
    progress.construction = {"focus_nodes": 1}
    progress.usage = {"llm": {"relation_validation": {"calls": 1}}}

    write_relink_progress(path, progress)
    loaded = load_relink_progress(path)

    assert loaded == progress
    assert not path.with_suffix(".json.tmp").exists()
    rendered = path.read_text(encoding="utf-8")
    assert "api_key" not in rendered
    assert json.loads(rendered) == asdict(progress)


def test_fingerprint_changes_with_node_content_or_link_config():
    first = _fingerprint([_node("a", "one")])

    assert _fingerprint([_node("a", "two")]) != first
    assert _fingerprint([_node("a", "one")], link_top_k=4) != first
    assert (
        _fingerprint([_node("a", "one")], construction_code_sha256="code-v2")
        != first
    )
    assert first.startswith("sha256:")


def test_progress_loader_rejects_unknown_schema(tmp_path):
    path = tmp_path / "relink_progress.json"
    path.write_text('{"schema_version": 99}', encoding="utf-8")

    with pytest.raises(RelinkProgressError, match="schema"):
        load_relink_progress(path)


def test_merge_relink_usage_sums_prior_and_current_metrics():
    prior = {"llm": {"relation_validation": {"calls": 3, "cost_usd": 0.2}}}
    current = {"llm": {"relation_validation": {"calls": 2, "cost_usd": 0.1}}}

    merged = merge_relink_usage(prior, current)

    assert merged["llm"]["relation_validation"]["calls"] == 5
    assert merged["llm"]["relation_validation"]["cost_usd"] == pytest.approx(0.3)
    assert prior["llm"]["relation_validation"]["calls"] == 3


def test_usage_delta_and_summary_cover_tokens_cost_time_and_failures():
    prior = {
        "llm": {
            "relation_validation": {
                "calls": 3,
                "input_tokens": 100,
                "output_tokens": 20,
                "cost_usd": 0.2,
                "elapsed_seconds": 5.0,
            }
        }
    }
    current = {
        "llm": {
            "relation_validation": {
                "calls": 5,
                "failures": 1,
                "input_tokens": 160,
                "output_tokens": 35,
                "reasoning_tokens": 12,
                "cached_input_tokens": 8,
                "cache_hits": 1,
                "cost_usd": 0.32,
                "elapsed_seconds": 8.5,
            }
        },
        "embedding": {
            "embedding": {
                "calls": 1,
                "input_tokens": 40,
                "cost_usd": 0.01,
                "elapsed_seconds": 0.5,
            }
        },
    }

    delta = diff_relink_usage(prior, current)
    totals = summarize_relink_usage(current)

    relation_delta = delta["llm"]["relation_validation"]
    assert relation_delta["calls"] == 2
    assert relation_delta["input_tokens"] == 60
    assert relation_delta["output_tokens"] == 15
    assert relation_delta["reasoning_tokens"] == 12
    assert relation_delta["cost_usd"] == pytest.approx(0.12)
    assert relation_delta["elapsed_seconds"] == pytest.approx(3.5)
    assert totals == {
        "calls": 6,
        "failures": 1,
        "cache_hits": 1,
        "input_tokens": 200,
        "cached_input_tokens": 8,
        "output_tokens": 35,
        "reasoning_tokens": 12,
        "cost_usd": pytest.approx(0.33),
        "elapsed_seconds": pytest.approx(9.0),
    }


def test_relink_event_log_is_append_only_jsonl_and_secret_free(tmp_path):
    path = tmp_path / "relink_events.jsonl"
    append_relink_event(
        path,
        {
            "event": "attempt_started",
            "run_id": "run-1",
            "attempt_id": "attempt-1",
            "api_key": "sk-or-secret",
            "prompt": "private prompt",
            "metrics": {"input_tokens": 12},
        },
    )
    append_relink_event(
        path,
        {
            "event": "checkpoint",
            "run_id": "run-1",
            "attempt_id": "attempt-1",
            "checkpoint": 1,
        },
    )

    rendered = path.read_text(encoding="utf-8")
    events = [json.loads(line) for line in rendered.splitlines()]

    assert [event["event"] for event in events] == [
        "attempt_started",
        "checkpoint",
    ]
    assert all(event["schema_version"] == 1 for event in events)
    assert all(event["timestamp"] for event in events)
    assert events[0]["metrics"]["input_tokens"] == 12
    assert "api_key" not in rendered
    assert "private prompt" not in rendered
    assert "sk-or-secret" not in rendered
