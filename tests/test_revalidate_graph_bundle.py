from __future__ import annotations

from evaluation.analysis.revalidate_graph_bundle import revalidate_bundle


def _skill(
    name: str,
    *,
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    domains: list[str] | None = None,
    description: str = "",
    tooling: list[str] | None = None,
) -> dict:
    return {
        "name": name,
        "description": description,
        "inputs": inputs or [],
        "outputs": outputs or [],
        "domain_tags": domains or [],
        "tooling": tooling or [],
    }


def test_revalidate_bundle_recomputes_deterministic_and_filters_cached_llm_edges():
    bundle = {
        "metadata": {
            "graph_fingerprint": "sha256:parent",
            "dependency_match_threshold": 0.6,
        },
        "skills": [
            _skill(
                "-2chat-automation",
                inputs=["tool execution requests"],
                outputs=["tool execution results"],
                domains=["messaging", "automation"],
                description="Automate 2Chat via Rube MCP.",
                tooling=["Rube MCP", "Composio 2Chat toolkit"],
            ),
            _skill(
                "ably-automation",
                inputs=["tool execution requests"],
                outputs=["tool execution results"],
                domains=["messaging", "automation"],
                description="Automate Ably via Rube MCP.",
                tooling=["Rube MCP", "Composio Ably toolkit"],
            ),
            _skill(
                "audio-extractor",
                outputs=["mono WAV audio"],
                domains=["audio processing"],
            ),
            _skill(
                "transcriber",
                inputs=["WAV audio file"],
                domains=["audio processing"],
            ),
        ],
        "edges": [
            {
                "source": "ably-automation",
                "target": "-2chat-automation",
                "type": "dependency",
                "provenance": "deterministic_io",
                "confidence": 1.0,
                "weight": 1.0,
                "evidence": "execution",
                "description": "generic execution result",
                "validator_model": "",
            },
            {
                "source": "-2chat-automation",
                "target": "ably-automation",
                "type": "semantic",
                "provenance": "llm_validated",
                "confidence": 0.92,
                "weight": 0.368,
                "evidence": "shared Rube MCP messaging workflow",
                "description": "Both are generic Rube wrappers.",
                "validator_model": "minimax",
            },
            {
                "source": "audio-extractor",
                "target": "transcriber",
                "type": "dependency",
                "provenance": "llm_validated",
                "confidence": 0.9,
                "weight": 0.9,
                "evidence": "WAV audio output is consumed as WAV audio input",
                "description": "Audio handoff.",
                "validator_model": "minimax",
            },
        ],
    }

    repaired, report = revalidate_bundle(bundle)

    assert [
        (edge["source"], edge["target"], edge["type"], edge["provenance"])
        for edge in repaired["edges"]
    ] == [("audio-extractor", "transcriber", "dependency", "deterministic_io")]
    assert report["old_edge_count"] == 3
    assert report["new_edge_count"] == 1
    assert report["rejected_llm_edge_count"] == 1
    assert report["rejected_llm_edges"][0]["type"] == "semantic"
    assert repaired["metadata"]["parent_graph_fingerprint"] == "sha256:parent"
    assert repaired["metadata"]["graph_source"] == "cached_llm_quality_gate_replay"
