from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER_PATH = (
    REPO_ROOT / "evaluation" / "skillsbench" / "scripts" / "run_traex_ablation.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location("traex_ablation_runner", RUNNER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_commands_pins_condition_task_agent_and_model(tmp_path):
    runner = _load_runner()

    commands = runner.build_commands(
        repo_root=REPO_ROOT,
        condition="lexical-forward-ppr",
        tasks=("dialogue-parser",),
        attempts=2,
        concurrency=1,
        skillset_name="skills_1000",
        output_root=tmp_path / "generated",
        jobs_root=tmp_path / "jobs",
        executable="traex",
        timeout_multiplier=5.0,
        skip_generate=False,
    )

    assert len(commands) == 2
    generate, harbor = commands
    assert "--retrieval-condition" in generate
    assert (
        generate[generate.index("--retrieval-condition") + 1] == "lexical-forward-ppr"
    )
    assert generate[generate.index("--task") + 1] == "dialogue-parser"
    assert "evaluation.skillsbench.agents.traex_host:TraexHostAgent" in harbor
    assert "traex/GPT-5.2" in harbor
    assert harbor[harbor.index("--agent-kwarg") + 1] == "executable=traex"
    assert harbor[harbor.index("--timeout-multiplier") + 1] == "5.0"


def test_build_commands_can_reuse_generated_tasks(tmp_path):
    runner = _load_runner()

    commands = runner.build_commands(
        repo_root=REPO_ROOT,
        condition="lexical-no-graph",
        tasks=(),
        attempts=1,
        concurrency=1,
        skillset_name="skills_1000",
        output_root=tmp_path / "generated",
        jobs_root=tmp_path / "jobs",
        executable="traex",
        timeout_multiplier=5.0,
        skip_generate=True,
    )

    assert len(commands) == 1
    assert "harbor" in commands[0]
    assert str(tmp_path / "generated" / "tasks_graph_skills") in commands[0]


def test_static_traex_configs_cover_smoke_and_all_conditions():
    config_dir = (
        REPO_ROOT / "evaluation" / "skillsbench" / "experiments" / "configs" / "traex"
    )

    assert {path.name for path in config_dir.glob("*.yaml")} == {
        "smoke.yaml",
        "lexical-reverse-ppr.yaml",
        "lexical-forward-ppr.yaml",
        "lexical-no-graph.yaml",
        "lexical-one-hop.yaml",
    }
    for path in config_dir.glob("*.yaml"):
        content = path.read_text(encoding="utf-8")
        assert "evaluation.skillsbench.agents.traex_host:TraexHostAgent" in content
        assert "model_name: traex/GPT-5.2" in content
        assert "n_concurrent_trials: 1" in content
        assert "timeout_multiplier: 5.0" in content
