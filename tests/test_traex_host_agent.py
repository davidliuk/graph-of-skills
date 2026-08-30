from __future__ import annotations

import importlib.util
import asyncio
import inspect
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parent.parent
AGENT_PATH = REPO_ROOT / "evaluation" / "skillsbench" / "agents" / "traex_host.py"


def _load_agent_module():
    spec = importlib.util.spec_from_file_location("traex_host_agent", AGENT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_traex_jsonl_extracts_usage_and_final_message():
    agent = _load_agent_module()
    text = "\n".join(
        [
            '{"type":"thread.started","thread_id":"thread-123"}',
            '{"type":"item.completed","item":{"type":"agent_message","text":"done"}}',
            "not-json",
            '{"type":"turn.completed","usage":{"input_tokens":100,"cached_input_tokens":25,"output_tokens":12}}',
        ]
    )

    summary = agent.parse_traex_jsonl(text)

    assert summary.thread_id == "thread-123"
    assert summary.final_message == "done"
    assert summary.input_tokens == 100
    assert summary.cached_input_tokens == 25
    assert summary.output_tokens == 12
    assert summary.malformed_lines == ("not-json",)


def test_parse_traex_jsonl_sums_multiple_completed_turns():
    agent = _load_agent_module()
    text = "\n".join(
        [
            '{"type":"turn.completed","usage":{"input_tokens":10,"cached_input_tokens":3,"output_tokens":2}}',
            '{"type":"turn.completed","usage":{"input_tokens":20,"cached_input_tokens":5,"output_tokens":4}}',
        ]
    )

    summary = agent.parse_traex_jsonl(text)

    assert summary.input_tokens == 30
    assert summary.cached_input_tokens == 8
    assert summary.output_tokens == 6


def test_build_traex_command_uses_argv_and_stdin_prompt(tmp_path):
    agent = _load_agent_module()

    argv = agent.build_traex_command(
        executable="/usr/local/bin/traex",
        model="GPT-5.2",
        workspace=tmp_path,
    )

    assert argv[0:2] == ["/usr/local/bin/traex", "exec"]
    assert argv[-1] == "-"
    assert argv[argv.index("-m") + 1] == "GPT-5.2"
    assert argv[argv.index("-C") + 1] == str(tmp_path)
    assert "--json" in argv
    assert "--ephemeral" in argv


def test_runner_contract_requires_container_command_wrapper():
    agent = _load_agent_module()

    contract = agent.render_runner_contract("./task-exec")

    assert "TRAE CLI (GPT-5.2)" in contract
    assert "./task-exec" in contract
    assert "Do not run Docker directly" in contract
    assert "edit files directly" in contract


def test_task_exec_script_targets_only_current_harbor_project():
    agent = _load_agent_module()

    script = agent.render_task_exec_script(
        project_name="demo-task__trial-1",
        container_workspace="/logs/agent/traex-workspace",
    )

    assert "com.docker.compose.project=demo-task__trial-1" in script
    assert "com.docker.compose.service=main" in script
    assert 'docker exec -i -w "/logs/agent/traex-workspace"' in script
    assert "docker compose" not in script


def test_traex_host_agent_mirrors_runs_syncs_and_populates_context(tmp_path):
    agent_module = _load_agent_module()

    class FakeEnvironment:
        session_id = "Demo.Task__Trial-1"

        def __init__(self):
            self.commands: list[str] = []

        async def exec(self, command: str, **_kwargs):
            self.commands.append(command)
            if command == "pwd -P":
                return SimpleNamespace(stdout="/root\n", stderr=None, return_code=0)
            if "cp -a /root/." in command:
                workspace = tmp_path / "traex-workspace"
                workspace.mkdir(parents=True, exist_ok=True)
                (workspace / "input.txt").write_text("task", encoding="utf-8")
            return SimpleNamespace(stdout="", stderr=None, return_code=0)

    class FakeTraexAgent(agent_module.TraexHostAgent):
        async def _run_traex(self, argv, prompt, workspace, output_path):
            assert argv[argv.index("-m") + 1] == "GPT-5.2"
            assert "solve the task" in prompt
            (workspace / "answer.txt").write_text("done", encoding="utf-8")
            output_path.write_text(
                "\n".join(
                    [
                        '{"type":"thread.started","thread_id":"thread-1"}',
                        '{"type":"item.completed","item":{"type":"agent_message","text":"finished"}}',
                        '{"type":"turn.completed","usage":{"input_tokens":50,"cached_input_tokens":10,"output_tokens":7}}',
                    ]
                ),
                encoding="utf-8",
            )
            return 0

    environment = FakeEnvironment()
    context = SimpleNamespace(
        n_input_tokens=None,
        n_cache_tokens=None,
        n_output_tokens=None,
        metadata=None,
    )
    traex_agent = FakeTraexAgent(
        logs_dir=tmp_path,
        model_name="traex/GPT-5.2",
        executable="/usr/local/bin/traex",
    )

    asyncio.run(traex_agent.run("solve the task", environment, context))

    assert context.n_input_tokens == 50
    assert context.n_cache_tokens == 10
    assert context.n_output_tokens == 7
    assert context.metadata["thread_id"] == "thread-1"
    assert (tmp_path / "traex.jsonl").exists()
    assert any(
        "cp -a /logs/agent/traex-workspace/. /root/" in command
        for command in environment.commands
    )


def test_traex_setup_accepts_harbor_environment_keyword():
    agent_module = _load_agent_module()

    parameters = inspect.signature(agent_module.TraexHostAgent.setup).parameters

    assert "environment" in parameters


def test_traex_agent_normalizes_relative_logs_directory(tmp_path, monkeypatch):
    agent_module = _load_agent_module()
    monkeypatch.chdir(tmp_path)

    traex_agent = agent_module.TraexHostAgent(
        logs_dir=Path("jobs/trial/agent"),
        model_name="traex/GPT-5.2",
    )

    assert traex_agent.logs_dir == (tmp_path / "jobs/trial/agent").resolve()


def test_run_traex_streams_output_to_diagnostic_file(tmp_path):
    agent_module = _load_agent_module()
    traex_agent = agent_module.TraexHostAgent(
        logs_dir=tmp_path,
        model_name="traex/GPT-5.2",
        executable="sh",
    )
    output_path = tmp_path / "traex.jsonl"

    return_code = asyncio.run(
        traex_agent._run_traex(
            ["sh", "-c", "printf '%s\\n' first second; cat >/dev/null"],
            "prompt",
            tmp_path,
            output_path,
        )
    )

    assert return_code == 0
    assert output_path.read_text(encoding="utf-8") == "first\nsecond\n"
