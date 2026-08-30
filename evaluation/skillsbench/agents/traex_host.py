from __future__ import annotations

import asyncio
import json
import shlex
import shutil
from pathlib import Path
from typing import NamedTuple

try:
    from harbor.agents.base import BaseAgent
    from harbor.environments.base import BaseEnvironment
    from harbor.models.agent.context import AgentContext
except ModuleNotFoundError:  # Keep pure helpers testable from the repository root.
    BaseEnvironment = object  # type: ignore[assignment,misc]
    AgentContext = object  # type: ignore[assignment,misc]

    class BaseAgent:  # type: ignore[no-redef]
        def __init__(self, logs_dir: Path, model_name: str | None = None, **_kwargs):
            self.logs_dir = Path(logs_dir)
            self.model_name = model_name


class TraexRunSummary(NamedTuple):
    thread_id: str | None
    final_message: str | None
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    malformed_lines: tuple[str, ...]


def parse_traex_jsonl(text: str) -> TraexRunSummary:
    thread_id: str | None = None
    final_message: str | None = None
    input_tokens = 0
    cached_input_tokens = 0
    output_tokens = 0
    malformed_lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            malformed_lines.append(line)
            continue
        if not isinstance(event, dict):
            continue

        event_type = event.get("type")
        if event_type == "thread.started" and isinstance(event.get("thread_id"), str):
            thread_id = event["thread_id"]
        elif event_type == "item.completed":
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "agent_message":
                text_value = item.get("text")
                if isinstance(text_value, str):
                    final_message = text_value
        elif event_type == "turn.completed":
            usage = event.get("usage")
            if not isinstance(usage, dict):
                continue
            input_tokens += int(usage.get("input_tokens") or 0)
            cached_input_tokens += int(usage.get("cached_input_tokens") or 0)
            output_tokens += int(usage.get("output_tokens") or 0)

    return TraexRunSummary(
        thread_id=thread_id,
        final_message=final_message,
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
        malformed_lines=tuple(malformed_lines),
    )


def build_traex_command(
    *,
    executable: str,
    model: str,
    workspace: Path,
) -> list[str]:
    return [
        executable,
        "exec",
        "--ephemeral",
        "--skip-git-repo-check",
        "--ignore-user-config",
        "--ignore-rules",
        "--permission-mode",
        "bypass_permissions",
        "--json",
        "--color",
        "never",
        "-m",
        model,
        "-C",
        str(workspace),
        "-",
    ]


def render_runner_contract(task_exec_path: str) -> str:
    return f"""# TRAE CLI (GPT-5.2) Harbor runner contract

You are editing a host mirror of the active SkillsBench task workspace.

- Read and edit files directly in the current workspace.
- Run every environment-dependent command through `{task_exec_path} -- <command>`.
- Do not run Docker directly; the wrapper is already bound to the current task container.
- Do not write task artifacts to absolute host paths such as `/root` or `/app`.
- When the task names an absolute container path, use `{task_exec_path}` to inspect it and
  create the corresponding artifact in this mirrored workspace so it can be synchronized.
- Before finishing, use `{task_exec_path}` for the task's verifier-aligned checks.
"""


def render_task_exec_script(
    *,
    project_name: str,
    container_workspace: str,
) -> str:
    quoted_workspace = container_workspace.replace('"', '\\"')
    return f"""#!/usr/bin/env bash
set -euo pipefail
if [[ "${{1:-}}" == "--" ]]; then
  shift
fi
if [[ $# -eq 0 ]]; then
  echo "usage: task-exec -- <command>" >&2
  exit 2
fi
container_id="$(docker ps \\
  --filter 'label=com.docker.compose.project={project_name}' \\
  --filter 'label=com.docker.compose.service=main' \\
  --format '{{{{.ID}}}}' | head -n 1)"
if [[ -z "$container_id" ]]; then
  echo "active Harbor task container not found for project {project_name}" >&2
  exit 3
fi
exec docker exec -i -w "{quoted_workspace}" "$container_id" bash -lc "$*"
"""


class TraexHostAgent(BaseAgent):
    """Harbor agent that runs the authenticated host Traex CLI against a task mirror."""

    SUPPORTS_ATIF = False
    _CONTAINER_AGENT_DIR = "/logs/agent"

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = "traex/GPT-5.2",
        executable: str = "traex",
        workspace_dir: str | None = None,
        *args,
        **kwargs,
    ):
        resolved_logs_dir = Path(logs_dir).expanduser().resolve()
        super().__init__(*args, logs_dir=resolved_logs_dir, model_name=model_name, **kwargs)
        self.logs_dir = resolved_logs_dir
        self.model_name = model_name
        self.executable = executable
        self.workspace_dir = workspace_dir
        self._version: str | None = None

    @staticmethod
    def name() -> str:
        return "traex-host"

    def version(self) -> str | None:
        return self._version

    async def setup(self, environment: BaseEnvironment) -> None:
        del environment
        resolved = shutil.which(self.executable)
        if resolved is None:
            raise FileNotFoundError(f"Traex executable not found: {self.executable}")
        self.executable = resolved
        process = await asyncio.create_subprocess_exec(
            self.executable,
            "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(f"Failed to inspect Traex version using {self.executable}")
        self._version = stdout.decode(errors="replace").strip() or "unknown"

    async def _run_traex(
        self,
        argv: list[str],
        prompt: str,
        workspace: Path,
        output_path: Path,
    ) -> int:
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=workspace,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        assert process.stdin is not None
        assert process.stdout is not None
        try:
            process.stdin.write(prompt.encode())
            await process.stdin.drain()
            process.stdin.close()
            await process.stdin.wait_closed()
            with output_path.open("wb") as output_file:
                while chunk := await process.stdout.read(64 * 1024):
                    output_file.write(chunk)
                    output_file.flush()
            return await process.wait()
        except asyncio.CancelledError:
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except TimeoutError:
                    process.kill()
                    await process.wait()
            raise

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        if self.workspace_dir:
            task_root = self.workspace_dir
        else:
            pwd_result = await environment.exec(command="pwd -P")
            if pwd_result.return_code != 0 or not pwd_result.stdout:
                raise RuntimeError("Unable to discover the Harbor task working directory")
            task_root = pwd_result.stdout.strip().splitlines()[-1]
        if not task_root.startswith("/"):
            raise RuntimeError(f"Harbor task working directory is not absolute: {task_root!r}")

        workspace = self.logs_dir / "traex-workspace"
        container_workspace = f"{self._CONTAINER_AGENT_DIR}/{workspace.name}"
        mirror_command = (
            f"rm -rf {shlex.quote(container_workspace)} && "
            f"mkdir -p {shlex.quote(container_workspace)} && "
            f"cp -a {shlex.quote(task_root)}/. {shlex.quote(container_workspace)}/"
        )
        mirror_result = await environment.exec(command=mirror_command)
        if mirror_result.return_code != 0:
            raise RuntimeError(f"Failed to mirror task workspace: {mirror_result.stderr or mirror_result.stdout}")
        if not workspace.exists():
            raise RuntimeError(f"Harbor did not expose the mirrored workspace at {workspace}")

        project_name = environment.session_id.lower().replace(".", "-")
        task_exec = workspace / "task-exec"
        task_exec.write_text(
            render_task_exec_script(
                project_name=project_name,
                container_workspace=container_workspace,
            ),
            encoding="utf-8",
        )
        task_exec.chmod(0o755)

        model = (self.model_name or "traex/GPT-5.2").split("/", maxsplit=1)[-1]
        argv = build_traex_command(
            executable=self.executable,
            model=model,
            workspace=workspace,
        )
        prompt = f"{render_runner_contract('./task-exec')}\n# Task instruction\n\n{instruction}\n"
        output_path = self.logs_dir / "traex.jsonl"
        return_code = await self._run_traex(argv, prompt, workspace, output_path)
        output = output_path.read_text(encoding="utf-8", errors="replace")
        summary = parse_traex_jsonl(output)

        task_exec.unlink(missing_ok=True)
        sync_command = f"cp -a {shlex.quote(container_workspace)}/. {shlex.quote(task_root)}/"
        sync_result = await environment.exec(command=sync_command)
        if sync_result.return_code != 0:
            raise RuntimeError(f"Failed to synchronize Traex task output: {sync_result.stderr or sync_result.stdout}")

        context.n_input_tokens = summary.input_tokens
        context.n_cache_tokens = summary.cached_input_tokens
        context.n_output_tokens = summary.output_tokens
        metadata = dict(getattr(context, "metadata", None) or {})
        metadata.update(
            {
                "harness": "TRAE CLI",
                "model": model,
                "thread_id": summary.thread_id,
                "final_message": summary.final_message,
                "return_code": return_code,
                "malformed_jsonl_lines": len(summary.malformed_lines),
            }
        )
        context.metadata = metadata

        if return_code != 0:
            raise RuntimeError(f"Traex exited with code {return_code}; see {output_path}")
