#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CONDITIONS = (
    "lexical-reverse-ppr",
    "lexical-forward-ppr",
    "lexical-no-graph",
    "lexical-one-hop",
)
AGENT_IMPORT_PATH = "evaluation.skillsbench.agents.traex_host:TraexHostAgent"


def build_commands(
    *,
    repo_root: Path,
    condition: str,
    tasks: tuple[str, ...],
    attempts: int,
    concurrency: int,
    skillset_name: str,
    output_root: Path,
    jobs_root: Path,
    executable: str,
    timeout_multiplier: float,
    skip_generate: bool,
) -> list[list[str]]:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}; choose one of: {', '.join(CONDITIONS)}")
    if attempts < 1 or concurrency < 1 or timeout_multiplier <= 0:
        raise ValueError("attempts, concurrency, and timeout multiplier must be positive")

    commands: list[list[str]] = []
    if not skip_generate:
        generate = [
            "uv",
            "run",
            "python",
            "evaluation/skillsbench/graphskills_benchmark.py",
            "--skillset-name",
            skillset_name,
            "--output-root",
            str(output_root),
            "--retrieval-condition",
            condition,
            "--skip-allskills",
            "--skip-vectorskills",
        ]
        for task in tasks:
            generate.extend(["--task", task])
        commands.append(generate)

    dataset_path = output_root / "tasks_graph_skills"
    harbor = [
        "uv",
        "run",
        "--project",
        "evaluation/skillsbench",
        "harbor",
        "run",
        "--job-name",
        f"traex-{condition}",
        "--jobs-dir",
        str(jobs_root),
        "--n-attempts",
        str(attempts),
        "--n-concurrent",
        str(concurrency),
        "--timeout-multiplier",
        str(timeout_multiplier),
        "--agent-import-path",
        AGENT_IMPORT_PATH,
        "--model",
        "traex/GPT-5.2",
        "--agent-kwarg",
        f"executable={executable}",
        "--path",
        str(dataset_path),
    ]
    for task in tasks:
        harbor.extend(["--task-name", task])
    commands.append(harbor)
    return commands


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and run a matched Traex retrieval ablation.")
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--timeout-multiplier",
        type=float,
        default=5.0,
        help="Scale task agent/verifier timeouts for local Traex latency.",
    )
    parser.add_argument("--skillset-name", default="skills_1000")
    parser.add_argument("--executable", default="traex")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--jobs-root", type=Path, default=None)
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Execute commands; otherwise print a dry run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = (args.output_root or REPO_ROOT / "evaluation" / "skillsbench" / "generated_traex" / args.condition).resolve()
    jobs_root = (args.jobs_root or REPO_ROOT / "evaluation" / "skillsbench" / "jobs" / "traex").resolve()
    commands = build_commands(
        repo_root=REPO_ROOT,
        condition=args.condition,
        tasks=tuple(args.task),
        attempts=args.attempts,
        concurrency=args.concurrency,
        skillset_name=args.skillset_name,
        output_root=output_root,
        jobs_root=jobs_root,
        executable=args.executable,
        timeout_multiplier=args.timeout_multiplier,
        skip_generate=args.skip_generate,
    )

    for command in commands:
        print(shlex.join(command))
    if not args.execute:
        return

    if shutil.which(args.executable) is None:
        raise SystemExit(f"Traex executable not found: {args.executable}")
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(item for item in (str(REPO_ROOT), existing_pythonpath) if item)
    for command in commands:
        completed = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
