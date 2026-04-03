#!/usr/bin/env python3
"""
Analyze token consumption across SkillsBench job runs.

Usage:
  # Analyze specific job directories
  python scripts/analyze-tokens.py jobs/civ6-gos-v2 jobs/civ6-allskills-v3

  # Analyze all jobs
  python scripts/analyze-tokens.py jobs/

  # Compare two runs side by side
  python scripts/analyze-tokens.py --compare jobs/civ6-gos-v2 jobs/civ6-allskills-v3
"""

import json
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Pricing per 1M tokens (Claude Sonnet 4.6 list price)
PRICE = {
    "input": 3.00,
    "cache_create": 3.75,
    "cache_read": 0.30,
    "output": 15.00,
}


@dataclass
class TokenStats:
    input: int = 0
    cache_create: int = 0
    cache_read: int = 0
    output: int = 0
    turns: int = 0

    def add(self, usage: dict) -> None:
        self.input += usage.get("input_tokens") or 0
        self.cache_create += usage.get("cache_creation_input_tokens") or 0
        self.cache_read += usage.get("cache_read_input_tokens") or 0
        self.output += usage.get("output_tokens") or 0
        if (usage.get("output_tokens") or 0) > 0:
            self.turns += 1

    @property
    def effective_input(self) -> int:
        return self.input + self.cache_create + self.cache_read

    @property
    def cost_usd(self) -> float:
        return (
            self.input * PRICE["input"]
            + self.cache_create * PRICE["cache_create"]
            + self.cache_read * PRICE["cache_read"]
            + self.output * PRICE["output"]
        ) / 1_000_000

    def __add__(self, other: "TokenStats") -> "TokenStats":
        return TokenStats(
            input=self.input + other.input,
            cache_create=self.cache_create + other.cache_create,
            cache_read=self.cache_read + other.cache_read,
            output=self.output + other.output,
            turns=self.turns + other.turns,
        )


@dataclass
class RunResult:
    job_name: str
    task_name: str
    reward: Optional[float]
    duration_sec: Optional[float]
    tokens: TokenStats
    n_agents: int  # main + subagents
    condition: str  # gos / all_skills / no_skills / unknown
    per_agent: list = field(default_factory=list)  # list of (label, TokenStats)


def parse_jsonl(path: Path) -> TokenStats:
    stats = TokenStats()
    try:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("type") == "assistant":
                usage = entry.get("message", {}).get("usage", {})
                if usage:
                    stats.add(usage)
    except Exception:
        pass
    return stats


def collect_all_tokens(session_dir: Path) -> tuple[TokenStats, int, list[tuple[str, TokenStats]]]:
    """Recursively collect tokens from main agent + all subagents.
    Returns (total, n_agents, per_agent_list).
    per_agent_list entries: (label, stats)
    """
    total = TokenStats()
    n_agents = 0
    per_agent: list[tuple[str, TokenStats]] = []

    # Find main session JSONL (directly under projects/-root/, not in subagents/)
    root_dir = session_dir / "projects" / "-root"
    if not root_dir.exists():
        # Fallback: scan everything
        for jsonl in session_dir.rglob("*.jsonl"):
            if jsonl.name.endswith(".meta.json"):
                continue
            stats = parse_jsonl(jsonl)
            if stats.turns > 0:
                total = total + stats
                n_agents += 1
                per_agent.append((jsonl.name[:16], stats))
        return total, n_agents, per_agent

    for jsonl in sorted(root_dir.iterdir()):
        if not jsonl.is_file() or not jsonl.name.endswith(".jsonl"):
            continue
        stats = parse_jsonl(jsonl)
        if stats.turns > 0:
            total = total + stats
            n_agents += 1
            per_agent.append(("main", stats))
        # Look for subagents dir alongside this JSONL
        sub_dir = root_dir / jsonl.stem / "subagents"
        if sub_dir.exists():
            for i, sub_jsonl in enumerate(sorted(sub_dir.glob("*.jsonl"))):
                if sub_jsonl.name.endswith(".meta.json"):
                    continue
                sub_stats = parse_jsonl(sub_jsonl)
                if sub_stats.turns > 0:
                    total = total + sub_stats
                    n_agents += 1
                    per_agent.append((f"subagent-{i+1}", sub_stats))

    return total, n_agents, per_agent


def infer_condition(task_id_path: str) -> str:
    if "tasks_gos" in task_id_path:
        return "gos"
    elif "tasks_all_skills" in task_id_path or "all_skills" in task_id_path:
        return "all_skills"
    elif "tasks-no-skills" in task_id_path or "no_skills" in task_id_path:
        return "no_skills"
    return "unknown"


def load_run(trial_dir: Path, job_name: str) -> Optional[RunResult]:
    result_file = trial_dir / "result.json"
    if not result_file.exists():
        return None

    result = json.loads(result_file.read_text())
    task_name = result.get("task_name", trial_dir.name)

    # Reward
    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    reward = rewards.get("reward")

    # Duration
    agent_exec = result.get("agent_execution") or {}
    duration = None
    if agent_exec.get("started_at") and agent_exec.get("finished_at"):
        from datetime import datetime, timezone
        fmt = "%Y-%m-%dT%H:%M:%S.%fZ"
        try:
            start = datetime.strptime(agent_exec["started_at"], fmt)
            end = datetime.strptime(agent_exec["finished_at"], fmt)
            duration = (end - start).total_seconds()
        except Exception:
            pass

    # Condition
    task_id = result.get("task_id", {})
    condition = infer_condition(task_id.get("path", ""))

    # Tokens (aggregate all agents)
    session_dir = trial_dir / "agent" / "sessions"
    if not session_dir.exists():
        return None
    tokens, n_agents, per_agent = collect_all_tokens(session_dir)

    return RunResult(
        job_name=job_name,
        task_name=task_name,
        reward=reward,
        duration_sec=duration,
        tokens=tokens,
        n_agents=n_agents,
        condition=condition,
        per_agent=per_agent,
    )


def find_runs(job_path: Path, job_name: str) -> list[RunResult]:
    runs = []
    # Direct trial dir: job/trial_dir/result.json
    if (job_path / "result.json").exists():
        r = load_run(job_path, job_name)
        if r:
            runs.append(r)
        return runs
    # Timestamped: job/<timestamp>/<trial>/result.json
    for ts_dir in sorted(job_path.iterdir()):
        if not ts_dir.is_dir():
            continue
        for trial_dir in sorted(ts_dir.iterdir()):
            if not trial_dir.is_dir():
                continue
            r = load_run(trial_dir, job_name)
            if r:
                runs.append(r)
    return runs


def fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def print_run(r: RunResult, verbose: bool = False, breakdown: bool = False) -> None:
    t = r.tokens
    reward_str = f"{r.reward:.3f}" if r.reward is not None else "  N/A"
    dur_str = f"{r.duration_sec:.0f}s" if r.duration_sec else "  N/A"
    print(
        f"  [{r.condition:10s}] {r.task_name:40s}  "
        f"reward={reward_str}  dur={dur_str:6s}  "
        f"turns={t.turns:3d}  agents={r.n_agents}  "
        f"eff_input={fmt_tokens(t.effective_input):7s}  "
        f"output={fmt_tokens(t.output):6s}  "
        f"cost=${t.cost_usd:.4f}"
    )
    if verbose:
        print(
            f"    {'':12s} input={fmt_tokens(t.input):7s}  "
            f"cache_create={fmt_tokens(t.cache_create):7s}  "
            f"cache_read={fmt_tokens(t.cache_read):7s}"
        )
    if breakdown and r.per_agent:
        # Table header
        print(f"    {'Role':<14} {'turns':>5}  {'input':>8}  {'cache_create':>12}  {'cache_read':>10}  {'output':>8}  {'eff_input':>9}  {'cost':>8}")
        print(f"    {'-'*14} {'-'*5}  {'-'*8}  {'-'*12}  {'-'*10}  {'-'*8}  {'-'*9}  {'-'*8}")
        for label, s in r.per_agent:
            print(
                f"    {label:<14} {s.turns:>5}  "
                f"{fmt_tokens(s.input):>8}  "
                f"{fmt_tokens(s.cache_create):>12}  "
                f"{fmt_tokens(s.cache_read):>10}  "
                f"{fmt_tokens(s.output):>8}  "
                f"{fmt_tokens(s.effective_input):>9}  "
                f"${s.cost_usd:>7.4f}"
            )
        print(
            f"    {'TOTAL':<14} {t.turns:>5}  "
            f"{fmt_tokens(t.input):>8}  "
            f"{fmt_tokens(t.cache_create):>12}  "
            f"{fmt_tokens(t.cache_read):>10}  "
            f"{fmt_tokens(t.output):>8}  "
            f"{fmt_tokens(t.effective_input):>9}  "
            f"${t.cost_usd:>7.4f}"
        )


def compare_runs(groups: dict[str, list[RunResult]]) -> None:
    """Print side-by-side comparison for runs with matching task names."""
    # Gather all task names
    all_tasks: dict[str, dict[str, RunResult]] = {}
    for job_name, runs in groups.items():
        for r in runs:
            all_tasks.setdefault(r.task_name, {})[job_name] = r

    job_names = list(groups.keys())
    print(f"\n{'Task':<40}  ", end="")
    for j in job_names:
        print(f"  {j:<50}", end="")
    print()
    print("-" * (42 + 52 * len(job_names)))

    for task, by_job in sorted(all_tasks.items()):
        print(f"{task:<40}  ", end="")
        for j in job_names:
            r = by_job.get(j)
            if r is None:
                print(f"  {'—':50}", end="")
            else:
                t = r.tokens
                s = (
                    f"reward={r.reward:.3f} "
                    f"dur={r.duration_sec:.0f}s "
                    f"input={fmt_tokens(t.effective_input)} "
                    f"out={fmt_tokens(t.output)} "
                    f"${t.cost_usd:.4f}"
                )
                print(f"  {s:<50}", end="")
        print()


def main() -> None:
    args = sys.argv[1:]
    compare_mode = "--compare" in args
    if compare_mode:
        args.remove("--compare")
    breakdown = "--breakdown" in args or "-b" in args
    if breakdown:
        for flag in ("--breakdown", "-b"):
            if flag in args:
                args.remove(flag)
    verbose = "--verbose" in args or "-v" in args
    if verbose:
        for flag in ("--verbose", "-v"):
            if flag in args:
                args.remove(flag)

    if not args:
        print(__doc__)
        sys.exit(1)

    # Resolve paths (support passing "jobs/" to scan all subdirs)
    paths: list[tuple[str, Path]] = []
    for arg in args:
        p = Path(arg)
        if not p.exists():
            print(f"Warning: {arg} not found, skipping")
            continue
        # If it's a parent "jobs/" dir, expand to children
        if p.is_dir() and not (p / "result.json").exists():
            has_timestamp = any(
                d.is_dir() and len(d.name) > 10 for d in p.iterdir()
                if d.is_dir()
            )
            if has_timestamp:
                paths.append((p.name, p))
            else:
                for child in sorted(p.iterdir()):
                    if child.is_dir():
                        paths.append((child.name, child))
        else:
            paths.append((p.name, p))

    groups: dict[str, list[RunResult]] = {}
    for job_name, job_path in paths:
        runs = find_runs(job_path, job_name)
        if runs:
            groups[job_name] = runs

    if compare_mode:
        compare_runs(groups)
        return

    for job_name, runs in groups.items():
        print(f"\n{'='*80}")
        print(f"Job: {job_name}  ({len(runs)} trial(s))")
        print(f"{'='*80}")
        for r in runs:
            print_run(r, verbose=verbose, breakdown=breakdown)

        if len(runs) > 1:
            total = TokenStats()
            for r in runs:
                total = total + r.tokens
            print(f"\n  {'TOTAL':>54}  turns={total.turns:3d}  "
                  f"eff_input={fmt_tokens(total.effective_input):7s}  "
                  f"output={fmt_tokens(total.output):6s}  "
                  f"cost=${total.cost_usd:.4f}")


if __name__ == "__main__":
    main()
