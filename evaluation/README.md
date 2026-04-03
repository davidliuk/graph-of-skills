# Evaluation

This directory contains benchmark runners and experiment infrastructure for Graph of Skills.

> **Before running evaluations**, download the benchmark data:
>
> ```bash
> ./scripts/download_data.sh
> ```
>
> See [DATA.md](../DATA.md) for selective download options.

## Overview

| Track | Type | Tasks | Runner |
|-------|------|-------|--------|
| [ALFWorld](#1-alfworld) | Interactive household tasks | 134 games | `alfworld_run.py` |
| [ScienceWorld](#2-scienceworld) | Science experiment simulation | varies | `scienceworld_run.py` |
| [SkillsBench](#3-skillsbench) | Dockerized coding tasks | 87 tasks | Harbor + `graphskills_benchmark.py` |

All tracks support four retrieval modes:

| Mode | Description |
|------|-------------|
| `gos` | Graph-backed retrieval (default) |
| `vector` | Vector-only retrieval baseline |
| `all_full` | Full skill library in context |
| `none` | No skills provided |

---

## 1. ALFWorld

Interactive household task benchmark. The runner wraps GoS retrieval into the agent decision loop.

**Prerequisites:**

```bash
uv sync --extra alfworld   # install ALFWorld dependencies
uv run alfworld-download    # download game data (~300 MB)
```

**Run a single game:**

```bash
API_KEY=<your-key> BASE_URL=<api-base-url> \
python evaluation/alfworld_run.py \
  --model <model-name> \
  --split eval_out_of_distribution \
  --use_skill \
  --mode gos \
  --gos_workspace data/gos_workspace/skills_1000_v1 \
  --skills_dir data/skillsets/skills_1000 \
  --max_workers 1 \
  --max_steps 30 \
  --max_games 1 \
  --exp_name my_experiment
```

The runner uses an OpenAI-compatible chat API (`API_KEY` + `BASE_URL`).

## 2. ScienceWorld

Science experiment simulation benchmark with the same retrieval integration.

```bash
python evaluation/scienceworld_run.py \
  --model <model-name> \
  --use_skill \
  --mode gos \
  --gos_workspace data/gos_workspace/skills_1000_v1 \
  --skills_dir data/skillsets/skills_1000 \
  --exp_name my_experiment
```

## 3. SkillsBench

Dockerized benchmark with 87 coding tasks. Each task runs inside an isolated Docker container managed by [Harbor](https://github.com/harbor-ai/harbor).

See [skillsbench/README.md](skillsbench/README.md) for full setup and run instructions.

**Quick example** (single task, Gemini CLI):

```bash
GEMINI_API_KEY=<key> harbor run \
  --agent gemini-cli \
  --model gemini/gemini-3-flash-preview \
  --force-build \
  -p evaluation/skillsbench/generated_skills200/tasks_all_skills/dialogue-parser \
  -o evaluation/skillsbench/jobs/my-test-run
```

---

## Environment Setup

GoS retrieval requires an embedding API. Set these variables before running any track:

```bash
# Example: Google Gemini
export GEMINI_API_KEY=<your-key>
export GOS_EMBEDDING_MODEL=gemini/gemini-embedding-001
export GOS_EMBEDDING_DIM=3072
```

Or for OpenRouter:

```bash
export OPENROUTER_API_KEY=<openrouter-key>
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
export GOS_EMBEDDING_MODEL=openrouter/openai/text-embedding-3-large
export GOS_EMBEDDING_DIM=3072
```

Or for OpenAI directly (no custom base URL):

```bash
export OPENAI_API_KEY=<your-key>
export GOS_EMBEDDING_MODEL=openai/text-embedding-3-large
export GOS_EMBEDDING_DIM=3072
```

> **Important:** The embedding model must match the one used when the workspace was indexed. On Azure, `GOS_EMBEDDING_MODEL` must be `openai/<deployment-name>`; see [`.env.example`](../.env.example).

## Building a GoS Workspace

If you don't have a prebuilt workspace (or want to rebuild with a different embedding model):

```bash
uv run gos index data/skillsets/skills_1000 \
  --workspace data/gos_workspace/skills_1000_v1 \
  --clear
```

Verify:

```bash
uv run gos status --workspace data/gos_workspace/skills_1000_v1
```

## Result Format

All runners produce JSON result files with these key fields:

| Field | Description |
|-------|-------------|
| `reward` | Task success score (0.0 -- 1.0) |
| `token_usage` / `agent_result.n_input_tokens` | Token consumption |
| `agent_execution.started_at` / `finished_at` | Agent-only execution time |
| `retrieval_status` | `SKILL_HIT` or `NO_SKILL_HIT` (GoS/vector modes) |

When reporting execution time, use the **agent-only** interval. Do not include Docker build or environment setup time.

## File Reference

| File | Purpose |
|------|---------|
| `alfworld_run.py` | ALFWorld benchmark runner |
| `scienceworld_run.py` | ScienceWorld benchmark runner |
| `skill.py` | `SkillModule` -- unified adapter wrapping GoS retrieval for all tracks |
| `prompt_generator.py` | Prompt construction utilities |
| `token_usage.py` | Token accounting helpers |
| `utils.py` | Shared utilities |
| `skills_ref/` | Skill document parsing and validation |
| `skillsbench/` | SkillsBench benchmark framework ([details](skillsbench/README.md)) |
