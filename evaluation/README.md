# Evaluation

This directory contains the experiment runners and benchmark framework for Graph of Skills.

**Before running evaluations**, download the benchmark data:

```bash
./scripts/download_data.sh
```

See [DATA.md](../DATA.md) for details.

## Architecture Boundary

- `gos/`: core Graph of Skills implementation (retrieval algorithm)
- `evaluation/`: benchmark runners, experiment configs, and task generation

If you want to understand or modify retrieval, start in `gos/`.
If you want to reproduce paper results, start here.

## Evaluation Tracks

### 1. ALFWorld

Interactive household task benchmark. The runner integrates GoS retrieval into the agent loop.

```bash
python evaluation/alfworld_run.py \
  --model <model> \
  --split dev \
  --use_skill \
  --mode gos \
  --gos_workspace data/gos_workspace/skills_1000_v1 \
  --skills_dir data/skillsets/skills_1000 \
  --exp_name my_experiment \
  --max_workers 1 \
  --max_steps 30
```

Supported modes: `gos` (graph retrieval), `vector` (vector-only baseline), `all_full` (flat full-library), `none` (no skills).

ALFWorld requires a separate installation of the [ALFWorld](https://github.com/alfworld/alfworld) environment and dataset. Set `ALFWORLD_DATA` to the dataset root and add ALFWorld to `PYTHONPATH`.

### 2. ScienceWorld

Science experiment simulation benchmark with the same retrieval integration.

```bash
python evaluation/scienceworld_run.py \
  --model <model> \
  --use_skill \
  --mode gos \
  --gos_workspace data/gos_workspace/skills_1000_v1 \
  --skills_dir data/skillsets/skills_1000 \
  --exp_name my_experiment
```

### 3. SkillsBench (Harbor)

Dockerized benchmark with 87 tasks comparing retrieval conditions:

| Condition | Description |
|-----------|-------------|
| `graphskills` | Graph-backed retrieval via `graphskills-query` |
| `allskills` | Full skill library mounted in the container |
| `vectorskills` | Vector-only retrieval baseline |
| `without` | No skills provided |

See [skillsbench/README.md](skillsbench/README.md) for setup and run commands.

## Environment Setup

Required environment variables for all tracks:

```bash
export OPENAI_API_KEY=<your-key>
export OPENAI_BASE_URL=https://openrouter.ai/api/v1  # if using OpenRouter
export GOS_EMBEDDING_MODEL=openai/text-embedding-3-large
export GOS_EMBEDDING_DIM=3072
```

The GoS workspace must be indexed with the same embedding model used at retrieval time.

## Building a GoS Workspace

```bash
uv run gos index data/skillsets/skills_1000 \
  --workspace data/gos_workspace/skills_1000_v1 \
  --clear
```

Verify with:

```bash
uv run gos status --workspace data/gos_workspace/skills_1000_v1
```

## Result Format

Both local runners and SkillsBench produce JSON result files with:

- `reward`: task success (0.0 or 1.0)
- `token_usage`: prompt/completion/total token counts
- `agent_runtime_seconds` or `agent_execution` timing
- `retrieval_status` and `retrieval_summary` (for GoS/vector modes)

When reporting execution time, use agent-only intervals. Do not include Docker build or environment setup time.

## Components

| File | Purpose |
|------|---------|
| `alfworld_run.py` | ALFWorld benchmark runner |
| `scienceworld_run.py` | ScienceWorld benchmark runner |
| `skill.py` | `SkillModule` adapter wrapping `gos.SkillGraphRAG` |
| `prompt_generator.py` | Prompt construction utilities |
| `token_usage.py` | Token accounting helpers |
| `utils.py` | Shared utilities |
| `skills_ref/` | Skill validation library |
| `skillsbench/` | SkillsBench benchmark framework |
