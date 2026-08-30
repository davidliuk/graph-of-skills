# SkillsBench Evaluation

SkillsBench is a Dockerized benchmark with **87 coding tasks**, each evaluated inside an isolated container via [Harbor](https://github.com/harbor-ai/harbor). This framework compares GoS retrieval against several baselines.

## Prerequisites

1. Install Harbor: `uv tool install harbor`
2. Ensure Docker (or OrbStack) is running
3. Download benchmark data: `./scripts/download_data.sh` (see [DATA.md](../../DATA.md))

## Benchmark Conditions

| Condition | Template | Description |
|-----------|----------|-------------|
| `graphskills` | `_gos_template/` | Agent uses `graphskills-query` to retrieve skills from the graph |
| `allskills` | `_allskills_template/` | Full skill library is mounted in the container |
| `vectorskills` | `_vectorskills_template/` | Vector-only retrieval (no graph reranking) |
| `without` | -- | No skills provided to the agent |

## Generating Task Variants

Generate condition-specific task directories from the base task set:

```bash
python evaluation/skillsbench/graphskills_benchmark.py \
  --skillset-name skills_200 \
  --conditions graphskills allskills
```

This creates directories like `generated_skills200/tasks_graph_skills/` and `generated_skills200/tasks_all_skills/` using the corresponding templates.

## Running Experiments

### Single Task

```bash
GEMINI_API_KEY=<key> harbor run \
  --agent gemini-cli \
  --model gemini/gemini-3-flash-preview \
  --force-build \
  -p evaluation/skillsbench/generated_skills200/tasks_graph_skills/dialogue-parser \
  -o evaluation/skillsbench/jobs/dialogue-parser-gos
```

### Batch via YAML Config

```bash
harbor run -c evaluation/skillsbench/experiments/configs/graphskills/codex.yaml
```

For experiments that need host-side API keys (verifier, oracle):

```bash
scripts/harbor_run_with_env.sh -c experiments/configs/graphskills/codex.yaml
```

## Supported Agents

| Agent | Config Example | Notes |
|-------|---------------|-------|
| OpenAI Codex | `codex.yaml` | Uses `--model openai/gpt-5.2-codex` |
| Claude Code | `claude-code.yaml` | Requires `ANTHROPIC_AUTH_TOKEN` |
| Gemini CLI | `gemini-cli.yaml` | Requires `GEMINI_API_KEY` |
| TRAE CLI | `configs/traex/*.yaml` | Uses the authenticated host `traex` executable; no OpenAI API key is required |

## Running with local TRAE CLI (GPT-5.2)

The custom Harbor agent in `agents/traex_host.py` runs the host-side TRAE CLI while
the benchmark remains isolated in Docker. It mirrors the active task workspace through
Harbor's `/logs/agent` mount and gives TRAE a task-scoped `./task-exec` wrapper for
commands that must run inside the benchmark container.

Requirements:

```bash
traex login status
traex --version
docker info
```

Run the end-to-end sanity task first. TRAE model startup can have a long tail, so the
config uses a 5x task timeout and stores live CLI output in each trial's
`agent/traex.jsonl`.

```bash
PYTHONPATH="$PWD" uv run --project evaluation/skillsbench harbor run \
  -c evaluation/skillsbench/experiments/configs/traex/smoke.yaml
```

Generate and run one matched lexical ablation condition:

```bash
uv run python evaluation/skillsbench/scripts/run_traex_ablation.py \
  --condition lexical-reverse-ppr \
  --task dialogue-parser \
  --attempts 1 \
  --execute
```

Omit `--execute` to print the exact generation and Harbor commands. Use
`--skip-generate` to reuse an existing generated dataset, and adjust slow or flaky local
model service behavior with `--timeout-multiplier`.

For paper reporting, label these runs **TRAE CLI (GPT-5.2; Codex harness mode)**. Do not
merge them into an OpenAI Codex/API row: the model family may match, but the provider,
authentication path, CLI version, and harness are separate experimental factors.

## Inspecting Results

**Job-level summary:**

```bash
cat evaluation/skillsbench/jobs/<job-name>/result.json | python -m json.tool
```

**Per-trial result:**

```bash
cat evaluation/skillsbench/jobs/<job-name>/<task>__<trial>/result.json
```

**Key fields:**

| Field | Meaning |
|-------|---------|
| `verifier_result.rewards.reward` | Task success (0.0 or 1.0) |
| `agent_result.n_input_tokens` | Input tokens consumed |
| `agent_result.n_output_tokens` | Output tokens generated |
| `agent_execution.started_at` | Agent start timestamp |
| `agent_execution.finished_at` | Agent end timestamp |

## Directory Layout

```
skillsbench/
├── _allskills_template/         # Docker template: all-skills condition
├── _gos_template/               # Docker template: graph-skills condition
├── _vectorskills_template/      # Docker template: vector-skills condition
├── graphskills_benchmark.py     # Task variant generator
├── graphskills_assets/          # Assets bundled during task generation
├── experiments/
│   └── configs/                 # Harbor YAML configs per condition & agent
├── agents/                      # Custom Harbor agents, including host TRAE CLI
├── generated_skills*/           # Generated task variants (gitignored)
├── scripts/                     # Benchmark maintenance scripts
├── tasks/                       # Base benchmark tasks (cloned from SkillsBench)
└── jobs/                        # Run outputs (gitignored)
```
