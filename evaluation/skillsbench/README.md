# SkillsBench Evaluation

This directory contains the SkillsBench benchmark framework used to evaluate Graph of Skills against baseline conditions.

## Prerequisites

- [Harbor](https://harborframework.com/) installed (`uv tool install harbor`)
- Docker or OrbStack running
- Benchmark data downloaded (see [DATA.md](../../DATA.md))

## Benchmark Conditions

| Condition | Config Directory | Description |
|-----------|-----------------|-------------|
| `graphskills` | `experiments/configs/graphskills/` | Graph-backed retrieval via `graphskills-query` |
| `allskills` | `experiments/configs/allskills/` | Full skill library mounted in container |
| `vectorskills` | `experiments/configs/vectorskills/` | Vector-only retrieval baseline |
| `without` | `experiments/configs/without/` | No skills provided |

## Task Generation

Generate benchmark task variants from the base task set:

```bash
python graphskills_benchmark.py
```

This creates task directories for each condition using the templates in `_gos_template/`, `_allskills_template/`, and `_vectorskills_template/`.

## Running Experiments

### Single task

```bash
harbor run \
  --agent codex \
  --model openai/gpt-5.2-codex \
  --force-build \
  -p <task-directory> \
  -o <output-directory>
```

### Batch via YAML config

```bash
harbor run -c experiments/configs/graphskills/codex.yaml
```

For experiments requiring host-side API keys (verifier, oracle):

```bash
scripts/harbor_run_with_env.sh -c experiments/configs/graphskills/codex.yaml
```

## Supported Agents

Each condition directory contains configs for multiple agents:

- `codex.yaml` -- OpenAI Codex
- `claude-code.yaml` -- Claude Code (Anthropic)
- `gemini-cli.yaml` -- Gemini CLI (Google)

## Result Inspection

Job-level result:

```bash
cat <output-dir>/result.json
```

Per-trial result:

```bash
cat <output-dir>/<task>__<trial_id>/result.json
```

Key fields: `verifier_result.rewards.reward`, `agent_result.n_input_tokens`, `agent_result.n_output_tokens`, `agent_execution.started_at`, `agent_execution.finished_at`.

## Directory Layout

```
skillsbench/
├── _allskills_template/        # Template for all-skills condition
├── _gos_template/              # Template for graph-skills condition
├── _vectorskills_template/     # Template for vector-skills condition
├── graphskills_benchmark.py    # Task variant generator
├── graphskills_assets/         # Assets used during generation
├── experiments/
│   └── configs/                # Harbor YAML configs per condition
├── scripts/                    # Benchmark maintenance scripts
└── tasks/                      # Base benchmark tasks (cloned from SkillsBench)
```

## License

[Apache 2.0](LICENSE)
