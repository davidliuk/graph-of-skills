# Benchmark Data

The evaluation benchmark requires data assets that are not included in this repository.

| Asset | Source | Destination | Size |
|-------|--------|-------------|------|
| Skill sets (`skills_200`, `skills_1000`) | [HuggingFace](https://huggingface.co/datasets/DLPenn/graph-of-skills-data) | `data/skillsets/` | ~160MB |
| Benchmark tasks (87 tasks) | [benchflow-ai/skillsbench](https://github.com/benchflow-ai/skillsbench) | `evaluation/skillsbench/tasks/` | ~580MB |
| Prebuilt GoS workspace | [HuggingFace](https://huggingface.co/datasets/DLPenn/graph-of-skills-data) | `data/gos_workspace/` | ~40MB |

## Quick Download

```bash
./scripts/download_data.sh
```

## Selective Download

```bash
./scripts/download_data.sh --skillsets    # skill libraries from HuggingFace
./scripts/download_data.sh --tasks        # 87 tasks from SkillsBench GitHub
./scripts/download_data.sh --workspace    # prebuilt GoS workspace from HuggingFace
```

## HuggingFace Authentication

If the HuggingFace dataset is gated or private, set your token:

```bash
HF_TOKEN=hf_... ./scripts/download_data.sh --skillsets
```

## Building the Workspace from Source

Instead of downloading the prebuilt workspace, build it yourself after downloading the skill sets:

```bash
uv run gos index data/skillsets/skills_1000 \
  --workspace data/gos_workspace/skills_1000_v1 \
  --clear
```

Verify:

```bash
uv run gos status --workspace data/gos_workspace/skills_1000_v1
```

Test retrieval:

```bash
uv run gos retrieve "parse binary STL file, calculate volume and mass" \
  --workspace data/gos_workspace/skills_1000_v1 \
  --max-skills 5
```

This requires an embedding API (see `.env.example` for configuration). The workspace must be indexed with the same embedding model used at retrieval time.
