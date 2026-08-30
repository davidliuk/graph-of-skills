# Benchmark Data

Evaluation data is **not** included in this repository. The download script fetches everything you need.

## Data Assets

| Asset | Source | Destination | Approx. Size |
|-------|--------|-------------|-------------|
| Skill libraries (`skills_200`, `skills_500`, `skills_1000`, `skills_2000`) | [HuggingFace](https://huggingface.co/datasets/DLPenn/graph-of-skills-data) | `data/skillsets/` | varies by archives on Hub |
| Benchmark tasks (87 coding tasks) | [benchflow-ai/skillsbench](https://github.com/benchflow-ai/skillsbench) | `evaluation/skillsbench/tasks/` | 580 MB |
| Prebuilt GoS workspaces (`gos_workspace_skills_*_v1.tar.gz`) | [HuggingFace](https://huggingface.co/datasets/DLPenn/graph-of-skills-data) | `data/gos_workspace/skills_{200,500,1000,2000}_v1/` | varies |

> **Legacy workspace warning:** the `*_v1` workspace archives contain
> undirected graphs from the earlier cleanup implementation. They may be used
> only for reproducing old results; the current typed-directed storage rejects
> them because dependency direction cannot be reconstructed reliably. Rebuild
> from the corresponding skill library for current experiments.

## Download Everything

```bash
./scripts/download_data.sh
```

## Selective Download

Download only what you need:

```bash
./scripts/download_data.sh --skillsets    # skill libraries only (tries all four; skips existing dirs; missing Hub archives log [skip])
./scripts/download_data.sh --tasks        # benchmark tasks only
./scripts/download_data.sh --workspace    # prebuilt workspaces (tries all four; skips existing; missing [skip])
```

## Authenticated Download

If the HuggingFace dataset is gated or private:

```bash
HF_TOKEN=hf_... ./scripts/download_data.sh --skillsets
```

You can also override the HuggingFace repository:

```bash
GOS_HF_REPO=your-org/your-dataset ./scripts/download_data.sh --skillsets
```

## Building the Workspace from Source

You can rebuild the GoS workspace yourself instead of downloading it. This is useful when using a different embedding model or skill set.

**Step 1.** Download the skill sets (if not already present):

```bash
./scripts/download_data.sh --skillsets
```

**Step 2.** Index the skill library:

```bash
uv run gos index data/skillsets/skills_1000 \
  --workspace data/gos_workspace/skills_1000_v1 \
  --clear
```

**Step 3.** Verify the workspace:

```bash
uv run gos status --workspace data/gos_workspace/skills_1000_v1
```

**Step 4.** Test retrieval:

```bash
uv run gos retrieve "parse binary STL file, calculate volume and mass" \
  --workspace data/gos_workspace/skills_1000_v1 \
  --max-skills 5
```

> **Note:** Building requires an embedding API. Configure your provider in `.env` (see [`.env.example`](.env.example)). The workspace must be queried with the same embedding model it was indexed with. On **Azure**, set `GOS_EMBEDDING_MODEL=openai/<deployment-name>` to match the name shown in the Azure portal, not necessarily `text-embedding-3-large`.

## Packaging prebuilt workspaces (maintainers)

After `data/gos_workspace/skills_{200,500,1000,2000}_v1/` are built, create Hub-ready tarballs and upload:

```bash
./scripts/pack_gos_workspaces.sh
HF_TOKEN=hf_... uv run --with huggingface_hub python scripts/upload_gos_workspaces_to_hf.py
```

Optional: `GOS_HF_REPO=org/dataset` and `--staging-dir` on the upload script. Archives land in `data/hf_staging/` (gitignored).
