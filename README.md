<h1 align="center">Graph of Skills (GoS)</h1>

<p align="center">
  <strong>Dependency-Aware Structural Retrieval for Massive Agent Skills</strong>
</p>

<p align="center">
  <em>Dawei Liu, Zongxia Li, Hongyang Du, Xiyang Wu, Shihang Gui, Yongbei Kuang, Lichao Sun</em>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/TODO"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv" alt="Paper"></a>
  <a href="https://huggingface.co/datasets/DLPenn/graph-of-skills-data"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Data-yellow" alt="Data"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10--3.12-3776ab?logo=python&logoColor=white" alt="Python"></a>
</p>

---

Graph of Skills builds a **skill graph** offline from a library of `SKILL.md` documents, then retrieves a small, ranked set of relevant skills at task time. Instead of flooding the agent context with an entire skill library, GoS surfaces only the skills most likely to help -- along with their prerequisites and related capabilities.

## How It Works

```
Offline (index)                           Online (retrieve)
┌──────────────┐                         ┌──────────────────────┐
│  SKILL.md    │──parse──▶ Skill Nodes   │  Task description    │
│  library     │──embed──▶ Vector Index  │         │            │
│              │──link───▶ Skill Graph   │    ┌────▼─────┐      │
└──────────────┘                         │    │ Semantic  │      │
                                         │    │ + Lexical │      │
                                         │    │  Seeds    │      │
                                         │    └────┬─────┘      │
                                         │    ┌────▼─────┐      │
                                         │    │  Graph    │      │
                                         │    │ Reranking │      │
                                         │    └────┬─────┘      │
                                         │    ┌────▼─────┐      │
                                         │    │ Bounded   │      │
                                         │    │ Skill     │      │
                                         │    │ Bundle    │      │
                                         │    └──────────┘      │
                                         └──────────────────────┘
```

**Retrieval pipeline:**

1. **Seed** -- retrieve semantic candidates (embedding similarity) and lexical candidates (exact-match tokens)
2. **Merge** -- combine both candidate pools
3. **Rerank** -- rerank using the skill-graph structure (dependencies, co-occurrence)
4. **Return** -- emit a capped, agent-readable skill bundle

## Installation

### Requirements

- Python 3.10 -- 3.12
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`
- An embedding API key (OpenAI, Gemini, or any OpenAI-compatible provider)

### Setup

```bash
git clone https://github.com/graph-of-skills/graph-of-skills.git
cd graph-of-skills
uv sync
cp .env.example .env   # then fill in your API keys
```

<details>
<summary><strong>Provider: OpenAI (direct)</strong></summary>

```bash
OPENAI_API_KEY=sk-...
# Use the ``openai/...`` prefix so LiteLLM targets the OpenAI API (omit OPENAI_BASE_URL).
GOS_EMBEDDING_MODEL=openai/text-embedding-3-large
GOS_EMBEDDING_DIM=3072
```
</details>

<details>
<summary><strong>Provider: OpenRouter</strong></summary>

```bash
OPENROUTER_API_KEY=<openrouter-key>
OPENAI_BASE_URL=https://openrouter.ai/api/v1
GOS_EMBEDDING_MODEL=openrouter/openai/text-embedding-3-large
GOS_EMBEDDING_DIM=3072
```
</details>

<details>
<summary><strong>Provider: Azure AI (OpenAI-compatible)</strong></summary>

```bash
OPENAI_API_KEY=<azure-api-key>
OPENAI_BASE_URL=https://YOUR-RESOURCE.services.ai.azure.com/openai/v1
# Must match your **deployment name** in Azure (not necessarily ``text-embedding-3-large``).
GOS_EMBEDDING_MODEL=openai/<your-deployment-name>
GOS_EMBEDDING_DIM=<vector-dimension-for-that-model>
```
</details>

<details>
<summary><strong>Provider: Google Gemini</strong></summary>

```bash
GEMINI_API_KEY=<your-key>
GOS_EMBEDDING_MODEL=gemini/gemini-embedding-001
GOS_EMBEDDING_DIM=3072
```
</details>

## Quick Start

**1. Index a skill library**

```bash
uv run gos index path/to/skills/ --workspace ./my_workspace --clear
```

**2. Retrieve skills for a task**

```bash
uv run gos retrieve "parse binary STL file, calculate volume and mass" \
  --workspace ./my_workspace --max-skills 5
```

**3. Check workspace status**

```bash
uv run gos status --workspace ./my_workspace
```

**4. Add a skill incrementally**

```bash
uv run gos add path/to/NEW_SKILL.md --workspace ./my_workspace
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `gos index <dir>` | Build a graph workspace from a skill directory |
| `gos add <file>` | Add a single skill to an existing workspace |
| `gos retrieve <query>` | Retrieve a ranked skill bundle for a query |
| `gos query <query>` | Compact retrieval output (for debugging) |
| `gos status` | Show workspace statistics |
| `gos experiment` | Run built-in experiment presets |
| `graphskills-query` | Agent-facing retrieval (rewrites `Source:` paths for containers) |
| `gos-server` | Start the MCP server for tool-based retrieval |

## Agent Integration

Inside a Docker container, an agent calls `graphskills-query` with a natural-language task description and receives a bounded skill bundle:

```bash
graphskills-query "parse binary STL file and calculate mass"
```

Each returned skill includes a `Source:` path the agent can open directly:

```
Source: /opt/graphskills/skills/mesh-analysis/SKILL.md
```

Set `GOS_SKILLS_DIR` to control path rewriting, so the same workspace can be indexed on a host and queried inside a container.

## Configuration

All runtime settings are driven by environment variables. See [`.env.example`](.env.example) for the full template.

| Variable | Default | Description |
|----------|---------|-------------|
| `GOS_EMBEDDING_MODEL` | `openai/text-embedding-3-large` | Embedding model for indexing and retrieval (use `openai/<deployment>` on Azure) |
| `GOS_EMBEDDING_DIM` | `3072` | Embedding dimension (must match the model output) |
| `GOS_PREBUILT_WORKING_DIR` | -- | Path to a prebuilt workspace for retrieval |
| `GOS_RETRIEVAL_TOP_N` | `8` | Maximum number of skills returned |
| `GOS_SEED_TOP_K` | `5` | Initial seed count before graph expansion |
| `GOS_MAX_CONTEXT_CHARS` | `12000` | Hard cap on total returned bundle size (chars) |
| `GOS_SKILLS_DIR` | -- | Container-side skill root (for `Source:` path rewriting) |

> **Note:** The embedding model at retrieval time **must** match the model used when the workspace was indexed.

## Repository Layout

```
graph-of-skills/
├── gos/                          # Core GoS package
│   ├── core/                     #   Engine, retrieval, parsing, schema
│   ├── interfaces/               #   CLI and MCP server
│   └── utils/                    #   Configuration (pydantic-settings)
├── data/                         # Downloaded data (gitignored; see DATA.md)
│   ├── skillsets/                #   Skill libraries (skills_200, skills_1000)
│   └── gos_workspace/            #   Prebuilt graph workspaces
├── evaluation/                   # Evaluation framework
│   ├── alfworld_run.py           #   ALFWorld benchmark runner
│   ├── scienceworld_run.py       #   ScienceWorld benchmark runner
│   ├── skill.py                  #   SkillModule adapter for GoS
│   └── skillsbench/              #   SkillsBench (Harbor-based, 87 tasks)
├── skills/                       # Agent bootstrap skills for retrieval
├── scripts/                      # Utility scripts (data download, etc.)
├── tests/                        # Test suite
├── pyproject.toml                # Package definition & CLI entry points
├── .env.example                  # Environment variable template
└── DATA.md                       # Benchmark data download guide
```

## Evaluation

We evaluate GoS on three benchmarks:

| Benchmark | Type | Tasks |
|-----------|------|-------|
| **ALFWorld** | Interactive household tasks | 134 games |
| **ScienceWorld** | Science experiment simulation | varies by task type |
| **SkillsBench** | Dockerized coding tasks | 87 tasks |

For **running these evaluations**, we recommend routing the agent’s chat / completion API through [OpenRouter](https://openrouter.ai/): use an OpenAI-compatible `BASE_URL` (for example `https://openrouter.ai/api/v1`) and the API key your runner documents. The GoS project’s own evaluation testing is done mainly this way. Embeddings for indexing and retrieval are separate; configure them in `.env` as in [`.env.example`](.env.example) (OpenRouter, direct OpenAI, Gemini, or Azure). See [evaluation/README.md](evaluation/README.md) for per-track commands and environment variables.

Benchmark data is hosted externally and **not** included in this repository:

```bash
./scripts/download_data.sh          # download all assets (~780 MB)
```

See [DATA.md](DATA.md) for selective downloads.

## Citation

Coming soon.

<!--
If you find this work useful, please cite:

```bibtex
@article{liu2026graphofskills,
  title         = {Graph of Skills: Dependency-Aware Structural Retrieval for Massive Agent Skills},
  author        = {Dawei Liu and Zongxia Li and Hongyang Du and Xiyang Wu and Shihang Gui and Yongbei Kuang and Lichao Sun},
  year          = {2026},
  eprint        = {TODO},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}
```
-->

## License

This project is licensed under the [MIT License](LICENSE).
