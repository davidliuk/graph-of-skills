# Graph of Skills (GoS)

[\[Paper\]](https://arxiv.org/abs/TODO)
[\[Data\]](https://huggingface.co/datasets/DLPenn/graph-of-skills-data)

**Dependency-aware structural retrieval for massive agent skills.**

> Dawei Liu, Zongxia Li, Hongyang Du, Xiyang Wu, Shihang Gui, Yongbei Kuang, Lichao Sun

Graph of Skills builds a skill graph offline from a `SKILL.md` library, then retrieves a small, ranked set of relevant skills at task time. Instead of flooding the agent context with an entire skill library, GoS surfaces only the skills most likely to help -- along with their prerequisites and related capabilities.

## Architecture

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

The retrieval pipeline:

1. Semantic candidate retrieval via embedding similarity
2. Lexical candidate retrieval for exact-match coverage
3. Merge both candidate pools
4. Rerank using the skill graph structure
5. Return a capped, agent-readable bundle

## Installation

```bash
git clone https://github.com/graph-of-skills/graph-of-skills.git
cd graph-of-skills
uv sync
cp .env.example .env
```

Edit `.env` with your API credentials. GoS supports any OpenAI-compatible embedding provider.

**Minimal configuration** (using OpenAI directly):

```bash
OPENAI_API_KEY=sk-...
GOS_EMBEDDING_MODEL=text-embedding-3-large
GOS_EMBEDDING_DIM=3072
```

**OpenRouter configuration**:

```bash
OPENAI_API_KEY=<openrouter-key>
OPENAI_BASE_URL=https://openrouter.ai/api/v1
GOS_EMBEDDING_MODEL=openai/text-embedding-3-large
GOS_EMBEDDING_DIM=3072
```

## Quick Start

### 1. Index a skill library

```bash
uv run gos index path/to/skills/ \
  --workspace ./my_workspace \
  --clear
```

### 2. Retrieve skills for a task

```bash
uv run gos retrieve "parse binary STL file, calculate volume and mass" \
  --workspace ./my_workspace \
  --max-skills 5
```

### 3. Check workspace status

```bash
uv run gos status --workspace ./my_workspace
```

### 4. Incrementally add a skill

```bash
uv run gos add path/to/NEW_SKILL.md --workspace ./my_workspace
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `gos index` | Build a workspace from a skill directory |
| `gos add` | Add a single skill to an existing workspace |
| `gos retrieve` | Retrieve a ranked skill bundle for a query |
| `gos query` | Compact query output for inspection |
| `gos status` | Show workspace statistics |
| `gos experiment` | Run built-in experiment presets |
| `graphskills-query` | Agent-facing retrieval wrapper (rewrites paths for containers) |
| `gos-server` | MCP server for tool-based retrieval |

## Agent Integration

In a typical setup, an agent calls `graphskills-query` with a task description and receives a bounded skill bundle with `Source:` paths pointing to the relevant skill documents:

```bash
graphskills-query "parse binary STL file and calculate mass"
```

The `GOS_SKILLS_DIR` environment variable rewrites `Source:` paths for use inside containers, so the same workspace can be indexed on a host and queried by an agent in Docker.

## Configuration

All runtime settings are driven by environment variables. See `.env.example` for the full list.

Key variables:

| Variable | Description |
|----------|-------------|
| `GOS_EMBEDDING_MODEL` | Embedding model for indexing and retrieval |
| `GOS_EMBEDDING_DIM` | Embedding dimension (must match the model) |
| `GOS_PREBUILT_WORKING_DIR` | Prebuilt workspace path for retrieval |
| `GOS_RETRIEVAL_TOP_N` | Maximum skills returned |
| `GOS_SEED_TOP_K` | Initial seed count for graph retrieval |
| `GOS_MAX_CONTEXT_CHARS` | Hard cap on returned bundle size |
| `GOS_SKILLS_DIR` | Path rewriting for container-side skill access |

Retrieval-time settings must match the workspace's index-time embedding model.

## Repository Layout

```
graph-of-skills/
├── gos/                          # Core GoS package
│   ├── core/                     # Engine, retrieval, parsing, schema
│   ├── interfaces/               # CLI and MCP server
│   └── utils/                    # Configuration
├── data/                         # Downloaded data (gitignored, see DATA.md)
│   ├── skillsets/                # Skill libraries (skills_200, skills_1000)
│   └── gos_workspace/            # Prebuilt graph workspaces
├── tests/                        # Test suite
├── agent_skills/                 # Agent bootstrap skills
├── evaluation/                   # Evaluation framework
│   ├── alfworld_run.py           # ALFWorld benchmark runner
│   ├── scienceworld_run.py       # ScienceWorld benchmark runner
│   ├── skill.py                  # Benchmark adapter for GoS
│   └── skillsbench/              # SkillsBench (Harbor-based evaluation)
├── scripts/                      # Utility scripts
├── pyproject.toml                # Package definition
├── .env.example                  # Environment template
└── DATA.md                       # Benchmark data download instructions
```

## Evaluation

The repo includes evaluation against three benchmarks: ALFWorld, ScienceWorld, and SkillsBench.

**Benchmark data** is not included in this repository. See [DATA.md](DATA.md) for download instructions.

```bash
./scripts/download_data.sh
```

For evaluation details, see [evaluation/README.md](evaluation/README.md).

## Citation

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

## License

[MIT](LICENSE)
