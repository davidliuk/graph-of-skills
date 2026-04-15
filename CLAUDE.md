# Graph of Skills — Claude Code Integration

GoS ships with a built-in [MCP](https://modelcontextprotocol.io/) server that gives Claude Code direct access to the skill graph — search, retrieve, explore, and manage skills without leaving the conversation.

## Setup

### Prerequisites

1. **Install dependencies:**

   ```bash
   uv sync
   ```

2. **Configure API keys** — copy `.env.example` to `.env` and fill in your embedding provider (OpenAI, Gemini, or Azure):

   ```bash
   cp .env.example .env
   # Edit .env: set OPENAI_API_KEY or GEMINI_API_KEY
   ```

3. **Get a workspace** — either download prebuilt or build from source:

   ```bash
   # Option A: Download prebuilt workspaces
   ./scripts/download_data.sh --workspace

   # Option B: Build locally (needs embedding API)
   ./scripts/download_data.sh --skillsets
   uv run gos index data/skillsets/skills_200 \
     --workspace data/gos_workspace/skills_200_v1 --clear
   ```

### Auto-Discovery (Recommended)

The `.mcp.json` at the project root auto-registers the MCP server. **Open the project directory in Claude Code** — the `graph-of-skills` server is immediately available, no extra steps.

Verify with:

```bash
claude mcp list
```

You should see `graph-of-skills` in the output.

### Manual Registration

If auto-discovery doesn't work (e.g. you're working from a different directory), register manually:

```bash
# Basic registration
claude mcp add graph-of-skills -- uv run --directory /path/to/graph-of-skills gos-claude

# With an explicit workspace
claude mcp add graph-of-skills -- uv run --directory /path/to/graph-of-skills gos-claude \
  --workspace /path/to/graph-of-skills/data/gos_workspace/skills_200_v1
```

### Switching Workspaces

The default workspace is set in `.mcp.json`. To use a different skill library size, edit the `--workspace` argument:

```jsonc
// .mcp.json
{
  "mcpServers": {
    "graph-of-skills": {
      "command": "uv",
      "args": ["run", "--directory", ".", "gos-claude",
               "--workspace", "data/gos_workspace/skills_2000_v1"]
    }
  }
}
```

Available prebuilt workspaces: `skills_200_v1`, `skills_500_v1`, `skills_1000_v1`, `skills_2000_v1`.

## Available MCP Tools

### Retrieval

| Tool | Purpose |
|------|---------|
| `search_skills` | Quick ranked summary — skill names, scores, graph relations. Use for a fast overview before deciding which skills to load. |
| `retrieve_skill_bundle` | Full agent-ready skill content: SKILL.md bodies, source paths, script entrypoints, graph evidence. **This is the primary tool.** |
| `hydrate_skills` | Load specific skills by exact name (comma-separated). Use when you already know which skills you need. |

### Exploration

| Tool | Purpose |
|------|---------|
| `get_status` | Workspace stats: skill count, edge count, embedding model, retrieval config. |
| `list_skills` | Browse every indexed skill — name, description, source path, domain tags. |
| `get_skill_detail` | Full metadata for one skill: I/O schema, domain tags, tooling, scripts, and all graph neighbors. |
| `get_skill_neighbors` | Incoming/outgoing edges for a skill — dependency, workflow, semantic, alternative relationships. |

### Management

| Tool | Purpose |
|------|---------|
| `index_skills` | Build the skill graph from a directory of SKILL.md files. Set `clear=true` to rebuild from scratch. |
| `add_skill` | Incrementally add new SKILL.md files to an existing graph without rebuilding. |

## Example Workflows

### 1. Find and Use Skills for a Task

Just describe what you need in natural language:

> "Find skills for processing 3D mesh files with GoS, then follow the skill instructions to complete the task."

Claude Code will:

1. Call `get_status` → check workspace (200 skills, 126 edges)
2. Call `search_skills("3D mesh")` → find `mesh-analysis`, `obj-exporter`, `threejs`
3. Call `retrieve_skill_bundle("3D mesh")` → get full SKILL.md content with scripts
4. Read the skill's script (`mesh_tool.py`), find task data, execute the solution

**Real output from a test run (1 min 16 sec):**

```
GoS retrieval — skills_200_v1 (200 skills, 126 edges), 3 relevant skills retrieved:

┌───────────────┬──────────────────────────────────────────┬───────┐
│     Skill     │               Purpose                    │ Score │
├───────────────┼──────────────────────────────────────────┼───────┤
│ mesh-analysis │ STL parsing, volume calc, component anal │ 0.146 │
│ obj-exporter  │ Three.js → OBJ format export             │ 0.613 │
│ threejs       │ Scene graph parsing & URDF export         │ 0.044 │
└───────────────┴──────────────────────────────────────────┴───────┘

Executed following mesh-analysis skill instructions:
1. Parse — MeshAnalyzer reads Binary STL, extracts 53 connected components
2. Filter — analyze_largest_component() isolates the main body
3. Extract — attribute bytes yield Material ID = 42
4. Compute — 6242.89 cm³ × 5.55 g/cm³ = 34648.04 g

Output → mass_report.json
```

### 2. Browse the Skill Library

> "List all skills related to audio processing."

Claude Code calls `list_skills` then filters by domain. Or more directly:

> "Search for FFmpeg-related skills and tell me what's available."

### 3. Explore Skill Relationships

> "What are the dependencies and related skills for mesh-analysis?"

Claude Code calls `get_skill_neighbors("mesh-analysis")` and `get_skill_detail("mesh-analysis")` to show the full graph context.

### 4. Index Your Own Skills

> "Index all skills from ~/my-skills/ into the GoS graph."

Claude Code calls `index_skills(path="~/my-skills/", clear=true)` to build a new workspace.

### 5. Add a New Skill Incrementally

> "Add this new SKILL.md to the graph."

Claude Code calls `add_skill(path="path/to/new/SKILL.md")`.

## Configuration

All settings use the `GOS_` prefix and are read from `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `GOS_WORKING_DIR` | `./gos_workspace` | Default workspace path |
| `GOS_EMBEDDING_MODEL` | `openai/text-embedding-3-large` | Embedding model (must match workspace) |
| `GOS_EMBEDDING_DIM` | `3072` | Embedding dimension |
| `GOS_LLM_MODEL` | `openrouter/google/gemini-2.5-flash` | LLM for extraction/linking |
| `GOS_RETRIEVAL_TOP_N` | `8` | Max skills returned per query |
| `GOS_SEED_TOP_K` | `5` | Initial seed count before PPR expansion |
| `GOS_MAX_SKILL_CHARS` | `2400` | Character budget per skill |
| `GOS_MAX_CONTEXT_CHARS` | `12000` | Total character budget for the skill bundle |

> **Note:** The embedding model at retrieval time **must match** the model used when the workspace was indexed.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `graph-of-skills` not in `claude mcp list` | Run `claude mcp add` manually (see [Manual Registration](#manual-registration)) |
| "0 skills" on first query | Check `--workspace` in `.mcp.json` points to a valid prebuilt workspace |
| Embedding dimension mismatch | Ensure `GOS_EMBEDDING_MODEL` / `GOS_EMBEDDING_DIM` in `.env` match the workspace |
| "LLM service is not configured" | Set the correct API key in `.env` (`OPENAI_API_KEY`, `GEMINI_API_KEY`, etc.) |
| Slow first query | The first query loads the graph into memory; subsequent queries are faster |
