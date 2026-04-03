---
name: vector-skills-retriever
description: Retrieve a bounded bundle of relevant external skills from the local skill workspace using vector embedding similarity only.
inputs:
  - task description or subproblem summary
outputs:
  - a ranked skill bundle with full instructions and source paths under /opt/graphskills/skills
compatibility:
  - claude-code
  - codex
  - gemini-cli
allowed-tools:
  - shell
  - python3
---

# Purpose

Use this skill when the task likely needs specialized domain knowledge, scripts, or references that are not already obvious from the current context.

This retriever uses embedding similarity only. It does not use graph edges, graph propagation, or lexical expansion.

# Retrieve Relevant Skills

Run:

```bash
OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-$OPENAI_API_KEY}" \
OPENAI_API_KEY="${OPENROUTER_API_KEY:-$OPENAI_API_KEY}" \
OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
GOS_EMBEDDING_MODEL=openai/text-embedding-3-large \
GOS_EMBEDDING_DIM=3072 \
vectorskills-query "short description of the task or current subproblem"
```

Diagnostic rule: do not suppress stderr and do not replace failures with fallback text. If `vectorskills-query` fails, keep the original stderr visible and note the real exit code so the failure remains diagnosable.

Useful flags:

```bash
vectorskills-query "debug spring boot jakarta migration build errors" --top-n 5 --max-context-chars 9000
vectorskills-query "extract text from receipts into xlsx" --json
```

# How To Use The Results

1. Start with a short task-level query.
2. Read the returned skill bundle.
3. Follow the retrieved skill instructions and inspect any referenced files under `/opt/graphskills/skills/<skill-name>/`.
4. Re-query with a narrower subproblem if needed.
