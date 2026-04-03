---
name: graph-skills-retriever
description: Retrieve a bounded bundle of relevant external skills from the local Graph Skills library instead of manually browsing hundreds of skills.
inputs:
  - task description or subproblem summary
outputs:
  - a ranked skill bundle with full instructions and source paths under /opt/graphskills/library
compatibility:
  - claude-code
  - codex
  - gemini-cli
allowed-tools:
  - shell
  - python3
---

# Purpose

Use this skill when the task seems to require specialized domain knowledge, scripts, or references that are not already obvious from the current context.

The full skill library is stored outside the auto-loaded harness context at:

`/opt/graphskills/library`

Do not manually scan that whole directory first. Retrieve a focused bundle.

# Retrieve Relevant Skills

Run:

```bash
graphskills-query "short description of the task or current subproblem"
```

Useful flags:

```bash
graphskills-query "debug spring boot jakarta migration build errors" --top-n 5 --seed-top-k 4 --max-context-chars 9000
graphskills-query "extract text from receipts into xlsx" --json
```

# How To Use The Results

1. Start with a short task-level query.
2. Read the returned skill bundle.
3. Follow the retrieved skill instructions and inspect any referenced files in `/opt/graphskills/library/<skill-name>/`.
4. Re-query with a narrower subproblem if needed.

# Guidance

- Prefer 1-2 retrieval calls over manually searching the entire library.
- The returned `Source:` path points to the canonical `SKILL.md` in `/opt/graphskills/library`.
- If you need scripts or references from a retrieved skill, inspect that skill directory directly.
- Reuse the graph retriever whenever the task shifts to a new subproblem.
