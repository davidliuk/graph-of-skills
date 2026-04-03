# Task Environment

This environment contains a prebuilt skill workspace for **vector-only retrieval**.

## Required First Step

Before attempting any task, retrieve relevant skills with a targeted query:

```bash
OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-$OPENAI_API_KEY}" \
OPENAI_API_KEY="${OPENROUTER_API_KEY:-$OPENAI_API_KEY}" \
OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
GOS_EMBEDDING_MODEL=openai/text-embedding-3-large \
GOS_EMBEDDING_DIM=3072 \
vectorskills-query "goal + artifact/format + operation/API + verifier-critical constraint"
```

This command uses embedding similarity only. It does not use graph edges, graph propagation, or lexical expansion.

Read and follow the returned skill bundle before writing any code.

Diagnostic rule: do not hide retrieval failures.

- Do not run `vectorskills-query` with `2>/dev/null`.
- Do not replace failures with fallback text like `|| echo "vectorskills-query not available"`.
- If the command fails, preserve stderr and explicitly report the real failure message and exit code in your notes.

You must read the retrieval status from that output.

- If it says `Retrieval Status: NO_SKILL_HIT`, explicitly state that no relevant skill was found and proceed without claiming skill usage.
- If it says `Retrieval Status: SKILL_HIT`, treat the retrieved skills as a narrowing device, not as permission to expand scope.

## Skill Scripts

Each retrieved skill shows a `Source:` path like:

```
Source: /opt/graphskills/skills/mesh-analysis/SKILL.md
```

The full skill directory is accessible at that path. Check whether scripts exist before re-implementing their logic from scratch.

Before implementing, inspect the task requirements, tests, and verifier and identify the minimum acceptance requirements.

Priorities:

1. Take the shortest path to passing the verifier.
2. Satisfy only the verifier's minimum required behavior first.
3. Use the exact `Source:` path returned by retrieval.
4. Reuse or adapt retrieved scripts and interfaces when they directly help.
5. Treat skills as a narrowing device, not permission to expand scope or open more implementation branches.
