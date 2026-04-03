# Task Environment

This environment contains a prebuilt skill workspace for **vector-only retrieval**.

## Required First Step

Before writing any code, retrieve relevant skills with a targeted query:

```bash
OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-$OPENAI_API_KEY}" \
OPENAI_API_KEY="${OPENROUTER_API_KEY:-$OPENAI_API_KEY}" \
OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
GOS_EMBEDDING_MODEL=openai/text-embedding-3-large \
GOS_EMBEDDING_DIM=3072 \
vectorskills-query "goal + artifact/format + operation/API + verifier-critical constraint"
```

This command uses embedding similarity only. It does not use graph edges, graph propagation, or lexical expansion.

Diagnostic rule: do not hide retrieval failures.

- Do not run `vectorskills-query` with `2>/dev/null`.
- Do not replace failures with fallback text like `|| echo "vectorskills-query not available"`.
- If the command fails, preserve stderr and explicitly report the real failure message and exit code in your notes.

When writing the query, include only the retrieval-critical task facts that are actually known:

- concrete goal
- artifact, file format, or main object
- key operation, algorithm, API, or library
- verifier-critical constraint or invariant

Avoid vague queries such as `solve this task` or `help with benchmark`.

You must read the retrieval status from that result.

- If it says `Retrieval Status: NO_SKILL_HIT`, explicitly note that no relevant skill was found and continue without claiming skill usage.
- If it says `Retrieval Status: SKILL_HIT`, use the retrieved skills only as constraints on how to solve the task.

## Reading the Output

Each returned skill includes a `Source:` path, e.g.:

```
Source: /opt/graphskills/skills/mesh-analysis/SKILL.md
```

The full skill directory is accessible from that path. Check for ready-made `scripts/` before implementing from scratch.

Before implementing, inspect the task requirements, tests, and verifier and identify the minimum acceptance requirements.

Priorities:

1. Take the shortest path to passing the verifier.
2. Pass only the verifier's minimum required behavior first.
3. Use the exact `Source:` path returned by retrieval.
4. Reuse or adapt retrieved scripts and interfaces when they directly fit the verifier target.
5. Treat retrieved skills as a way to shrink the search space, not permission to expand scope.
