---
name: graph-skills-retriever
description: Retrieve a bounded bundle of relevant external skills from a prebuilt Graph of Skills workspace. Use when the task may need specialized skills, scripts, or references that are not already obvious from current context, especially in containerized eval environments that mount a prebuilt GoS graph.
inputs:
  - task description or subproblem summary
outputs:
  - a ranked skill bundle with full instructions and source paths
compatibility:
  - claude-code
  - codex
  - gemini-cli
allowed-tools:
  - shell
---

# Purpose

Use this skill instead of manually browsing a large skill library.

It assumes the environment already provides:

- the `graphskills-query` CLI
- a GoS workspace or a prebuilt workspace configured through `GOS_WORKING_DIR` and optionally `GOS_PREBUILT_WORKING_DIR`

If that wiring is missing, read `references/container-layout.md`.

# Retrieve Relevant Skills

Construct the query yourself. Do not rely on the retrieval system to infer missing task structure for you.

A good query should usually include only the retrieval-critical fields that are actually known:

- the concrete goal
- the main artifact or file format
- the key operation or algorithm
- the required library, API, protocol, or tool name if known
- the verifier-critical constraint or invariant
- the task object being edited, parsed, generated, optimized, or validated

Keep it short, but make it specific. Prefer a compact noun/verb phrase over a long paragraph.

Good patterns:

```text
update embedded xlsx in pptx and preserve formulas
parallel tfidf indexing with processpoolexecutor deterministic ranking
civ6 district adjacency exact calculator for verifier
parse branching dialogue script into graph export
```

Bad patterns:

```text
please solve this task for me
I need help with a benchmark task
fix the project and make everything work
```

Run:

```bash
graphskills-query "short specific query with goal + artifact + operation + constraint"
```

Useful flags:

```bash
graphskills-query "debug spring boot jakarta migration build errors" --top-n 5 --seed-top-k 4 --max-context-chars 9000
graphskills-query "extract text from receipts into xlsx" --json
graphskills-query "review paper references and improve the draft" --workspace /opt/graphskills/runtime
```

# How To Use The Results

1. Start with a short task-level query.
2. Read the returned bundle carefully and check the retrieval status.
3. If the result says `Retrieval Status: NO_SKILL_HIT`, explicitly state that no relevant skill was retrieved and continue on a no-skill path. Do not imply that you used a retrieved skill.
4. If the result says `Retrieval Status: SKILL_HIT`, inspect the task requirements, tests, and verifier first. Write down the minimum acceptance requirements before implementing.
5. Use the exact `Source:` paths returned by retrieval. Do not reconstruct paths from the skill name or scan the whole skill library if a `Source:` path is already available.
6. Follow the retrieved skill instructions and inspect the referenced `Source:` paths when you need scripts or references from a specific skill.
7. Use the skill bundle to narrow the solution space. Prefer the shortest path to verifier pass, and prefer adapting an existing script or interface over inventing a broader replacement.
8. Re-query with a narrower subproblem if the task shifts.

# Guidance

- Prefer 1-2 targeted retrieval calls over scanning the whole library.
- Keep the query focused on the current task or subproblem, not the whole conversation history.
- Query content priority: `goal > artifact/format > operation/API > verifier constraint`.
- Include filenames, formats, protocols, or library names when they are part of the task signal.
- Include exact invariants when they matter, e.g. `preserve formulas`, `deterministic ranking`, `exact total`, `match verifier`.
- Do not include benchmark names, generic filler, or conversation meta-text unless they are truly task-relevant.
- If the result is too broad, narrow the query and reduce `--top-n`.
- If the result is empty, retry with simpler keywords before giving up.
- If no skill is retrieved after retrying, say so explicitly and solve without pretending a skill was used.
- After a skill hit, take the shortest path to verifier pass and satisfy only the verifier's minimum requirement first.
- Use the exact `Source:` paths already returned before searching elsewhere.
- Do not add extra features, side outputs, UI panels, or refactors unless the task explicitly requires them.
- Treat retrieved skills as a constraint on implementation choices, not permission to explore more branches.
- Use `--json` only when you need structured fields like scores or edge evidence; plain text is usually enough.
