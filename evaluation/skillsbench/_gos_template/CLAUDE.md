# Task Environment

This environment contains a prebuilt **Graph of Skills** (GoS) workspace.

## Required First Step

Before attempting any task, retrieve relevant skills with a targeted query:

```bash
graphskills-query "goal + artifact/format + operation/API + verifier-critical constraint"
```

The `/graph-skills-retriever` skill explains how to use this command in detail.

When writing the query, include only the retrieval-critical task facts that are actually known:

- concrete goal
- artifact, file format, or main object
- key operation, algorithm, API, or library
- verifier-critical constraint or invariant

Examples:

```text
update embedded xlsx in pptx preserve formulas
parallel tfidf search processpoolexecutor deterministic ranking
exact civ6 district adjacency calculator
```

Avoid vague queries such as `solve this task` or `help with benchmark`.

Read and follow the returned skill bundle before writing any code.

You must read the retrieval status from that output.

- If it says `Retrieval Status: NO_SKILL_HIT`, explicitly state that no relevant skill was found and proceed without claiming skill usage.
- If it says `Retrieval Status: SKILL_HIT`, treat the retrieved skills as a narrowing device, not as permission to expand scope.

## Skill Scripts

Each retrieved skill shows a `Source:` path like:

```
Source: /opt/graphskills/skills/mesh-analysis/SKILL.md
```

The full skill directory (including `scripts/`, `references/`) is accessible at that path.
For example: `/opt/graphskills/skills/mesh-analysis/scripts/mesh_tool.py`

Check whether scripts exist before re-implementing their logic from scratch.

Before implementing, inspect the task requirements, tests, and verifier and identify the minimum acceptance requirements.

Priorities:

1. Take the shortest path to passing the verifier.
2. Satisfy only the verifier's minimum required behavior first.
3. Use the exact `Source:` path returned by retrieval. Do not reconstruct paths from the skill name or scan the whole skill library if a `Source:` path is already available.
4. Reuse or adapt retrieved scripts and interfaces when they directly help.
5. If a retrieved skill contains an authoritative calculator, validator, parser, or pack/unpack workflow, use that exact interface for the final output and final self-check.
6. Treat skills as a narrowing device, not permission to expand scope or open more implementation branches.
7. Do not add extra features, panels, outputs, or refactors unless the task explicitly requires them.
