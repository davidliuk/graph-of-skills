# Task Environment

This environment contains a prebuilt **Graph of Skills** (GoS) workspace.

## Required First Step

Before writing any code, retrieve relevant skills by running a targeted query:

```bash
graphskills-query "goal + artifact/format + operation/API + verifier-critical constraint"
```

This command is available on PATH. It queries a prebuilt skill graph and returns
a ranked bundle of skills relevant to your task.

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

You must read the retrieval status from that result.

- If it says `Retrieval Status: NO_SKILL_HIT`, explicitly note that no relevant skill was found and continue without claiming skill usage.
- If it says `Retrieval Status: SKILL_HIT`, use the retrieved skills only as constraints on how to solve the task.

## Reading the Output

Each returned skill includes a `Source:` path, e.g.:

```
Source: /opt/graphskills/skills/mesh-analysis/SKILL.md
```

The full skill directory (including `scripts/` and `references/`) is at that path.
For example: `/opt/graphskills/skills/mesh-analysis/scripts/mesh_tool.py`

**Always check whether a ready-made script exists before implementing from scratch.**

Before implementing, inspect the task requirements, tests, and verifier and identify the minimum acceptance requirements.

Priorities:

1. Take the shortest path to passing the verifier.
2. Pass only the verifier's minimum required behavior first.
3. Use the exact `Source:` path returned by retrieval. Do not reconstruct paths from the skill name or scan the whole skill library if a `Source:` path is already available.
4. Reuse or adapt retrieved scripts and interfaces when they directly fit the verifier target.
5. If a retrieved skill contains an authoritative calculator, validator, parser, or pack/unpack workflow, use that exact interface for the final output and final self-check.
6. Treat skills as a way to shrink the search space, not as permission to explore more implementation branches.
7. Avoid extra features, UI expansion, side outputs, or generalization unless explicitly required.

## Workflow

1. Inspect the task and form a short retrieval query containing `goal + artifact/format + operation/API + verifier-critical constraint`
2. Run `graphskills-query "<targeted query>"`
3. Record whether retrieval was a skill hit or no-hit
4. Inspect task requirements/tests/verifier and write down the minimum acceptance requirements
5. Read the returned skill bundle and use the exact `Source:` paths it provides
6. Check for scripts at the parent directory of each returned `Source:` path
7. Use or adapt those scripts only if they directly help satisfy the minimum requirements
8. If no returned script/interface directly fits, stay on the shortest no-frills path to verifier pass
9. Before finalizing, run one verifier-aligned self-check with the retrieved skill's authoritative interface when available
