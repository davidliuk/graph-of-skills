# Task Environment

This environment provides the full local skills library, mounted on the filesystem for direct inspection.

## How To Use Skills

Read the task instruction first, inspect the workspace, then decide whether any local skill is useful.

Do not rely on `graphskills-query` or any Graph of Skills runtime command in this control setting.

Use skills only when they directly help satisfy the task verifier or tests.

Before implementing, inspect the task requirements, tests, and verifier and identify the minimum acceptance requirements.

Priorities:

1. Pass the verifier's minimum required behavior.
2. Reuse or adapt an existing local skill script or workflow when it directly helps.
3. Prefer the simplest verifier-aligned path over broader generalization.
4. Avoid extra features, UI expansion, side outputs, or refactors unless explicitly required.

## Workflow

1. Inspect the task and identify the minimum acceptance requirements.
2. Search the mounted skills library only if it is likely to help with those requirements.
3. If you use a skill, read the relevant `SKILL.md` and check whether a ready-made script exists.
4. Use or adapt that script or workflow only when it improves the verifier-aligned solution path.
5. Before finalizing, run a verifier-aligned self-check.
