# Task Environment

This environment contains a prebuilt skill workspace for **vector-only retrieval**.

## Required First Step

Before writing any code, retrieve relevant skills by running:

```bash
vectorskills-query "short description of what you need to do"
```

This command uses embedding similarity only. It does not use graph propagation or lexical expansion.

## Reading the Output

Each returned skill includes a `Source:` path, e.g.:

```
Source: /opt/graphskills/skills/mesh-analysis/SKILL.md
```

Use that exact `Source:` path. Check the parent directory's `scripts/` subdirectory before implementing logic from scratch.
