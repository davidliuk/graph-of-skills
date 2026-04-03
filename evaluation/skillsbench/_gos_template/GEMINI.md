# Task Environment

This environment contains a prebuilt **Graph of Skills** (GoS) workspace.

## Required First Step

Before writing any code, retrieve relevant skills by running a shell command:

```bash
graphskills-query "short description of what you need to do"
```

Use `run_shell_command` to execute this. The command is on PATH inside the container.

## Reading the Output

Each returned skill includes a `Source:` path, e.g.:

```
Source: /opt/graphskills/skills/mesh-analysis/SKILL.md
```

Use that exact `Source:` path. Do not reconstruct a path from the skill name or scan the whole skill library if a `Source:` path is already available.

Read that file with `read_file` (path is under `/opt/graphskills/`, which is
accessible via shell commands). Check the parent directory's `scripts/` subdirectory for ready-made
tools before implementing logic from scratch.

## Path Notes

- Task data is available at both `/data/` (shell access) and `/root/data/` (file tool access)
- Write output files to `/root/output/<filename>` or use `run_shell_command` to write to `/output/<filename>`

## Workflow

1. Run `graphskills-query "<task description>"` via `run_shell_command`
2. Inspect the task tests/verifier and identify the minimum acceptance requirements
3. Read the returned skill bundle and use the exact `Source:` paths it provides
4. Check for scripts at the parent directory of each returned `Source:` path via `run_shell_command`
5. Prefer the shortest path to verifier pass; do not open extra implementation branches unless the verifier forces them
6. Execute returned scripts only when they directly help satisfy the minimum requirements
