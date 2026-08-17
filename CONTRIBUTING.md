# Contributing

Thank you for your interest in contributing to Graph of Skills!

## Getting Started

```bash
git clone https://github.com/davidliuk/graph-of-skills.git
cd graph-of-skills
uv sync
cp .env.example .env    # fill in your API keys
uv run pytest           # verify everything works
```

## Development Workflow

1. **Create a branch** from `main` for your change.
2. **Make your changes** with tests.
3. **Run checks** before pushing:

```bash
uv run ruff check gos/ tests/     # lint
uv run black gos/ tests/          # format
uv run pytest                      # test
```

4. **Open a pull request** with a clear description of what changed and why.

## Project Structure

| Directory | Contents |
|-----------|----------|
| `gos/core/` | Retrieval engine, graph construction, parsing, schema |
| `gos/interfaces/` | CLI (`typer`) and MCP server entry points |
| `gos/utils/` | Configuration (`pydantic-settings`) and helpers |
| `evaluation/` | Benchmark runners (ALFWorld, SkillsBench) |
| `tests/` | Unit and integration tests |

## Adding a New Skill

Skill documents follow a standard Markdown format. Place your `SKILL.md` file in a skill set directory and rebuild the workspace:

```bash
uv run gos add path/to/MY_SKILL.md --workspace data/gos_workspace/skills_1000_v1
```

## Reporting Issues

Please open a GitHub issue with:

- A short description of the problem
- Steps to reproduce (commands, input data)
- Expected vs. actual behavior
- Environment: Python version, OS, embedding provider

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
