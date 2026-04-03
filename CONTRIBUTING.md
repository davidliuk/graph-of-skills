# Contributing to Graph of Skills

Thank you for your interest in contributing to Graph of Skills.

## Getting Started

1. Fork and clone the repository.
2. Install dependencies:
   ```bash
   uv sync
   cp .env.example .env
   ```
3. Run the test suite:
   ```bash
   uv run pytest
   ```

## Development

### Code Style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and [black](https://black.readthedocs.io/) for formatting:

```bash
uv run ruff check gos/ tests/
uv run black gos/ tests/
```

### Testing

Add tests for new functionality under `tests/`. Run the full suite with:

```bash
uv run pytest
```

### Project Structure

- `gos/core/` -- retrieval engine, parsing, schema, graph logic
- `gos/interfaces/` -- CLI and MCP server entry points
- `gos/utils/` -- configuration and helpers
- `evaluation/` -- benchmark runners and experiment framework

## Submitting Changes

1. Create a branch for your changes.
2. Ensure tests pass and code style is clean.
3. Open a pull request with a clear description of the change.

## Reporting Issues

Open a GitHub issue with:

- A concise description of the problem
- Steps to reproduce
- Expected vs. actual behavior
- Environment details (Python version, OS)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
