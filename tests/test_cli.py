from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from gos.core.engine import SkillGraphRAG
from gos.interfaces import cli


class FakeEmbeddingService:
    def __init__(self, embedding_dim: int = 16) -> None:
        self.embedding_dim = embedding_dim
        self.model = "fake-embedding"

    async def encode(self, texts, model=None):
        vectors = []
        for text in texts:
            vector = np.zeros(self.embedding_dim, dtype=np.float32)
            for token in text.lower().split():
                vector[hash(token) % self.embedding_dim] += 1.0
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            vectors.append(vector)
        return np.vstack(vectors)


class FakeLLMService:
    model = "fake-llm"

    async def send_message(
        self,
        prompt,
        system_prompt=None,
        history_messages=None,
        response_model=None,
        **kwargs,
    ):
        if response_model is None:
            return "", []
        return response_model(), []


SKILLS = [
    (
        """---
name: read_csv
description: Read a CSV file into a dataset.
inputs:
  - csv_path
outputs:
  - dataset
---
# Usage
Load a CSV file.
""",
        "read_csv/SKILL.md",
    ),
    (
        """---
name: analyze_trend
description: Analyze a dataset and produce a trend report.
inputs:
  - dataset
outputs:
  - trend_report
---
# Usage
Analyze a dataset.
""",
        "analyze_trend/SKILL.md",
    ),
]


def _build_test_engine(
    *,
    workspace: Path | None = None,
    retrieval_top_n: int | None = None,
    seed_top_k: int | None = None,
    max_skill_chars: int | None = None,
    max_context_chars: int | None = None,
    enable_semantic_linking: bool | None = None,
) -> SkillGraphRAG:
    return SkillGraphRAG(
        config=SkillGraphRAG.Config(
            llm_service=FakeLLMService(),
            embedding_service=FakeEmbeddingService(),
            working_dir=str(workspace or Path(cli.settings.WORKING_DIR)),
            prebuilt_working_dir=cli.settings.PREBUILT_WORKING_DIR,
            use_full_markdown=False,
            enable_semantic_linking=(
                False if enable_semantic_linking is None else enable_semantic_linking
            ),
            retrieval_top_n=(
                retrieval_top_n if retrieval_top_n is not None else cli.settings.RETRIEVAL_TOP_N
            ),
            seed_top_k=seed_top_k if seed_top_k is not None else cli.settings.SEED_TOP_K,
            max_skill_chars=(
                max_skill_chars if max_skill_chars is not None else cli.settings.MAX_SKILL_CHARS
            ),
            max_context_chars=(
                max_context_chars
                if max_context_chars is not None
                else cli.settings.MAX_CONTEXT_CHARS
            ),
        )
    )


def _write_skill(base_dir: Path, relative_path: str, content: str) -> Path:
    skill_path = base_dir / relative_path
    skill_path.parent.mkdir(parents=True, exist_ok=True)
    skill_path.write_text(content, encoding="utf-8")
    return skill_path


def test_graphskills_query_cli_reads_prebuilt_workspace(monkeypatch):
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as tmpdir:
        source_workspace = Path(tmpdir) / "source_workspace"
        target_workspace = Path(tmpdir) / "target_workspace"

        engine = SkillGraphRAG(
            config=SkillGraphRAG.Config(
                llm_service=FakeLLMService(),
                embedding_service=FakeEmbeddingService(),
                working_dir=str(source_workspace),
                use_full_markdown=False,
                enable_semantic_linking=False,
            )
        )

        contents = [content for content, _ in SKILLS]
        metadatas = [
            {
                "source_path": str(Path(tmpdir) / relative_path),
                "raw_content": content,
            }
            for content, relative_path in SKILLS
        ]

        import asyncio

        asyncio.run(engine.async_insert_skills(contents, metadatas))

        monkeypatch.setattr(cli.settings, "PREBUILT_WORKING_DIR", str(source_workspace))
        monkeypatch.setattr(cli.settings, "WORKING_DIR", str(target_workspace))

        result = runner.invoke(
            cli.graphskills_query_app,
            [
                "analyze csv trends",
                "--workspace",
                str(target_workspace),
            ],
        )

        assert result.exit_code == 0
        assert "## Skill: analyze_trend" in result.stdout


def test_gos_add_command_persists_incremental_graph_updates(monkeypatch):
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir) / "workspace"
        library = Path(tmpdir) / "library"
        skill_paths = [
            _write_skill(library, relative_path, content)
            for content, relative_path in SKILLS
        ]

        monkeypatch.setattr(cli, "_build_engine", _build_test_engine)
        monkeypatch.setattr(cli.settings, "WORKING_DIR", str(workspace))
        monkeypatch.setattr(cli.settings, "PREBUILT_WORKING_DIR", None)

        first_add = runner.invoke(
            cli.app,
            [
                "add",
                str(skill_paths[0]),
                "--workspace",
                str(workspace),
            ],
        )

        assert first_add.exit_code == 0
        assert "inserted=1" in first_add.stdout
        assert "final_skills=1" in first_add.stdout
        assert (workspace / "graph_igraph_data.pklz").exists()
        assert (workspace / "entities_hnsw_index_16.bin").exists()

        second_add = runner.invoke(
            cli.app,
            [
                "add",
                str(skill_paths[1]),
                "--workspace",
                str(workspace),
            ],
        )

        assert second_add.exit_code == 0
        assert "inserted=1" in second_add.stdout
        assert "final_skills=2" in second_add.stdout

        query_result = runner.invoke(
            cli.app,
            [
                "query",
                "analyze csv trends",
                "--raw",
                "--workspace",
                str(workspace),
            ],
        )

        assert query_result.exit_code == 0
        assert "## Skill: analyze_trend" in query_result.stdout


def test_query_output_includes_rewritten_query_summary(monkeypatch):
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir) / "workspace"

        engine = SkillGraphRAG(
            config=SkillGraphRAG.Config(
                llm_service=FakeLLMService(),
                embedding_service=FakeEmbeddingService(),
                working_dir=str(workspace),
                use_full_markdown=False,
                enable_semantic_linking=False,
            )
        )

        contents = [content for content, _ in SKILLS]
        metadatas = [
            {
                "source_path": str(Path(tmpdir) / relative_path),
                "raw_content": content,
            }
            for content, relative_path in SKILLS
        ]

        import asyncio

        asyncio.run(engine.async_insert_skills(contents, metadatas))

        monkeypatch.setattr(cli.settings, "PREBUILT_WORKING_DIR", None)
        monkeypatch.setattr(cli.settings, "WORKING_DIR", str(workspace))

        result = runner.invoke(
            cli.app,
            [
                "query",
                "analyze csv trends and make a chart in python",
                "--workspace",
                str(workspace),
            ],
        )

        assert result.exit_code == 0
        assert "### Rewritten Query" in result.stdout
        assert "task_name:" in result.stdout


def test_gos_retrieve_command_returns_rendered_context(monkeypatch):
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir) / "workspace"

        engine = SkillGraphRAG(
            config=SkillGraphRAG.Config(
                llm_service=FakeLLMService(),
                embedding_service=FakeEmbeddingService(),
                working_dir=str(workspace),
                use_full_markdown=False,
                enable_semantic_linking=False,
            )
        )

        contents = [content for content, _ in SKILLS]
        metadatas = [
            {
                "source_path": str(Path(tmpdir) / relative_path),
                "raw_content": content,
            }
            for content, relative_path in SKILLS
        ]

        import asyncio

        asyncio.run(engine.async_insert_skills(contents, metadatas))

        monkeypatch.setattr(cli.settings, "WORKING_DIR", str(workspace))
        monkeypatch.setattr(cli.settings, "PREBUILT_WORKING_DIR", None)

        result = runner.invoke(
            cli.app,
            [
                "retrieve",
                "analyze csv trends",
                "--workspace",
                str(workspace),
            ],
        )

        assert result.exit_code == 0
        assert "## Skill: analyze_trend" in result.stdout
