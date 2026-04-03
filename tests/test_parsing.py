import tempfile
import unittest

from gos.core.parsing import parse_skill_document


SAMPLE_SKILL = """---
name: analyze_csv
description: Analyze a CSV file and summarize trends.
compatibility:
  - codex
allowed-tools:
  - python
metadata:
  owner: research
---
# Usage

Load a CSV file, compute summary stats, and describe the trend.

## Inputs
- csv_path: str

## Outputs
- analysis_report: markdown
"""


class ParseSkillDocumentTest(unittest.TestCase):
    def test_parses_frontmatter_and_body_sections(self) -> None:
        document = parse_skill_document(
            SAMPLE_SKILL,
            source_path="/tmp/skills/analyze_csv/SKILL.md",
            snippet_chars=240,
        )

        self.assertIsNotNone(document)
        assert document is not None
        self.assertEqual(document.name, "analyze_csv")
        self.assertEqual(document.description, "Analyze a CSV file and summarize trends.")
        self.assertEqual(document.inputs, ["csv_path: str"])
        self.assertEqual(document.outputs, ["analysis_report: markdown"])
        self.assertEqual(document.compatibility, ["codex"])
        self.assertEqual(document.allowed_tools, ["python"])
        self.assertEqual(document.metadata["owner"], "research")
        self.assertIn("Load a CSV file", document.rendered_snippet)


    def test_parses_richer_metadata_and_script_entrypoints(self) -> None:
        skill = """---
name: tfidf_parallel_search
description: Parallel TF-IDF indexing for deterministic ranked search.
summary: Build and query a multiprocessing TF-IDF index.
domain:
  - information retrieval
tags:
  - tfidf
  - processpoolexecutor
libraries:
  - sklearn
metadata:
  example_tasks:
    - parallelize tfidf batch search
---
# Examples
- preserve deterministic ranking across workers
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            from pathlib import Path

            skill_dir = Path(tmpdir) / "tfidf_parallel_search"
            scripts_dir = skill_dir / "scripts"
            scripts_dir.mkdir(parents=True)
            (scripts_dir / "run.py").write_text("print('ok')\n", encoding="utf-8")

            document = parse_skill_document(
                skill,
                source_path=str(skill_dir / "SKILL.md"),
                snippet_chars=240,
            )

            self.assertIsNotNone(document)
            assert document is not None
            self.assertEqual(document.one_line_capability, "Build and query a multiprocessing TF-IDF index.")
            self.assertIn("information retrieval", document.domain_tags)
            self.assertIn("sklearn", document.tooling)
            self.assertIn("parallelize tfidf batch search", document.example_tasks)
            self.assertIn("scripts/run.py", document.script_entrypoints)


if __name__ == "__main__":
    unittest.main()
