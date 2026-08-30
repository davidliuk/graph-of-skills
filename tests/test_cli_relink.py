from pathlib import Path

from typer.testing import CliRunner

from gos.core.relink import RelinkProgress, RelinkResult
from gos.interfaces import cli


class FakeRelinkEngine:
    def __init__(self):
        self.calls = []

    async def async_relink_all(self, **kwargs):
        self.calls.append(kwargs)
        progress = RelinkProgress.new(
            fingerprint="sha256:test",
            total_focus_nodes=4,
            concurrency=kwargs["concurrency"],
            checkpoint_every=kwargs["checkpoint_every"],
        )
        progress.completed_focus_names = ["a", "b", "c", "d"]
        progress.persisted_edge_count = 5
        progress.checkpoint_count = 2
        progress.usage = {
            "llm": {
                "relation_validation": {
                    "calls": 4,
                    "input_tokens": 120,
                    "output_tokens": 30,
                    "cost_usd": 0.0123,
                }
            }
        }
        kwargs["progress_callback"](progress)
        return RelinkResult(
            total_focus_count=4,
            resumed_focus_count=0,
            processed_focus_count=4,
            completed_focus_count=4,
            failed_focus={},
            checkpoint_count=2,
            edge_count=5,
            elapsed_seconds=1.25,
        )


def test_relink_cli_wires_resume_concurrency_checkpoints_and_report(
    monkeypatch,
    tmp_path,
):
    engine = FakeRelinkEngine()
    report_calls = []

    monkeypatch.setattr(cli, "_build_engine", lambda **kwargs: engine)

    async def write_report(passed_engine, path):
        report_calls.append((passed_engine, path))
        path.write_text("{}\n", encoding="utf-8")
        return {}

    monkeypatch.setattr(cli, "write_construction_report", write_report)
    result = CliRunner().invoke(
        cli.app,
        [
            "relink",
            "--workspace",
            str(tmp_path),
            "--concurrency",
            "2",
            "--checkpoint-every",
            "2",
            "--resume",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "Relink checkpoint: completed=4/4 edges=5" in result.stdout
    assert "calls=4 input_tokens=120 output_tokens=30 cost=$0.0123" in result.stdout
    assert "Relink complete: completed=4/4" in result.stdout
    assert f"Relink event log: {tmp_path / 'relink_events.jsonl'}" in result.stdout
    assert engine.calls[0]["concurrency"] == 2
    assert engine.calls[0]["checkpoint_every"] == 2
    assert engine.calls[0]["resume"] is True
    assert engine.calls[0]["restart"] is False
    assert report_calls == [(engine, Path(tmp_path) / "construction_report.json")]
