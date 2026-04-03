from pathlib import Path

from gos.experiments import get_experiment_preset, resolve_preset_documents


def test_research_subset_preset_resolves_expected_skill_documents():
    preset = get_experiment_preset("research-subset")

    documents = resolve_preset_documents(
        preset,
        base_dir=Path(__file__).resolve().parent.parent,
    )

    assert len(documents) == 12
    parsed_names = {parsed.name for _, _, _, parsed in documents}
    assert "academic-researcher" in parsed_names
    assert "inno-paper-reviewer" in parsed_names
    assert "scientific-writing" in parsed_names
