from __future__ import annotations

from evaluation.analysis.metadata_stress import corrupt_bundle


def test_corruption_is_deterministic_non_mutating_and_mode_specific():
    bundle = {
        "metadata": {},
        "skills": [
            {
                "name": f"skill-{index}",
                "description": "specific description",
                "rendered_snippet": "specific snippet",
                "inputs": ["csv"],
                "outputs": ["report"],
            }
            for index in range(10)
        ],
        "edges": [],
    }

    first, selected = corrupt_bundle(
        bundle, mode="drop-io", fraction=0.3, salt="fixed"
    )
    second, selected_again = corrupt_bundle(
        bundle, mode="drop-io", fraction=0.3, salt="fixed"
    )

    assert selected == selected_again
    assert len(selected) == 3
    assert first == second
    assert bundle["skills"][0]["inputs"] == ["csv"]
    for skill in first["skills"]:
        if skill["name"] in selected:
            assert skill["inputs"] == []
            assert skill["outputs"] == []
        else:
            assert skill["inputs"] == ["csv"]


def test_broad_description_corruption_preserves_io_fields():
    bundle = {
        "metadata": {},
        "skills": [
            {
                "name": "skill",
                "description": "specific",
                "rendered_snippet": "specific",
                "inputs": ["csv"],
                "outputs": ["report"],
            }
        ],
        "edges": [],
    }

    corrupted, _ = corrupt_bundle(
        bundle, mode="broad-description", fraction=1.0, salt="fixed"
    )

    assert "general-purpose" in corrupted["skills"][0]["description"].lower()
    assert corrupted["skills"][0]["inputs"] == ["csv"]
