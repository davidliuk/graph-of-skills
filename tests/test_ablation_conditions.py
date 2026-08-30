from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
CONDITIONS_PATH = (
    REPO_ROOT / "evaluation" / "skillsbench" / "experiments" / "ablation_conditions.py"
)
GENERATOR_PATH = REPO_ROOT / "evaluation" / "skillsbench" / "graphskills_benchmark.py"


def _load_conditions():
    spec = importlib.util.spec_from_file_location(
        "ablation_conditions", CONDITIONS_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_generator():
    spec = importlib.util.spec_from_file_location("condition_generator", GENERATOR_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ablation_registry_has_four_matched_lexical_conditions():
    conditions = _load_conditions()

    assert set(conditions.ABLATION_CONDITIONS) == {
        "lexical-reverse-ppr",
        "lexical-forward-ppr",
        "lexical-no-graph",
        "lexical-one-hop",
    }
    assert {item.seed_mode for item in conditions.ABLATION_CONDITIONS.values()} == {
        "lexical"
    }
    assert not any(
        item.requires_vector_store for item in conditions.ABLATION_CONDITIONS.values()
    )


def test_condition_environment_pins_all_retrieval_modes():
    conditions = _load_conditions()

    environment = conditions.get_condition("lexical-forward-ppr").environment()

    assert environment == {
        "GOS_LIGHT_SEED_MODE": "lexical",
        "GOS_LIGHT_PROPAGATION_MODE": "ppr",
        "GOS_LIGHT_REVERSE_MODE": "none",
    }


def test_render_condition_environment_replaces_passthrough_entries():
    conditions = _load_conditions()
    compose = """services:
  main:
    environment:
      - GOS_LIGHT_SEED_MODE
      - GOS_LIGHT_PROPAGATION_MODE
      - GOS_LIGHT_REVERSE_MODE
"""

    rendered = conditions.render_condition_environment(
        compose,
        conditions.get_condition("lexical-one-hop"),
    )

    assert "- GOS_LIGHT_SEED_MODE=lexical" in rendered
    assert "- GOS_LIGHT_PROPAGATION_MODE=one-hop" in rendered
    assert "- GOS_LIGHT_REVERSE_MODE=full" in rendered


def test_graphskills_compose_render_pins_selected_condition(tmp_path):
    generator = _load_generator()
    env_dir = tmp_path / "task" / "environment"
    env_dir.mkdir(parents=True)
    skills_root = tmp_path / "skills"
    skills_root.mkdir()

    rendered = generator.render_compose_template(
        generator.GOS_TEMPLATE_DIR / "docker-compose.yaml",
        destination_env_dir=env_dir,
        skills_root=skills_root,
        retrieval_condition="lexical-forward-ppr",
    )

    assert "- GOS_LIGHT_SEED_MODE=lexical" in rendered
    assert "- GOS_LIGHT_PROPAGATION_MODE=ppr" in rendered
    assert "- GOS_LIGHT_REVERSE_MODE=none" in rendered
