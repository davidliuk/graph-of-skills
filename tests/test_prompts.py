from gos.core.prompts import PROMPTS


def test_search_and_link_prompt_preserves_json_example():
    prompt = PROMPTS["search_and_link_prompt"].format(
        new_skill="skill-a",
        candidate_skills="- skill-b: example",
    )

    assert '{"relations": []}' in prompt


def test_skill_extraction_prompt_is_minimal_retrieval_schema():
    system_prompt = PROMPTS["skill_extraction_system"]

    assert "one_line_capability" in system_prompt
    assert "domain_tags" in system_prompt
    assert "tooling" in system_prompt
    assert "example_tasks" in system_prompt
    assert "script_entrypoints" not in system_prompt
    assert "allowed_tools" not in system_prompt
