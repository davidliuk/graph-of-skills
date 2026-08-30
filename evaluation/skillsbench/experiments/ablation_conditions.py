from __future__ import annotations

from typing import NamedTuple


class RetrievalCondition(NamedTuple):
    name: str
    seed_mode: str
    propagation_mode: str
    reverse_mode: str

    @property
    def requires_vector_store(self) -> bool:
        return self.seed_mode in {"embedding", "hybrid"}

    def environment(self) -> dict[str, str]:
        return {
            "GOS_LIGHT_SEED_MODE": self.seed_mode,
            "GOS_LIGHT_PROPAGATION_MODE": self.propagation_mode,
            "GOS_LIGHT_REVERSE_MODE": self.reverse_mode,
        }


ABLATION_CONDITIONS = {
    condition.name: condition
    for condition in (
        RetrievalCondition("lexical-reverse-ppr", "lexical", "ppr", "full"),
        RetrievalCondition("lexical-forward-ppr", "lexical", "ppr", "none"),
        RetrievalCondition("lexical-no-graph", "lexical", "none", "full"),
        RetrievalCondition("lexical-one-hop", "lexical", "one-hop", "full"),
    )
}


def get_condition(name: str) -> RetrievalCondition:
    try:
        return ABLATION_CONDITIONS[name]
    except KeyError as exc:
        available = ", ".join(ABLATION_CONDITIONS)
        raise ValueError(f"Unknown retrieval condition {name!r}; choose one of: {available}") from exc


def render_condition_environment(
    compose_text: str,
    condition: RetrievalCondition,
) -> str:
    rendered = compose_text
    for key, value in condition.environment().items():
        passthrough = f"- {key}"
        pinned = f"- {key}={value}"
        if passthrough not in rendered:
            raise ValueError(f"Compose template is missing retrieval setting {key}")
        rendered = rendered.replace(passthrough, pinned)
    return rendered
