__all__ = ["SkillGraphRAG", "SkillNode", "SkillEdge", "GOSSkill", "GOSGraph"]


def __getattr__(name: str):
    if name == "SkillGraphRAG":
        from .core.engine import SkillGraphRAG

        return SkillGraphRAG
    if name in {"SkillNode", "SkillEdge", "GOSSkill", "GOSGraph"}:
        from .core.schema import GOSGraph, GOSSkill, SkillEdge, SkillNode

        exports = {
            "SkillNode": SkillNode,
            "SkillEdge": SkillEdge,
            "GOSSkill": GOSSkill,
            "GOSGraph": GOSGraph,
        }
        return exports[name]
    raise AttributeError(name)
