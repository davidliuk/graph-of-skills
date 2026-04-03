from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Iterable

from fast_graphrag._types import BTNode, BTEdge, TSerializable
from pydantic import BaseModel, Field


def _split_multivalue(text: str) -> list[str]:
    if not text:
        return []
    return [part.strip() for part in text.split("\n") if part.strip()]


def _serialize_list(values: list[str]) -> str:
    return "\n".join(value.strip() for value in values if value and value.strip())


def _parse_json_list(payload: str, fallback: str = "") -> list[str]:
    if payload:
        try:
            value = json.loads(payload)
        except json.JSONDecodeError:
            value = None
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]

    return _split_multivalue(fallback)


@dataclass
class SkillNode(BTNode, TSerializable):
    description: str = ""
    one_line_capability: str = ""
    inputs: str = ""
    outputs: str = ""
    input_schema_json: str = "[]"
    output_schema_json: str = "[]"
    domain_tags: str = ""
    tooling: str = ""
    example_tasks: str = ""
    script_entrypoints: str = ""
    compatibility: str = ""
    allowed_tools: str = ""
    source_path: str = ""
    rendered_snippet: str = ""
    raw_content: str = ""
    metadata_json: str = "{}"
    skill_id: str = ""
    type: str = "Skill"

    F_TO_CONTEXT = [
        "name",
        "description",
        "one_line_capability",
        "inputs",
        "outputs",
        "domain_tags",
        "tooling",
        "example_tasks",
        "script_entrypoints",
        "compatibility",
        "allowed_tools",
        "source_path",
        "rendered_snippet",
    ]

    @property
    def input_types(self) -> list[str]:
        return _parse_json_list(self.input_schema_json, self.inputs)

    @property
    def output_types(self) -> list[str]:
        return _parse_json_list(self.output_schema_json, self.outputs)

    @property
    def domain_tags_list(self) -> list[str]:
        return _split_multivalue(self.domain_tags)

    @property
    def tooling_list(self) -> list[str]:
        return _split_multivalue(self.tooling)

    @property
    def example_tasks_list(self) -> list[str]:
        return _split_multivalue(self.example_tasks)

    @property
    def script_entrypoints_list(self) -> list[str]:
        return _split_multivalue(self.script_entrypoints)

    @property
    def compatibility_list(self) -> list[str]:
        return _split_multivalue(self.compatibility)

    @property
    def allowed_tools_list(self) -> list[str]:
        return _split_multivalue(self.allowed_tools)

    @property
    def metadata(self) -> dict[str, Any]:
        try:
            value = json.loads(self.metadata_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    def to_str(self) -> str:
        parts = [f"[{self.type}] {self.name}"]
        if self.description:
            parts.append(f"[DESCRIPTION] {self.description}")
        if self.one_line_capability:
            parts.append(f"[CAPABILITY] {self.one_line_capability}")
        if self.inputs:
            parts.append(f"[INPUTS] {self.inputs}")
        if self.outputs:
            parts.append(f"[OUTPUTS] {self.outputs}")
        if self.domain_tags:
            parts.append(f"[DOMAIN_TAGS] {self.domain_tags}")
        if self.tooling:
            parts.append(f"[TOOLING] {self.tooling}")
        if self.example_tasks:
            parts.append(f"[EXAMPLE_TASKS] {self.example_tasks}")
        if self.script_entrypoints:
            parts.append(f"[SCRIPT_ENTRYPOINTS] {self.script_entrypoints}")
        if self.compatibility:
            parts.append(f"[COMPATIBILITY] {self.compatibility}")
        if self.allowed_tools:
            parts.append(f"[ALLOWED_TOOLS] {self.allowed_tools}")
        if self.rendered_snippet:
            parts.append(f"[SNIPPET] {self.rendered_snippet}")
        return "\n".join(parts)

    def render_for_agent(self, max_chars: int | None = None) -> str:
        content = self.raw_content or self.rendered_snippet or self.to_str()
        if max_chars is not None and max_chars > 0 and len(content) > max_chars:
            content = f"{content[: max_chars - 3].rstrip()}..."

        header = [
            f"## Skill: {self.name}",
            f"Source: {self.source_path or 'inline'}",
        ]
        return "\n".join(header + [content])

    @staticmethod
    def from_lists(
        *,
        name: str,
        description: str,
        one_line_capability: str = "",
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        domain_tags: list[str] | None = None,
        tooling: list[str] | None = None,
        example_tasks: list[str] | None = None,
        script_entrypoints: list[str] | None = None,
        compatibility: list[str] | None = None,
        allowed_tools: list[str] | None = None,
        source_path: str = "",
        rendered_snippet: str = "",
        raw_content: str = "",
        metadata: dict[str, Any] | None = None,
        skill_id: str = "",
        type: str = "Skill",
    ) -> "SkillNode":
        input_values = inputs or []
        output_values = outputs or []
        domain_tag_values = domain_tags or []
        tooling_values = tooling or []
        example_task_values = example_tasks or []
        script_entrypoint_values = script_entrypoints or []
        compatibility_values = compatibility or []
        allowed_tool_values = allowed_tools or []
        metadata_value = metadata or {}

        return SkillNode(
            name=name,
            description=description,
            one_line_capability=one_line_capability,
            inputs=_serialize_list(input_values),
            outputs=_serialize_list(output_values),
            input_schema_json=json.dumps(input_values),
            output_schema_json=json.dumps(output_values),
            domain_tags=_serialize_list(domain_tag_values),
            tooling=_serialize_list(tooling_values),
            example_tasks=_serialize_list(example_task_values),
            script_entrypoints=_serialize_list(script_entrypoint_values),
            compatibility=_serialize_list(compatibility_values),
            allowed_tools=_serialize_list(allowed_tool_values),
            source_path=source_path,
            rendered_snippet=rendered_snippet,
            raw_content=raw_content,
            metadata_json=json.dumps(metadata_value),
            skill_id=skill_id or source_path or name,
            type=type,
        )

    @staticmethod
    def to_attrs(
        node: "SkillNode | None" = None,
        nodes: Iterable["SkillNode"] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if node is not None:
            return {
                "description": node.description,
                "one_line_capability": node.one_line_capability,
                "inputs": node.inputs,
                "outputs": node.outputs,
                "input_schema_json": node.input_schema_json,
                "output_schema_json": node.output_schema_json,
                "domain_tags": node.domain_tags,
                "tooling": node.tooling,
                "example_tasks": node.example_tasks,
                "script_entrypoints": node.script_entrypoints,
                "compatibility": node.compatibility,
                "allowed_tools": node.allowed_tools,
                "source_path": node.source_path,
                "rendered_snippet": node.rendered_snippet,
                "raw_content": node.raw_content,
                "metadata_json": node.metadata_json,
                "skill_id": node.skill_id,
                "type": node.type,
            }
        if nodes is not None:
            nodes_list = list(nodes)
            return {
                "description": [item.description for item in nodes_list],
                "one_line_capability": [item.one_line_capability for item in nodes_list],
                "inputs": [item.inputs for item in nodes_list],
                "outputs": [item.outputs for item in nodes_list],
                "input_schema_json": [item.input_schema_json for item in nodes_list],
                "output_schema_json": [item.output_schema_json for item in nodes_list],
                "domain_tags": [item.domain_tags for item in nodes_list],
                "tooling": [item.tooling for item in nodes_list],
                "example_tasks": [item.example_tasks for item in nodes_list],
                "script_entrypoints": [item.script_entrypoints for item in nodes_list],
                "compatibility": [item.compatibility for item in nodes_list],
                "allowed_tools": [item.allowed_tools for item in nodes_list],
                "source_path": [item.source_path for item in nodes_list],
                "rendered_snippet": [item.rendered_snippet for item in nodes_list],
                "raw_content": [item.raw_content for item in nodes_list],
                "metadata_json": [item.metadata_json for item in nodes_list],
                "skill_id": [item.skill_id for item in nodes_list],
                "type": [item.type for item in nodes_list],
            }
        return {}


@dataclass
class SkillEdge(BTEdge, TSerializable):
    description: str = ""
    type: str = "dependency"
    weight: float = 1.0
    confidence: float = 1.0
    chunks: list[Any] = field(default_factory=list)

    F_TO_CONTEXT = ["source", "target", "description", "type", "weight", "confidence"]

    @staticmethod
    def to_attrs(
        edge: "SkillEdge | None" = None,
        edges: Iterable["SkillEdge"] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if edge is not None:
            return {
                "description": edge.description,
                "type": edge.type,
                "weight": edge.weight,
                "confidence": edge.confidence,
                "chunks": edge.chunks if edge.chunks is not None else [],
            }
        if edges is not None:
            edges_list = list(edges)
            return {
                "description": [item.description for item in edges_list],
                "type": [item.type for item in edges_list],
                "weight": [item.weight for item in edges_list],
                "confidence": [item.confidence for item in edges_list],
                "chunks": [item.chunks if item.chunks is not None else [] for item in edges_list],
            }
        return {}


class GOSSkill(BaseModel):
    name: str = Field(..., description="The unique name of the skill")
    description: str = Field(..., description="Detailed description of what the skill does")
    one_line_capability: str = ""
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    domain_tags: list[str] = Field(default_factory=list)
    tooling: list[str] = Field(default_factory=list)
    example_tasks: list[str] = Field(default_factory=list)
    script_entrypoints: list[str] = Field(default_factory=list)
    compatibility: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    source_path: str = ""
    rendered_snippet: str = ""
    raw_content: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    skill_id: str = ""


class GOSRelation(BaseModel):
    source: str = Field(..., description="The name of the source skill")
    target: str = Field(..., description="The name of the target skill")
    description: str = Field(..., description="Why these two skills are related")
    type: str = Field(
        default="dependency",
        description="Type: 'dependency', 'workflow', 'semantic', or 'alternative'",
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class GOSRelationList(BaseModel):
    relations: list[GOSRelation] = Field(default_factory=list)


class GOSGraph(BaseModel):
    nodes: list[GOSSkill] = Field(default_factory=list)
    edges: list[GOSRelation] = Field(default_factory=list)


class QuerySchema(BaseModel):
    goal: str = Field(default="", description="Concise statement of the task intent")
    task_name: str = Field(default="", description="Short task slug when obvious")
    domain: list[str] = Field(default_factory=list, description="Narrow technical domains")
    operations: list[str] = Field(default_factory=list, description="Concrete operations, algorithms, or APIs")
    artifacts: list[str] = Field(default_factory=list, description="Files, formats, interfaces, or concrete objects")
    constraints: list[str] = Field(default_factory=list, description="Acceptance constraints and invariants")
    keywords: list[str] = Field(default_factory=list, description="High-value retrieval terms or short phrases")

    def to_query_text(self) -> str:
        parts: list[str] = []
        if self.goal:
            parts.append(self.goal)
        for values in (self.domain, self.operations, self.artifacts, self.constraints, self.keywords):
            if values:
                parts.extend(values)
        if self.task_name:
            parts.append(self.task_name)
        return "\n".join(part.strip() for part in parts if part and part.strip())


class RetrievalBudget(BaseModel):
    seed_top_k: int
    seed_candidate_top_k_semantic: int
    seed_candidate_top_k_lexical: int
    top_n: int
    max_chars_per_skill: int
    max_context_chars: int
    ppr_damping: float


class SkillSeed(BaseModel):
    name: str
    source_path: str = ""
    seed_weight: float
    semantic_rank: int


class RetrievedSkill(BaseModel):
    name: str
    description: str
    source_path: str = ""
    one_line_capability: str = ""
    score: float
    rerank_score: float = 0.0
    semantic_rank: int | None = None
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    domain_tags: list[str] = Field(default_factory=list)
    tooling: list[str] = Field(default_factory=list)
    example_tasks: list[str] = Field(default_factory=list)
    script_entrypoints: list[str] = Field(default_factory=list)
    compatibility: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    rendered_snippet: str = ""
    payload: str = ""


class RetrievedRelation(BaseModel):
    source: str
    target: str
    description: str
    type: str
    weight: float
    confidence: float = 1.0


class SkillRetrievalResult(BaseModel):
    query: str
    rewritten_query: QuerySchema = Field(default_factory=QuerySchema)
    budget: RetrievalBudget
    seeds: list[SkillSeed] = Field(default_factory=list)
    skills: list[RetrievedSkill] = Field(default_factory=list)
    relations: list[RetrievedRelation] = Field(default_factory=list)
    rendered_context: str = ""
    summary: str = ""


class SkillSyncResult(BaseModel):
    requested_skill_count: int
    existing_skill_count: int
    final_skill_count: int
    reused_count: int = 0
    inserted_count: int = 0
    updated_count: int = 0
    inserted_skill_names: list[str] = Field(default_factory=list)
    updated_skill_names: list[str] = Field(default_factory=list)
    prebuilt_working_dir: str = ""
