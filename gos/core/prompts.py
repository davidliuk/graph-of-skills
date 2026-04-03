from typing import Any, Dict

PROMPTS: Dict[str, Any] = {}

PROMPTS["skill_extraction_system"] = """# DOMAIN
{domain}

# GOAL
You are normalizing a single agent skill document into structured fields.
The input document is usually a `SKILL.md` file with YAML frontmatter and a markdown body.

# INSTRUCTIONS
1. Extract exactly one skill node from the document.
2. Preserve the canonical `name` and `description` from the document if they are provided.
3. Infer only the retrieval-critical fields from the full markdown body: `one_line_capability`, `inputs`, `outputs`, `domain_tags`, `tooling`, `example_tasks`.
4. Prefer semantic understanding of the task and implementation details over copying raw lines.
5. Do not include filesystem-only or bookkeeping fields in the semantic extraction; those are handled outside the model.
6. Use high precision. If a field is uncertain, leave it empty.
7. Do not invent relationships during this phase. Return an empty `edges` list.

# OUTPUT FORMAT
Return strictly valid JSON with:
- `nodes`: a single normalized skill object containing only `name`, `description`, `one_line_capability`, `inputs`, `outputs`, `domain_tags`, `tooling`, `example_tasks`
- `edges`: always an empty list
"""

PROMPTS["skill_extraction_prompt"] = """**STRICT JSON RULES**:
- No markdown fences.
- No trailing commas.
- Double quotes only for strings.

# INPUT
{input_text}

OUTPUT:
"""

PROMPTS["search_and_link_system"] = """You are validating candidate relationships for a skill graph.

# RELATIONSHIP TYPES
- `dependency`: Skill A produces something Skill B consumes.
- `workflow`: Skill A and Skill B are commonly chained in a concrete multi-step workflow.
- `semantic`: Skill A and Skill B are in the same narrow capability cluster.
- `alternative`: Skill A and Skill B solve the same task via different implementations.

# RULES
- Prefer sparse, high-precision edges.
- Only emit an edge when the evidence is explicit or strongly implied.
- Dependency edges are preferred over semantic edges when I/O compatibility exists.
- If uncertain, emit no edge.
- Do not try to make the graph dense.
- `source` and `target` must match the skill names exactly.

# OUTPUT FORMAT
Return valid JSON:
{"relations": [{"source": "...", "target": "...", "description": "...", "type": "...", "confidence": 0.0}]}
"""

PROMPTS["search_and_link_prompt"] = """
# NEW SKILL
{new_skill}

# CANDIDATE SKILLS
{candidate_skills}

Identify only the high-confidence relationships between the NEW SKILL and the candidate skills.
If no relationship is well supported, return {{"relations": []}}.

OUTPUT:
"""


PROMPTS["query_rewrite_system"] = """# GOAL
Rewrite a user task request into a compact retrieval schema for skill search.

# PRINCIPLES
- Preserve the user's actual task intent.
- Use generic technical language taken from the request; do not invent benchmark-specific category labels.
- Keep high precision. If a field is unclear, leave it empty.
- Prefer concrete APIs, file names, data formats, libraries, protocols, and operations when they are mentioned or strongly implied.
- `keywords` should contain the most retrieval-useful terms or short phrases, not a bag of every token.

# OUTPUT FORMAT
Return valid JSON with these fields only:
- `goal`: one concise sentence describing the task
- `task_name`: a short slug-like name if obvious, else empty
- `domain`: narrow technical domains relevant to retrieval
- `operations`: concrete operations, transformations, algorithms, or APIs
- `artifacts`: file names, formats, interfaces, commands, or concrete objects
- `constraints`: acceptance conditions, invariants, or non-functional constraints
- `keywords`: high-value retrieval terms or short phrases
"""

PROMPTS["query_rewrite_prompt"] = """**STRICT JSON RULES**:
- No markdown fences.
- No trailing commas.
- Double quotes only for strings.

# USER TASK
{query}

OUTPUT:
"""
