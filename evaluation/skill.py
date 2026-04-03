import json
import os
import re
import asyncio
import inspect
from pathlib import Path
import yaml
from typing import Any
import sys


_SKILLS_REF_SRC = str(Path(__file__).resolve().parent)
if _SKILLS_REF_SRC not in sys.path:
    sys.path.insert(0, _SKILLS_REF_SRC)

# Try to import GoS engine
try:
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from gos import SkillGraphRAG
    from gos.core.engine import build_default_embedding_service, build_default_llm_service
    from gos.core.schema import QuerySchema
except ImportError:
    SkillGraphRAG = None
    build_default_embedding_service = None
    build_default_llm_service = None
    QuerySchema = None

try:
    from .utils import get_llm_response
    from .skills_ref import to_prompt as skills_ref_to_prompt
except ImportError:
    from utils import get_llm_response
    from skills_ref import to_prompt as skills_ref_to_prompt

class SkillModule:
    def __init__(self, **kwargs):
        self.skills_dir = Path(kwargs.get("skills_dir", "skills"))
        self.model = kwargs.get("model", "gpt-4o")
        self.mode = kwargs.get("mode", "gos") # "all_full", "gos", "vector", "none"
        self.gos_workspace = kwargs.get("gos_workspace", None)
        self.enable_alfworld_gating = bool(kwargs.get("enable_alfworld_gating", False))

        self.last_retrieval_result: Any = None
        self.last_retrieval_status = "NOT_RUN"
        self.last_retrieval_summary = ""
        self.last_retrieved_skill_names = []
        self.last_retrieval_query = ""
        self.runtime_skill_events = []
        self.runtime_skill_count = 0
        self.runtime_last_injection_step = -999

        self.metadata = self._load_metadata()

        # Initialize GoS if needed
        if self.mode in {"gos", "vector"} and not self.gos_workspace:
            raise ValueError(f"{self.mode} mode requires `gos_workspace`.")

        if self.mode in {"gos", "vector"} and SkillGraphRAG is None:
            raise ImportError("Failed to import `gos.SkillGraphRAG`; retrieval is unavailable.")

        if self.mode in {"gos", "vector"} and SkillGraphRAG and self.gos_workspace:
            gos_workspace = str(Path(self.gos_workspace).expanduser().resolve())
            self.gos_workspace = gos_workspace
            self.rag = SkillGraphRAG(
                working_dir=gos_workspace,
                config=SkillGraphRAG.Config(
                    working_dir=gos_workspace,
                    prebuilt_working_dir=gos_workspace,
                    llm_service=build_default_llm_service() if build_default_llm_service else None,
                    embedding_service=build_default_embedding_service() if build_default_embedding_service else None,
                    # ALFWorld / ScienceWorld already construct a retrieval-oriented query.
                    # Skip GoS internal LLM rewrite here to avoid schema-format drift.
                    enable_query_rewrite=False,
                )
            )
        else:
            self.rag = None

    def _log(self, message):
        print(f"[SkillModule] {message}")

    def _is_alfworld_task(self, task):
        task_lower = task.lower()
        return "your task is to:" in task_lower or "you are in the middle of a room" in task_lower

    def _is_scienceworld_task(self, task):
        task_lower = task.lower()
        scienceworld_markers = [
            "scienceworld",
            "task description:",
            "focus on ",
            "located around the",
            "degrees celsius",
            "electrically conductive",
            "electrically nonconductive",
            "living thing",
            "animal",
            "teleport to",
        ]
        scienceworld_rooms = [
            "kitchen",
            "foundry",
            "workshop",
            "bathroom",
            "outside",
            "living room",
            "bedroom",
            "greenhouse",
            "art studio",
            "hallway",
        ]
        if any(marker in task_lower for marker in scienceworld_markers):
            return True
        if "box" in task_lower and any(room in task_lower for room in scienceworld_rooms):
            return True
        return False

    def _extract_alfworld_goal(self, task):
        match = re.search(r"your task is to:\s*(.+)", task, re.IGNORECASE)
        if match:
            return match.group(1).splitlines()[0].strip().rstrip('.')
        return task.strip()

    def _extract_scienceworld_goal(self, task):
        patterns = [
            r"task description:\s*(.+)",
            r"goal:\s*(.+)",
            r"objective:\s*(.+)",
            r"your task is to\s*(.+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, task, re.IGNORECASE)
            if match:
                return match.group(1).splitlines()[0].strip().rstrip('.')
        compact = " ".join(task.split())
        return compact[:400]

    def _infer_scienceworld_task_type(self, goal):
        goal_lower = goal.lower()
        if "electrically conductive" in goal_lower or "conductivity" in goal_lower:
            return "conductivity_classification"
        if "melting point" in goal_lower:
            return "melting_point_measurement"
        if "boiling point" in goal_lower:
            return "boiling_point_measurement"
        if "temperature" in goal_lower:
            return "temperature_measurement"
        if "living thing" in goal_lower:
            return "living_thing_identification"
        if "animal" in goal_lower:
            return "animal_identification"
        if "plant" in goal_lower:
            return "plant_identification"
        if "box" in goal_lower:
            return "scientific_sorting"
        return "scientific_task"

    def _extract_scienceworld_primary_target(self, goal):
        goal_lower = goal.lower().rstrip('.')
        patterns = [
            r"change the state of matter of\s+(.+?)(?:\.| first| then|,|$)",
            r"boil\s+(.+?)(?:\.| first| then|,|$)",
            r"create\s+(.+?)(?:\.| when| first| then|,|$)",
            r"grow\s+(?:a|an)\s+(.+?)\s+from seed(?:\.| first| then|,|$)",
            r"determine if\s+(.+?)\s+is electrically conductive",
            r"measure the melting point of\s+(.+?)(?:,|\.|$)",
            r"measure the boiling point of\s+(.+?)(?:,|\.|$)",
            r"measure the temperature of\s+(.+?)(?:,|\.|$)",
            r"find a\(n\)\s+(.+?)(?:\.| first| then|,|$)",
            r"focus on the\s+(.+?)(?:\.| if| then|,|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, goal_lower)
            if match:
                candidate = " ".join(match.group(1).split())
                if candidate:
                    return candidate
        return self._extract_primary_object(goal)

    def _extract_scienceworld_rooms(self, goal):
        goal_lower = goal.lower()
        ordered_rooms = [
            "kitchen",
            "foundry",
            "workshop",
            "bathroom",
            "outside",
            "living room",
            "bedroom",
            "greenhouse",
            "art studio",
            "hallway",
        ]
        found = [room for room in ordered_rooms if room in goal_lower]
        return found or ["unknown"]

    def _extract_scienceworld_property(self, goal):
        goal_lower = goal.lower()
        if "electrically conductive" in goal_lower or "conductivity" in goal_lower:
            return "electrical_conductivity"
        if "melting point" in goal_lower:
            return "melting_point"
        if "boiling point" in goal_lower:
            return "boiling_point"
        if "temperature" in goal_lower:
            return "temperature"
        if "living thing" in goal_lower:
            return "living_thing"
        if "animal" in goal_lower:
            return "animal"
        return "unknown"

    def _extract_scienceworld_boxes(self, goal):
        matches = re.findall(r"\b([a-z]+ box)\b", goal.lower())
        ordered = []
        for match in matches:
            if match not in ordered:
                ordered.append(match)
        return ordered or ["unknown"]

    def _extract_scienceworld_tools(self, goal, task_type):
        goal_lower = goal.lower()
        tools = []
        if "conduct" in goal_lower or task_type == "conductivity_classification":
            tools.extend(["wire", "battery", "light bulb", "circuit"])
        if "focus on" in goal_lower:
            tools.append("focus")
        deduped = []
        for tool in tools:
            if tool not in deduped:
                deduped.append(tool)
        return deduped or ["unknown"]

    def _infer_task_type(self, goal):
        goal_lower = goal.lower()
        if "look at" in goal_lower or "examine" in goal_lower:
            return "examine"
        if "find two" in goal_lower or "put two" in goal_lower:
            return "put_two"
        if "clean" in goal_lower:
            return "clean_and_place"
        if "cool" in goal_lower:
            return "cool_and_place"
        if "heat" in goal_lower or "hot " in goal_lower:
            return "heat_and_place"
        if "put" in goal_lower:
            return "put"
        return "other"

    def _extract_required_state(self, goal):
        goal_lower = goal.lower()
        if "clean" in goal_lower:
            return "clean"
        if "cool" in goal_lower:
            return "cool"
        if "heat" in goal_lower or "hot " in goal_lower:
            return "hot"
        return "none"

    def _extract_count(self, goal):
        goal_lower = goal.lower()
        if "find two" in goal_lower or "put two" in goal_lower:
            return "2"
        if re.search(r"\bsome\b", goal_lower):
            return "some"
        if re.search(r"\ban?\b", goal_lower):
            return "1"
        return "unspecified"

    def _extract_target_receptacle(self, goal):
        patterns = [
            r"\b(?:in|into|inside|on|onto|under)\s+([a-z0-9]+)",
            r"\bto\s+([a-z0-9]+)$",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, goal, re.IGNORECASE)
            if matches:
                return matches[-1].lower()
        return "unknown"

    def _extract_device(self, goal):
        devices = [
            "desklamp",
            "microwave",
            "fridge",
            "sinkbasin",
            "coffeemachine",
            "stoveburner",
            "cabinet",
            "drawer",
            "dresser",
            "garbagecan",
            "diningtable",
            "countertop",
            "desk",
            "sidetable",
            "table",
            "toilet",
        ]
        goal_lower = goal.lower()
        for device in devices:
            if device in goal_lower:
                return device
        return "none"

    def _extract_primary_object(self, goal):
        goal_lower = goal.lower().rstrip('.')
        patterns = [
            r"(?:look at|examine)\s+the\s+([a-z0-9]+)",
            r"(?:look at|examine)\s+([a-z0-9]+)",
            r"(?:put|find|clean|cool|heat)\s+(?:a|an|some|two)?\s*(?:clean|cool|hot|heated)?\s*([a-z0-9]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, goal_lower)
            if match:
                candidate = match.group(1).lower()
                if candidate not in {"clean", "cool", "hot", "heated"}:
                    return candidate
        tokens = re.findall(r"[a-z0-9]+", goal_lower)
        stop = {
            "put", "find", "clean", "cool", "heat", "hot", "heated", "look", "at", "examine",
            "the", "a", "an", "some", "two", "in", "into", "inside", "on", "onto",
            "under", "with", "and", "it", "them"
        }
        for token in tokens:
            if token not in stop:
                return token
        return "unknown"

    def _build_alfworld_structured_query(self, task):
        goal = self._extract_alfworld_goal(task)
        task_type = self._infer_task_type(goal)
        obj = self._extract_primary_object(goal)
        required_state = self._extract_required_state(goal)
        target_receptacle = self._extract_target_receptacle(goal)
        count = self._extract_count(goal)
        device = self._extract_device(goal)
        if QuerySchema is None:
            return (
                "environment=alfworld; "
                f"task_type={task_type}; "
                f"goal={goal}; "
                f"object={obj}; "
                f"required_state={required_state}; "
                f"target_receptacle={target_receptacle}; "
                f"count={count}; "
                f"device={device}; "
                "actions=navigate,take,move,open,close,heat,cool,clean,use,look; "
                "optimization=shortest_valid_action_sequence"
            )

        schema = QuerySchema(
            goal=goal,
            task_name=f"alfworld-{task_type}",
            domain=["alfworld", "household manipulation", "embodied task planning"],
            operations=[
                task_type,
                "navigate",
                "take",
                "move",
                "open",
                "close",
                "heat",
                "cool",
                "clean",
                "look",
            ],
            artifacts=[obj, target_receptacle, device],
            constraints=[
                f"required_state={required_state}",
                f"count={count}",
                "optimize for shortest valid action sequence",
            ],
            keywords=[
                "environment=alfworld",
                f"object={obj}",
                f"target_receptacle={target_receptacle}",
                f"device={device}",
            ],
        )
        return schema.to_query_text()

    def _build_scienceworld_structured_query(self, task):
        goal = self._extract_scienceworld_goal(task)
        task_type = self._infer_scienceworld_task_type(goal)
        property_name = self._extract_scienceworld_property(goal)
        primary_target = self._extract_scienceworld_primary_target(goal)
        rooms = self._extract_scienceworld_rooms(goal)
        boxes = self._extract_scienceworld_boxes(goal)
        tools = self._extract_scienceworld_tools(goal, task_type)

        keywords = []

        def add_keyword(value):
            value = " ".join(str(value).replace("_", " ").split())
            if not value or value == "unknown":
                return
            if value not in keywords:
                keywords.append(value)

        add_keyword(task_type)
        add_keyword(property_name)
        add_keyword(primary_target)
        for room in rooms:
            add_keyword(room)
        for box in boxes:
            add_keyword(box)
        for tool in tools:
            add_keyword(tool)

        action_hints = []

        def add_action_hint(value):
            if value and value not in action_hints:
                action_hints.append(value)

        if task_type == "conductivity_classification":
            add_action_hint("focus on target")
            add_action_hint("build circuit with battery wire light bulb")
            add_action_hint("activate switch")
            add_action_hint("place target into destination box based on result")
        elif task_type in {"melting_point_measurement", "boiling_point_measurement", "temperature_measurement"}:
            add_action_hint("acquire thermometer")
            add_action_hint("use thermometer on target")
            add_action_hint("heat or cool target if needed")
        elif task_type in {"animal_identification", "living_thing_identification", "plant_identification"}:
            add_action_hint("navigate to relevant room")
            add_action_hint("inspect visible entities")
            add_action_hint("focus on target entity")
        elif task_type == "scientific_sorting":
            add_action_hint("focus on target object")
            add_action_hint("measure or test required property")
            add_action_hint("place object into the correct box")
        else:
            goal_lower = goal.lower()
            if "change the state of matter" in goal_lower or "boil" in goal_lower or "melt" in goal_lower:
                add_action_hint("focus on target substance")
                add_action_hint("manipulate the target until the requested state change is achieved")
            if "paint" in goal_lower or "mix" in goal_lower:
                add_action_hint("combine relevant materials and inspect the resulting mixture")
            if "grow" in goal_lower or "life stage" in goal_lower or "reproduction" in goal_lower:
                add_action_hint("set up the target for growth and monitor stage changes")

        query_parts = [
            "environment=scienceworld",
            f"task_type={task_type}",
            f"goal={goal}",
            f"primary_target={primary_target}",
            f"property={property_name}",
            f"rooms={', '.join(rooms)}",
            f"tools={', '.join(tools)}",
        ]

        if boxes and boxes != ["unknown"]:
            query_parts.append(f"destination_containers={', '.join(boxes)}")
        if keywords:
            query_parts.append(f"keywords={'; '.join(keywords)}")
        if action_hints:
            query_parts.append(f"required_actions={'; '.join(action_hints)}")

        return "\n".join(query_parts)

    def _build_targeted_retrieval_query(self, task):
        if self._is_alfworld_task(task):
            return self._build_alfworld_structured_query(task)

        if self._is_scienceworld_task(task):
            return self._build_scienceworld_structured_query(task)

        return task.strip()

    def _skill_confident_enough(self, skill):
        rerank_score = float(getattr(skill, "rerank_score", 0.0) or 0.0)
        score = float(getattr(skill, "score", 0.0) or 0.0)
        semantic_rank = getattr(skill, "semantic_rank", None)

        if rerank_score >= 0.60:
            return True
        if rerank_score >= 0.45 and semantic_rank is not None and semantic_rank <= 2:
            return True
        if score >= 0.30 and semantic_rank is not None and semantic_rank <= 2:
            return True
        return False

    def _effective_top_k(self, task, requested_top_k):
        if self._is_alfworld_task(task):
            return min(requested_top_k, 4)
        return requested_top_k

    def _extract_vector_skill_payloads(self, result):
        skill_names = [skill.name for skill in result.skills]
        skill_payloads = [skill.payload for skill in result.skills]
        return skill_payloads, skill_names

    def _filter_skills_for_task(self, task, result, *, source_label="retrieval"):
        if not self._is_alfworld_task(task) or not self.enable_alfworld_gating:
            skill_names = [skill.name for skill in result.skills]
            skill_payloads = [skill.payload for skill in result.skills]
            return skill_payloads, skill_names

        confident_skills = [skill for skill in result.skills if self._skill_confident_enough(skill)]
        if not confident_skills:
            self._log(f"alfworld gating pruned all {source_label} results; returning NO_SKILL_HIT")
            return [], []

        selected_skills = confident_skills
        skill_names = [skill.name for skill in selected_skills]
        skill_payloads = [skill.payload for skill in selected_skills]

        self._log(
            f"alfworld gating kept {len(selected_skills)}/{len(result.skills)} {source_label} skills after confidence pruning"
        )
        return skill_payloads, skill_names

    def should_generate_procedure(self, task):
        return False

    def _reset_retrieval_state(self):
        self.last_retrieval_result = None
        self.last_retrieval_status = "NOT_RUN"
        self.last_retrieval_summary = ""
        self.last_retrieved_skill_names = []
        self.last_retrieval_query = ""
        self.runtime_skill_events = []
        self.runtime_skill_count = 0
        self.runtime_last_injection_step = -999

    def _set_retrieval_state(self, status, summary="", skill_names=None, result=None):
        self.last_retrieval_status = status
        self.last_retrieval_summary = summary or ""
        self.last_retrieved_skill_names = list(skill_names or [])
        self.last_retrieval_result = result

    def _all_metadata_entries(self):
        return [
            {
                "name": name,
                "description": data.get("description", ""),
                "skill_dir": data.get("skill_dir", ""),
            }
            for name, data in sorted(self.metadata.items())
        ]

    def _all_metadata_context(self):
        lines = []
        for item in self._all_metadata_entries():
            lines.append(f"- {item['name']}: {item['description']}")
        return "\n".join(lines)

    def _all_metadata_skill_bundle(self):
        metadata_context = self._all_metadata_context()
        if not metadata_context:
            return []
        return [
            "=== Full Skill Library Metadata ===\n"
            "The following is the full available skill library. Treat it as capability exposure, not as a pre-filtered retrieval result.\n\n"
            f"{metadata_context}"
        ]

    def get_all_full_exposure_messages(self):
        if self.mode != "all_full":
            return []

        skill_dirs = [Path(item["skill_dir"]) for item in self._all_metadata_entries() if item.get("skill_dir")]
        if not skill_dirs:
            return []

        prompt_block = skills_ref_to_prompt(skill_dirs)
        return [
            "The following block lists the full available skill library in Anthropic skills-ref format. "
            "This is not a pre-filtered retrieval result. Use it as a catalog of available capabilities. "
            "If a skill looks relevant, prefer reading only the few most relevant skills by exact name.\n\n"
            f"{prompt_block}"
        ]

    def get_all_full_exposure_message(self):
        messages = self.get_all_full_exposure_messages()
        if not messages:
            return ""
        return messages[0]

    def get_agent_skill_request_message(self):
        if self.mode == "none":
            return ""

        lines = [
            "Tool-style skill access is available in this run.",
            "Use it when you are blocked, the syntax is unclear, the retrieved skills look mismatched to the current blocker, or 1-2 actions already failed.",
            "Use skills conditionally, not by default: if the next environment action is already obvious from the current observation, act directly instead of retrieving.",
            "Prefer retrieval when the exact syntax is unclear, the task needs a multi-step procedure or tool setup, the current shortlist looks mismatched, or 1-2 recent actions failed.",
            "Prefer READ_SKILL when you already have a promising exact skill name. For measurement, electrical connection, conditional placement, or any unfamiliar procedure, do not guess the syntax twice in a row; retrieve first, then read the single best skill before continuing.",
            "Mirror the current benchmark vocabulary in retrieval queries. Reuse the task's own object, property, tool, room, and container words instead of naming a different environment.",
            "In a request turn, output exactly two lines: `Thought: ...` and `SkillRequest: ...`. Do not output an `Action:` line in the same turn.",
        ]

        if self.mode == "gos":
            lines.extend([
                "Available requests:",
                "- `SkillRequest: GOS_RETRIEVE <short focused query>` to search GoS again. Prefer this first when you are blocked or the current shortlist looks noisy, generic, or off-task.",
                "- `SkillRequest: READ_SKILL <exact skill name>` to read one concrete skill after GoS has surfaced a promising candidate.",
                "Examples:",
                "- `Thought: I already have a good shortlist and need the exact instructions from one candidate.`",
                "  `SkillRequest: READ_SKILL <exact shortlisted skill name>`",
                "- `Thought: The current shortlist looks noisy. I need a narrower retrieval grounded in the current task.`",
                "  `SkillRequest: GOS_RETRIEVE <target object> <property or subgoal> <tool if needed> <room> <destination container>`",
                "- `Thought: I failed twice and need retrieval that mirrors the current blocker instead of guessing again.`",
                "  `SkillRequest: GOS_RETRIEVE <task-specific keywords from the current benchmark only>`",
                "Use skill requests sparingly, only when they directly help the next action. Prefer a two-step pattern: `GOS_RETRIEVE` to shortlist candidates, then `READ_SKILL` for the single best candidate before guessing again.",
            ])
        elif self.mode == "vector":
            lines.extend([
                "Available requests:",
                "- `SkillRequest: VECTOR_RETRIEVE <short focused query>` to run vector-only retrieval again. This uses embedding similarity only, without graph propagation or lexical expansion.",
                "- `SkillRequest: READ_SKILL <exact skill name>` to read a known skill file. Use this only when you already know the exact skill you want.",
                "Examples:",
                "- `Thought: I already have a good shortlist and need the exact instructions from one candidate.`",
                "  `SkillRequest: READ_SKILL <exact shortlisted skill name>`",
                "- `Thought: The current shortlist looks noisy. I need a narrower vector retrieval grounded in the current task.`",
                "  `SkillRequest: VECTOR_RETRIEVE <target object> <property or subgoal> <tool if needed> <room> <destination container>`",
                "- `Thought: I failed twice and need vector retrieval that mirrors the current blocker instead of guessing again.`",
                "  `SkillRequest: VECTOR_RETRIEVE <task-specific keywords from the current benchmark only>`",
                "Use skill requests sparingly, only when they directly help the next action. In vector mode, prefer `VECTOR_RETRIEVE` before guessing again, and `READ_SKILL` only after a specific skill name looks relevant.",
            ])
        elif self.mode == "all_full":
            lines.extend([
                "Available requests:",
                "- `SkillRequest: READ_SKILL <exact skill name>` to read a known skill file.",
                "Examples:",
                "- `Thought: The full catalog already shows a likely match and I need its exact instructions.`",
                "  `SkillRequest: READ_SKILL <exact skill name already visible in the catalog>`",
                "Use skill requests sparingly, only when they directly help the next action. In all_full mode, do not attempt retrieval; read a specific skill only when the full catalog already reveals a directly relevant candidate.",
            ])

        else:
            return ""
        return "\n".join(lines)

    def _skill_catalog_entries(self, skill_names):
        entries = []
        for name in skill_names or []:
            meta = self.metadata.get(name, {})
            entries.append(
                {
                    "name": name,
                    "description": meta.get("description", ""),
                    "skill_dir": meta.get("skill_dir", ""),
                }
            )
        return entries

    def _format_retrieval_shortlist(self, header, query, skill_names, source_label):
        if not skill_names:
            return f"{header}\n\nNo relevant skills were retrieved."

        lines = [
            header,
            f"Query: {query}",
            f"Shortlisted {source_label} candidates:",
        ]
        for entry in self._skill_catalog_entries(skill_names[:3]):
            description = entry["description"] or "No description available."
            lines.append(f"- {entry['name']}: {description}")
            if entry["skill_dir"]:
                lines.append(f"  Source: {entry['skill_dir']}/SKILL.md")
        lines.extend([
            "Do not assume these summaries are enough to execute correctly.",
            "If one candidate looks directly relevant to the current blocker, issue `SkillRequest: READ_SKILL <exact skill name>` before trying another uncertain action.",
        ])
        return "\n".join(lines)

    def _load_metadata(self):
        """Load existing metadata from file."""
        metadata = {}
        if not self.skills_dir.exists():
            return metadata
            
        for skill_dir in self.skills_dir.iterdir():
            if skill_dir.is_dir():
                skill_md_path = skill_dir / "SKILL.md"
                if skill_md_path.exists():
                    try:
                        content = skill_md_path.read_text(encoding="utf-8")
                        if content.strip().startswith('---'):
                            parts = content.split('---', 2)
                            if len(parts) >= 3:
                                header_data = yaml.safe_load(parts[1])
                                if isinstance(header_data, dict) and header_data.get('name') and header_data.get('description'):
                                    metadata[header_data['name']] = {
                                        'description': header_data['description'],
                                        'skill_dir': str(skill_dir)
                                    }
                    except Exception as e:
                        print(f"[ERROR] Failed to parse SKILL.md for {skill_dir.name}: {e}")
        return metadata
    
    def retrieve_relevant_skills(self, task, top_k=15):
        self._reset_retrieval_state()
        effective_top_k = self._effective_top_k(task, top_k)
        retrieval_query = self._build_targeted_retrieval_query(task)
        self.last_retrieval_query = retrieval_query
        self._log(
            f"retrieve_relevant_skills start mode={self.mode} top_k={top_k} effective_top_k={effective_top_k} task_chars={len(task)} retrieval_query={retrieval_query!r}"
        )

        if self.mode == "none":
            self._log("mode=none, skipping retrieval")
            return []
            
        if self.mode in {"gos", "vector"} and self.rag:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            if self.mode == "gos":
                self._log(f"starting GoS async_retrieve workspace={self.gos_workspace}")
                result = loop.run_until_complete(self.rag.async_retrieve(retrieval_query, top_n=effective_top_k))
                skill_payloads, skill_names = self._filter_skills_for_task(task, result, source_label="gos")
                status = "SKILL_HIT" if skill_names else "NO_SKILL_HIT"
                summary = result.summary
                if not skill_names:
                    status = "NO_SKILL_HIT"
                    summary = "ALFWorld retrieval gating pruned all retrieved skills; proceeding without injected skills."
                self._log(f"GoS async_retrieve finished status={status} n_skills={len(skill_names)}")
            else:
                self._log(f"starting vector async_retrieve workspace={self.gos_workspace}")
                result = loop.run_until_complete(self.rag.async_retrieve_vector(retrieval_query, top_n=effective_top_k))
                skill_payloads, skill_names = self._filter_skills_for_task(task, result, source_label="vector")
                status = "SKILL_HIT" if skill_names else "NO_SKILL_HIT"
                summary = result.summary
                self._log(f"vector async_retrieve finished status={status} n_skills={len(skill_names)}")

            self._set_retrieval_state(
                status=status,
                summary=summary,
                skill_names=skill_names,
                result=result,
            )
            return skill_payloads

        if self.mode == "all_full":
            metadata_entries = self._all_metadata_entries()
            skill_names = [entry["name"] for entry in metadata_entries]
            status = "SKILL_HIT" if skill_names else "NO_SKILL_HIT"
            summary = (
                f"Exposed full skill metadata library in a single initial dialogue message ({len(skill_names)} skills). "
                "This matches the all-skills capability-exposure baseline rather than retrieval-time shortlisting."
            )
            self._set_retrieval_state(
                status=status,
                summary=summary,
                skill_names=skill_names,
                result={"skill_names": skill_names, "mode": "all_full"},
            )
            self._log(f"all_full exposure finished status={status} n_skills={len(skill_names)}")
            return []

        self._set_retrieval_state(status="NO_SKILL_HIT", summary="No retrieval configured for this mode.")
        return []

    def get_retrieval_guidance(self):
        if self.mode not in {"gos", "vector"} or self.last_retrieval_result is None:
            return ""

        if self.last_retrieval_status != "SKILL_HIT" or not self.last_retrieved_skill_names:
            return ""

        top_skills = self.last_retrieved_skill_names[:3]
        title = "Graph of Skills retrieval guidance:" if self.mode == "gos" else "Vector-skills retrieval guidance:"
        is_scienceworld_context = (
            "environment=scienceworld" in (self.last_retrieval_query or "")
            or any(name.startswith("scienceworld-") for name in (self.last_retrieved_skill_names or []))
        )
        content_parts = [
            title,
            f"Retrieval Status: {self.last_retrieval_status}",
        ]
        if top_skills:
            content_parts.append("Top retrieved skills: " + ", ".join(top_skills))
            skill_lines = ["Retrieved skill summaries:"]
            for entry in self._skill_catalog_entries(top_skills):
                description = entry["description"] or "No description available."
                skill_lines.append(f"- {entry['name']}: {description}")
            content_parts.append("\n".join(skill_lines))
        content_parts.append(
            "Use retrieval only as weak high-level guidance. Prioritize the shortest path from current observation to task completion."
        )
        content_parts.append(
            "Do not follow a rigid room-wide search checklist if the current observation already reveals the target object or target receptacle."
        )
        content_parts.append(
            "If the environment feedback or reward indicates the task is complete, stop issuing new actions immediately."
        )
        if is_scienceworld_context:
            content_parts.append(
                "For ScienceWorld, keep retrieval grounded in the current scientific task: mention the target material or organism, the property to test or measure, the required tool, the relevant room, and the destination box or container."
            )
            content_parts.append(
                "Do not write retrieval queries that mention another benchmark such as alfworld."
            )
            content_parts.append(
                "If the task involves measurement, electrical connection, conditional placement, or another unfamiliar procedure, do not guess the syntax twice in a row. Retrieve first, then read the single best skill before continuing."
            )
        else:
            content_parts.append(
                "For ALFWorld action syntax: first navigate to the destination receptacle, then use the exact action form 'move {obj} to {recep}'."
            )
            content_parts.append(
                "Do not use 'use {obj}' unless the task explicitly requires turning on, heating, cooling, or cleaning something."
            )
        if self.mode == "gos":
            content_parts.append(
                "If the current retrieved skills look mismatched to the blocker, or 1-2 actions already failed, issue `SkillRequest: GOS_RETRIEVE <short focused query>`. Treat retrieval as a shortlist step and prefer `READ_SKILL` for the single best candidate before another uncertain action."
            )
        elif self.mode == "vector":
            content_parts.append(
                "If the current retrieved skills look mismatched to the blocker, or 1-2 actions already failed, issue `SkillRequest: VECTOR_RETRIEVE <short focused query>`. After vector retrieval surfaces a plausible exact skill name, prefer `READ_SKILL` for that single candidate before another uncertain action."
            )
        return "\n\n".join(part for part in content_parts if part)

    def _get_skill_contents(self, skill_names):
        skill_contents = []
        for name in skill_names:
            if name in self.metadata:
                skill_dir = Path(self.metadata[name]['skill_dir'])
                combined_text = f"=== Skill: {name} ===\n"
                for file_path in skill_dir.rglob('*'):
                    if file_path.is_file():
                        try:
                            content = file_path.read_text(encoding='utf-8')
                            combined_text += f"\n[File: {file_path.name}]\n{content}\n"
                        except: continue
                skill_contents.append(combined_text)
        return skill_contents

    def _parse_skill_request(self, response):
        if not isinstance(response, str):
            return None, ""

        patterns = [
            r"^SkillRequest:\s*(.+)$",
            r"^Action:\s*SkillRequest:\s*(.+)$",
        ]
        match = None
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                break
        if not match:
            return None, ""

        payload = match.group(1).strip()
        if not payload:
            return None, ""

        upper = payload.upper()
        if upper.startswith("READ_SKILL "):
            return "read_skill", payload[len("READ_SKILL "):].strip()
        if upper.startswith("GOS_RETRIEVE "):
            return "gos_retrieve", payload[len("GOS_RETRIEVE "):].strip()
        if upper.startswith("VECTOR_RETRIEVE "):
            return "vector_retrieve", payload[len("VECTOR_RETRIEVE "):].strip()
        return None, payload

    def _record_runtime_skill_event(self, step, trigger, query, skill_names):
        self.runtime_skill_count += 1
        self.runtime_last_injection_step = step
        self.runtime_skill_events.append(
            {
                "step": step,
                "trigger": trigger,
                "query": query,
                "skill_names": list(skill_names or []),
            }
        )

    def _format_agent_skill_response(self, header, skill_names, skill_payloads):
        if not skill_payloads:
            return ""
        clipped_payloads = [self._clip_text(payload, 1200) for payload in skill_payloads[:2]]
        lines = [header, "Use this only if it directly improves the next action."]
        if skill_names:
            lines.append("Selected skills: " + ", ".join(skill_names[:2]))
        return "\n\n".join(lines + clipped_payloads)

    def handle_agent_skill_request(self, task, response, current_step):
        request_type, payload = self._parse_skill_request(response)
        if not request_type:
            return ""

        if request_type == "read_skill":
            skill_name = payload
            skill_payloads = self._get_skill_contents([skill_name])[:1]
            if not skill_payloads:
                return (
                    f"Skill request could not be fulfilled: skill `{skill_name}` was not found. "
                    "Use an exact skill name from the available skill list or retrieval results."
                )
            self._record_runtime_skill_event(current_step, "agent_request:read_skill", skill_name, [skill_name])
            return self._format_agent_skill_response(
                f"Skill request fulfilled: READ_SKILL {skill_name}",
                [skill_name],
                skill_payloads,
            )

        if request_type == "gos_retrieve":
            if self.mode != "gos" or not self.rag:
                return "Skill request could not be fulfilled: GOS_RETRIEVE is only available in gos mode."
            query = payload
            if not query:
                return "Skill request could not be fulfilled: empty GOS_RETRIEVE query."
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.rag.async_retrieve(query, top_n=2))
            skill_payloads, skill_names = self._filter_skills_for_task(task, result, source_label="gos")
            skill_names = skill_names[:2]
            if not skill_names:
                return f"Skill request fulfilled: GOS_RETRIEVE {query}\n\nNo relevant skills were retrieved."
            self._record_runtime_skill_event(current_step, "agent_request:gos_retrieve", query, skill_names)
            return self._format_retrieval_shortlist(
                f"Skill request fulfilled: GOS_RETRIEVE {query}",
                query,
                skill_names,
                "GoS",
            )

        if request_type == "vector_retrieve":
            if self.mode != "vector" or not self.rag:
                return "Skill request could not be fulfilled: VECTOR_RETRIEVE is only available in vector mode."
            query = payload
            if not query:
                return "Skill request could not be fulfilled: empty VECTOR_RETRIEVE query."
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.rag.async_retrieve_vector(query, top_n=2))
            skill_payloads, skill_names = self._filter_skills_for_task(task, result, source_label="vector")
            skill_payloads = skill_payloads[:2]
            skill_names = skill_names[:2]
            if not skill_names:
                return f"Skill request fulfilled: VECTOR_RETRIEVE {query}\n\nNo relevant skills were retrieved."
            self._record_runtime_skill_event(current_step, "agent_request:vector_retrieve", query, skill_names)
            return self._format_retrieval_shortlist(
                f"Skill request fulfilled: VECTOR_RETRIEVE {query}",
                query,
                skill_names,
                "vector",
            )

        return ""

    @staticmethod
    def _clip_text(text, max_chars=1800):
        if not text or len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."

    @staticmethod
    def _recent_actions(messages, limit=2):
        actions = []
        for message in reversed(messages or []):
            if message.get("role") != "assistant":
                continue
            content = message.get("content", "")
            if not isinstance(content, str):
                continue
            match = re.search(r"Action:\s*(.+)", content, re.IGNORECASE)
            if match:
                actions.append(match.group(1).strip())
            if len(actions) >= limit:
                break
        actions.reverse()
        return actions

    def _runtime_trigger_reason(self, observation, current_step):
        observation_lower = (observation or "").lower()
        if current_step - self.runtime_last_injection_step < 3:
            return ""
        if self.runtime_skill_count >= 2:
            return ""
        failure_markers = [
            "nothing happens",
            "nothing happened",
            "you can't",
            "cannot",
            "can't",
            "not found",
            "don't see",
            "do not see",
        ]
        for marker in failure_markers:
            if marker in observation_lower:
                return f"runtime_failure:{marker}"
        return ""

    def _build_runtime_retrieval_query(self, task, messages, observation):
        base_query = self._build_targeted_retrieval_query(task)
        recent_actions = self._recent_actions(messages)
        parts = [base_query]
        if recent_actions:
            parts.append("recent_actions=" + ", ".join(recent_actions))
        compact_observation = " ".join((observation or "").split())
        if compact_observation:
            parts.append("runtime_observation=" + compact_observation[:400])
        return "\n".join(part for part in parts if part)

    def _format_runtime_skill_hint(self, skill_names, skill_payloads, trigger):
        if not skill_payloads:
            return ""
        clipped_payloads = [self._clip_text(payload, 1200) for payload in skill_payloads[:2]]
        header = [
            f"Additional runtime skill support was injected because: {trigger}.",
            "Use the following skill details only if they directly help recover and reach the shortest path to completion.",
        ]
        if skill_names:
            header.append("Selected skills: " + ", ".join(skill_names[:2]))
        return "\n\n".join(header + clipped_payloads)

    def maybe_get_runtime_skill_hint(self, task, messages, observation, current_step):
        trigger = self._runtime_trigger_reason(observation, current_step)
        if not trigger:
            return ""

        dynamic_query = self._build_runtime_retrieval_query(task, messages, observation)
        skill_names = []
        skill_payloads = []

        if self.mode in {"gos", "vector"} and self.rag:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            if self.mode == "gos":
                result = loop.run_until_complete(self.rag.async_retrieve(dynamic_query, top_n=2))
                skill_payloads, skill_names = self._filter_skills_for_task(task, result, source_label="gos")
            else:
                result = loop.run_until_complete(self.rag.async_retrieve_vector(dynamic_query, top_n=2))
                skill_payloads, skill_names = self._filter_skills_for_task(task, result, source_label="vector")

            skill_payloads = skill_payloads[:2]
            skill_names = skill_names[:2]
        if not skill_payloads:
            return ""

        self._record_runtime_skill_event(current_step, trigger, dynamic_query, skill_names)
        self._log(
            f"runtime skill injection triggered step={current_step} trigger={trigger} n_skills={len(skill_names)}"
        )
        return self._format_runtime_skill_hint(skill_names, skill_payloads, trigger)

    def get_runtime_skill_events(self):
        return list(self.runtime_skill_events)
