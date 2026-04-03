def retrieve_relevant_skills_prompt(metadata, task):
    template = "You are an expert in task analysis and skills retrieval."
    prompt = f'''
Your task is to analyze the given task description and retrieve the most relevant skills from the provided metadata (name and description pairs).

# Principles
1.  **Contextual Understanding**: Thoroughly analyze the task description to grasp its requirements.
2.  **Relevance Matching**: Match the task requirements with the most relevant skills based on the metadata descriptions. Only include skills that are most helpful for accomplishing the task. (No more than 5 unless necessary)
3.  **Clarity and Precision**: Clearly list the names of the relevant skills in your output.
4.  **Output Format**: Return the skill names as a JSON list of strings. Must strictly match names in the provided metadata.
5.  **Allow Empty List**: If no relevant skills are found, return an empty list `[]`.

# Input Data
1.  **Metadata:** 
{metadata}
2.  **Task Description:**
{task}

Keep your output in the format below:
<Analysis> your analysis here </Analysis>
<Relevant_Skill_Names> relevant skill names list in JSON format here </Relevant_Skill_Names>
'''
    return [{"role": "system", "content": template}, {"role": "user", "content": prompt}]


def generate_overall_procedure_prompt(task, overall_procedure_examples, skill_contents):
    template = "You are a Senior Systems Architect."
    prompt = f'''
You are given a complex task description, relevant skill contents, and examples of well-structured procedural guidance texts.
Your task is to create a structured "Procedural Guidance" text. This text will serve as lightweight guidance for an AI Agent solving the task from environment feedback.

# CRITICAL INSTRUCTIONS (Follow these strictly)
1.  **Keep It Very Short**: Keep the guidance concise. Prefer 2-4 short phases. Do NOT write a long room-wide checklist.
1a. **Policy, Not Script**: Write this like lightweight decision guidance, not a fixed action script.
1b. **No Exhaustive Search Orders**: Do not enumerate long receptacle sequences like cabinet 1/2/3/4 or shelf 1..12 unless absolutely required.
2.  **Observation-Driven**: Tell the agent to pivot immediately when the current observation reveals the target object, target receptacle, or a faster path.
3.  **Use Exact Syntax Only**: When describing actions, you MUST quote only exact actions from the available action set. Do not invent verbs like "put", "toggle", "inspect", or "observe" unless they are explicitly available.
4.  **State & Preconditions**: Explicitly mention only necessary preconditions (e.g., "If the fridge is closed, use 'open fridge 1' before cooling").
5.  **Hard Early Stop Rule**: If environment feedback or reward indicates the task is complete, the agent must stop issuing new actions immediately and must not add verification steps.
6.  **Avoid Overbinding to Skills**: Relevant skills may be noisy. Use them only if they directly shorten the path to completion.
7.  **ALFWorld Bias Toward Minimal Plans**: For ALFWorld-style tasks, prefer the shortest valid subgoal sequence over exhaustive search.
8.  **SkillRequest Friendly**: If the agent may need more targeted skill help during execution, explicitly say it can pause and issue a `SkillRequest:` instead of guessing.
9.  **Request Before Guess**: If the exact interaction syntax or device usage pattern is unclear after a failed attempt, instruct the agent to issue `SkillRequest:` rather than repeating invalid actions.
10. **Prefer Re-Retrieval Over Wrong Skill Commitment**: If the retrieved skills appear generic, off-task, or mismatched to the current blocker, prefer `SkillRequest: GOS_RETRIEVE <short focused query>` over forcing the agent to read or follow the wrong skill.

# Input Data
1.  **Task Description:**
{task}
2.  **Relevant Skill Contents:**
{skill_contents}

# Examples
Here are some examples of well-structured Procedural Guidance texts you should emulate (focus on style, structure, and level of detail instead of content):
{overall_procedure_examples}


Keep your output in the format below:
<Analysis> your analysis here </Analysis>
<Overall_Procedure> your generated overall procedure guidance here </Overall_Procedure>
'''
    return [{"role": "system", "content": template}, {"role": "user", "content": prompt}]

def generate_overall_procedure_code_prompt(task, overall_procedure, procedure_code_template):
    template = "You are a Senior Software Engineer specializing in AI-driven automation and safe code generation."
    prompt = f'''
You are given a complex task description, overall procedural guidance, and a strict procedure code template.
Your task is to implement a Python function `overall_procedure_code` that follows the provided procedure code template and incorporates the overall procedure guidance to help Agent solve a complex task.

# Principles
1.  **Adherence to Template**: Strictly follow the provided procedure code template. Input arguments and return types must align with the template. (Most Important)
2.  **Inject Overall Procedural Guidance**: Help the agent accomplish the task by injecting the overall procedure guidance into the function.
3.  **Preserve Tool-Style Skill Access**: If the agent outputs `SkillRequest: ...`, the procedure should let the runner fulfill it and continue, instead of forcing an environment action that turn.
4.  **Error Handling**: Implement robust error handling for LLM calls and environment interactions.
5.  **Function Naming**: Name the function `overall_procedure_code` as per the template.

# Input Data
1.  **Task Description:**
{task}
2.  **Overall Procedural Guidance:**
{overall_procedure}
3.  **Procedure Code Template:**
```python
{procedure_code_template}
```

Keep your output in the format below:
<Analysis> your analysis here </Analysis>
<Overall_Procedure_Code> your generated overall procedure workflow code here </Overall_Procedure_Code>
'''
    return [{"role": "system", "content": template}, {"role": "user", "content": prompt}]
