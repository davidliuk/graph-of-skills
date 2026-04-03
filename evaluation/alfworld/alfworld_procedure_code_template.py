#v2
def overall_procedure_code(
    env,
    llm,
    model: str,
    process_ob,
    parse_action,
    messages: list = [],
    max_steps: int = 30,
    skill_module=None,
    task_text: str = ""
):
    """
    Procedure code template for solving a household interactive task using iterative LLM-guided actions. 
    
    INPUTS:
    - env: The environment instance to interact with. (Predefined, call directly)
    - llm: The language model function to generate agent responses. (Predefined, call directly)
    - model: The specific model name or identifier to use with the llm. (Predefined, pass directly)
    - process_ob: A function to process raw environment observations. (Predefined, call directly)
    - parse_action: A function to extract executable actions from LLM responses. (Predefined, call directly)
    - messages: Pre-populated dialogue context as a list of message dicts. (Contains messages describing the environment background and the task)
    - max_steps: Maximum number of interaction steps before termination. (Predefined, pass directly)
    RETURNS:
    - messages: The updated dialogue context including all interactions.
    - task_done: Boolean indicating if the task was successfully completed.
    - reward: The final reward received from the environment.
    - current_steps: The total number of steps taken in the interaction loop.
    These Input Arguments and Return Types MUST NOT be changed.

    This function serves as a STRICT TEMPLATE to enforce grounded, environment-driven task execution.
    """

    # ------------------------------------------------------------------
    # [SECTION] Overall Procedural Guidance INJECTION
    # INSTRUCTION: You should add detailed Overall Procedural Guidance here to help the agent better complete the task. 
    #              These guidelines will be visible to the agent as user messages.
    #              You may define additional helper variables and functions as needed to help the generated code running correctly and efficiently.
    # ------------------------------------------------------------------
    # Example: 
    procedure_guidelines = "<detailed_overall_procedural_guidelines_here>"
    messages.append({"role": "user", "content": procedure_guidelines})
    messages.append({
        "role": "user",
        "content": (
            "ALFWorld execution policy: prefer the shortest valid path from the current observation to task completion. "
            "Do not follow a rigid checklist if the target object, device, or destination is already visible. "
            "Use only exact available actions such as 'go to {recep}', 'take {obj} from {recep}', 'move {obj} to {recep}', 'open {recep}', 'use {obj}', 'cool {obj} with {recep}'. "
            "If one attempt fails and the correct interaction syntax or skill pattern is unclear, issue a SkillRequest instead of guessing another invalid action. "
            "If the retrieved skills look mismatched to the current blocker, prefer `SkillRequest: GOS_RETRIEVE <short focused query>` over reading another likely-wrong skill. "
            "If the task is complete or reward is true, stop immediately and do not add verification steps."
        ),
    })
    # ------------------------------------------------------------------

    # --- Core task execution state ---
    task_done = False
    current_steps = 0
    reward = 0  # always track latest reward

    # --- Main agent loop (IMMUTABLE STRUCTURE) ---
    while not task_done and current_steps < max_steps:

        # ============================================================
        # 1) THOUGHT PHASE — Query LLM based on current message state
        #    - DO NOT change this call signature
        # ============================================================
        try:
            response = llm(messages, model)
            print(f'\033[92mAgent response: \n{response}\033[0m')
        except Exception as e:
            print(f'\033[91mError in LLM call: {e}\033[0m')
            break

        # ============================================================
        # 2) Persist assistant response into dialogue
        # ============================================================
        messages.append({"role": "assistant", "content": response})

        if skill_module is not None:
            skill_reply = skill_module.handle_agent_skill_request(task_text, response, current_steps + 1)
            if skill_reply:
                print('\033[94mHandled agent-requested skill access.\033[0m')
                messages.append({"role": "user", "content": skill_reply})
                continue

        # ============================================================
        # 3) ACTION PARSING PHASE — Extract executable command
        #    Expected format: "Action: <command>"
        # ============================================================
        action = parse_action(response)

        # ============================================================
        # 4) ENVIRONMENT INTERACTION PHASE — Execute Action
        #    - Must use env.step()
        #    - Do not alter the list wrapping behavior
        #    - Task completion (task_done) and reward MUST ONLY be
        #      derived from env.step() outputs. DO NOT INFER.
        # ============================================================
        action_list = [action]
        observation, reward, task_done, info = env.step(action_list)

        # ============================================================
        # 5) OBSERVATION NORMALIZATION PHASE
        #    - Convert env raw outputs into semantic signals
        #    - Do NOT modify this section
        # ============================================================
        observation, reward, task_done = (
            process_ob(observation[0]),
            info['won'][0],
            task_done[0]
        )

        if reward or task_done:
            task_done = True

        print(f'\033[93mObservation: \n{observation}\033[0m')

        # ============================================================
        # 6) FEEDBACK PHASE — Feed environment signal back to agent
        # ============================================================
        messages.append({"role": "user", "content": f"Observation: {observation}"})

        if task_done:
            print(f'\033[92mWhole Task completed! Reward: {reward}\033[0m')
            current_steps += 1
            break

        if observation:
            observation_lower = observation.lower()
            if 'nothing happens' in observation_lower and len(messages) >= 2:
                prev = messages[-2].get('content', '')
                if isinstance(prev, str):
                    prev_lower = prev.lower()
                    if 'task is complete' in prev_lower or 'i will stop here' in prev_lower:
                        task_done = True
                        break
            elif any(marker in observation_lower for marker in [
                'you move the',
                'you cool the',
                'you heat the',
                'you clean the',
                'you turn on the',
            ]):
                messages.append({
                    "role": "user",
                    "content": "If this observation already satisfied the goal, stop immediately. Do not add verification steps or extra navigation.",
                })

        # ==============================================================================
        # [SECTION] RUNTIME ANALYSIS & INTERVENTION (Optional, Leave empty if not needed.)
        # INSTRUCTION: Analyze 'observation' or 'response'. 
        #              You can append EXTRA hints to 'messages' here.
        #              WARNING: Do NOT use 'continue', 'break', or 'return'.
        # ==============================================================================
        # Example (change as needed):
        if False and "<specific_condition>" in observation:
            extra_hint = "<extra_hint_here>"
            messages.append({"role": "user", "content": extra_hint})
        # ==============================================================

        # ============================================================
        # 7) STEP ACCOUNTING
        # ============================================================
        current_steps += 1

    return messages, task_done, reward, current_steps
