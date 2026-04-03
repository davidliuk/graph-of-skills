import os
from openai import OpenAI
import re
import inspect
import time
from retry import retry
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json
import argparse
from datetime import datetime, timezone
import yaml
import alfworld
import alfworld.agents.environment
from alfworld.agents.environment import get_environment
import sys
from pathlib import Path

# Add evaluation and project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent)
evaluation_dir = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if evaluation_dir not in sys.path:
    sys.path.insert(0, evaluation_dir)

from evaluation.alfworld.prompts.system_prompt import alfworld_system_prompt
from evaluation.skill import SkillModule
from evaluation.token_usage import (
    clear_token_usage_tracker as _clear_token_usage_tracker,
    get_usage_debug_fields as _get_usage_debug_fields,
    new_token_usage as _new_token_usage,
    record_usage as _record_usage,
    set_token_usage_tracker as _set_token_usage_tracker,
)


DEFAULT_SKILLS_DIR = "data/skillsets/skills_200"
DEFAULT_GOS_WORKSPACE = "data/gos_workspace/skills_200_v1"
LLM_REQUEST_TIMEOUT_SECS = float(os.environ.get("LLM_REQUEST_TIMEOUT_SECS", "90"))

client = OpenAI(
    api_key=os.environ["API_KEY"],
    base_url=os.environ["BASE_URL"]
)

def _message_stats(messages):
    total_chars = 0
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
    return len(messages), total_chars


def _last_message_preview(messages, limit=240):
    if not messages:
        return "<empty>"

    content = messages[-1].get("content", "")
    if not isinstance(content, str):
        return "<non-string content>"

    compact = " ".join(content.split())
    if len(compact) > limit:
        return compact[:limit] + "..."
    return compact

@retry(tries=5, delay=5, backoff=2, jitter=(1, 3))
def llm(prompt, model="YOUR_MODEL_NAME"):
    if isinstance(prompt, list):
        messages = prompt
    elif isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        raise ValueError(f'prompt must be a list or a string, but got {type(prompt)}')

    message_count, total_chars = _message_stats(messages)
    print(
        f'Calling LLM with model: {model} '
        f'(messages={message_count}, chars={total_chars}, timeout={LLM_REQUEST_TIMEOUT_SECS}s)'
    )
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            timeout=LLM_REQUEST_TIMEOUT_SECS,
        )
    except Exception as exc:
        print(
            f'{Colors.RED}LLM request failed '
            f'(type={type(exc).__name__}, model={model}, timeout={LLM_REQUEST_TIMEOUT_SECS}s, '
            f'messages={message_count}, chars={total_chars}). '
            f'Last message preview: {_last_message_preview(messages)}. '
            f'Error: {exc}{Colors.RESET}'
        )
        raise

    usage = getattr(response, "usage", None)
    _record_usage(usage, bucket="agent")
    if usage is not None:
        usage_fields = _get_usage_debug_fields(usage)
        usage_parts = [
            f"prompt={usage_fields.get('prompt_tokens')}",
            f"completion={usage_fields.get('completion_tokens')}",
            f"total={usage_fields.get('total_tokens')}",
        ]
        if "cached_prompt_tokens" in usage_fields:
            usage_parts.append(f"cached_prompt={usage_fields['cached_prompt_tokens']}")
        if "cache_creation_input_tokens" in usage_fields:
            usage_parts.append(f"cache_create={usage_fields['cache_creation_input_tokens']}")
        if "reasoning_tokens" in usage_fields:
            usage_parts.append(f"reasoning={usage_fields['reasoning_tokens']}")
        print(
            f"{Colors.BLUE}LLM usage: {'; '.join(usage_parts)}{Colors.RESET}"
        )

    content = response.choices[0].message.content
    if content is not None:
        return content
    return "Output Error"

def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]    
    return ob

class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

def parse_action(response: str) -> str:
    pattern = re.compile(r"Action:\s*(.+)", re.IGNORECASE)
    match = pattern.search(response)
    if match:
        return match.group(1).strip().strip('"\'*`')
    return ""


def build_skill_config(args):
    gos_workspace = args.gos_workspace
    if args.mode in {'gos', 'vector'} and not gos_workspace:
        gos_workspace = DEFAULT_GOS_WORKSPACE

    resolved_skills_dir = str(Path(args.skills_dir).expanduser().resolve())
    resolved_workspace = str(Path(gos_workspace).expanduser().resolve()) if gos_workspace else None

    if args.mode in {'gos', 'vector'} and resolved_workspace:
        workspace_name = Path(resolved_workspace).name
        skills_name = Path(resolved_skills_dir).name
        if workspace_name.startswith('skills_') and skills_name.startswith('skills_'):
            valid_names = {skills_name, f'{skills_name}_v1'}
            if not (workspace_name in valid_names or workspace_name.startswith(f'{skills_name}_v')):
                raise ValueError(
                    f'goss workspace / skills_dir mismatch: workspace={resolved_workspace}, skills_dir={resolved_skills_dir}'
                )

    return {
        "skills_dir": resolved_skills_dir,
        "model": args.model,
        "mode": args.mode,
        "gos_workspace": resolved_workspace,
        "enable_alfworld_gating": args.enable_alfworld_gating,
    }


def parse_task_indices(raw_value):
    if not raw_value:
        return None

    parsed = []
    for chunk in raw_value.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        parsed.append(int(chunk))
    return parsed or None

def run_standard_procedure(env, llm, model, process_ob, messages, max_steps):
    task_done = False
    task_reward = 0
    current_steps = 0

    while not task_done and current_steps < max_steps:
        current_steps += 1
        try:
            response = llm(messages, model)
            print(f'{Colors.GREEN}Agent response: \n{response}{Colors.RESET}')
        except Exception as e:
            print(f'{Colors.RED}Error in LLM call: {e}{Colors.RESET}')
            break

        messages.append({"role": "assistant", "content": response})
        action = parse_action(response)
        action_list = [action]
        
        observation, task_reward, done, info = env.step(action_list)
        observation, task_reward, task_done = (
            process_ob(observation[0]),
            info["won"][0],
            done[0]
        )
        print(f'{Colors.YELLOW}Observation: \n{observation}{Colors.RESET}')
        messages.append({"role": "user", "content": f"Observation: {observation}"})

        if task_done:
            print(f'{Colors.GREEN}Whole Task completed! Reward: {task_reward}{Colors.RESET}')
            break

    return messages, task_done, task_reward, current_steps


def _maybe_handle_skill_request(messages, response, skill_module, task_text, current_steps):
    if skill_module is None:
        return False

    skill_reply = skill_module.handle_agent_skill_request(task_text, response, current_steps)
    if not skill_reply:
        return False

    print(f'{Colors.BLUE}Handled agent-requested skill access.{Colors.RESET}')
    messages.append({"role": "user", "content": skill_reply})
    return True


def run_standard_procedure_with_skill_module(env, llm, model, process_ob, messages, max_steps, skill_module, task_text):
    task_done = False
    task_reward = 0
    current_steps = 0

    while not task_done and current_steps < max_steps:
        current_steps += 1
        try:
            response = llm(messages, model)
            print(f'{Colors.GREEN}Agent response: \n{response}{Colors.RESET}')
        except Exception as e:
            print(f'{Colors.RED}Error in LLM call: {e}{Colors.RESET}')
            break

        messages.append({"role": "assistant", "content": response})
        if _maybe_handle_skill_request(messages, response, skill_module, task_text, current_steps):
            continue

        action = parse_action(response)
        action_list = [action]

        observation, task_reward, done, info = env.step(action_list)
        observation, task_reward, task_done = (
            process_ob(observation[0]),
            info["won"][0],
            done[0]
        )
        print(f'{Colors.YELLOW}Observation: \n{observation}{Colors.RESET}')
        messages.append({"role": "user", "content": f"Observation: {observation}"})

        runtime_hint = skill_module.maybe_get_runtime_skill_hint(task_text, messages, observation, current_steps)
        if runtime_hint:
            print(f'{Colors.BLUE}Injected runtime skill hint after observation failure.{Colors.RESET}')
            messages.append({"role": "user", "content": runtime_hint})

        if task_done:
            print(f'{Colors.GREEN}Whole Task completed! Reward: {task_reward}{Colors.RESET}')
            break

    return messages, task_done, task_reward, current_steps

def alfworld_run_single(env, obs=[], names=[], max_steps=30, model=None, Skill_Module=None):
    results = []
    for task_idx, (ob, name) in enumerate(zip(obs, names)):
        print(f'{Colors.RED}Processing task {task_idx + 1}/{len(obs)}: {name}{Colors.RESET}')
        query = ob.split('Your task is to: ')[-1].split('\n')[0].strip()
        messages = [{"role": "system", "content": alfworld_system_prompt}]

        if Skill_Module is not None:
            all_full_exposures = Skill_Module.get_all_full_exposure_messages()
            if all_full_exposures:
                print(f'{Colors.BLUE}Injected full skill metadata exposure into initial dialogue ({len(all_full_exposures)} messages).{Colors.RESET}')
                for exposure_message in all_full_exposures:
                    messages.append({"role": "user", "content": exposure_message})
            skill_request_message = Skill_Module.get_agent_skill_request_message()
            if skill_request_message:
                print(f'{Colors.BLUE}Injected skill request protocol into initial dialogue.{Colors.RESET}')
                messages.append({"role": "user", "content": skill_request_message})

        messages.append({"role": "user", "content": ob})
        
        task_done = False
        task_reward = 0
        steps = 0
        relevant_skill_names = []
        retrieval_status = "NOT_RUN"
        retrieval_summary = ""
        retrieval_query = ""
        runtime_skill_events = []
        token_usage = _new_token_usage()
        started_at = datetime.now(timezone.utc).isoformat()
        agent_start_time = time.perf_counter()
        finished_at = started_at
        agent_runtime_seconds = 0.0

        _set_token_usage_tracker(token_usage)
        try:
            if Skill_Module is not None:
                Skill_Module.retrieve_relevant_skills(ob)
                relevant_skill_names = list(Skill_Module.last_retrieved_skill_names)
                retrieval_status = Skill_Module.last_retrieval_status
                retrieval_summary = Skill_Module.last_retrieval_summary
                retrieval_query = Skill_Module.last_retrieval_query

                retrieval_guidance = Skill_Module.get_retrieval_guidance()
                if retrieval_guidance:
                    print(f'{Colors.BLUE}Injected GoS retrieval guidance into dialogue.{Colors.RESET}')
                    messages.append({"role": "user", "content": retrieval_guidance})

                if relevant_skill_names:
                    print(f'{Colors.BLUE}Retrieved relevant skills.{Colors.RESET}')
                    print(f'{Colors.BLUE}Using lightweight retrieval guidance only; no procedure generation.{Colors.RESET}')
                else:
                    print(f"[INFO] No relevant skills found. Falling back.")

            if Skill_Module is not None and Skill_Module.mode in {'gos', 'vector', 'all_full'}:
                messages, task_done, task_reward, steps = run_standard_procedure_with_skill_module(
                    env, llm, model, process_ob, messages, max_steps, Skill_Module, ob
                )
            else:
                messages, task_done, task_reward, steps = run_standard_procedure(
                    env, llm, model, process_ob, messages, max_steps
                )

            if Skill_Module is not None:
                runtime_skill_events = Skill_Module.get_runtime_skill_events()
        finally:
            finished_at = datetime.now(timezone.utc).isoformat()
            agent_runtime_seconds = round(time.perf_counter() - agent_start_time, 3)
            _clear_token_usage_tracker()
        
        results.append({
            "query": query,
            "name": name,
            "task_done": task_done,
            "reward": task_reward,
            "steps": steps,
            "messages": messages,
            "relevant_skill_names": relevant_skill_names,
            "retrieval_status": retrieval_status,
            "retrieval_summary": retrieval_summary,
            "retrieval_query": retrieval_query,
            "runtime_skill_events": runtime_skill_events,
            "token_usage": token_usage,
            "started_at": started_at,
            "finished_at": finished_at,
            "agent_runtime_seconds": agent_runtime_seconds,
        })
    return results

def eval_single_game(game_idx, args, config, split, output_path):
    env = None
    try:
        env = get_environment(config["env"]["type"])(config, train_eval=split)
        env = env.init_env(batch_size=1)
        obs_list = []
        info = {}
        for _ in range(game_idx + 1):
            obs_list, info = env.reset()
            
        Skill_Module = None
        if args.use_skill:
            Skill_Module = SkillModule(**build_skill_config(args))

        ob_str = '\n'.join(obs_list[0].split('\n\n')[1:])
        game_name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
        
        batch_results = alfworld_run_single(
            env=env,
            obs=[ob_str], 
            names=[game_name], 
            max_steps=args.max_steps,
            model=args.model,
            Skill_Module=Skill_Module
        )
        result = batch_results[0]
        save_file = f'{output_path}/idx_{game_idx}.json'
        with open(save_file, 'w') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        return result
    except Exception as e:
        print(f"Error in game {game_idx}: {e}")
        return None
    finally:
        if env:
            env.close()

def main(args):
    model_name = args.model
    with open('evaluation/alfworld/base_config.yaml') as reader:
        config = yaml.safe_load(reader)
    split = "eval_in_distribution" if args.split == 'dev' else "eval_out_of_distribution"
    output_path = f'results/alfworld/{model_name}/{args.split}_{args.exp_name}_mode_{args.mode}'
    os.makedirs(output_path, exist_ok=True)

    temp_env = get_environment(config["env"]["type"])(config, train_eval=split)
    temp_env = temp_env.init_env(batch_size=1)
    num_games = len(temp_env.gamefiles)
    del temp_env 

    tasks_to_run = []
    finished_games = 0
    all_rewards = 0
    all_steps = 0
    existing_files = set()
    if os.path.exists(output_path):
        for file in os.listdir(output_path):
            if file.endswith('.json') and file.startswith('idx_'):
                try:
                    idx = int(file.split('_')[1].split('.')[0])
                    existing_files.add(idx)
                    with open(f'{output_path}/{file}', 'r') as f:
                        res = json.load(f)
                        all_rewards += res['reward']
                        all_steps += res['steps']
                    finished_games += 1
                except: continue

    requested_indices = parse_task_indices(args.task_indices)
    candidate_indices = requested_indices if requested_indices is not None else list(range(num_games))
    if args.max_games is not None:
        candidate_indices = candidate_indices[: args.max_games]

    for idx in candidate_indices:
        if idx not in existing_files:
            tasks_to_run.append(idx)

    max_workers = args.max_workers
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(eval_single_game, idx, args, config, split, output_path): idx for idx in tasks_to_run}
        pbar = tqdm(total=len(tasks_to_run), desc="Evaluating ALFWorld")
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                if result:
                    finished_games += 1
                    all_rewards += result['reward']
                    all_steps += result['steps']
                    pbar.set_postfix({'Avg R': f'{all_rewards/finished_games:.2f}'})
            except Exception as exc: print(f'\nGame {idx} error: {exc}')
            pbar.update(1)
        pbar.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt-4o')
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--max_workers', type=int, default=5)
    parser.add_argument('--max_steps', type=int, default=30)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--use_skill', action='store_true')
    parser.add_argument('--mode', type=str, default='gos', choices=['all_full', 'gos', 'vector', 'none'])
    parser.add_argument('--gos_workspace', type=str, default=None)
    parser.add_argument('--skills_dir', type=str, default=DEFAULT_SKILLS_DIR)
    parser.add_argument('--max_games', type=int, default=None)
    parser.add_argument('--task_indices', type=str, default=None)
    parser.add_argument('--enable_alfworld_gating', action='store_true')
    args = parser.parse_args()
    main(args)
