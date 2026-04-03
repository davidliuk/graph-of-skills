import os
from openai import OpenAI
import json
from retry import retry

try:
    from .token_usage import get_usage_debug_fields, record_usage
except ImportError:
    from token_usage import get_usage_debug_fields, record_usage

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
def get_llm_response(messages, is_string=False, model="gpt-4o"):
    message_count, total_chars = _message_stats(messages)
    print(
        f'[evaluation.utils] Calling LLM model={model} '
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
            f'[evaluation.utils] LLM request failed '
            f'(type={type(exc).__name__}, model={model}, timeout={LLM_REQUEST_TIMEOUT_SECS}s, '
            f'messages={message_count}, chars={total_chars}). '
            f'Last message preview: {_last_message_preview(messages)}. '
            f'Error: {exc}'
        )
        raise

    usage = getattr(response, "usage", None)
    record_usage(usage, bucket="aux")
    if usage is not None:
        usage_fields = get_usage_debug_fields(usage)
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
            f"[evaluation.utils] LLM usage: {'; '.join(usage_parts)}"
        )

    if not hasattr(response, "error"):
        ans = response.choices[0].message.content
        if is_string:
            return ans
        else:
            cleaned_text = ans.strip("`json\n").strip("`\n").strip("```\n")
            ans = json.loads(cleaned_text)
            return ans
    else:
        raise Exception(response.error.message)
