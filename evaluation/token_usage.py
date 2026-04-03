import threading


_LLM_USAGE_TRACKER = threading.local()
_DETAIL_PATHS = {
    "cached_prompt_tokens": [
        ("prompt_tokens_details", "cached_tokens"),
        ("input_tokens_details", "cached_tokens"),
        ("cache_read_input_tokens",),
        ("cache_read_tokens",),
    ],
    "cache_creation_input_tokens": [
        ("prompt_tokens_details", "cache_creation_tokens"),
        ("input_tokens_details", "cache_creation_tokens"),
        ("cache_creation_input_tokens",),
    ],
    "reasoning_tokens": [
        ("completion_tokens_details", "reasoning_tokens"),
        ("output_tokens_details", "reasoning_tokens"),
    ],
}


def _new_bucket():
    return {
        "llm_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "last_prompt_tokens": 0,
        "last_completion_tokens": 0,
        "last_total_tokens": 0,
        "max_prompt_tokens": 0,
        "max_completion_tokens": 0,
        "max_total_tokens": 0,
        "cached_prompt_tokens": 0,
        "cache_creation_input_tokens": 0,
        "reasoning_tokens": 0,
    }


def new_token_usage():
    stats = _new_bucket()
    stats["buckets"] = {
        "agent": _new_bucket(),
        "aux": _new_bucket(),
    }
    return stats


def set_token_usage_tracker(stats):
    _LLM_USAGE_TRACKER.stats = stats


def clear_token_usage_tracker():
    if hasattr(_LLM_USAGE_TRACKER, "stats"):
        delattr(_LLM_USAGE_TRACKER, "stats")


def _safe_int(value):
    return value if isinstance(value, int) else None


def _get_usage_detail(usage, key):
    for path in _DETAIL_PATHS.get(key, []):
        current = usage
        for part in path:
            current = getattr(current, part, None)
            if current is None:
                break
        if isinstance(current, int):
            return current
    return None


def _apply_usage(stats, usage):
    stats["llm_calls"] += 1
    if usage is None:
        return

    numeric_keys = {
        "prompt_tokens": _safe_int(getattr(usage, "prompt_tokens", None)),
        "completion_tokens": _safe_int(getattr(usage, "completion_tokens", None)),
        "total_tokens": _safe_int(getattr(usage, "total_tokens", None)),
    }
    for key, value in numeric_keys.items():
        if value is None:
            continue
        stats[key] += value
        stats[f"last_{key}"] = value
        stats[f"max_{key}"] = max(stats[f"max_{key}"], value)

    for detail_key in ("cached_prompt_tokens", "cache_creation_input_tokens", "reasoning_tokens"):
        value = _get_usage_detail(usage, detail_key)
        if value is not None:
            stats[detail_key] += value


def record_usage(usage, bucket="agent"):
    stats = getattr(_LLM_USAGE_TRACKER, "stats", None)
    if stats is None:
        return

    _apply_usage(stats, usage)
    bucket_stats = stats.setdefault("buckets", {}).setdefault(bucket, _new_bucket())
    _apply_usage(bucket_stats, usage)


def get_usage_debug_fields(usage):
    if usage is None:
        return {}

    fields = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = _safe_int(getattr(usage, key, None))
        if value is not None:
            fields[key] = value

    for key in ("cached_prompt_tokens", "cache_creation_input_tokens", "reasoning_tokens"):
        value = _get_usage_detail(usage, key)
        if value is not None:
            fields[key] = value
    return fields
