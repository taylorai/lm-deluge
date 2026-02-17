import functools
import os
import warnings

WARNINGS: dict[str, str] = {
    "WARN_JSON_MODE_UNSUPPORTED": "JSON mode requested for {model_name} but response_format parameter not supported.",
    "WARN_REASONING_UNSUPPORTED": "Ignoring reasoning_effort param for non-reasoning model: {model_name}.",
    "WARN_CACHING_UNSUPPORTED": "Cache parameter '{cache_param}' is not supported, ignoring for {model_name}.",
    "WARN_LOGPROBS_UNSUPPORTED": "Ignoring logprobs param for non-logprobs model: {model_name}",
    "WARN_MINIMAL_TO_LOW": "'minimal' reasoning effort only allowed for gpt-5 models. Setting to 'low' for {model_name}.",
    "WARN_MINIMAL_TO_NONE": "GPT-5.1 models don't support 'minimal' reasoning effort. Converting to 'none' for {model_name}.",
    "WARN_XHIGH_TO_HIGH": "'xhigh' reasoning effort only supported for gpt-5.2 and gpt-5.1-codex-max. Using 'high' for {model_name}.",
    "WARN_MEDIA_RESOLUTION_UNSUPPORTED": "media_resolution parameter is only supported for Gemini 3 models, ignoring for {model_name}.",
    "WARN_GEMINI3_MISSING_SIGNATURE": "Gemini 3 thought signature missing in {part_type}, injecting dummy signature 'context_engineering_is_the_way_to_go' to avoid API error.",
    "WARN_GEMINI3_NO_REASONING": "Gemini 3 requires reasoning (thinkingConfig). Setting thinkingConfig to low.",
    "WARN_THINKING_BUDGET_AND_REASONING_EFFORT": "`reasoning_effort` and `thinking_budget` both provided. `thinking_budget` will take priority.",
    "WARN_KIMI_THINKING_NO_REASONING": "kimi-k2-thinking works best with thinking enabled. set thinking_budget > 0 or reasoning_effort to anything but none",
    "WARN_CLAUDE_46_BUDGET_TOKENS_DEPRECATED": "thinking budget_tokens is deprecated on Claude 4.6 models and will be removed in a future release. Use adaptive thinking (default) with the effort parameter instead.",
}


def maybe_warn(warning: str, **kwargs):
    if os.getenv(warning):
        pass
    else:
        warnings.warn(WARNINGS[warning].format(**kwargs))
        os.environ[warning] = "1"


def deprecated(replacement: str):
    """Decorator to mark methods as deprecated and suggest replacement.

    Only shows the warning once per method to avoid spam.

    Args:
        replacement: The name of the replacement method to suggest
    """

    def decorator(func):
        warning_key = f"DEPRECATED_{func.__module__}_{func.__qualname__}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not os.getenv(warning_key):
                warnings.warn(
                    f"{func.__name__} is deprecated, use {replacement} instead",
                    DeprecationWarning,
                    stacklevel=2,
                )
                os.environ[warning_key] = "1"
            return func(*args, **kwargs)

        return wrapper

    return decorator
