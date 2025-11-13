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
