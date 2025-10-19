import os
import warnings

WARNINGS: dict[str, str] = {
    "WARN_JSON_MODE_UNSUPPORTED": "JSON mode requested for {model_name} but response_format parameter not supported.",
    "WARN_REASONING_UNSUPPORTED": "Ignoring reasoning_effort param for non-reasoning model: {model_name}.",
    "WARN_CACHING_UNSUPPORTED": "Cache parameter '{cache_param}' is not supported for Gemini models, ignoring for {model_name}.",
    "WARN_LOGPROBS_UNSUPPORTED": "Ignoring logprobs param for non-logprobs model: {model_name}",
}


def maybe_warn(warning: str, **kwargs):
    if os.getenv(warning):
        pass
    else:
        warnings.warn(WARNINGS[warning].format(**kwargs))
        os.environ[warning] = "1"
