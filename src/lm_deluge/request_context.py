from dataclasses import dataclass, field
from typing import Any, Callable

from .config import SamplingParams
from .prompt import CachePattern, Conversation
from .tracker import StatusTracker


@dataclass
class RequestContext:
    """
    Encapsulates all the parameters needed for an API request.
    This reduces parameter shuttling and makes the request lifecycle clearer.
    """

    # Core request parameters
    task_id: int
    model_name: str
    prompt: Conversation
    sampling_params: SamplingParams

    # Request configuration
    attempts_left: int = 5
    request_timeout: int = 30

    # Infrastructure
    status_tracker: StatusTracker | None = None
    results_arr: list[Any] | None = (
        None  # list["APIRequestBase"] but avoiding circular import
    )
    callback: Callable | None = None

    # Optional features
    tools: list | None = None
    cache: CachePattern | None = None
    use_responses_api: bool = False
    extra_headers: dict[str, str] | None = None

    # Computed properties
    cache_key: str = field(init=False)
    num_tokens: int = field(init=False)

    def __post_init__(self):
        # Compute cache key from prompt fingerprint
        self.cache_key = self.prompt.fingerprint

        # Compute token count
        self.num_tokens = self.prompt.count_tokens(self.sampling_params.max_new_tokens)

    def maybe_callback(self, response, tracker):
        if not self.callback:
            return
        self.callback(response, tracker)

    def copy(self, **overrides):
        """Create a copy of this RequestContext with optional field overrides."""
        # Get all current field values
        current_values = {
            "task_id": self.task_id,
            "model_name": self.model_name,
            "prompt": self.prompt,
            "sampling_params": self.sampling_params,
            "attempts_left": self.attempts_left,
            "request_timeout": self.request_timeout,
            "status_tracker": self.status_tracker,
            "results_arr": self.results_arr,
            "callback": self.callback,
            "tools": self.tools,
            "cache": self.cache,
            "use_responses_api": self.use_responses_api,
        }

        # Update with any overrides
        current_values.update(overrides)

        return RequestContext(**current_values)
