from dataclasses import dataclass, field
from typing import Callable, Any

from .prompt import Conversation, CachePattern
from .config import SamplingParams
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

    # Model fallback configuration
    all_model_names: list[str] | None = None
    all_sampling_params: list[SamplingParams] | None = None

    # Optional features
    tools: list | None = None
    cache: CachePattern | None = None
    computer_use: bool = False
    display_width: int = 1024
    display_height: int = 768
    use_responses_api: bool = False

    # Computed properties
    cache_key: str = field(init=False)
    num_tokens: int = field(init=False)

    def __post_init__(self):
        # Compute cache key from prompt fingerprint
        self.cache_key = self.prompt.fingerprint

        # Compute token count
        self.num_tokens = self.prompt.count_tokens(self.sampling_params.max_new_tokens)

        # Validate required fields
        if self.all_model_names is None:
            self.all_model_names = [self.model_name]

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
            "all_model_names": self.all_model_names,
            "all_sampling_params": self.all_sampling_params,
            "tools": self.tools,
            "cache": self.cache,
            "computer_use": self.computer_use,
            "display_width": self.display_width,
            "display_height": self.display_height,
            "use_responses_api": self.use_responses_api,
        }

        # Update with any overrides
        current_values.update(overrides)

        return RequestContext(**current_values)
