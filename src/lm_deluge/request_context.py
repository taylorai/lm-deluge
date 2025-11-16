from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Callable, TYPE_CHECKING

from .config import SamplingParams
from .prompt import CachePattern, Conversation
from .tracker import StatusTracker

if TYPE_CHECKING:
    from pydantic import BaseModel


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
    # avoiding circular import
    results_arr: list[Any] | None = None  # list["APIRequestBase"]
    callback: Callable | None = None

    # Optional features
    tools: list | None = None
    output_schema: "type[BaseModel] | dict | None" = None
    cache: CachePattern | None = None
    use_responses_api: bool = False
    background: bool = False
    service_tier: str | None = None
    extra_headers: dict[str, str] | None = None
    extra_body: dict[str, Any] | None = None
    force_local_mcp: bool = False

    # Computed properties
    cache_key: str = field(init=False)

    @cached_property
    def num_tokens(self):
        return self.prompt.count_tokens(self.sampling_params.max_new_tokens)

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
            "output_schema": self.output_schema,
            "cache": self.cache,
            "use_responses_api": self.use_responses_api,
            "background": self.background,
            "service_tier": self.service_tier,
            "extra_headers": self.extra_headers,
            "extra_body": self.extra_body,
            "force_local_mcp": self.force_local_mcp,
        }

        # Update with any overrides
        current_values.update(overrides)

        return RequestContext(**current_values)
