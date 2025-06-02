import json
import random
from dataclasses import dataclass

from lm_deluge.prompt import Conversation, Message
from lm_deluge.usage import Usage

from ..config import SamplingParams
from ..models import APIModel


@dataclass
class APIResponse:
    # request information
    id: int  # should be unique to the request within a given prompt-processing call
    model_internal: str  # our internal model tag
    prompt: Conversation
    sampling_params: SamplingParams

    # http response information
    status_code: int | None
    is_error: bool | None
    error_message: str | None

    # completion information - unified usage tracking
    usage: Usage | None = None

    # response content - structured format
    content: Message | None = None

    # optional or calculated automatically
    thinking: str | None = None  # if model shows thinking tokens
    model_external: str | None = None  # the model tag used by the API
    region: str | None = None
    logprobs: list | None = None
    finish_reason: str | None = None  # make required later
    cost: float | None = None  # calculated automatically
    cache_hit: bool = False  # manually set if true
    # set to true if is_error and should be retried with a different model
    retry_with_different_model: bool | None = False
    # set to true if should NOT retry with the same model (unrecoverable error)
    give_up_if_no_other_models: bool | None = False
    # OpenAI Responses API specific - used for computer use continuation
    response_id: str | None = None
    # Raw API response for debugging
    raw_response: dict | None = None

    @property
    def completion(self) -> str | None:
        """Backward compatibility: extract text from content Message."""
        if self.content is not None:
            return self.content.completion
        return None

    @property
    def input_tokens(self) -> int | None:
        """Get input tokens from usage object."""
        return self.usage.input_tokens if self.usage else None

    @property
    def output_tokens(self) -> int | None:
        """Get output tokens from usage object."""
        return self.usage.output_tokens if self.usage else None

    @property
    def cache_read_tokens(self) -> int | None:
        """Get cache read tokens from usage object."""
        return self.usage.cache_read_tokens if self.usage else None

    @property
    def cache_write_tokens(self) -> int | None:
        """Get cache write tokens from usage object."""
        return self.usage.cache_write_tokens if self.usage else None

    def __post_init__(self):
        # calculate cost & get external model name
        self.id = int(self.id)
        api_model = APIModel.from_registry(self.model_internal)
        self.model_external = api_model.name
        self.cost = None
        if (
            self.usage is not None
            and api_model.input_cost is not None
            and api_model.output_cost is not None
        ):
            self.cost = (
                self.usage.input_tokens * api_model.input_cost / 1e6
                + self.usage.output_tokens * api_model.output_cost / 1e6
            )
        elif self.content is not None and self.completion is not None:
            print(
                f"Warning: Completion provided without token counts for model {self.model_internal}."
            )

    def to_dict(self):
        return {
            "id": self.id,
            "model_internal": self.model_internal,
            "model_external": self.model_external,
            "region": self.region,
            "prompt": self.prompt.to_log(),  # destroys image if present
            "sampling_params": self.sampling_params.__dict__,
            "status_code": self.status_code,
            "is_error": self.is_error,
            "error_message": self.error_message,
            "completion": self.completion,  # computed property
            "content": self.content.to_log() if self.content else None,
            "usage": self.usage.to_dict() if self.usage else None,
            "finish_reason": self.finish_reason,
            "cost": self.cost,
        }

    @classmethod
    def from_dict(cls, data: dict):
        # Handle backward compatibility for content/completion
        content = None
        if "content" in data and data["content"] is not None:
            # Reconstruct message from log format
            content = Message.from_log(data["content"])
        elif "completion" in data and data["completion"] is not None:
            # Backward compatibility: create a Message with just text
            content = Message.ai(data["completion"])

        usage = None
        if "usage" in data and data["usage"] is not None:
            usage = Usage.from_dict(data["usage"])

        return cls(
            id=data.get("id", random.randint(0, 1_000_000_000)),
            model_internal=data["model_internal"],
            prompt=Conversation.from_log(data["prompt"]),
            sampling_params=SamplingParams(**data["sampling_params"]),
            status_code=data["status_code"],
            is_error=data["is_error"],
            error_message=data["error_message"],
            usage=usage,
            content=content,
            thinking=data.get("thinking"),
            model_external=data.get("model_external"),
            region=data.get("region"),
            logprobs=data.get("logprobs"),
            finish_reason=data.get("finish_reason"),
            cost=data.get("cost"),
            cache_hit=data.get("cache_hit", False),
        )

    def write_to_file(self, filename):
        """
        Writes the APIResponse as a line to a file.
        If file exists, appends to it.
        """
        with open(filename, "a") as f:
            f.write(json.dumps(self.to_dict()) + "\n")
