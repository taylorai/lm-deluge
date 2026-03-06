from typing import Literal, TypeAlias

from pydantic import BaseModel, model_validator

ReasoningEffort: TypeAlias = Literal[
    "none", "minimal", "low", "medium", "high", "xhigh"
]
GlobalEffort: TypeAlias = Literal[
    "none", "minimal", "low", "medium", "high", "xhigh", "max"
]
MediaResolution: TypeAlias = Literal[
    "media_resolution_low", "media_resolution_medium", "media_resolution_high"
]


class SamplingParams(BaseModel):
    temperature: float = 1.0  # more typical for new models
    top_p: float = 1.0
    json_mode: bool = False
    max_new_tokens: int = 2_048
    global_effort: GlobalEffort | None = None
    verbosity: GlobalEffort | None = None
    reasoning_effort: ReasoningEffort | None = None
    thinking_budget: int | None = None
    logprobs: bool = False
    top_logprobs: int | None = None
    strict_tools: bool = True
    # Gemini 3 only - controls multimodal vision processing fidelity
    media_resolution: MediaResolution | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_output_effort_aliases(cls, data):
        if not isinstance(data, dict):
            return data

        verbosity_present = "verbosity" in data
        global_effort_present = "global_effort" in data
        verbosity = data.get("verbosity")
        global_effort = data.get("global_effort")

        if (
            verbosity_present
            and global_effort_present
            and verbosity is not None
            and global_effort is not None
            and verbosity != global_effort
        ):
            raise ValueError(
                "verbosity and global_effort must match when both are provided"
            )

        if verbosity is not None and global_effort is None:
            data["global_effort"] = verbosity
        elif global_effort is not None and verbosity is None:
            data["verbosity"] = global_effort

        return data
