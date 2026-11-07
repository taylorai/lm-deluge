from typing import Annotated, Literal, TypeAlias, overload

from pydantic import BaseModel, Field, model_validator

ReasoningEffort: TypeAlias = Literal[
    "none", "minimal", "low", "medium", "high", "xhigh", "max"
]
# Numeric effort is only accepted by providers that support it (Tinker).
FloatReasoningEffort: TypeAlias = Annotated[float, Field(ge=0.0, le=0.99)]
ReasoningEffortValue: TypeAlias = ReasoningEffort | FloatReasoningEffort


@overload
def named_reasoning_effort(effort: ReasoningEffortValue) -> ReasoningEffort: ...
@overload
def named_reasoning_effort(effort: None) -> None: ...
@overload
def named_reasoning_effort(
    effort: ReasoningEffortValue | None,
) -> ReasoningEffort | None: ...
def named_reasoning_effort(
    effort: ReasoningEffortValue | None,
) -> ReasoningEffort | None:
    """Narrow to a named effort level for providers that only accept those.

    APIModel.make_request already rejects numeric effort for such models; this
    keeps provider code type-safe and fails loudly if that gate is bypassed.
    """
    if isinstance(effort, (int, float)):
        raise TypeError(f"numeric reasoning_effort {effort!r} is not supported here")
    return effort


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
    reasoning_effort: ReasoningEffortValue | None = None
    thinking_budget: int | None = None
    logprobs: bool = False
    top_logprobs: int | None = None
    strict_tools: bool = True
    # Gemini 3 only - controls multimodal vision processing fidelity
    media_resolution: MediaResolution | None = None
    # Anthropic Opus 4.7+ only - advisory token budget for the full agentic loop.
    # Minimum 20_000 per Anthropic's API; we don't enforce that here.
    task_budget: int | None = None

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
