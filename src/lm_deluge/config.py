from typing import Literal, TypeAlias

from pydantic import BaseModel

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
    global_effort: GlobalEffort = "high"  # for Anthropic Opus global effort
    reasoning_effort: ReasoningEffort | None = None
    thinking_budget: int | None = None
    logprobs: bool = False
    top_logprobs: int | None = None
    strict_tools: bool = True
    # Gemini 3 only - controls multimodal vision processing fidelity
    media_resolution: MediaResolution | None = None
