from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..request_context import RequestContext

# Import and register all provider models
from .anthropic import ANTHROPIC_MODELS
from .bedrock import BEDROCK_MODELS
from .cerebras import CEREBRAS_MODELS
from .cohere import COHERE_MODELS
from .deepseek import DEEPSEEK_MODELS
from .fireworks import FIREWORKS_MODELS
from .google import GOOGLE_MODELS
from .grok import XAI_MODELS
from .groq import GROQ_MODELS
from .kimi import KIMI_MODELS
from .meta import META_MODELS
from .minimax import MINIMAX_MODELS
from .mistral import MISTRAL_MODELS
from .openai import OPENAI_MODELS
from .openrouter import OPENROUTER_MODELS
from .together import TOGETHER_MODELS


@dataclass
class APIModel:
    id: str
    name: str
    api_base: str
    api_key_env_var: str
    api_spec: str
    cached_input_cost: float | None = 0  # $ per million cached/read input tokens
    cache_write_cost: float | None = 0  # $ per million cache write tokens
    input_cost: float | None = 0  # $ per million input tokens
    output_cost: float | None = 0  # $ per million output tokens
    supports_json: bool = False
    supports_logprobs: bool = False
    supports_responses: bool = False
    reasoning_model: bool = False
    regions: list[str] | dict[str, int] = field(default_factory=list)
    # tokens_per_minute: int | None = None
    # requests_per_minute: int | None = None
    # gpus: list[str] | None = None

    @classmethod
    def from_registry(cls, name: str):
        if name not in registry:
            raise ValueError(f"Model {name} not found in registry")
        cfg = registry[name]
        if isinstance(cfg, APIModel):
            return cfg
        return cls(**cfg)

    def sample_region(self):
        if isinstance(self.regions, list):
            regions = self.regions
            weights = [1] * len(regions)
        elif isinstance(self.regions, dict):
            regions = list(self.regions.keys())
            weights = self.regions.values()
        else:
            raise ValueError("no regions to sample")
        random.sample(regions, 1, counts=weights)[0]

    def make_request(self, context: RequestContext):
        from ..api_requests.common import CLASSES

        api_spec = self.api_spec
        if (
            context.use_responses_api
            and self.supports_responses
            and api_spec == "openai"
        ):
            api_spec = "openai-responses"

        request_class = CLASSES.get(api_spec, None)
        if request_class is None:
            raise ValueError(f"Unsupported API spec: {api_spec}")
        return request_class(context=context)


registry: dict[str, APIModel] = {}


def register_model(
    id: str,
    name: str,
    api_base: str,
    api_key_env_var: str,
    api_spec: str = "openai",
    input_cost: float | None = 0,  # $ per million input tokens
    cached_input_cost: float | None = 0,
    cache_write_cost: float | None = 0,  # $ per million cache write tokens
    output_cost: float | None = 0,  # $ per million output tokens
    supports_json: bool = False,
    supports_logprobs: bool = False,
    supports_responses: bool = False,
    reasoning_model: bool = False,
    regions: list[str] | dict[str, int] = field(default_factory=list),
    # tokens_per_minute: int | None = None,
    # requests_per_minute: int | None = None,
) -> APIModel:
    """Register a model configuration and return the created APIModel."""
    model = APIModel(
        id=id,
        name=name,
        api_base=api_base,
        api_key_env_var=api_key_env_var,
        api_spec=api_spec,
        cached_input_cost=cached_input_cost,
        cache_write_cost=cache_write_cost,
        input_cost=input_cost,
        output_cost=output_cost,
        supports_json=supports_json,
        supports_logprobs=supports_logprobs,
        supports_responses=supports_responses,
        reasoning_model=reasoning_model,
        regions=regions,
        # tokens_per_minute=tokens_per_minute,
        # requests_per_minute=requests_per_minute,
    )
    registry[model.id] = model
    return model


# Register all models from all providers
for model_dict in [
    ANTHROPIC_MODELS,
    BEDROCK_MODELS,
    COHERE_MODELS,
    DEEPSEEK_MODELS,
    FIREWORKS_MODELS,
    GOOGLE_MODELS,
    XAI_MODELS,
    KIMI_MODELS,
    META_MODELS,
    MINIMAX_MODELS,
    MISTRAL_MODELS,
    OPENAI_MODELS,
    OPENROUTER_MODELS,
    TOGETHER_MODELS,
    GROQ_MODELS,
    CEREBRAS_MODELS,
]:
    for cfg in model_dict.values():
        register_model(**cfg)


# print("Valid models:", registry.keys())
