from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..api_requests.context import RequestContext

# Import and register all provider models
from .anthropic import ANTHROPIC_MODELS
from .arcee import ARCEE_MODELS
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
from .zai import ZAI_MODELS


@dataclass
class APIModel:
    id: str
    name: str
    api_base: str
    api_key_env_var: str
    api_spec: str
    provider: str = ""  # The actual provider (anthropic, openai, together, etc.)
    cached_input_cost: float | None = 0  # $ per million cached/read input tokens
    cache_write_cost: float | None = 0  # $ per million cache write tokens
    input_cost: float | None = 0  # $ per million input tokens
    output_cost: float | None = 0  # $ per million output tokens
    supports_json: bool = False
    supports_images: bool = False
    supports_logprobs: bool = False
    supports_responses: bool = False
    reasoning_model: bool = False
    supports_xhigh: bool = False
    regions: list[str] | dict[str, int] = field(default_factory=list)

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
        return random.sample(regions, 1, counts=weights)[0]

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
    provider: str = "",
    input_cost: float | None = 0,  # $ per million input tokens
    cached_input_cost: float | None = 0,
    cache_write_cost: float | None = 0,  # $ per million cache write tokens
    output_cost: float | None = 0,  # $ per million output tokens
    supports_json: bool = False,
    supports_images: bool = False,
    supports_logprobs: bool = False,
    supports_responses: bool = False,
    reasoning_model: bool = False,
    supports_xhigh: bool = False,
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
        provider=provider,
        cached_input_cost=cached_input_cost,
        cache_write_cost=cache_write_cost,
        input_cost=input_cost,
        output_cost=output_cost,
        supports_json=supports_json,
        supports_images=supports_images,
        supports_logprobs=supports_logprobs,
        supports_responses=supports_responses,
        reasoning_model=reasoning_model,
        supports_xhigh=supports_xhigh,
        regions=regions,
        # tokens_per_minute=tokens_per_minute,
        # requests_per_minute=requests_per_minute,
    )
    registry[model.id] = model
    return model


# Register all models from all providers
# Maps each model dict to its provider name
_PROVIDER_MODELS = [
    (ANTHROPIC_MODELS, "anthropic"),
    (ZAI_MODELS, "zai"),
    (ARCEE_MODELS, "arcee"),
    (BEDROCK_MODELS, "bedrock"),
    (COHERE_MODELS, "cohere"),
    (DEEPSEEK_MODELS, "deepseek"),
    (FIREWORKS_MODELS, "fireworks"),
    (GOOGLE_MODELS, "google"),
    (XAI_MODELS, "xai"),
    (KIMI_MODELS, "kimi"),
    (META_MODELS, "meta"),
    (MINIMAX_MODELS, "minimax"),
    (MISTRAL_MODELS, "mistral"),
    (OPENAI_MODELS, "openai"),
    (OPENROUTER_MODELS, "openrouter"),
    (TOGETHER_MODELS, "together"),
    (GROQ_MODELS, "groq"),
    (CEREBRAS_MODELS, "cerebras"),
]

for model_dict, provider_name in _PROVIDER_MODELS:
    for cfg in model_dict.values():
        register_model(**cfg, provider=provider_name)  # type: ignore[arg-type]


# print("Valid models:", registry.keys())


def find_models(
    *,
    provider: str | None = None,
    supports_json: bool | None = None,
    supports_images: bool | None = None,
    supports_logprobs: bool | None = None,
    reasoning_model: bool | None = None,
    min_input_cost: float | None = None,
    max_input_cost: float | None = None,
    min_output_cost: float | None = None,
    max_output_cost: float | None = None,
    name_contains: str | None = None,
    sort_by: str | None = None,
    limit: int | None = None,
) -> list[APIModel]:
    """
    Find models matching the given criteria.

    All parameters are optional filters. Only models matching ALL specified
    criteria are returned.

    Args:
        provider: Filter by provider (e.g., "openai", "anthropic", "together", "fireworks")
        supports_json: Filter by JSON mode support
        supports_images: Filter by image input support
        supports_logprobs: Filter by logprobs support
        reasoning_model: Filter by reasoning model capability
        min_input_cost: Minimum input cost ($ per million tokens)
        max_input_cost: Maximum input cost ($ per million tokens)
        min_output_cost: Minimum output cost ($ per million tokens)
        max_output_cost: Maximum output cost ($ per million tokens)
        name_contains: Filter by substring in model ID (case-insensitive)
        sort_by: Sort results by "input_cost", "output_cost", "-input_cost", "-output_cost"
        limit: Maximum number of results to return

    Returns:
        List of APIModel objects matching all criteria
    """
    results = list(registry.values())

    if provider is not None:
        results = [m for m in results if m.provider == provider]

    if supports_json is not None:
        results = [m for m in results if m.supports_json == supports_json]

    if supports_images is not None:
        results = [m for m in results if m.supports_images == supports_images]

    if supports_logprobs is not None:
        results = [m for m in results if m.supports_logprobs == supports_logprobs]

    if reasoning_model is not None:
        results = [m for m in results if m.reasoning_model == reasoning_model]

    if min_input_cost is not None:
        results = [
            m
            for m in results
            if m.input_cost is not None and m.input_cost >= min_input_cost
        ]

    if max_input_cost is not None:
        results = [
            m
            for m in results
            if m.input_cost is not None and m.input_cost <= max_input_cost
        ]

    if min_output_cost is not None:
        results = [
            m
            for m in results
            if m.output_cost is not None and m.output_cost >= min_output_cost
        ]

    if max_output_cost is not None:
        results = [
            m
            for m in results
            if m.output_cost is not None and m.output_cost <= max_output_cost
        ]

    if name_contains is not None:
        name_lower = name_contains.lower()
        results = [m for m in results if name_lower in m.id.lower()]

    if sort_by is not None:
        reverse = sort_by.startswith("-")
        field = sort_by.lstrip("-")
        if field == "input_cost":
            results = [m for m in results if m.input_cost is not None]
            results.sort(key=lambda m: m.input_cost or 0, reverse=reverse)
        elif field == "output_cost":
            results = [m for m in results if m.output_cost is not None]
            results.sort(key=lambda m: m.output_cost or 0, reverse=reverse)

    if limit is not None:
        results = results[:limit]

    return results
