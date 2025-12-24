from __future__ import annotations

import random
from typing import Callable, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from lm_deluge.models import APIModel, registry

RouteStrategy = Literal["round_robin", "random", "weighted"]
PolicyMode = Literal["allow_user_pick", "force_default", "alias_only"]


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


class RouteConfig(BaseModel):
    models: list[str]
    strategy: RouteStrategy = "round_robin"
    weights: list[float] | None = None

    @model_validator(mode="after")
    def _validate_route(self) -> "RouteConfig":
        if not self.models:
            raise ValueError("route models must not be empty")
        if self.strategy == "weighted":
            if not self.weights:
                raise ValueError("weighted strategy requires weights")
            if len(self.weights) != len(self.models):
                raise ValueError("weights must match models length")
            if any(weight < 0 for weight in self.weights):
                raise ValueError("weights must be non-negative")
            if sum(self.weights) <= 0:
                raise ValueError("weights must sum to a positive value")
        return self


class ProxyModelPolicy(BaseModel):
    mode: PolicyMode = "allow_user_pick"
    allowed_models: list[str] | None = None
    default_model: str | None = None
    routes: dict[str, RouteConfig] = Field(default_factory=dict)
    expose_aliases: bool = False

    def validate_against_registry(self, model_registry: dict[str, APIModel]) -> None:
        registry_keys = set(model_registry.keys())
        allowed_models = (
            _dedupe_keep_order(self.allowed_models)
            if self.allowed_models is not None
            else None
        )

        if allowed_models is not None:
            unknown = [model for model in allowed_models if model not in registry_keys]
            if unknown:
                raise ValueError(f"Unknown allowed models: {', '.join(unknown)}")

        for alias, route in self.routes.items():
            if alias in registry_keys:
                raise ValueError(
                    f"Route alias '{alias}' conflicts with a registry model id"
                )
            for model_id in route.models:
                if model_id not in registry_keys:
                    raise ValueError(
                        f"Route '{alias}' references unknown model '{model_id}'"
                    )
                if allowed_models is not None and model_id not in allowed_models:
                    raise ValueError(
                        f"Route '{alias}' uses model '{model_id}' not in allowlist"
                    )

        if self.mode == "force_default" and not self.default_model:
            raise ValueError("force_default mode requires default_model")

        if self.default_model:
            if self.default_model not in self.routes:
                if self.default_model not in registry_keys:
                    raise ValueError(f"Default model '{self.default_model}' is unknown")
                if (
                    allowed_models is not None
                    and self.default_model not in allowed_models
                ):
                    raise ValueError(
                        f"Default model '{self.default_model}' not in allowlist"
                    )

        if self.mode == "alias_only" and not self.routes:
            raise ValueError("alias_only mode requires at least one route alias")

    def allowed_raw_models(self, model_registry: dict[str, APIModel]) -> list[str]:
        if self.allowed_models is None:
            return list(model_registry.keys())
        return _dedupe_keep_order(self.allowed_models)


class ModelRouter:
    def __init__(
        self,
        policy: ProxyModelPolicy,
        model_registry: dict[str, APIModel] | None = None,
        *,
        rng: random.Random | None = None,
    ) -> None:
        self.policy = policy
        self.model_registry = model_registry or registry
        self.policy.validate_against_registry(self.model_registry)
        self._rng = rng or random.Random()
        self._round_robin_index: dict[str, int] = {}

    def resolve(self, requested_model: str) -> str:
        target = requested_model
        if self.policy.mode == "force_default":
            if not self.policy.default_model:
                raise ValueError("No default model configured")
            target = self.policy.default_model

        if target in self.policy.routes:
            return self._select_from_route(target)

        if self.policy.mode == "alias_only":
            raise ValueError(f"Model '{requested_model}' is not an exposed alias")

        if target not in self.model_registry:
            raise ValueError(f"Model '{target}' not found in registry")
        if (
            self.policy.allowed_models is not None
            and target not in self.policy.allowed_models
        ):
            raise ValueError(f"Model '{target}' is not allowed by proxy policy")
        return target

    def list_model_ids(
        self,
        *,
        only_available: bool,
        is_available: Callable[[str], bool],
    ) -> list[str]:
        models: list[str] = []

        if self.policy.mode != "alias_only":
            raw_models = self.policy.allowed_raw_models(self.model_registry)
            if only_available:
                raw_models = [model for model in raw_models if is_available(model)]
            models.extend(raw_models)

        if self.policy.mode == "alias_only" or self.policy.expose_aliases:
            aliases = list(self.policy.routes.keys())
            if only_available:
                aliases = [
                    alias
                    for alias in aliases
                    if any(
                        is_available(model)
                        for model in self.policy.routes[alias].models
                    )
                ]
            models.extend(aliases)

        return models

    def _select_from_route(self, alias: str) -> str:
        route = self.policy.routes[alias]
        models = route.models
        if len(models) == 1:
            return models[0]

        if route.strategy == "round_robin":
            index = self._round_robin_index.get(alias, 0)
            selected = models[index % len(models)]
            self._round_robin_index[alias] = index + 1
            return selected

        if route.strategy == "weighted":
            assert route.weights is not None
            return self._rng.choices(models, weights=route.weights, k=1)[0]

        return self._rng.choice(models)


def load_policy_data(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        return {}
    if "model_policy" in data:
        return data["model_policy"] or {}
    proxy_block = data.get("proxy")
    if isinstance(proxy_block, dict) and "model_policy" in proxy_block:
        return proxy_block["model_policy"] or {}
    return data


def build_policy(
    *,
    path: str | None = None,
    overrides: dict | None = None,
) -> ProxyModelPolicy:
    data = load_policy_data(path)
    if overrides:
        data.update(overrides)
    policy = ProxyModelPolicy(**data)
    policy.validate_against_registry(registry)
    return policy
