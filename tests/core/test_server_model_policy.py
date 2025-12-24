import random
import tempfile

from lm_deluge.models import registry
from lm_deluge.server.model_policy import ModelRouter, ProxyModelPolicy, build_policy


def test_build_policy_from_yaml_and_override():
    yaml_config = """
mode: allow_user_pick
allowed_models:
  - gpt-4.1
  - claude-4-sonnet
"""
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml", delete=True) as handle:
        handle.write(yaml_config)
        handle.flush()
        policy = build_policy(
            path=handle.name,
            overrides={"mode": "force_default", "default_model": "gpt-4.1"},
        )

    assert policy.mode == "force_default"
    assert policy.default_model == "gpt-4.1"
    assert policy.allowed_models == ["gpt-4.1", "claude-4-sonnet"]


def test_router_round_robin():
    policy = ProxyModelPolicy(
        routes={
            "rr": {
                "strategy": "round_robin",
                "models": ["gpt-4.1", "claude-4-sonnet"],
            }
        },
        expose_aliases=True,
    )
    router = ModelRouter(policy, registry, rng=random.Random(0))
    first = router.resolve("rr")
    second = router.resolve("rr")
    third = router.resolve("rr")

    assert first == "gpt-4.1"
    assert second == "claude-4-sonnet"
    assert third == "gpt-4.1"


def test_router_weighted():
    policy = ProxyModelPolicy(
        routes={
            "weighted": {
                "strategy": "weighted",
                "models": ["gpt-4.1", "claude-4-sonnet"],
                "weights": [1.0, 0.0],
            }
        }
    )
    router = ModelRouter(policy, registry, rng=random.Random(0))
    assert router.resolve("weighted") == "gpt-4.1"


def test_alias_only_blocks_raw_models():
    policy = ProxyModelPolicy(
        mode="alias_only",
        routes={"only": {"models": ["gpt-4.1"]}},
    )
    router = ModelRouter(policy, registry)
    assert router.resolve("only") == "gpt-4.1"
    try:
        router.resolve("gpt-4.1")
        raise AssertionError("Expected alias_only to reject raw model ids")
    except ValueError:
        pass


def test_list_models_respects_allowlist_and_aliases():
    policy = ProxyModelPolicy(
        allowed_models=["gpt-4.1", "claude-4-sonnet"],
        routes={"mix": {"models": ["gpt-4.1", "claude-4-sonnet"]}},
        expose_aliases=True,
    )
    router = ModelRouter(policy, registry)

    def is_available(model_id: str) -> bool:
        return model_id == "gpt-4.1"

    listed_all = router.list_model_ids(
        only_available=False,
        is_available=is_available,
    )
    assert listed_all == ["gpt-4.1", "claude-4-sonnet", "mix"]

    listed_available = router.list_model_ids(
        only_available=True,
        is_available=is_available,
    )
    assert listed_available == ["gpt-4.1", "mix"]


if __name__ == "__main__":
    test_build_policy_from_yaml_and_override()
    test_router_round_robin()
    test_router_weighted()
    test_alias_only_blocks_raw_models()
    test_list_models_respects_allowlist_and_aliases()
    print("All tests passed!")
