"""Tests for model alias resolution."""

from lm_deluge.client import LLMClient
from lm_deluge.models import APIModel, find_models, registry


# --- Alias resolves to same APIModel instance ---


def test_anthropic_aliases_resolve():
    """Anthropic API-style names resolve to the same model as canonical names."""
    pairs = [
        ("claude-4.6-opus", "claude-opus-4-6"),
        ("claude-4.6-opus", "claude-opus-4.6"),
        ("claude-4.6-sonnet", "claude-sonnet-4-6"),
        ("claude-4.6-sonnet", "claude-sonnet-4.6"),
        ("claude-4.5-opus", "claude-opus-4-5"),
        ("claude-4.5-opus", "claude-opus-4.5"),
        ("claude-4.5-sonnet", "claude-sonnet-4-5"),
        ("claude-4.5-haiku", "claude-haiku-4-5"),
        ("claude-4.1-opus", "claude-opus-4-1"),
        ("claude-4-opus", "claude-opus-4"),
        ("claude-4-sonnet", "claude-sonnet-4"),
        ("claude-3.7-sonnet", "claude-sonnet-3.7"),
        ("claude-3.5-haiku", "claude-haiku-3.5"),
    ]
    for canonical, alias in pairs:
        m1 = APIModel.from_registry(canonical)
        m2 = APIModel.from_registry(alias)
        assert m1 is m2, f"{alias} should resolve to same object as {canonical}"
    print("All alias resolution tests passed")


# --- Alias with reasoning suffix ---


def test_alias_with_reasoning_suffix():
    """Alias + reasoning effort suffix resolves correctly."""
    client = LLMClient("claude-sonnet-4-6-high")
    model = APIModel.from_registry(client.models[0])
    assert model.id == "claude-4.6-sonnet"
    assert client.sampling_params[0].reasoning_effort == "high"

    client = LLMClient("claude-opus-4-6-low")
    model = APIModel.from_registry(client.models[0])
    assert model.id == "claude-4.6-opus"
    assert client.sampling_params[0].reasoning_effort == "low"

    client = LLMClient("claude-haiku-4.5-medium")
    model = APIModel.from_registry(client.models[0])
    assert model.id == "claude-4.5-haiku"
    assert client.sampling_params[0].reasoning_effort == "medium"
    print("Alias + reasoning suffix tests passed")


# --- find_models deduplication ---


def test_find_models_no_duplicates():
    """find_models() returns each model exactly once despite aliases."""
    all_models = find_models()
    ids = [m.id for m in all_models]
    assert len(ids) == len(
        set(ids)
    ), f"Duplicates: {[x for x in ids if ids.count(x) > 1]}"
    print(f"find_models: {len(all_models)} unique models, no duplicates")


# --- Unknown names still fail ---


def test_unknown_name_raises():
    """Names that don't match any canonical id or alias still raise."""
    try:
        APIModel.from_registry("claude-nonexistent-model")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("Unknown name correctly raises ValueError")


# --- Registry size is canonical + aliases ---


def test_registry_has_aliases():
    """Registry should have more entries than unique model ids."""
    unique_ids = {m.id for m in registry.values()}
    assert len(registry) > len(
        unique_ids
    ), "Registry should contain alias entries beyond canonical ids"
    print(f"Registry: {len(registry)} entries, {len(unique_ids)} unique models")


if __name__ == "__main__":
    test_anthropic_aliases_resolve()
    test_alias_with_reasoning_suffix()
    test_find_models_no_duplicates()
    test_unknown_name_raises()
    test_registry_has_aliases()
    print("\nAll model alias tests passed!")
