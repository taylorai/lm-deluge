"""Tests for the find_models utility function."""

from lm_deluge.models import find_models, registry


def test_find_all_models():
    """No filters returns all models."""
    all_models = find_models()
    assert len(all_models) == len(registry)


def test_find_by_provider():
    """Filter by provider/api_spec."""
    anthropic = find_models(provider="anthropic")
    assert len(anthropic) > 0
    assert all(m.api_spec == "anthropic" for m in anthropic)

    openai = find_models(provider="openai")
    assert len(openai) > 0
    assert all(m.api_spec == "openai" for m in openai)


def test_find_json_models():
    """Filter by JSON support."""
    json_models = find_models(supports_json=True)
    assert len(json_models) > 0
    assert all(m.supports_json for m in json_models)


def test_find_reasoning_models():
    """Filter by reasoning capability."""
    reasoning = find_models(reasoning_model=True)
    assert len(reasoning) > 0
    assert all(m.reasoning_model for m in reasoning)


def test_find_image_models():
    """Filter by image support."""
    image_models = find_models(supports_images=True)
    # All models with image support should have the flag set
    assert all(m.supports_images for m in image_models)


def test_find_by_cost_range():
    """Filter by input/output cost range."""
    # Find cheap models (input cost <= $1 per million tokens)
    cheap = find_models(max_input_cost=1.0)
    assert all(m.input_cost is not None and m.input_cost <= 1.0 for m in cheap)

    # Find expensive models (output cost >= $50 per million tokens)
    expensive = find_models(min_output_cost=50.0)
    assert all(m.output_cost is not None and m.output_cost >= 50.0 for m in expensive)


def test_find_by_name():
    """Filter by name substring."""
    claude_models = find_models(name_contains="claude")
    assert len(claude_models) > 0
    assert all("claude" in m.id.lower() for m in claude_models)

    gpt_models = find_models(name_contains="gpt")
    assert len(gpt_models) > 0
    assert all("gpt" in m.id.lower() for m in gpt_models)


def test_sort_by_cost():
    """Sort by cost ascending/descending."""
    # Cheapest input cost first
    cheapest = find_models(sort_by="input_cost", limit=5)
    for i in range(len(cheapest) - 1):
        assert (cheapest[i].input_cost or 0) <= (cheapest[i + 1].input_cost or 0)

    # Most expensive output cost first
    expensive = find_models(sort_by="-output_cost", limit=5)
    for i in range(len(expensive) - 1):
        assert (expensive[i].output_cost or 0) >= (expensive[i + 1].output_cost or 0)


def test_limit():
    """Limit number of results."""
    limited = find_models(limit=3)
    assert len(limited) == 3


def test_combined_filters():
    """Multiple filters combine with AND logic."""
    # Find cheap reasoning models from OpenAI
    results = find_models(
        provider="openai",
        reasoning_model=True,
        max_input_cost=2.0,
    )
    for m in results:
        assert m.api_spec == "openai"
        assert m.reasoning_model
        assert m.input_cost is not None and m.input_cost <= 2.0


def test_find_cheapest_reasoning_models():
    """Example: find the 5 cheapest reasoning models by input cost."""
    cheapest_reasoning = find_models(
        reasoning_model=True,
        sort_by="input_cost",
        limit=5,
    )
    assert len(cheapest_reasoning) <= 5
    assert all(m.reasoning_model for m in cheapest_reasoning)


def test_find_json_models_by_provider():
    """Example: find all JSON-capable models from a specific provider."""
    openai_json = find_models(provider="openai", supports_json=True)
    assert all(m.api_spec == "openai" and m.supports_json for m in openai_json)


if __name__ == "__main__":
    test_find_all_models()
    test_find_by_provider()
    test_find_json_models()
    test_find_reasoning_models()
    test_find_image_models()
    test_find_by_cost_range()
    test_find_by_name()
    test_sort_by_cost()
    test_limit()
    test_combined_filters()
    test_find_cheapest_reasoning_models()
    test_find_json_models_by_provider()
    print("All tests passed!")
