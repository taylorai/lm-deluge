import asyncio

from lm_deluge.api_requests.gemini import _build_gemini_request
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel
from lm_deluge.prompt import Conversation


def test_gemini_3_thinking_level_high():
    """Gemini 3 should use thinkingLevel=high for reasoning_effort=high."""
    model = APIModel.from_registry("gemini-3-pro-preview")
    convo = Conversation.user("Hello")
    request = asyncio.run(
        _build_gemini_request(
            model,
            convo,
            None,
            SamplingParams(reasoning_effort="high"),
        )
    )
    thinking_config = request["generationConfig"].get("thinkingConfig")
    assert thinking_config is not None
    assert thinking_config.get("thinkingLevel") == "high"
    # Should NOT have thinkingBudget for Gemini 3
    assert "thinkingBudget" not in thinking_config


def test_gemini_3_thinking_level_low():
    """Gemini 3 should use thinkingLevel=low for reasoning_effort=low/minimal."""
    model = APIModel.from_registry("gemini-3-pro-preview")
    convo = Conversation.user("Hello")

    # Test low
    request = asyncio.run(
        _build_gemini_request(
            model,
            convo,
            None,
            SamplingParams(reasoning_effort="low"),
        )
    )
    thinking_config = request["generationConfig"].get("thinkingConfig")
    assert thinking_config is not None
    assert thinking_config.get("thinkingLevel") == "low"

    # Test minimal maps to low
    request = asyncio.run(
        _build_gemini_request(
            model,
            convo,
            None,
            SamplingParams(reasoning_effort="minimal"),
        )
    )
    thinking_config = request["generationConfig"].get("thinkingConfig")
    assert thinking_config is not None
    assert thinking_config.get("thinkingLevel") == "low"


def test_gemini_3_thinking_level_medium():
    """Gemini 3 should use thinkingLevel=medium for reasoning_effort=medium."""
    model = APIModel.from_registry("gemini-3-pro-preview")
    convo = Conversation.user("Hello")
    request = asyncio.run(
        _build_gemini_request(
            model,
            convo,
            None,
            SamplingParams(reasoning_effort="medium"),
        )
    )
    thinking_config = request["generationConfig"].get("thinkingConfig")
    assert thinking_config is not None
    assert thinking_config.get("thinkingLevel") == "medium"


def test_gemini_3_default_thinking_level():
    """Gemini 3 should default to high thinking level when reasoning_effort is None."""
    model = APIModel.from_registry("gemini-3-pro-preview")
    convo = Conversation.user("Hello")
    request = asyncio.run(
        _build_gemini_request(
            model,
            convo,
            None,
            SamplingParams(reasoning_effort=None),
        )
    )
    thinking_config = request["generationConfig"].get("thinkingConfig")
    assert thinking_config is not None
    assert thinking_config.get("thinkingLevel") == "high"


def test_gemini_25_still_uses_thinking_budget():
    """Gemini 2.5 models should still use thinkingBudget (legacy behavior)."""
    model = APIModel.from_registry("gemini-2.5-pro")
    convo = Conversation.user("Hello")
    request = asyncio.run(
        _build_gemini_request(
            model,
            convo,
            None,
            SamplingParams(reasoning_effort="high"),
        )
    )
    thinking_config = request["generationConfig"].get("thinkingConfig")
    assert thinking_config is not None
    # Should have thinkingBudget for Gemini 2.5
    assert "includeThoughts" in thinking_config
    # Should NOT have thinkingLevel for Gemini 2.5
    assert "thinkingLevel" not in thinking_config


if __name__ == "__main__":
    test_gemini_3_thinking_level_high()
    test_gemini_3_thinking_level_low()
    test_gemini_3_thinking_level_medium()
    test_gemini_3_default_thinking_level()
    test_gemini_25_still_uses_thinking_budget()
    print("All tests passed!")
