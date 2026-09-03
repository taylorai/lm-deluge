import asyncio

from lm_deluge.api_requests.gemini import _build_gemini_request
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel
from lm_deluge.prompt import Conversation


def test_gemini_3_thinking_level_high():
    """Gemini 3 should use thinkingLevel=high for reasoning_effort=high."""
    model = APIModel.from_registry("gemini-3-pro-preview")
    convo = Conversation().user("Hello")
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
    convo = Conversation().user("Hello")

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
    """Gemini 3 should map medium effort to high until medium is supported."""
    model = APIModel.from_registry("gemini-3-pro-preview")
    convo = Conversation().user("Hello")
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
    assert thinking_config.get("thinkingLevel") == "high"


def test_gemini_3_thinking_level_none():
    """Gemini 3 should use thinkingLevel=low for reasoning_effort='none'."""
    model = APIModel.from_registry("gemini-3-pro-preview")
    convo = Conversation().user("Hello")
    request = asyncio.run(
        _build_gemini_request(
            model,
            convo,
            None,
            SamplingParams(reasoning_effort="none"),
        )
    )
    thinking_config = request["generationConfig"].get("thinkingConfig")
    assert thinking_config is not None
    assert thinking_config.get("thinkingLevel") == "low"


def test_gemini_3_default_thinking_level():
    """Gemini 3 should default to low thinking level when reasoning_effort is None."""
    model = APIModel.from_registry("gemini-3-pro-preview")
    convo = Conversation().user("Hello")
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
    assert thinking_config.get("thinkingLevel") == "low"


def test_gemini_25_still_uses_thinking_budget():
    """Gemini 2.5 models should still use thinkingBudget (legacy behavior)."""
    model = APIModel.from_registry("gemini-2.5-pro")
    convo = Conversation().user("Hello")
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


def test_gemini_3_flash_thinking_level_minimal():
    """Gemini 3 Flash should support thinkingLevel=minimal directly."""
    model = APIModel.from_registry("gemini-3-flash-preview")
    convo = Conversation().user("Hello")
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
    assert thinking_config.get("thinkingLevel") == "minimal"


def test_gemini_3_flash_thinking_level_medium():
    """Gemini 3 Flash should support thinkingLevel=medium directly."""
    model = APIModel.from_registry("gemini-3-flash-preview")
    convo = Conversation().user("Hello")
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


def test_gemini_3_flash_thinking_level_low():
    """Gemini 3 Flash should use thinkingLevel=low for reasoning_effort=low."""
    model = APIModel.from_registry("gemini-3-flash-preview")
    convo = Conversation().user("Hello")
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


def test_gemini_3_flash_thinking_level_high():
    """Gemini 3 Flash should use thinkingLevel=high for reasoning_effort=high."""
    model = APIModel.from_registry("gemini-3-flash-preview")
    convo = Conversation().user("Hello")
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


def test_gemini_35_flash_default_thinking_level():
    """Gemini 3.5 Flash should default to thinkingLevel=medium."""
    model = APIModel.from_registry("gemini-3.5-flash")
    convo = Conversation().user("Hello")
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
    assert thinking_config.get("thinkingLevel") == "medium"


def test_gemini_35_flash_thinking_level_minimal():
    """Gemini 3.5 Flash should support thinkingLevel=minimal directly."""
    model = APIModel.from_registry("gemini-3.5-flash")
    convo = Conversation().user("Hello")
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
    assert thinking_config.get("thinkingLevel") == "minimal"


def test_gemini_35_flash_thinking_level_medium():
    """Gemini 3.5 Flash should support thinkingLevel=medium directly."""
    model = APIModel.from_registry("gemini-3.5-flash")
    convo = Conversation().user("Hello")
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


def test_new_gemini_flash_defaults_and_sampling_params():
    """New Gemini Flash models default to medium and reject sampling params."""
    convo = Conversation().user("Hello")
    for model_name in ("gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash"):
        model = APIModel.from_registry(model_name)
        request = asyncio.run(
            _build_gemini_request(
                model,
                convo,
                None,
                SamplingParams(temperature=0.2, top_p=0.8),
            )
        )

        generation_config = request["generationConfig"]
        assert generation_config["thinkingConfig"]["thinkingLevel"] == "medium"
        assert "temperature" not in generation_config
        assert "topP" not in generation_config


def test_gemini_37_and_38_map_minimal_thinking_to_low():
    convo = Conversation().user("Hello")
    for model_name in ("gemini-3.7-flash", "gemini-3.8-flash"):
        request = asyncio.run(
            _build_gemini_request(
                APIModel.from_registry(model_name),
                convo,
                None,
                SamplingParams(reasoning_effort="minimal"),
            )
        )
        thinking_config = request["generationConfig"]["thinkingConfig"]
        assert thinking_config["thinkingLevel"] == "low"


if __name__ == "__main__":
    test_gemini_3_thinking_level_high()
    test_gemini_3_thinking_level_low()
    test_gemini_3_thinking_level_medium()
    test_gemini_3_thinking_level_none()
    test_gemini_3_default_thinking_level()
    test_gemini_25_still_uses_thinking_budget()
    # Gemini 3 Flash specific tests
    test_gemini_3_flash_thinking_level_minimal()
    test_gemini_3_flash_thinking_level_medium()
    test_gemini_3_flash_thinking_level_low()
    test_gemini_3_flash_thinking_level_high()
    test_gemini_35_flash_default_thinking_level()
    test_gemini_35_flash_thinking_level_minimal()
    test_gemini_35_flash_thinking_level_medium()
    test_new_gemini_flash_defaults_and_sampling_params()
    test_gemini_37_and_38_map_minimal_thinking_to_low()
    print("All tests passed!")
