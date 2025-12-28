import asyncio

from lm_deluge.api_requests.gemini import _build_gemini_request
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel
from lm_deluge.prompt import Conversation


def test_gemini_3_media_resolution_high():
    """Test that Gemini 3 accepts media_resolution_high parameter."""
    model = APIModel.from_registry("gemini-3-pro-preview")
    convo = Conversation().user("Analyze this image")

    request = asyncio.run(
        _build_gemini_request(
            model,
            convo,
            None,
            SamplingParams(media_resolution="media_resolution_high"),
        )
    )

    media_res = request["generationConfig"].get("mediaResolution")
    assert media_res is not None
    assert media_res["level"] == "media_resolution_high"


def test_gemini_3_media_resolution_medium():
    """Test that Gemini 3 accepts media_resolution_medium parameter."""
    model = APIModel.from_registry("gemini-3-pro-preview")
    convo = Conversation().user("Analyze this image")

    request = asyncio.run(
        _build_gemini_request(
            model,
            convo,
            None,
            SamplingParams(media_resolution="media_resolution_medium"),
        )
    )

    media_res = request["generationConfig"].get("mediaResolution")
    assert media_res is not None
    assert media_res["level"] == "media_resolution_medium"


def test_gemini_3_media_resolution_low():
    """Test that Gemini 3 accepts media_resolution_low parameter."""
    model = APIModel.from_registry("gemini-3-pro-preview")
    convo = Conversation().user("Analyze this image")

    request = asyncio.run(
        _build_gemini_request(
            model,
            convo,
            None,
            SamplingParams(media_resolution="media_resolution_low"),
        )
    )

    media_res = request["generationConfig"].get("mediaResolution")
    assert media_res is not None
    assert media_res["level"] == "media_resolution_low"


def test_gemini_3_no_media_resolution():
    """Test that Gemini 3 works without media_resolution (default behavior)."""
    model = APIModel.from_registry("gemini-3-pro-preview")
    convo = Conversation().user("Hello")

    request = asyncio.run(
        _build_gemini_request(
            model,
            convo,
            None,
            SamplingParams(),
        )
    )

    # Should not have mediaResolution if not specified
    assert "mediaResolution" not in request["generationConfig"]


def test_gemini_25_ignores_media_resolution():
    """Test that Gemini 2.5 doesn't add mediaResolution (should warn)."""
    model = APIModel.from_registry("gemini-2.5-pro")
    convo = Conversation().user("Analyze this image")

    request = asyncio.run(
        _build_gemini_request(
            model,
            convo,
            None,
            SamplingParams(media_resolution="media_resolution_high"),
        )
    )

    # Gemini 2.5 should NOT have mediaResolution
    assert "mediaResolution" not in request["generationConfig"]


def test_gemini_3_combined_params():
    """Test that Gemini 3 can combine media_resolution with other parameters."""
    model = APIModel.from_registry("gemini-3-pro-preview")
    convo = Conversation().user("Analyze this image and reason about it")

    request = asyncio.run(
        _build_gemini_request(
            model,
            convo,
            None,
            SamplingParams(
                media_resolution="media_resolution_high",
                reasoning_effort="high",
                temperature=0.7,
            ),
        )
    )

    # Check all parameters are present
    gen_config = request["generationConfig"]

    # Media resolution
    assert "mediaResolution" in gen_config
    assert gen_config["mediaResolution"]["level"] == "media_resolution_high"

    # Thinking level
    assert "thinkingConfig" in gen_config
    assert gen_config["thinkingConfig"]["thinkingLevel"] == "high"

    # Temperature
    assert gen_config["temperature"] == 0.7


if __name__ == "__main__":
    test_gemini_3_media_resolution_high()
    test_gemini_3_media_resolution_medium()
    test_gemini_3_media_resolution_low()
    test_gemini_3_no_media_resolution()
    test_gemini_25_ignores_media_resolution()
    test_gemini_3_combined_params()
    print("All tests passed!")
