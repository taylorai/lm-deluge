import os
import warnings

import asyncio

from lm_deluge.api_requests.gemini import _build_gemini_request
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel
from lm_deluge.prompt import Conversation


def test_gemini_no_reasoning_effort():
    """Gemini 2.5 Pro should still think even when reasoning_effort is None."""
    model = APIModel.from_registry("gemini-2.5-pro")
    convo = Conversation.user("Hello")
    request = asyncio.run(
        _build_gemini_request(
            model,
            convo,
            None,
            SamplingParams(reasoning_effort=None),
        )
    )
    thinking = request["generationConfig"].get("thinkingConfig")
    assert thinking is not None
    assert thinking.get("includeThoughts") is True
    assert thinking.get("thinkingBudget") == 128


def test_gemini_thinking_budget_overrides_reasoning_effort():
    """thinking_budget should take priority over reasoning_effort for Gemini 2.5."""
    os.environ.pop("WARN_THINKING_BUDGET_AND_REASONING_EFFORT", None)
    model = APIModel.from_registry("gemini-2.5-flash")
    convo = Conversation.user("Hello")
    with warnings.catch_warnings(record=True) as caught:
        request = asyncio.run(
            _build_gemini_request(
                model,
                convo,
                None,
                SamplingParams(reasoning_effort="low", thinking_budget=2048),
            )
        )
    thinking = request["generationConfig"].get("thinkingConfig")
    assert thinking is not None
    assert thinking.get("includeThoughts") is True
    assert thinking.get("thinkingBudget") == 2048
    assert any("thinking_budget" in str(w.message) for w in caught)


def test_gemini_flash_lite_min_budget():
    """Flash lite models should honor minimum thinking budget."""
    model = APIModel.from_registry("gemini-2.5-flash-lite")
    convo = Conversation.user("Hello")
    request = asyncio.run(
        _build_gemini_request(
            model,
            convo,
            None,
            SamplingParams(reasoning_effort="minimal"),
        )
    )
    thinking = request["generationConfig"].get("thinkingConfig")
    assert thinking is not None
    assert thinking.get("includeThoughts") is True
    assert thinking.get("thinkingBudget") == 512
