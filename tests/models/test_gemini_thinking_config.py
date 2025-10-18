import asyncio

from lm_deluge.api_requests.gemini import _build_gemini_request
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel
from lm_deluge.prompt import Conversation


def test_gemini_no_reasoning_effort():
    """Gemini request should disable thoughts when reasoning_effort is None."""
    model = APIModel.from_registry("gemini-2.5-pro-gemini")
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
    assert thinking.get("includeThoughts") is False
    assert thinking.get("thinkingBudget") == 0
