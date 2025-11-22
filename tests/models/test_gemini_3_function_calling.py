import asyncio

from lm_deluge.api_requests.gemini import _build_gemini_request
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel
from lm_deluge.prompt import Conversation, Message, Text, ToolCall, ToolResult


def test_gemini_3_missing_signature_gets_dummy():
    """Test that missing thought signatures in function calls get dummy signature injected."""
    model = APIModel.from_registry("gemini-3-pro-preview")

    # Create conversation with a function call WITHOUT signature
    convo = Conversation(
        [
            Message("user", [Text("What's the weather?")]),
            Message(
                "assistant",
                [
                    ToolCall(
                        id="call1",
                        name="get_weather",
                        arguments={"city": "Paris"},
                        # No thought_signature
                    )
                ],
            ),
        ]
    )

    request = asyncio.run(
        _build_gemini_request(
            model,
            convo,
            None,
            SamplingParams(),
        )
    )

    # Check that dummy signature was injected
    messages = request["contents"]
    assistant_msg = messages[1]
    assert "parts" in assistant_msg
    function_call_part = assistant_msg["parts"][0]
    assert "functionCall" in function_call_part
    assert "thoughtSignature" in function_call_part
    assert (
        function_call_part["thoughtSignature"] == "context_engineering_is_the_way_to_go"
    )


def test_gemini_3_existing_signature_preserved():
    """Test that existing thought signatures in function calls are preserved."""
    model = APIModel.from_registry("gemini-3-pro-preview")

    # Create conversation with a function call WITH signature
    convo = Conversation(
        [
            Message("user", [Text("What's the weather?")]),
            Message(
                "assistant",
                [
                    ToolCall(
                        id="call1",
                        name="get_weather",
                        arguments={"city": "Paris"},
                        thought_signature="original_signature",
                    )
                ],
            ),
        ]
    )

    request = asyncio.run(
        _build_gemini_request(
            model,
            convo,
            None,
            SamplingParams(),
        )
    )

    # Check that original signature was preserved
    messages = request["contents"]
    assistant_msg = messages[1]
    function_call_part = assistant_msg["parts"][0]
    assert "thoughtSignature" in function_call_part
    assert function_call_part["thoughtSignature"] == "original_signature"


def test_gemini_3_multi_step_function_calling():
    """Test multi-step sequential function calling with accumulated signatures."""
    model = APIModel.from_registry("gemini-3-pro-preview")

    # Simulate multi-step: user -> assistant (tool call 1) -> user (result) ->
    # assistant (tool call 2)
    convo = Conversation(
        [
            Message("user", [Text("Check flight AA100 and book a taxi")]),
            Message(
                "assistant",
                [
                    ToolCall(
                        id="call_flight",
                        name="check_flight",
                        arguments={"flight": "AA100"},
                        thought_signature="sig_flight",
                    )
                ],
            ),
            Message("tool", [ToolResult("call_flight", "Flight delayed 2 hours")]),
            Message(
                "assistant",
                [
                    ToolCall(
                        id="call_taxi",
                        name="book_taxi",
                        arguments={"delay": "2h"},
                        thought_signature="sig_taxi",
                    )
                ],
            ),
        ]
    )

    request = asyncio.run(
        _build_gemini_request(
            model,
            convo,
            None,
            SamplingParams(),
        )
    )

    messages = request["contents"]

    # Check first assistant message has signature
    first_assistant = messages[1]
    assert first_assistant["parts"][0]["thoughtSignature"] == "sig_flight"

    # Check second assistant message has signature
    # (Note: In Gemini format, tool results become user messages)
    second_assistant = messages[3]
    assert second_assistant["parts"][0]["thoughtSignature"] == "sig_taxi"


def test_gemini_3_parallel_function_calling():
    """Test parallel function calls - only first should have signature."""
    model = APIModel.from_registry("gemini-3-pro-preview")

    # Parallel calls: both in same assistant message
    convo = Conversation(
        [
            Message("user", [Text("Check weather in Paris and London")]),
            Message(
                "assistant",
                [
                    ToolCall(
                        id="call_paris",
                        name="check_weather",
                        arguments={"city": "Paris"},
                        thought_signature="sig_paris",
                    ),
                    ToolCall(
                        id="call_london",
                        name="check_weather",
                        arguments={"city": "London"},
                        # No signature for second parallel call (per Gemini 3 spec)
                    ),
                ],
            ),
        ]
    )

    request = asyncio.run(
        _build_gemini_request(
            model,
            convo,
            None,
            SamplingParams(),
        )
    )

    messages = request["contents"]
    assistant_msg = messages[1]
    parts = assistant_msg["parts"]

    # First call should have signature
    assert "thoughtSignature" in parts[0]
    assert parts[0]["thoughtSignature"] == "sig_paris"

    # Second call shouldn't have signature (or gets dummy)
    # According to spec, only first has signature in parallel calls
    if "thoughtSignature" in parts[1]:
        # If it exists, it should be the dummy (injected for missing)
        assert parts[1]["thoughtSignature"] == "context_engineering_is_the_way_to_go"


def test_gemini_25_no_dummy_signature_injection():
    """Test that Gemini 2.5 doesn't get dummy signatures injected."""
    model = APIModel.from_registry("gemini-2.5-pro")

    convo = Conversation(
        [
            Message("user", [Text("What's the weather?")]),
            Message(
                "assistant",
                [
                    ToolCall(
                        id="call1",
                        name="get_weather",
                        arguments={"city": "Paris"},
                        # No thought_signature
                    )
                ],
            ),
        ]
    )

    request = asyncio.run(
        _build_gemini_request(
            model,
            convo,
            None,
            SamplingParams(reasoning_effort="high"),
        )
    )

    # Gemini 2.5 should NOT have thoughtSignature injected
    messages = request["contents"]
    assistant_msg = messages[1]
    function_call_part = assistant_msg["parts"][0]
    # Should not have thoughtSignature for Gemini 2.5
    assert "thoughtSignature" not in function_call_part


if __name__ == "__main__":
    test_gemini_3_missing_signature_gets_dummy()
    test_gemini_3_existing_signature_preserved()
    test_gemini_3_multi_step_function_calling()
    test_gemini_3_parallel_function_calling()
    test_gemini_25_no_dummy_signature_injection()
    print("All tests passed!")
