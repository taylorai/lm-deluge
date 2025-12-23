"""
Basic tests for the proxy server.
"""


def test_server_import():
    """Test that server module imports correctly."""
    from lm_deluge.server import create_app

    app = create_app()
    assert app is not None
    print("Server import: OK")


def test_openai_models():
    """Test OpenAI Pydantic models."""
    from lm_deluge.server.models_openai import (
        OpenAIChatCompletionsRequest,
        OpenAIChatCompletionsResponse,
        OpenAIMessage,
    )

    # Test request parsing
    req = OpenAIChatCompletionsRequest(
        model="gpt-4.1",
        messages=[
            OpenAIMessage(role="user", content="Hello!"),
        ],
        temperature=0.7,
    )
    assert req.model == "gpt-4.1"
    assert len(req.messages) == 1
    assert req.stream is False
    print("OpenAI request model: OK")

    # Test response creation
    from lm_deluge.server.models_openai import (
        OpenAIChoice,
        OpenAIResponseMessage,
        OpenAIUsage,
    )

    resp = OpenAIChatCompletionsResponse(
        model="gpt-4.1",
        choices=[
            OpenAIChoice(
                message=OpenAIResponseMessage(content="Hello back!"),
                finish_reason="stop",
            )
        ],
        usage=OpenAIUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    assert resp.choices[0].message.content == "Hello back!"
    print("OpenAI response model: OK")


def test_anthropic_models():
    """Test Anthropic Pydantic models."""
    from lm_deluge.server.models_anthropic import (
        AnthropicMessage,
        AnthropicMessagesRequest,
        AnthropicMessagesResponse,
        AnthropicResponseContentBlock,
        AnthropicUsage,
    )

    # Test request parsing
    req = AnthropicMessagesRequest(
        model="claude-sonnet-4",
        max_tokens=1024,
        messages=[
            AnthropicMessage(role="user", content="Hello!"),
        ],
    )
    assert req.model == "claude-sonnet-4"
    assert req.max_tokens == 1024
    print("Anthropic request model: OK")

    # Test response creation
    resp = AnthropicMessagesResponse(
        model="claude-sonnet-4",
        content=[AnthropicResponseContentBlock(type="text", text="Hello back!")],
        stop_reason="end_turn",
        usage=AnthropicUsage(input_tokens=10, output_tokens=5),
    )
    assert resp.content[0].text == "Hello back!"
    print("Anthropic response model: OK")


def test_adapters():
    """Test request/response adapters."""
    from lm_deluge.server.adapters import (
        openai_request_to_conversation,
        openai_request_to_sampling_params,
    )
    from lm_deluge.server.models_openai import (
        OpenAIChatCompletionsRequest,
        OpenAIMessage,
    )

    req = OpenAIChatCompletionsRequest(
        model="gpt-4.1",
        messages=[
            OpenAIMessage(role="system", content="You are helpful."),
            OpenAIMessage(role="user", content="Hello!"),
        ],
        temperature=0.5,
        max_tokens=100,
    )

    # Test conversation conversion
    conv = openai_request_to_conversation(req)
    assert len(conv.messages) == 2
    assert conv.messages[0].role == "system"
    assert conv.messages[1].role == "user"
    print("OpenAI to Conversation adapter: OK")

    # Test sampling params extraction
    params = openai_request_to_sampling_params(req)
    assert params.temperature == 0.5
    assert params.max_new_tokens == 100
    print("OpenAI to SamplingParams adapter: OK")


def test_tool_conversion():
    """Test tool definition conversion."""
    from lm_deluge.server.adapters import openai_tools_to_lm_deluge
    from lm_deluge.server.models_openai import OpenAITool, OpenAIToolFunction

    tools = [
        OpenAITool(
            function=OpenAIToolFunction(
                name="get_weather",
                description="Get the weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                    },
                    "required": ["location"],
                },
            )
        )
    ]

    lm_tools = openai_tools_to_lm_deluge(tools)
    assert len(lm_tools) == 1
    assert lm_tools[0].name == "get_weather"
    assert lm_tools[0].description == "Get the weather"
    assert "location" in lm_tools[0].parameters
    print("Tool conversion: OK")


if __name__ == "__main__":
    test_server_import()
    test_openai_models()
    test_anthropic_models()
    test_adapters()
    test_tool_conversion()
    print("\nAll tests passed!")
