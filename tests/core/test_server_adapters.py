import base64

from lm_deluge.api_requests.response import APIResponse
from lm_deluge.config import SamplingParams
from lm_deluge.prompt import (
    File,
    Image,
    Conversation,
    Message,
    Text,
    ThoughtSignature,
    Thinking,
    ToolCall,
)
from lm_deluge.server.adapters import (
    anthropic_request_to_conversation,
    anthropic_request_to_output_schema,
    anthropic_request_to_sampling_params,
    anthropic_tools_to_lm_deluge,
    api_response_to_anthropic,
    openai_tools_to_lm_deluge,
)
from lm_deluge.server.models_anthropic import (
    AnthropicContentBlock,
    AnthropicMessage,
    AnthropicMessagesRequest,
)


def test_openai_tools_to_lm_deluge_handles_missing_parameters():
    tools = [{"type": "function", "function": {"name": "ping"}}]
    lm_tools = openai_tools_to_lm_deluge(tools)
    assert len(lm_tools) == 1
    assert lm_tools[0].parameters is None
    assert lm_tools[0].required == []


def test_anthropic_tools_to_lm_deluge_handles_missing_schema():
    tools = [{"name": "ping"}]
    lm_tools = anthropic_tools_to_lm_deluge(tools)
    assert len(lm_tools) == 1
    assert lm_tools[0].parameters is None
    assert lm_tools[0].required == []


def test_anthropic_request_to_conversation_rich_content():
    image_data = base64.b64encode(b"fake-image").decode()
    doc_data = base64.b64encode(b"fake-document").decode()

    req = AnthropicMessagesRequest(
        model="claude-4-sonnet",
        max_tokens=10,
        system=[AnthropicContentBlock(type="text", text="System prompt")],
        messages=[
            AnthropicMessage(
                role="user",
                content=[
                    AnthropicContentBlock(type="text", text="Hello"),
                    AnthropicContentBlock(
                        type="image",
                        source={
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_data,
                        },
                    ),
                    AnthropicContentBlock(
                        type="document",
                        source={
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": doc_data,
                        },
                        title="Spec",
                    ),
                ],
            ),
            AnthropicMessage(
                role="assistant",
                content=[
                    AnthropicContentBlock(
                        type="thinking",
                        thinking="Plan",
                        signature="sig",
                    ),
                    AnthropicContentBlock(type="text", text="Done"),
                ],
            ),
        ],
    )

    conversation = anthropic_request_to_conversation(req)

    system_msg = next(msg for msg in conversation.messages if msg.role == "system")
    assert any(isinstance(part, Text) for part in system_msg.parts)

    user_msg = next(msg for msg in conversation.messages if msg.role == "user")
    assert any(isinstance(part, Image) for part in user_msg.parts)
    assert any(isinstance(part, File) for part in user_msg.parts)

    assistant_msg = next(
        msg for msg in conversation.messages if msg.role == "assistant"
    )
    assert any(
        isinstance(part, Thinking) and part.content == "Plan"
        for part in assistant_msg.parts
    )


def test_anthropic_request_preserves_thinking_signature_for_tools():
    req = AnthropicMessagesRequest(
        model="claude-4-sonnet",
        max_tokens=10,
        messages=[
            AnthropicMessage(role="user", content="Run tool."),
            AnthropicMessage(
                role="assistant",
                content=[
                    AnthropicContentBlock(
                        type="thinking",
                        thinking="Plan",
                        signature="sig",
                    ),
                    AnthropicContentBlock(
                        type="tool_use",
                        id="tool_1",
                        name="get_weather",
                        input={"location": "Paris"},
                    ),
                ],
            ),
        ],
    )

    conversation = anthropic_request_to_conversation(req)
    assistant_msg = next(
        msg for msg in conversation.messages if msg.role == "assistant"
    )

    thinking_part = next(
        part for part in assistant_msg.parts if isinstance(part, Thinking)
    )
    tool_part = next(part for part in assistant_msg.parts if isinstance(part, ToolCall))

    assert isinstance(thinking_part.thought_signature, ThoughtSignature)
    assert thinking_part.thought_signature.value == "sig"
    assert thinking_part.thought_signature.provider == "anthropic"
    assert isinstance(tool_part.thought_signature, ThoughtSignature)
    assert tool_part.thought_signature.value == "sig"
    assert tool_part.thought_signature.provider == "anthropic"


def test_anthropic_request_to_sampling_params_parses_effort_and_adaptive():
    req = AnthropicMessagesRequest(
        model="claude-opus-4-6",
        max_tokens=64,
        output_config={"effort": "max"},
        thinking={"type": "adaptive"},
        messages=[AnthropicMessage(role="user", content="Hello")],
    )

    params = anthropic_request_to_sampling_params(req)
    assert params.global_effort == "max"
    assert params.reasoning_effort == "high"


def test_anthropic_request_to_output_schema_prefers_output_config_format():
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    req = AnthropicMessagesRequest(
        model="claude-opus-4-6",
        max_tokens=64,
        output_config={"format": {"type": "json_schema", "schema": schema}},
        messages=[AnthropicMessage(role="user", content="Hello")],
    )

    extracted = anthropic_request_to_output_schema(req)
    assert extracted == schema


def test_anthropic_request_to_output_schema_supports_deprecated_output_format():
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    req = AnthropicMessagesRequest(
        model="claude-opus-4-6",
        max_tokens=64,
        output_format={"type": "json_schema", "schema": schema},
        messages=[AnthropicMessage(role="user", content="Hello")],
    )

    extracted = anthropic_request_to_output_schema(req)
    assert extracted == schema


def test_api_response_to_anthropic_maps_stop_reason_and_thinking():
    response = APIResponse(
        id=1,
        model_internal="gpt-4.1",
        prompt=Conversation().user("Hello"),
        sampling_params=SamplingParams(),
        status_code=200,
        is_error=False,
        error_message=None,
        content=Message(
            "assistant",
            [
                Text("Hello"),
                Thinking(
                    content="Plan",
                    thought_signature=ThoughtSignature("sig", provider="anthropic"),
                ),
            ],
        ),
        finish_reason="length",
    )

    converted = api_response_to_anthropic(response, "gpt-4.1")
    assert converted.stop_reason == "max_tokens"
    assert any(
        block.type == "thinking"
        and block.thinking == "Plan"
        and block.signature == "sig"
        for block in converted.content
    )


def test_api_response_to_anthropic_adds_signature_thinking_for_tool_call():
    response = APIResponse(
        id=3,
        model_internal="gemini-3-flash-preview",
        prompt=Conversation().user("Hello"),
        sampling_params=SamplingParams(),
        status_code=200,
        is_error=False,
        error_message=None,
        content=Message(
            "assistant",
            [
                ToolCall(
                    id="tool_1",
                    name="get_weather",
                    arguments={"location": "Paris"},
                    thought_signature=ThoughtSignature("sig", provider="anthropic"),
                )
            ],
        ),
        finish_reason="tool_calls",
    )

    converted = api_response_to_anthropic(response, "gemini-3-flash-preview")
    thinking_blocks = [
        block
        for block in converted.content
        if block.type == "thinking" and block.signature == "sig"
    ]
    tool_blocks = [block for block in converted.content if block.type == "tool_use"]
    assert thinking_blocks
    assert tool_blocks
    assert converted.content.index(thinking_blocks[0]) < converted.content.index(
        tool_blocks[0]
    )


def test_api_response_to_anthropic_skips_non_anthropic_signature():
    response = APIResponse(
        id=4,
        model_internal="gemini-3-flash-preview",
        prompt=Conversation().user("Hello"),
        sampling_params=SamplingParams(),
        status_code=200,
        is_error=False,
        error_message=None,
        content=Message(
            "assistant",
            [
                ToolCall(
                    id="tool_2",
                    name="get_weather",
                    arguments={"location": "Paris"},
                    thought_signature=ThoughtSignature("sig", provider="gemini"),
                )
            ],
        ),
        finish_reason="tool_calls",
    )

    converted = api_response_to_anthropic(response, "gemini-3-flash-preview")
    thinking_blocks = [block for block in converted.content if block.type == "thinking"]
    tool_blocks = [block for block in converted.content if block.type == "tool_use"]
    assert not thinking_blocks
    assert tool_blocks


def test_api_response_to_anthropic_prefers_raw_stop_reason():
    response = APIResponse(
        id=2,
        model_internal="gpt-4.1",
        prompt=Conversation().user("Hello"),
        sampling_params=SamplingParams(),
        status_code=200,
        is_error=False,
        error_message=None,
        content=Message.ai("Done"),
        finish_reason=None,
        raw_response={"stop_reason": "tool_use", "stop_sequence": "<END>"},
    )

    converted = api_response_to_anthropic(response, "gpt-4.1")
    assert converted.stop_reason == "tool_use"
    assert converted.stop_sequence == "<END>"


if __name__ == "__main__":
    test_openai_tools_to_lm_deluge_handles_missing_parameters()
    test_anthropic_tools_to_lm_deluge_handles_missing_schema()
    test_anthropic_request_to_conversation_rich_content()
    test_anthropic_request_to_sampling_params_parses_effort_and_adaptive()
    test_anthropic_request_to_output_schema_prefers_output_config_format()
    test_anthropic_request_to_output_schema_supports_deprecated_output_format()
    test_api_response_to_anthropic_maps_stop_reason_and_thinking()
    test_api_response_to_anthropic_prefers_raw_stop_reason()
    print("All tests passed!")
