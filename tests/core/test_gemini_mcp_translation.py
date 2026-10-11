"""Offline regression tests for MCP schemas/results and cross-provider history."""

import asyncio
import copy
import io
import json
import os
from typing import Any, cast
from unittest.mock import patch

from gemini_mcp_fixtures import STATULAR_RESULT_MESSAGE, STATULAR_TOOLS
from PIL import Image as PILImage

from lm_deluge import Conversation, Message, Tool
from lm_deluge.api_requests.context import RequestContext
from lm_deluge.api_requests.gemini import GeminiRequest, VertexGeminiRequest
from lm_deluge.config import SamplingParams
from lm_deluge.prompt import Image, Text, Thinking, ToolCall, ToolResult
from lm_deluge.prompt.signatures import ThoughtSignature


def test_schemas():
    original = copy.deepcopy(STATULAR_TOOLS)
    for spec in STATULAR_TOOLS:
        tool = Tool(**spec)
        declaration = tool.for_google()
        assert "parameters" not in declaration
        schema = declaration["parametersJsonSchema"]
        assert schema["properties"] == spec["parameters"]
        assert schema["required"] == spec["required"]
        assert (
            tool.for_anthropic(strict=False)["input_schema"]["properties"]
            == spec["parameters"]
        )
    assert STATULAR_TOOLS == original
    batch = Tool(**STATULAR_TOOLS[2]).for_google()["parametersJsonSchema"]
    choices = batch["properties"]["commands"]["items"]["anyOf"]
    assert [choice["properties"]["name"]["const"] for choice in choices] == [
        "search",
        "fetch",
    ]
    tool = Tool(
        name="references",
        description="References and numeric constants",
        parameters={
            "code": {"$ref": "#/$defs/Code"},
            "flag": {"type": "boolean", "const": True},
            "label": {
                "type": "string",
                "format": "uuid",
                "title": "Label",
                "examples": ["ok"],
                "default": "ok",
            },
        },
        definitions={"Code": {"type": "integer", "const": 7}},
    )
    schema = tool.for_google()["parametersJsonSchema"]
    assert schema["$defs"]["Code"]["const"] == 7
    assert schema["properties"]["flag"]["const"] is True
    assert schema["properties"]["code"]["$ref"] == "#/$defs/Code"
    assert schema["properties"]["label"]["format"] == "uuid"


def test_results():
    persisted = Message.from_log(STATULAR_RESULT_MESSAGE)
    result = persisted.tool_results[0]
    assert isinstance(result.result, list)
    assert isinstance(result.result[0], Text)
    assert (
        result.gemini()["functionResponse"]["response"]["result"]
        == result.result[0].text
    )
    text_result = ToolResult("id", [Text("first"), Text("second")]).gemini()[
        "functionResponse"
    ]
    assert text_result["response"]["result"] == "first\n\n---\n\nsecond"
    assert ToolResult("id", []).gemini()["functionResponse"]["response"] == {
        "result": ""
    }
    assert ToolResult("id", {"count": 8}).gemini()["functionResponse"]["response"] == {
        "count": 8
    }
    raw = ToolResult(
        "id",
        cast(
            Any,
            [
                {"type": "text", "text": "Found 8 results"},
                {"type": "resource", "uri": "mcp://test"},
            ],
        ),
    )
    text = raw.gemini()["functionResponse"]["response"]["result"]
    assert text.startswith("Found 8 results")
    assert json.loads(text.split("\n\n---\n\n")[1]) == {
        "type": "resource",
        "uri": "mcp://test",
    }
    buf = io.BytesIO()
    PILImage.new("RGB", (2, 2), "red").save(buf, format="PNG")
    image = Image(buf.getvalue())
    mixed = ToolResult("id", [Text("Screenshot"), image]).gemini()["functionResponse"]
    assert mixed["parts"] == [image.gemini()]
    raw_image = {
        "type": "image",
        "mimeType": "image/png",
        "data": image.gemini()["inlineData"]["data"],
    }
    assert ToolResult("id", cast(Any, [raw_image])).gemini()["functionResponse"][
        "parts"
    ] == [image.gemini()]
    cu = ToolResult(
        "id",
        {
            "response": {"url": "https://example.com"},
            "inline_data": {"data": raw_image["data"]},
        },
        built_in_type="gemini_computer_use",
    )
    assert cu.gemini()["functionResponse"]["parts"] == [image.gemini()]
    return image


async def test_history(image):
    result = Message.from_log(STATULAR_RESULT_MESSAGE)
    call_id = result.tool_results[0].tool_call_id
    result.tool_results[0].files = [
        {"file_id": "file_abc", "filename": "output.txt", "media_type": "text/plain"}
    ]
    prompt = Conversation(
        messages=[
            Message.system("Use tools"),
            Message.user("Find clauses", image=image.data),
            Message(
                "assistant",
                [
                    Text(""),
                    Thinking(
                        "Search for clauses",
                        thought_signature=ThoughtSignature(
                            "foreign", provider="anthropic"
                        ),
                    ),
                    ToolCall(
                        call_id,
                        "search",
                        {"index": "sample_clauses", "queries": ["distribution"]},
                    ),
                ],
            ),
            result,
            Message.ai(""),
        ]
    )
    tools = [Tool(**spec) for spec in STATULAR_TOOLS]
    with patch.dict(os.environ, {"VERTEX_API_KEY": "fake", "GEMINI_API_KEY": "fake"}):
        requests = []
        for cls, model in [
            (GeminiRequest, "gemini-3.8-flash"),
            (VertexGeminiRequest, "gemini-3.8-flash-vertex"),
        ]:
            request = cls(
                RequestContext(
                    task_id=1,
                    model_name=model,
                    prompt=prompt,
                    tools=tools,
                    sampling_params=SamplingParams(max_new_tokens=512),
                )
            )
            await request.build_request()
            requests.append(request.request_json)
        assert requests[0] == requests[1]
    contents = requests[0]["contents"]
    assert len(contents) == 3
    assert contents[-1]["role"] == "user"
    assert contents[-1]["parts"][0]["functionResponse"]["name"] == "search"
    assert (
        contents[1]["parts"][-1]["thoughtSignature"]
        == "context_engineering_is_the_way_to_go"
    )
    assert all(
        part.get("thoughtSignature") != "foreign"
        for message in contents
        for part in message["parts"]
    )
    assert "inlineData" in contents[0]["parts"][1]


async def main():
    test_schemas()
    image = test_results()
    await test_history(image)
    print("PASS: Gemini MCP schemas, content blocks, and fallback history")


if __name__ == "__main__":
    asyncio.run(main())
