"""Sequential Vertex MCP compatibility tests.

Run: bop run --environment ci52dbovvf5haguktv2klydx4u -- .venv/bin/python tests/one_off/test_gemini_mcp_tools_live.py
Use --probe-schemas to compare native Schema with parametersJsonSchema.
"""

import asyncio
import copy
import os
import sys
from pathlib import Path
from unittest.mock import patch

import aiohttp

from lm_deluge import Conversation, LLMClient, Tool
from lm_deluge.api_requests.bedrock import BedrockRequest
from lm_deluge.api_requests.response import APIResponse
from lm_deluge.prompt import Text

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from gemini_mcp_fixtures import STATULAR_TOOLS


async def probe_schemas():
    declarations = [
        {
            "name": item["name"],
            "description": item["description"],
            "parametersJsonSchema": {
                "type": "object",
                "properties": item["parameters"],
                "required": item["required"],
            },
        }
        for item in STATULAR_TOOLS
    ]
    constraint_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "$defs": {"Code": {"type": "integer", "const": 7}},
        "properties": {
            "code": {"$ref": "#/$defs/Code"},
            "flag": {"type": "boolean", "const": True},
            "label": {
                "type": "string",
                "title": "Label",
                "default": "ok",
                "examples": ["ok"],
                "format": "uuid",
            },
        },
        "additionalProperties": False,
    }
    cases = [
        ("json_schema_statular", declarations),
        (
            "json_schema_keywords",
            [{"name": "probe", "parametersJsonSchema": constraint_schema}],
        ),
    ]
    for enum in ([7], ["7"]):
        cases.append(
            (
                f"native_integer_enum_{enum}",
                [
                    {
                        "name": "probe",
                        "parameters": {
                            "type": "object",
                            "properties": {"code": {"type": "integer", "enum": enum}},
                        },
                    }
                ],
            )
        )
    async with aiohttp.ClientSession() as session:
        for label, functions in cases:
            body = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": "Reply OK without calling tools."}],
                    }
                ],
                "tools": [{"functionDeclarations": copy.deepcopy(functions)}],
                "generationConfig": {
                    "maxOutputTokens": 256,
                    "thinkingConfig": {"thinkingLevel": "low"},
                },
            }
            async with session.post(
                "https://aiplatform.googleapis.com/v1beta1/publishers/google/models/gemini-3.8-flash:generateContent",
                headers={"x-goog-api-key": os.environ["VERTEX_API_KEY"]},
                json=body,
            ) as response:
                data = await response.json()
                print(
                    label,
                    response.status,
                    data.get("error", {}).get("message", "accepted"),
                    flush=True,
                )
                expected_status = 400 if label == "native_integer_enum_[7]" else 200
                assert response.status == expected_status, (label, data)


PRIMARY = "claude-5.5-opus-bedrock-global"
VERTEX = "gemini-3.8-flash-vertex"


async def agent_loop(fallback=False):
    calls = []
    rounds = []

    def fake_runner(name):
        async def run(**kwargs):
            calls.append((name, kwargs))
            # Tool.from_mcp_config normalizes MCP text blocks to Text objects.
            return [
                Text("Found 8 results. Verification code: VERTEX_MCP_827."),
                Text("Batch completed successfully."),
            ]

        return run

    tools = [Tool(**item, run=fake_runner(item["name"])) for item in STATULAR_TOOLS]
    client = LLMClient(
        PRIMARY if fallback else VERTEX,
        fallback_models=[VERTEX] if fallback else None,
        reasoning_effort="high" if fallback else "low",
        global_effort="high" if fallback else None,
        max_new_tokens=4096,
        max_attempts=2,
        request_timeout=120,
        progress="manual",
    )
    original = BedrockRequest.execute_once

    async def inject(request):
        if rounds and request.context.model_name == PRIMARY:
            context = request.context
            return APIResponse(
                id=context.task_id,
                model_internal=context.model_name,
                prompt=context.prompt,
                sampling_params=context.sampling_params,
                status_code=503,
                is_error=True,
                error_message="Injected Bedrock outage",
                retry_with_different_model=True,
            )
        return await original(request)

    async def record(conversation, response, number):
        rounds.append(response.model_internal)
        assert not response.is_error, response.error_message
        print(f"ROUND {number + 1}: {response.model_internal}", flush=True)
        if fallback and number == 0:
            assert response.model_internal == PRIMARY
            assert response.content and response.content.thinking_parts, (
                "Round 1 needs real Opus reasoning"
            )

    try:
        with patch.object(BedrockRequest, "execute_once", inject):
            conversation, response = await client.run_agent_loop(
                Conversation().user(
                    "First reason internally about the smallest positive integer n such that "
                    "n modulo 7 is 2 and n modulo 11 is 3. Do not put the reasoning in your answer. "
                    "Then retrieve sample clauses. You MUST call batch exactly once "
                    "with two commands: search index sample_clauses for queries ['distribution to children in equal shares'], "
                    "and fetch index sample_clauses document_ids ['clause-id-466']. "
                    "After receiving the result, answer with its verification code and the number of results. "
                    "Do not call any other tool or call batch again."
                ),
                tools=tools,
                max_rounds=3,
                on_round_complete=record,
            )
        assert not response.is_error, response.error_message
        assert [name for name, _ in calls] == ["batch"], calls
        commands = calls[0][1]["commands"]
        assert [command["name"] for command in commands] == ["search", "fetch"], (
            commands
        )
        assert rounds == ([PRIMARY, VERTEX] if fallback else [VERTEX, VERTEX]), rounds
        assert "VERTEX_MCP_827" in (response.completion or ""), response.completion
        assert "8" in (response.completion or ""), response.completion
        assert conversation.model_used == VERTEX
        print(f"PASS fallback={fallback}: {response.completion}", flush=True)
    finally:
        client.close()


async def main():
    if "--probe-schemas" in sys.argv:
        await probe_schemas()
    else:
        await agent_loop()
        await agent_loop(fallback=True)


if __name__ == "__main__":
    asyncio.run(main())
