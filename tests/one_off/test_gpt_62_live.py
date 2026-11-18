"""Live smoke tests for GPT-6.2 Sol and GPT-6.2 Astra.

    bop run lm-deluge -- .venv/bin/python tests/one_off/test_gpt_62_live.py

Covers chat and Responses at each reasoning effort, a tool round-trip on the
Responses API, and cost accounting. Two raw probes report (never fail) whether
the API accepts reasoning effort 'none', and whether chat completions accepts
function tools together with reasoning, which decides the registry flags and the
chat-tools guard in RequestContext.validate_request_config.
"""

import asyncio
import json
import os
import sys
import urllib.error
import urllib.request

from lm_deluge import Conversation, LLMClient, Tool
from lm_deluge.config import ReasoningEffort
from lm_deluge.models import APIModel

MODELS = sys.argv[1:] or ["gpt-6.2-sol", "gpt-6.2-astra"]


def _check(response, marker: str) -> None:
    assert response is not None, "no response"
    assert not response.is_error, response.error_message
    assert response.completion and marker in response.completion, (
        f"expected {marker!r}, got {response.completion!r}"
    )


async def test_basic(model: str, api: str, effort: ReasoningEffort) -> str:
    marker = f"G62_{api.upper()}_{effort.upper()}_OK"
    client = LLMClient(
        model,
        max_new_tokens=1024,
        max_attempts=1,
        reasoning_effort=effort,
        request_timeout=300,
        use_responses_api=api == "responses",
    )
    try:
        response = await client.start(
            Conversation().user(f"Reply with exactly: {marker}")
        )
    finally:
        client.close()
    _check(response, marker)
    assert response.usage is not None and response.usage.input_tokens > 0
    assert response.cost is not None and response.cost > 0
    assert response.stop_reason == "end", response.stop_reason
    return f"cost=${response.cost:.6f}"


async def test_tool_loop(model: str) -> str:
    marker = "G62_TOOL_OK"

    async def get_secret(name: str) -> str:
        return marker if name == "alpha" else "unknown"

    tool = Tool(
        name="get_secret",
        description="Look up a secret value by name.",
        run=get_secret,
        parameters={"name": {"type": "string", "description": "Secret name"}},
        required=["name"],
    )
    client = LLMClient(
        model,
        max_new_tokens=2048,
        max_attempts=1,
        reasoning_effort="low",
        request_timeout=300,
        use_responses_api=True,
    )
    try:
        _, response = await client.run_agent_loop(
            Conversation().user(
                "Call get_secret with name 'alpha', then reply with exactly the "
                "value it returns."
            ),
            tools=[tool],
            max_rounds=4,
        )
    finally:
        client.close()
    _check(response, marker)
    return "tool round-trip"


def _post(path: str, body: dict) -> str:
    request = urllib.request.Request(
        f"https://api.openai.com/v1/{path}",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as resp:
            return f"accepted (HTTP {resp.status})"
    except urllib.error.HTTPError as exc:
        return f"rejected (HTTP {exc.code}): {exc.read().decode()[:300]}"


def probe_raw_none(model: str) -> str:
    return _post(
        "responses",
        {
            "model": model,
            "input": "Reply with exactly: OK",
            "reasoning": {"effort": "none"},
            "max_output_tokens": 64,
        },
    )


def probe_chat_tools_with_reasoning(model: str) -> str:
    return _post(
        "chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
            "reasoning_effort": "low",
            "max_completion_tokens": 256,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "noop",
                        "description": "Does nothing.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        },
    )


async def main() -> None:
    failures: list[str] = []
    total = 0
    for model in MODELS:
        efforts: list[ReasoningEffort] = ["low", "medium", "high", "xhigh", "max"]
        if APIModel.from_registry(model).supports_reasoning_none:
            efforts.insert(0, "none")
        cases = [
            (f"{model} {api}/{effort}", test_basic(model, api, effort))
            for api in ("chat", "responses")
            for effort in efforts
        ]
        cases.append((f"{model} responses/tools", test_tool_loop(model)))
        total += len(cases)
        for label, case in cases:
            try:
                detail = await case
                print(f"PASS {label} ({detail})")
            except Exception as exc:  # noqa: BLE001 - run full matrix
                failures.append(f"{label}: {exc}")
                print(f"FAIL {failures[-1]}")
        print(f"{model} raw reasoning effort 'none': {probe_raw_none(model)}")
        print(
            f"{model} chat tools + reasoning: {probe_chat_tools_with_reasoning(model)}"
        )

    print(f"\n{total - len(failures)} passed, {len(failures)} failed")
    if failures:
        raise AssertionError("\n".join(failures))


if __name__ == "__main__":
    asyncio.run(main())
