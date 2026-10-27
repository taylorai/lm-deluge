"""Live smoke tests for GPT-6.1 Terra.

    bop run lm-deluge -- .venv/bin/python tests/one_off/test_gpt_61_terra_live.py

Covers chat and responses, each reasoning effort, a tool round-trip on the
Responses API, and cost accounting. It also sends a raw reasoning effort of
'none' to report whether the model accepts it (informational, never fails).
"""

import asyncio
import json
import os
import urllib.error
import urllib.request

from lm_deluge import Conversation, LLMClient, Tool

MODEL = "gpt-6.1-terra"


def _check(response, marker: str) -> None:
    assert response is not None, "no response"
    assert not response.is_error, response.error_message
    assert response.completion and marker in response.completion, (
        f"expected {marker!r}, got {response.completion!r}"
    )


async def test_basic(api: str, effort: str) -> str:
    marker = f"TERRA_{api.upper()}_{effort.upper()}_OK"
    client = LLMClient(
        MODEL,
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
    return f"cost=${response.cost:.6f}"


async def test_tool_loop() -> str:
    marker = "TERRA_TOOL_OK"

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
        MODEL,
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


def probe_raw_none() -> str:
    body = json.dumps(
        {
            "model": MODEL,
            "input": "Reply with exactly: OK",
            "reasoning": {"effort": "none"},
            "max_output_tokens": 64,
        }
    ).encode()
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=body,
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


async def main() -> None:
    cases = [
        (f"{api}/{effort}", test_basic(api, effort))
        for api in ("chat", "responses")
        for effort in ("none", "low", "medium", "high", "xhigh", "max")
    ]
    cases.append(("responses/tools", test_tool_loop()))

    failures: list[str] = []
    for label, case in cases:
        try:
            detail = await case
            print(f"PASS {label} ({detail})")
        except Exception as exc:  # noqa: BLE001 - run full matrix
            failures.append(f"{label}: {exc}")
            print(f"FAIL {failures[-1]}")

    print(f"raw reasoning effort 'none': {probe_raw_none()}")
    print(f"\n{len(cases) - len(failures)} passed, {len(failures)} failed")
    if failures:
        raise AssertionError("\n".join(failures))


if __name__ == "__main__":
    asyncio.run(main())
