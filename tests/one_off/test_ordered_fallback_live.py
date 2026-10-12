"""Live Opus -> fallback -> Opus tool loops and injected Bedrock-wide 503s.

Run: bop run --environment [env] -- .venv/bin/python tests/one_off/test_ordered_fallback_live.py
Only the injected 503s are mocked; all successful rounds call real providers.
"""

import asyncio
from unittest.mock import patch

from lm_deluge import Conversation, LLMClient, Tool
from lm_deluge.api_requests.bedrock import BedrockRequest
from lm_deluge.api_requests.response import APIResponse

PRIMARY = "claude-5.5-opus-bedrock"
FALLBACKS = ["claude-5-sonnet-bedrock", "gpt-6-luna-bedrock", "gemini-3.8-flash-vertex"]


def failure(context):
    return APIResponse(
        id=context.task_id,
        model_internal=context.model_name,
        prompt=context.prompt,
        sampling_params=context.sampling_params,
        status_code=503,
        is_error=True,
        error_message="Bedrock is unable to process your request. (injected)",
        retry_with_different_model=True,
    )


async def mid_loop(fallback, responses=False, primary=PRIMARY):
    client = LLMClient(
        primary,
        fallback_models=[fallback],
        global_effort="high",
        reasoning_effort="high",
        max_new_tokens=2048,
        max_attempts=2,
        request_timeout=120,
        use_responses_api=responses,
        progress="manual",
    )
    rounds = []
    tool_calls = []
    original = BedrockRequest.execute_once

    async def add(a: int, b: int) -> str:
        """Add two integers."""
        tool_calls.append((a, b))
        return str(a + b)

    async def inject(request):
        if len(rounds) == 1 and request.context.model_name == primary:
            return failure(request.context)
        return await original(request)

    async def record(conversation, response, number):
        rounds.append(response.model_internal)
        thinking = response.content.thinking_parts if response.content else []
        print(
            f"ROUND {number + 1}: {response.model_internal}, thinking_blocks={len(thinking)}, tools={len(response.content.tool_calls) if response.content else 0}",
            flush=True,
        )
        if number == 0:
            assert response.model_internal == primary, (
                "Live Opus request failed; cannot validate Opus history"
            )
            assert thinking, "Opus round 1 must contain reasoning"
        assert conversation.model_used == response.model_internal

    try:
        with patch.object(BedrockRequest, "execute_once", inject):
            conversation, response = await client.run_agent_loop(
                Conversation().user(
                    "Think carefully to find the smallest positive integer n such that "
                    "n modulo 7 is 2 and n modulo 11 is 3. Explain your reasoning internally. "
                    "Then call add(n, 5). Only after seeing its result, call add(result, 7). "
                    "Only after that second result, answer with the final number. "
                    "Use exactly one tool call per round; use the tools for the additions."
                ),
                tools=[Tool.from_function(add)],
                max_rounds=3,
                cache="last_2_user_messages",
                on_round_complete=record,
            )
        assert not response.is_error, response.error_message
        assert rounds == [primary, fallback, primary], rounds
        assert tool_calls == [(58, 5), (63, 7)], tool_calls
        assert "70" in (response.completion or ""), response.completion
        assert conversation.model_used == primary
        print(
            f"PASS mid_loop {primary} -> {fallback} responses={responses}: {response.completion!r}",
            flush=True,
        )
    finally:
        client.close()


async def full_chain(primary=PRIMARY):
    client = LLMClient(
        primary,
        fallback_models=FALLBACKS,
        max_attempts=1,
        max_new_tokens=512,
        progress="manual",
        request_timeout=120,
    )
    order = []

    async def inject(request):
        order.append(request.context.model_name)
        return failure(request.context)

    try:
        with patch.object(BedrockRequest, "execute_once", inject):
            response = await client.start(
                Conversation().user("What is 2+3? Answer with one number."),
                cache="last_2_user_messages",
            )
        order.append(response.model_internal)
        assert order == [primary] + [f for f in FALLBACKS if f != primary], order
        assert not response.is_error, response.error_message
        assert "5" in (response.completion or ""), response.completion
        print(
            f"PASS full_chain: {' -> '.join(order)}: {response.completion!r}",
            flush=True,
        )
    finally:
        client.close()


async def main():
    failures = []
    for fallback in FALLBACKS:
        for responses in (False, True):
            try:
                await mid_loop(fallback, responses)
            except Exception as exc:  # noqa: BLE001 - report all live cases before failing
                failures.append((fallback, responses, str(exc)))
                print(
                    f"FAIL mid_loop {fallback} responses={responses}: {exc}", flush=True
                )
    try:
        await full_chain()
    except Exception as exc:  # noqa: BLE001 - report all live cases before failing
        failures.append(("full_chain", False, str(exc)))
        print(f"FAIL full_chain: {exc}", flush=True)
    assert not failures, failures
    print("All live ordered fallback tests passed", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
