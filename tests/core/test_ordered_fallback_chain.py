"""Offline retry-chain regression tests through the actual request runner."""

import asyncio
from contextlib import contextmanager
from itertools import pairwise
from unittest.mock import patch

from lm_deluge import Conversation, LLMClient, Message, Tool
from lm_deluge.api_requests.base import APIRequestBase
from lm_deluge.api_requests.bedrock import BedrockRequest
from lm_deluge.api_requests.context import RequestContext
from lm_deluge.api_requests.gemini import _build_gemini_request
from lm_deluge.api_requests.response import APIResponse
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel
from lm_deluge.prompt import Thinking, ThoughtSignature
from lm_deluge.prompt.tool_calls import ToolCall, ToolResult

PRIMARY = "claude-5.5-opus-bedrock"
SONNET = "claude-5-sonnet-bedrock"
LUNA = "gpt-6-luna-bedrock"
GEMINI = "gemini-3.8-flash-vertex"
CHAIN = [PRIMARY, SONNET, LUNA, GEMINI]


@contextmanager
def fake_requests(execute):
    with (
        patch.object(APIRequestBase, "execute_once", execute),
        patch.object(BedrockRequest, "execute_once", execute),
    ):
        yield


def response(context, *, switch=True, block=False, content=None):
    return APIResponse(
        id=context.task_id,
        model_internal=context.model_name,
        prompt=context.prompt,
        sampling_params=context.sampling_params,
        status_code=200 if content else 503,
        is_error=content is None,
        error_message=None if content else "injected failure",
        retry_with_different_model=switch,
        give_up_if_no_other_models=block,
        content=content,
    )


async def walk(
    attempts, same=(), blocked=(), models=PRIMARY, fallbacks=None, start=PRIMARY
):
    client = LLMClient(
        models,
        fallback_models=CHAIN[1:] if fallbacks is None else fallbacks,
        max_attempts=attempts,
        progress="manual",
    )
    client._blocklisted_models.update(blocked)
    seen = []

    async def execute(request):
        seen.append(request.context.model_name)
        return response(request.context, switch=len(seen) not in same)

    with fake_requests(execute):
        result = await client.start(Conversation().user("hi"), prefer_model=start)
    assert result.is_error
    return seen


async def test_order_budget_and_same_model():
    assert await walk(5) == CHAIN + [PRIMARY]
    assert await walk(1) == CHAIN
    assert await walk(6, same=(1,)) == [PRIMARY, PRIMARY] + CHAIN[1:] + [PRIMARY]
    # With only four attempts, a timeout must advance to preserve coverage.
    assert await walk(1, same=(1, 2, 3, 4)) == CHAIN
    assert await walk(5, blocked=(SONNET,)) == [PRIMARY, LUNA, GEMINI, PRIMARY, LUNA]
    assert await walk(5, blocked=tuple(CHAIN[1:])) == [PRIMARY] * 5
    # Other weighted primaries enter only if explicitly named as fallbacks.
    assert await walk(3, models=[PRIMARY, SONNET], fallbacks=[LUNA]) == [
        PRIMARY,
        LUNA,
        PRIMARY,
    ]
    assert await walk(3, models=[PRIMARY, SONNET], fallbacks=[LUNA], start=SONNET) == [
        SONNET,
        LUNA,
        SONNET,
    ]
    assert await walk(3, models=[PRIMARY, SONNET], fallbacks=[PRIMARY, SONNET]) in (
        [PRIMARY, SONNET, PRIMARY],
        [SONNET, PRIMARY, SONNET],
    )
    assert await walk(
        3, models=[PRIMARY, SONNET], fallbacks=[PRIMARY, SONNET], start=SONNET
    ) == [SONNET, PRIMARY, SONNET]
    # Identical primary fallback doesn't increase the chain length.
    assert await walk(1, fallbacks=[PRIMARY]) == [PRIMARY]


async def test_legacy_selection_and_blocklist():
    client = LLMClient([PRIMARY, SONNET, LUNA], max_attempts=4, progress="manual")
    seen = []

    async def execute(request):
        seen.append(request.context.model_name)
        return response(request.context)

    cls = type(client)
    with (
        fake_requests(execute),
        patch.object(
            cls, "_select_different_model", wraps=client._select_different_model
        ) as select,
    ):
        await client.start(Conversation().user("hi"), prefer_model=PRIMARY)
    assert select.call_count == 3
    assert len(seen) == 4
    assert all(a != b for a, b in pairwise(seen))

    client = LLMClient(PRIMARY, fallback_models=CHAIN[1:], progress="manual")
    seen.clear()

    async def block(request):
        seen.append(request.context.model_name)
        return response(request.context, switch=False, block=True)

    with fake_requests(block):
        await client.start(Conversation().user("hi"))
    assert seen == CHAIN


async def test_concurrent_positions():
    client = LLMClient(
        PRIMARY, fallback_models=CHAIN[1:], max_attempts=5, progress="manual"
    )
    seen = {}

    async def execute(request):
        context = request.context
        calls = seen.setdefault(context.task_id, [])
        calls.append(context.model_name)
        await asyncio.sleep(0)
        return response(
            context,
            content=Message.ai("ok") if len(calls) == context.task_id + 2 else None,
        )

    with fake_requests(execute):
        results = await client.process_prompts_async(
            [Conversation().user("hi") for _ in range(3)], show_progress=False
        )
    assert list(seen.values()) == [CHAIN[:2], CHAIN[:3], CHAIN[:4]]
    assert all(result and not result.is_error for result in results)


async def test_agent_round_reset_and_transport():
    client = LLMClient(
        PRIMARY, fallback_models=[LUNA], max_attempts=2, progress="manual"
    )
    seen = []

    async def echo(value: str):
        return value

    async def execute(request):
        context = request.context
        seen.append((context.model_name, context.use_responses_api))
        if len(context.prompt.messages) == 3 and context.model_name == PRIMARY:
            return response(context)
        if len(context.prompt.messages) < 5:
            return response(
                context,
                content=Message(
                    "assistant",
                    [
                        ToolCall(
                            id=f"call_{len(seen)}",
                            name="echo",
                            arguments={"value": "ok"},
                        )
                    ],
                ),
            )
        return response(context, content=Message.ai("done"))

    with fake_requests(execute):
        conv, result = await client.run_agent_loop(
            Conversation().user("hi"), tools=[Tool.from_function(echo)], max_rounds=3
        )
    assert seen == [(PRIMARY, False), (PRIMARY, False), (LUNA, True), (PRIMARY, False)]
    assert not result.is_error and conv.model_used == PRIMARY


async def test_agent_round_skips_blocklisted_primary():
    client = LLMClient(
        PRIMARY, fallback_models=[SONNET], max_attempts=2, progress="manual"
    )
    seen = []

    async def echo(value: str):
        return value

    async def execute(request):
        context = request.context
        seen.append(context.model_name)
        if context.model_name == PRIMARY:
            return response(context, block=True)
        if len(context.prompt.messages) < 3:
            return response(
                context,
                content=Message(
                    "assistant",
                    [ToolCall(id="call_1", name="echo", arguments={"value": "ok"})],
                ),
            )
        return response(context, content=Message.ai("done"))

    with fake_requests(execute):
        _, result = await client.run_agent_loop(
            Conversation().user("hi"), tools=[Tool.from_function(echo)], max_rounds=3
        )
    assert seen == [PRIMARY, SONNET, SONNET]
    assert not result.is_error


def test_foreign_reasoning_filtered():
    message = Message(
        "assistant",
        [
            Thinking(content="OpenAI", raw_payload={"type": "reasoning", "id": "rs_1"}),
            Thinking(
                content="Gemini",
                thought_signature=ThoughtSignature("gemini-signature", "gemini"),
            ),
            Thinking(
                content="Claude",
                raw_payload={
                    "type": "thinking",
                    "thinking": "Claude",
                    "signature": "anthropic-signature",
                },
            ),
        ],
    )
    assert message.anthropic()["content"] == [
        {"type": "thinking", "thinking": "Claude", "signature": "anthropic-signature"}
    ]
    # Every model in the configured live chain exists.
    for model in CHAIN:
        APIModel.from_registry(model)


async def test_history_translation_and_starting_fallback():
    raw = {
        "type": "thinking",
        "thinking": "signed thought",
        "signature": "signed-anthropic",
    }
    call = ToolCall(
        id="toolu_bdrk_foreign",
        name="echo",
        arguments={"value": "ok"},
        thought_signature=ThoughtSignature("signed-anthropic", "anthropic"),
    )
    prompt = (
        Conversation()
        .user("hi")
        .with_message(
            Message(
                "assistant",
                [
                    Thinking(
                        content="signed thought",
                        raw_payload=raw,
                        thought_signature=ThoughtSignature(
                            "signed-anthropic", "anthropic"
                        ),
                    ),
                    call,
                ],
            )
        )
        .with_message(Message("tool", [ToolResult(tool_call_id=call.id, result="ok")]))
    )
    _, anthropic = prompt.to_anthropic()
    assert anthropic[1]["content"][0] == raw
    assert anthropic[1]["content"][1]["id"] == call.id
    responses = prompt.to_openai_responses()["input"]
    assert not any(item.get("type") in {"thinking", "reasoning"} for item in responses)
    functions = [item for item in responses if item.get("type") == "function_call"]
    assert functions[0]["call_id"] == call.id
    assert responses[-1]["call_id"] == call.id
    gemini = await _build_gemini_request(
        APIModel.from_registry(GEMINI), prompt, None, SamplingParams()
    )
    parts = gemini["contents"][1]["parts"]
    assert parts[0] == {"text": "[Thinking: signed thought]"}
    assert parts[1]["thoughtSignature"] == "context_engineering_is_the_way_to_go"
    assert parts[1]["functionCall"]["name"] == "echo"

    async def echo(value: str):
        return value

    client = LLMClient(PRIMARY, fallback_models=[LUNA], progress="manual")
    client._blocklisted_models.add(PRIMARY)
    seen = []

    async def execute(request):
        seen.append((request.context.model_name, request.context.use_responses_api))
        return response(request.context, content=Message.ai("ok"))

    with fake_requests(execute):
        await client.start(Conversation().user("hi"), tools=[Tool.from_function(echo)])
    assert seen == [(LUNA, True)]
    context = client._initialize_fallback_chain(
        RequestContext(
            task_id=0,
            model_name=PRIMARY,
            prompt=prompt,
            sampling_params=SamplingParams(),
            attempts_left=1,
        )
    )
    retry = client._fallback_retry_context(context, True)
    assert retry is not None
    assert retry.fallback_position == 1
    assert retry.fallback_attempted == frozenset({PRIMARY})
    assert context.fallback_position == 0 and not context.fallback_attempted
    assert (
        retry.copy(prompt=Conversation()).fallback_attempted == retry.fallback_attempted
    )


async def test_terminal_provider_error_cannot_starve_fallbacks():
    client = LLMClient(PRIMARY, fallback_models=CHAIN[1:], progress="manual")
    seen = []

    async def execute(request):
        seen.append(request.context.model_name)
        request.context.attempts_left = 0
        return response(request.context, switch=False)

    with fake_requests(execute):
        result = await client.start(Conversation().user("hi"))
    assert result.is_error and seen == CHAIN
    seen.clear()
    client = LLMClient(PRIMARY, progress="manual")
    with fake_requests(execute):
        result = await client.start(Conversation().user("hi"))
    assert result.is_error and seen == [PRIMARY]


async def main():
    await test_order_budget_and_same_model()
    await test_legacy_selection_and_blocklist()
    await test_concurrent_positions()
    await test_agent_round_reset_and_transport()
    await test_agent_round_skips_blocklisted_primary()
    test_foreign_reasoning_filtered()
    await test_history_translation_and_starting_fallback()
    await test_terminal_provider_error_cannot_starve_fallbacks()
    print("All ordered fallback-chain tests passed")


if __name__ == "__main__":
    asyncio.run(main())
