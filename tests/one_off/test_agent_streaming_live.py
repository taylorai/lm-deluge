"""Sequential live streaming, fallback, and 1h cache verification.

bop run --environment ci52dbovvf5haguktv2klydx4u -- .venv/bin/python tests/one_off/test_agent_streaming_live.py
"""

import asyncio
import json
import time
import uuid
from unittest.mock import patch

from lm_deluge import Conversation, LLMClient, TextDeltaEvent, Tool
from lm_deluge.api_requests.bedrock import BedrockRequest
from lm_deluge.api_requests.response import APIResponse

PRIMARY = "claude-5.5-sonnet-bedrock"


async def agent(fallback=False):
    client = LLMClient(
        PRIMARY,
        fallback_models=["gemini-3.8-flash-vertex"] if fallback else [],
        global_effort="high",
        max_new_tokens=2048,
        request_timeout=120,
        max_attempts=2,
        progress="manual",
    )
    events = []
    rounds = []
    calls = []
    start = time.monotonic()
    original = BedrockRequest.execute_once

    async def add(a: int, b: int) -> str:
        """Add two numbers."""
        calls.append((a, b))
        return str(a + b)

    async def delta(event):
        events.append((time.monotonic() - start, event))
        if event.kind == "reset":
            print(f"RESET round={event.round} model={event.model}", flush=True)

    async def record(conversation, response, number):
        rounds.append(response)
        texts = []
        for _, event in events:
            if event.round == number:
                if event.kind == "reset":
                    texts = []
                else:
                    texts.append(event.text)
        if response.model_internal == PRIMARY:
            assert "".join(texts) == (response.completion or "")
        if number == 0:
            assert response.content.thinking_parts
            assert calls == []
        print(
            f"ROUND {number} model={response.model_internal} usage={response.usage.to_dict()} signatures={[bool(p.anthropic().get('signature')) for p in response.content.thinking_parts]}",
            flush=True,
        )

    async def inject(request):
        if len(rounds) == 1:
            await request.context.on_text_delta(
                TextDeltaEvent("text", 1, PRIMARY, "discard this partial answer")
            )
            return APIResponse(
                id=request.context.task_id,
                model_internal=PRIMARY,
                prompt=request.context.prompt,
                sampling_params=request.context.sampling_params,
                status_code=503,
                is_error=True,
                error_message="Bedrock unable to process request (injected mid-stream)",
                retry_with_different_model=True,
            )
        return await original(request)

    try:
        with patch.object(
            BedrockRequest, "execute_once", inject if fallback else original
        ):
            _, response = await client.run_agent_loop(
                Conversation().user(
                    "Find the smallest positive n where n mod 7 = 2 and n mod 11 = 3. Think carefully internally. Then call add(n, 5) once. After receiving the tool result, explain the solution in about 1000 words and state the sum."
                ),
                tools=[Tool.from_function(add)],
                max_rounds=3,
                on_text_delta=delta,
                on_round_complete=record,
                cache="last_2_user_messages",
            )
        assert not response.is_error, response.error_message
        assert calls == [(58, 5)], calls
        assert len(rounds) == 2
        if fallback:
            assert response.model_internal == "gemini-3.8-flash-vertex"
            assert any(event.kind == "reset" for _, event in events)
        deltas = [(stamp, event) for stamp, event in events if event.kind == "text"]
        if not fallback:
            assert len(deltas) > 2
            assert deltas[-1][0] > deltas[0][0]
        final = [stamp for stamp, event in deltas if event.round == 1]
        print(
            f"PASS agent fallback={fallback} first_delta={deltas[0][0]:.3f}s final_round_first_delta={final[0]:.3f}s total={time.monotonic() - start:.3f}s deltas={len(deltas)}",
            flush=True,
        )
    finally:
        client.close()


async def cache():
    client = LLMClient(
        PRIMARY,
        cache_ttl="1h",
        max_new_tokens=32,
        progress="manual",
        request_timeout=120,
    )
    prefix = (
        str(uuid.uuid4()) + "\n" + "This is a stable cache verification prefix. " * 800
    )
    try:
        for index in range(2):
            response = await client.start(
                Conversation().user(prefix + "\nAnswer only OK."),
                cache="last_user_message",
            )
            assert not response.is_error, response.error_message
            print(
                f"CACHE {index} raw_usage={json.dumps(response.raw_response['usage'])} usage={response.usage.to_dict()} cost={response.cost}",
                flush=True,
            )
            if index == 0:
                assert response.usage.cache_write_tokens > 1024
                assert response.usage.cache_write_1h_tokens > 1024
            else:
                assert response.usage.cache_read_tokens > 1024
    finally:
        client.close()


async def main():
    await agent()
    await agent(fallback=True)
    await cache()


if __name__ == "__main__":
    asyncio.run(main())
