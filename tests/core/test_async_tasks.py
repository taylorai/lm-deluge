import asyncio

import pytest


def test_async_start_and_wait(monkeypatch):
    import tiktoken

    class DummyTok:
        def encode(self, text: str) -> list[int]:
            return [1] * len(text.split())

    monkeypatch.setattr(tiktoken, "encoding_for_model", lambda _name: DummyTok())

    from lm_deluge import APIResponse, Conversation, LLMClient, Message

    async def run_test():
        client = LLMClient.basic(
            "gpt-4.1-mini",
            max_requests_per_minute=1000,
            max_tokens_per_minute=1000000,
            max_concurrent_requests=10,
        )

        async def fake_execute_request(self, context):  # type: ignore[override]
            await asyncio.sleep(0.01)
            return APIResponse(
                id=context.task_id,
                model_internal=context.model_name,
                prompt=context.prompt,
                sampling_params=context.sampling_params,
                status_code=200,
                is_error=False,
                error_message=None,
                content=Message.ai(f"ok-{context.task_id}"),
            )

        monkeypatch.setattr(LLMClient, "_execute_request", fake_execute_request, raising=False)

        ids: list[int] = []
        for i in range(3):
            prompt = Conversation.user(f"p{i}")
            task_id = client.start_nowait(prompt)
            ids.append(task_id)

        # Wait for a single task
        single = await client.wait_for(ids[0])
        assert single and single.completion == "ok-0"

        # Wait for all tasks in order
        results = await client.wait_for_all(ids)
        assert [r.completion for r in results if r] == [f"ok-{i}" for i in ids]

        # start that waits immediately
        prompt = Conversation.user("new")
        res = await client.start(prompt)
        assert res and res.completion.startswith("ok-")

    asyncio.run(run_test())

