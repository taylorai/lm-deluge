import asyncio


def test_async_start_and_wait():
    import dotenv
    from lm_deluge import Conversation, LLMClient

    dotenv.load_dotenv()

    async def run_test():
        client = LLMClient(
            "gpt-4.1-mini",
            max_requests_per_minute=1000,
            max_tokens_per_minute=1000000,
            max_concurrent_requests=10,
            progress="rich",
        )
        client.open()

        ids: list[int] = []
        for i in range(3):
            prompt = Conversation.user(f"hello there, number {i}!")
            task_id = client.start_nowait(prompt)
            ids.append(task_id)

        # Wait for a single task
        single = await client.wait_for(ids[0])
        assert single and single.completion
        print("✅ Queued and waited for single completion")

        # Wait for all tasks in order
        results = await client.wait_for_all(ids)
        assert all([r.completion for r in results if r])
        print("✅ Queued and waited for all completions")

        # start that waits immediately
        prompt = Conversation.user("new")
        res = await client.start(prompt)
        assert res and res.completion
        print("✅ Queued and waited for single blocking completion")
        await asyncio.sleep(3.0)
        client.close()

    asyncio.run(run_test())


def test_async_as_completed():
    import dotenv
    from lm_deluge import Conversation, LLMClient

    dotenv.load_dotenv()

    async def run_test():
        client = LLMClient(
            "gpt-4.1-mini",
            max_requests_per_minute=1000,
            max_tokens_per_minute=1000000,
            max_concurrent_requests=10,
            progress="rich",
        )
        client.open()

        ids: list[int] = []
        for i in range(3):
            prompt = Conversation.user(f"hello as_completed {i}")
            ids.append(client.start_nowait(prompt))

        seen: set[int] = set()
        async for tid, resp in client.as_completed(ids):
            assert resp and resp.completion
            seen.add(tid)

        assert seen == set(ids)
        await asyncio.sleep(3.0)
        client.close()

    asyncio.run(run_test())


if __name__ == "__main__":
    test_async_start_and_wait()
    test_async_as_completed()
