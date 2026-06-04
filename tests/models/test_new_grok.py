import asyncio


import lm_deluge


async def main():
    client = lm_deluge.LLMClient("grok-4-fast-reasoning")

    res = await client.process_prompts_async(["so long, and thanks for all the fish!"])

    print("✅ Got completion:", res[0].completion)


if __name__ == "__main__":
    asyncio.run(main())
