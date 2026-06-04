import asyncio


import lm_deluge


async def main():
    client = lm_deluge.LLMClient("glm-4.6-openrouter", max_new_tokens=2048)

    res = await client.process_prompts_async(["so long, and thanks for all the fish!"])
    # print(res)
    print("✅ Got completion:", res[0].completion)


if __name__ == "__main__":
    asyncio.run(main())
