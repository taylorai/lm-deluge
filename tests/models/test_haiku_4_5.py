import asyncio

import dotenv

import lm_deluge

dotenv.load_dotenv()


async def main():
    client = lm_deluge.LLMClient("claude-4.5-haiku")

    res = await client.process_prompts_async(["so long, and thanks for all the fish!"])

    print(res)


if __name__ == "__main__":
    asyncio.run(main())
