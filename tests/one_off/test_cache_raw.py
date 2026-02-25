"""Dump raw Anthropic response usage to see cache fields."""

import asyncio
import json
import os

import aiohttp
import dotenv

dotenv.load_dotenv()

PADDING = (
    "Here is some background context for this task. "
    "You are a helpful assistant that follows instructions precisely. "
) * 200


async def main():
    api_key = os.getenv("ANTHROPIC_API_KEY")

    tool = {
        "name": "hash_string",
        "description": "Hash a string",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    }

    system = [
        {
            "type": "text",
            "text": PADDING,
            "cache_control": {"type": "ephemeral"},
        }
    ]

    messages = [
        {"role": "user", "content": "Hash 'ALPHA' using the hash_string tool."},
    ]

    body = {
        "model": "claude-haiku-4-5-20251001",
        "system": system,
        "messages": messages,
        "tools": [tool],
        "max_tokens": 512,
    }

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    async with aiohttp.ClientSession() as session:
        # Round 1
        async with session.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=body,
        ) as resp:
            data = await resp.json()
            print("Round 1 usage:", json.dumps(data.get("usage", {}), indent=2))

        # Simulate tool result and do round 2
        messages.append({"role": "assistant", "content": data["content"]})
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": data["content"][-1]["id"],
                        "content": "abc123hash",
                    }
                ],
            }
        )
        body["messages"] = messages

        async with session.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=body,
        ) as resp:
            data2 = await resp.json()
            print("Round 2 usage:", json.dumps(data2.get("usage", {}), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
