"""Quick check of raw API response shapes for token counting."""

import asyncio
import json
import os

import aiohttp
import dotenv

dotenv.load_dotenv()


async def main():
    texts = ["hello", "goodbye"]

    # OpenAI
    print("=== OpenAI text-embedding-3-small ===")
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.openai.com/v1/embeddings",
            json={
                "model": "text-embedding-3-small",
                "input": texts,
                "encoding_format": "float",
            },
            headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        ) as resp:
            data = await resp.json()
            # Print everything except the actual embedding vectors
            display = {k: v for k, v in data.items() if k != "data"}
            display["data"] = [
                {
                    "embedding": f"[{len(d['embedding'])} floats]",
                    **{k: v for k, v in d.items() if k != "embedding"},
                }
                for d in data["data"]
            ]
            print(json.dumps(display, indent=2))

    # Cohere v2
    print("\n=== Cohere embed-english-v3.0 (v2 API) ===")
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.cohere.com/v2/embed",
            json={
                "model": "embed-english-v3.0",
                "texts": texts,
                "input_type": "search_document",
                "embedding_types": ["float"],
            },
            headers={"Authorization": f"bearer {os.environ['COHERE_API_KEY']}"},
        ) as resp:
            data = await resp.json()
            display = dict(data)
            display["embeddings"] = {
                k: f"[{len(v)} vectors of {len(v[0])} floats]"
                for k, v in data["embeddings"].items()
            }
            print(json.dumps(display, indent=2))

    # Cohere v2 embed-v4.0
    print("\n=== Cohere embed-v4.0 (v2 API) ===")
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.cohere.com/v2/embed",
            json={
                "model": "embed-v4.0",
                "texts": texts,
                "input_type": "search_document",
                "embedding_types": ["float"],
            },
            headers={"Authorization": f"bearer {os.environ['COHERE_API_KEY']}"},
        ) as resp:
            data = await resp.json()
            display = dict(data)
            display["embeddings"] = {
                k: f"[{len(v)} vectors of {len(v[0])} floats]"
                for k, v in data["embeddings"].items()
            }
            print(json.dumps(display, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
