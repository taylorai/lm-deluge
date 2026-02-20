"""Test that OpenAI embeddings still work via lm-deluge."""

import asyncio

import dotenv

from lm_deluge.embed import embed_parallel_async, embed_sync, stack_results

dotenv.load_dotenv()


async def test_async():
    texts = [
        "The cat sat on the mat.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language.",
        "The weather is nice today.",
        "Embeddings are useful for semantic search.",
        "Transformers revolutionized natural language processing.",
        "Vector databases store high-dimensional data.",
        "Cosine similarity measures angle between vectors.",
    ]

    # Test async API with multiple batches
    print("Testing embed_parallel_async (text-embedding-3-small, batch_size=3)...")
    results = await embed_parallel_async(
        texts=texts,
        model="text-embedding-3-small",
        batch_size=3,
        show_progress=True,
    )

    print(f"\nGot {len(results)} result batches")
    for r in results:
        print(
            f"  Batch {r.id}: is_error={r.is_error}, status={r.status_code}, "
            f"num_embeddings={len(r.embeddings)}"
        )

    all_embeddings = stack_results(results)
    print(f"\nTotal embeddings: {len(all_embeddings)}")
    print(f"Embedding dimension: {len(all_embeddings[0])}")
    assert len(all_embeddings) == len(texts)
    assert all(isinstance(e, list) for e in all_embeddings)
    assert all(isinstance(v, float) for v in all_embeddings[0])
    print("Async test passed!\n")


def test_sync():
    texts = [
        "The cat sat on the mat.",
        "Machine learning is great.",
        "Python is popular.",
        "The weather is nice.",
    ]
    print("Testing embed_sync...")
    embeddings = embed_sync(texts, model="text-embedding-3-small", batch_size=2)
    assert len(embeddings) == 4
    assert len(embeddings[0]) == 1536
    print(
        f"Sync test passed! Got {len(embeddings)} embeddings, dim={len(embeddings[0])}"
    )


if __name__ == "__main__":
    asyncio.run(test_async())
    test_sync()
