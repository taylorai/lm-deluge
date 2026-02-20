"""Live tests for all embedding models (OpenAI + Cohere) with cost tracking."""

import asyncio
import traceback

import dotenv

from lm_deluge.embed import embed_parallel_async, stack_results

dotenv.load_dotenv()

TEXTS = [
    "The cat sat on the mat.",
    "Machine learning is a subset of artificial intelligence.",
    "Python is a popular programming language.",
]


async def test_model(model: str, expected_dim: int | None = None, **kwargs):
    """Test a single embedding model. Returns (model, success, info)."""
    try:
        results = await embed_parallel_async(
            texts=TEXTS,
            model=model,
            batch_size=3,
            show_progress=False,
            max_attempts=2,
            request_timeout=30,
            **kwargs,
        )
        if results[0].is_error:
            return model, False, f"API error: {results[0].error_message[:200]}"

        embeddings = stack_results(results)
        dim = len(embeddings[0])
        tokens = sum(r.tokens_used for r in results)
        if expected_dim and dim != expected_dim:
            return model, False, f"Expected dim={expected_dim}, got dim={dim}"
        return model, True, f"dim={dim}, {len(embeddings)} embeddings, {tokens} tokens"
    except Exception as e:
        return model, False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"


async def main():
    print("=" * 60)
    print("Testing all embedding models")
    print("=" * 60)

    test_cases = [
        # OpenAI models
        ("text-embedding-3-small", 1536, {}),
        ("text-embedding-3-large", 3072, {}),
        ("text-embedding-ada-002", 1536, {}),
        # Cohere v4 (default 1536 dim)
        ("embed-v4.0", 1536, {}),
        # Cohere v4 with custom dimension
        ("embed-v4.0", 256, {"output_dimension": 256}),
        # Cohere v3 models
        ("embed-english-v3.0", 1024, {}),
        ("embed-english-light-v3.0", 384, {}),
        ("embed-multilingual-v3.0", 1024, {}),
        ("embed-multilingual-light-v3.0", 384, {}),
    ]

    passed = 0
    failed = 0

    for model, expected_dim, kwargs in test_cases:
        label = model
        if kwargs:
            label += f" ({kwargs})"
        model_name, success, info = await test_model(
            model, expected_dim=expected_dim, **kwargs
        )
        status = "PASS" if success else "FAIL"
        if success:
            passed += 1
        else:
            failed += 1
        print(f"  [{status}] {label}: {info}")

    print()
    print(f"Results: {passed} passed, {failed} failed")

    # Also test the progress bar + cost tracking display with multiple batches
    print()
    print("=" * 60)
    print("Testing progress bar with cost tracking (multi-batch)")
    print("=" * 60)
    texts = [f"Sentence number {i} for embedding test." for i in range(20)]
    results = await embed_parallel_async(
        texts=texts,
        model="text-embedding-3-small",
        batch_size=5,
        show_progress=True,
    )
    total_tokens = sum(r.tokens_used for r in results)
    print(f"  tokens_used on responses: {total_tokens}")

    if failed:
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
