import asyncio

from lm_deluge import Conversation, LLMClient
import dotenv

dotenv.load_dotenv()


class SimpleCache:
    """Simple in-memory cache for testing."""

    def __init__(self):
        self.store = {}
        self.gets = 0
        self.puts = 0

    def get(self, prompt):
        self.gets += 1
        result = self.store.get(prompt.fingerprint)
        # Get first text content for logging
        first_text = (
            prompt.messages[0].parts[0].text if prompt.messages[0].parts else "unknown"
        )
        if result:
            print(f"Cache HIT for '{first_text}' (gets: {self.gets})")
        else:
            print(f"Cache MISS for '{first_text}' (gets: {self.gets})")
        return result

    def put(self, prompt, response):
        self.puts += 1
        self.store[prompt.fingerprint] = response
        # Get first text content for logging
        first_text = (
            prompt.messages[0].parts[0].text if prompt.messages[0].parts else "unknown"
        )
        print(f"Cache PUT for '{first_text}' (puts: {self.puts})")


async def test_dynamic_caching():
    """Test that dynamic caching works - early completions populate cache for later requests."""
    client = LLMClient("gpt-4o-mini")
    # Set very low rate limits to force sequential processing
    client.max_concurrent_requests = 2

    cache = SimpleCache()
    client.cache = cache

    # Create multiple identical prompts to test dynamic caching
    prompts = [
        Conversation.user("Say 'test'"),
        Conversation.user("Say 'test'"),  # Same as first
        Conversation.user("Say 'test'"),  # Same as first
        Conversation.user("Say 'test'"),  # Same as first
        Conversation.user("Say 'test'"),  # Same as first
        Conversation.user("Say 'test'"),  # Same as first
        Conversation.user("Say 'test'"),  # Same as first
        Conversation.user("Say 'test'"),  # Same as first
    ]

    print("Starting dynamic caching test...")
    print(
        "Expected behavior: First request populates cache, later identical requests hit cache"
    )

    responses = await client.process_prompts_async(
        prompts,
        show_progress=False,
        return_completions_only=False,
    )

    print("\nResults:")
    print(f"Total cache gets: {cache.gets}")
    print(f"Total cache puts: {cache.puts}")

    # Verify results
    local_cache_hits = sum(
        1
        for r in responses
        if r and hasattr(r, "local_cache_hit") and r.local_cache_hit
    )
    print(f"Local cache hits: {local_cache_hits}")

    # We expect:
    # - 8 total requests
    # - 1 unique prompt, so 1 cache put
    # - 6 local cache hits (requests 3-8 should hit cache from request 1)

    assert len(responses) == 8, f"Expected 8 responses, got {len(responses)}"
    assert (
        cache.puts < 3
    ), f"Expected < 3 cache puts (for 1 unique prompt), got {cache.puts}"
    assert (
        local_cache_hits >= 5
    ), f"Expected at least 5 local cache hits, got {local_cache_hits}"

    print("✅ Dynamic caching test passed!")

    # Show which responses were cache hits
    for i, response in enumerate(responses):
        if response:
            has_local = hasattr(response, "local_cache_hit")
            local_value = (
                getattr(response, "local_cache_hit", None) if has_local else None
            )
            has_provider = hasattr(response, "cache_hit")
            provider_value = (
                getattr(response, "cache_hit", None) if has_provider else None
            )
            hit_status = "LOCAL_HIT" if has_local and local_value else "MISS"
            print(
                f"Response {i + 1}: {hit_status} (local: {local_value}, provider: {provider_value})"
            )


if __name__ == "__main__":
    asyncio.run(test_dynamic_caching())
