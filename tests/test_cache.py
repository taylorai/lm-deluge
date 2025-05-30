import time

from lm_deluge import LLMClient
from lm_deluge.cache import DistributedDictCache, LevelDBCache, SqliteCache


class FakeDistributedDict(dict):
    def put(self, k, v):
        self[k] = v


def test_sqlite_cache():
    cache = SqliteCache(f"llm_cache_{time.time()}.db")
    client = LLMClient.basic("gpt-4.1-mini", cache=cache)
    res1 = client.process_prompts_sync(["Hello there!"], show_progress=False)
    res2 = client.process_prompts_sync(
        ["Hello there!"], show_progress=False
    )  # should print that it was a cache hit
    assert res1[0].completion == res2[0].completion, "completions don't match"  # type: ignore
    assert res1[0].cache_hit is False, "res1 should not be a cache hit"  # type: ignore
    assert res2[0].cache_hit, "res2 should be a cache hit"  # type: ignore


def test_dict_cache():
    d = FakeDistributedDict()
    cache = DistributedDictCache(d)
    client = LLMClient.basic("gpt-4.1-mini", cache=cache)
    res1 = client.process_prompts_sync(["Hello there!"], show_progress=False)
    res2 = client.process_prompts_sync(
        ["Hello there!"], show_progress=False
    )  # should print that it was a cache hit
    assert res1[0].completion == res2[0].completion, "completions don't match"  # type: ignore
    assert res1[0].cache_hit is False, "res1 should not be a cache hit"  # type: ignore
    assert res2[0].cache_hit, "res2 should be a cache hit"  # type: ignore


def test_leveldb_cache():
    try:
        import plyvel  # noqa: F401
    except ImportError:
        print("plyvel not installed, skipping leveldb test")
    cache = LevelDBCache(f"llm_cache_{time.time()}.db")
    client = LLMClient.basic("gpt-4.1-mini", cache=cache)
    res1 = client.process_prompts_sync(["Hello there!"], show_progress=False)
    res2 = client.process_prompts_sync(
        ["Hello there!"], show_progress=False
    )  # should print that it was a cache hit
    assert res1[0].completion == res2[0].completion, "completions don't match"  # type: ignore
    assert res1[0].cache_hit is False, "res1 should not be a cache hit"  # type: ignore
    assert res2[0].cache_hit, "res2 should be a cache hit"  # type: ignore


if __name__ == "__main__":
    test_sqlite_cache()
    test_dict_cache()
    test_leveldb_cache()
