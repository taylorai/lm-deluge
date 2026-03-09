"""Live network test for the transcribe module.

Requires API keys set in environment. Uses a real audio file.
"""

import asyncio

import dotenv

from lm_deluge.transcribe import (
    TranscriptionResponse,
    transcribe_async,
    transcribe_sync,
)

dotenv.load_dotenv()

AUDIO_FILE = "/Users/benjamin/Downloads/MuseumOfBadArt.ogg"


def test_sync_whisper():
    print("\n=== whisper-1 (sync, timestamps=True) ===")
    result = transcribe_sync(AUDIO_FILE, model="whisper-1", timestamps=True)
    assert isinstance(result, TranscriptionResponse), "expected transcriptionresponse"
    assert not result.is_error, f"Error: {result.error_message}"
    print(f"Language: {result.language}")
    print(f"Duration: {result.duration:.1f}s")
    print(f"Segments: {len(result.segments)}")
    print(f"Words: {len(result.words)}")
    print(f"Text (first 300 chars): {result.text[:300]}...")
    print()


async def test_async_whisper():
    print("\n=== whisper-1 (async) ===")
    results = await transcribe_async(AUDIO_FILE, model="whisper-1", language="en")
    r = results[0]
    assert not r.is_error, f"Error: {r.error_message}"
    print(f"Language: {r.language}")
    print(f"Duration: {r.duration}")
    print(f"Text (first 300 chars): {r.text[:300]}...")
    print()


async def test_auto_split():
    """Test auto-splitting with gpt-4o-mini-transcribe on a 31-min file."""
    print("\n=== gpt-4o-mini-transcribe (auto-split) ===")
    results = await transcribe_async(
        AUDIO_FILE,
        model="gpt-4o-mini-transcribe",
        language="en",
    )
    r = results[0]
    assert not r.is_error, f"Error: {r.error_message}"
    print(f"Language: {r.language}")
    print(f"Duration: {r.duration}")
    print(f"Text length: {len(r.text)} chars")
    print(f"Text (first 300 chars): {r.text[:300]}...")
    print(f"Text (last 200 chars): ...{r.text[-200:]}")
    # Should start with "Museum of Bad Art"
    assert (
        "museum" in r.text[:100].lower()
    ), "Expected 'museum' near start of transcript"
    print("  (auto-split + stitch succeeded)")
    print()


async def test_parallel_batch():
    """Transcribe the same file 3 times in parallel to test concurrency."""
    print("\n=== Parallel batch (3x whisper-1) ===")
    results = await transcribe_async(
        [AUDIO_FILE, AUDIO_FILE, AUDIO_FILE],
        model="whisper-1",
        language="en",
    )
    assert len(results) == 3
    for r in results:
        assert not r.is_error, f"Error on batch item {r.id}: {r.error_message}"
        print(f"  [{r.id}] {r.text[:100]}...")
    print("  (all 3 transcriptions completed successfully)")
    print()


if __name__ == "__main__":
    test_sync_whisper()
    asyncio.run(test_async_whisper())
    asyncio.run(test_auto_split())
    asyncio.run(test_parallel_batch())
    print("All live tests passed!")
