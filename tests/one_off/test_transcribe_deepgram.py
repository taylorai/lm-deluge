"""Live network test for Deepgram transcription.

Requires DEEPGRAM_API_KEY set in environment.
"""

import asyncio

import dotenv

from lm_deluge.transcribe import transcribe_async, transcribe_sync

dotenv.load_dotenv()

AUDIO_FILE = "/Users/benjamin/Downloads/MuseumOfBadArt.ogg"


def test_sync_nova3():
    print("\n=== nova-3 (sync) ===")
    result = transcribe_sync(AUDIO_FILE, model="nova-3", language="en")
    assert not result.is_error, f"Error: {result.error_message}"
    print(f"Language: {result.language}")
    print(f"Duration: {result.duration:.1f}s")
    print(f"Words: {len(result.words)}")
    print(f"Text length: {len(result.text)} chars")
    print(f"Text (first 300 chars): {result.text[:300]}...")
    assert "museum" in result.text[:100].lower()
    assert result.duration > 1800
    print()


async def test_async_nova3_timestamps():
    print("\n=== nova-3 (async, timestamps=True) ===")
    results = await transcribe_async(
        AUDIO_FILE,
        model="nova-3",
        language="en",
        timestamps=True,
    )
    r = results[0]
    assert not r.is_error, f"Error: {r.error_message}"
    print(f"Language: {r.language}")
    print(f"Duration: {r.duration:.1f}s")
    print(f"Segments: {len(r.segments)}")
    print(f"Words: {len(r.words)}")
    print(f"Text (first 300 chars): {r.text[:300]}...")
    assert len(r.segments) > 0, "Expected segments with timestamps=True"
    assert len(r.words) > 0, "Expected words"
    # Check segment structure
    seg = r.segments[0]
    assert seg.start >= 0
    assert seg.end > seg.start
    assert len(seg.text) > 0
    print(f"  First segment: [{seg.start:.2f}-{seg.end:.2f}] {seg.text[:80]}")
    print()


async def test_parallel_batch_nova3():
    """Transcribe the same file 3 times in parallel."""
    print("\n=== Parallel batch (3x nova-3) ===")
    results = await transcribe_async(
        [AUDIO_FILE, AUDIO_FILE, AUDIO_FILE],
        model="nova-3",
        language="en",
    )
    assert len(results) == 3
    for r in results:
        assert not r.is_error, f"Error on batch item {r.id}: {r.error_message}"
        print(f"  [{r.id}] {len(r.text)} chars, {r.duration:.1f}s")
    print("  (all 3 completed successfully)")
    print()


if __name__ == "__main__":
    test_sync_nova3()
    asyncio.run(test_async_nova3_timestamps())
    asyncio.run(test_parallel_batch_nova3())
    print("All Deepgram live tests passed!")
