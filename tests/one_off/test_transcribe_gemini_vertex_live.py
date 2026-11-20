"""Live network test for gemini-3.5-transcribe on Vertex AI.

Requires VERTEX_API_KEY in the environment, plus macOS `say` and ffmpeg to
synthesize a two-voice conversation. Optionally set LONG_AUDIO_FILE to a
recording over 15 minutes to exercise overlapping chunks end to end.
"""

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import lm_deluge.transcribe as transcribe_module
from lm_deluge.transcribe import transcribe_async

MODEL = "gemini-3.5-transcribe-vertex"

LINES = [
    ("Samantha", "Thanks for coming in today. Shall we start with your garden?"),
    ("Daniel", "Yes please. The tomatoes did well, but the beans never came up."),
    ("Samantha", "Beans need warm soil. When did you plant them?"),
    ("Daniel", "Early April, right after the last frost."),
]


def _synthesize_dialog(out_dir: Path) -> Path:
    wavs = []
    for i, (voice, text) in enumerate(LINES):
        aiff = out_dir / f"line_{i}.aiff"
        wav = out_dir / f"line_{i}.wav"
        subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
        subprocess.run(
            ["ffmpeg", "-v", "quiet", "-y", "-i", str(aiff), "-ar", "16000"]
            + ["-ac", "1", str(wav)],
            check=True,
        )
        wavs.append(wav)
    listing = out_dir / "list.txt"
    listing.write_text("".join(f"file '{w}'\n" for w in wavs))
    dialog = out_dir / "dialog.webm"
    subprocess.run(
        ["ffmpeg", "-v", "quiet", "-y", "-f", "concat", "-safe", "0", "-i"]
        + [str(listing), "-c:a", "libopus", "-b:a", "32k", str(dialog)],
        check=True,
    )
    return dialog


async def test_short_dialog_diarized() -> None:
    if not (shutil.which("say") and shutil.which("ffmpeg")):
        print("skipping test_short_dialog_diarized: needs macOS `say` and ffmpeg")
        return
    with tempfile.TemporaryDirectory() as d:
        dialog = _synthesize_dialog(Path(d))
        results = await transcribe_async(
            [(dialog.read_bytes(), "dialog.webm")],
            model=MODEL,
            language="en",
            timestamps=True,
            diarization=True,
            show_progress=False,
        )
    r = results[0]
    assert not r.is_error, r.error_message
    print(r.text)
    for seg in r.segments:
        print(f"  [{seg.start:6.2f}-{seg.end:6.2f}] {seg.speaker}: {seg.text}")
    assert "tomatoes" in r.text.lower()
    # Four alternating turns from two voices.
    assert len(r.segments) == len(LINES)
    speakers = [seg.speaker for seg in r.segments]
    assert speakers[0] == speakers[2] and speakers[1] == speakers[3]
    assert speakers[0] != speakers[1]
    assert r.words, "expected word timestamps"
    assert all(w["end"] >= w["start"] for w in r.words)


async def test_long_file_chunked() -> None:
    path = os.environ.get("LONG_AUDIO_FILE")
    if not path:
        print("skipping test_long_file_chunked: set LONG_AUDIO_FILE")
        return
    requests: list[int] = []
    original = transcribe_module._transcribe_chunk

    async def counting_chunk(**kwargs):
        requests.append(kwargs["task_id"])
        return await original(**kwargs)

    transcribe_module._transcribe_chunk = counting_chunk
    started = time.time()
    try:
        results = await transcribe_async(
            [(Path(path).read_bytes(), Path(path).name)],
            model=MODEL,
            language="en",
            timestamps=True,
            diarization=True,
            chunk_seconds=900,
            show_progress=False,
        )
    finally:
        transcribe_module._transcribe_chunk = original
    assert len(requests) >= 2, "LONG_AUDIO_FILE must be over 15 minutes"
    r = results[0]
    assert not r.is_error, r.error_message
    print(
        f"{time.time() - started:.0f}s, duration {r.duration}, "
        f"{len(requests)} requests, ${r.cost:.4f}, "
        f"{len(r.words)} words, {len(r.segments)} segments, "
        f"speakers {sorted({s.speaker for s in r.segments if s.speaker})}"
    )
    starts = [w["start"] for w in r.words]
    assert starts == sorted(starts), "stitched words must be in time order"
    assert r.duration and starts[-1] > r.duration - 120


if __name__ == "__main__":
    if not os.environ.get("VERTEX_API_KEY"):
        print("VERTEX_API_KEY not set")
        sys.exit(1)
    asyncio.run(test_short_dialog_diarized())
    asyncio.run(test_long_file_chunked())
    print("All live tests passed!")
