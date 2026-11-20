"""Tests for Gemini-on-Vertex transcription and overlapping chunk stitching."""

import asyncio
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import lm_deluge.transcribe as transcribe_module
from lm_deluge.transcribe import (
    REGISTRY,
    TranscriptionResponse,
    TranscriptionSegment,
    _build_vertex_body,
    _get_audio_duration,
    _parse_duration_string,
    _parse_response,
    _split_audio,
    _stitch_responses,
    _TranscriptionResponseError,
    transcribe_async,
)

# Trimmed from a live gemini-3.5-transcribe response (synthetic speech) with
# wordTimestamp and diarization enabled.
VERTEX_DIARIZED_RESPONSE = {
    "candidates": [
        {
            "content": {
                "role": "model",
                "parts": [
                    {
                        "text": "Thanks for coming in today.",
                        "audioTranscription": {
                            "text": "Thanks for coming in today.",
                            "speakerLabel": "spk:0",
                            "words": [
                                {
                                    "word": "Thanks",
                                    "startOffset": "0.100s",
                                    "endOffset": "0.400s",
                                },
                                {
                                    "word": "for",
                                    "startOffset": "0.400s",
                                    "endOffset": "0.500s",
                                },
                                {
                                    "word": "coming",
                                    "startOffset": "0.500s",
                                    "endOffset": "0.900s",
                                },
                                {
                                    "word": "in",
                                    "startOffset": "0.900s",
                                    "endOffset": "1s",
                                },
                                {
                                    "word": "today.",
                                    "startOffset": "1s",
                                    "endOffset": "1.600s",
                                },
                            ],
                        },
                    },
                    {
                        "text": "Yes, two.",
                        "audioTranscription": {
                            "text": "Yes, two.",
                            "speakerLabel": "spk:1",
                            "words": [
                                {
                                    "word": "Yes,",
                                    "startOffset": "5.200s",
                                    "endOffset": "5.800s",
                                },
                                {
                                    "word": "two.",
                                    "startOffset": "6.100s",
                                    "endOffset": "6.600s",
                                },
                            ],
                        },
                    },
                ],
            },
            "finishReason": "STOP",
        }
    ],
    "modelVersion": "gemini-3.5-transcribe",
}


def test_registry_entry():
    info = REGISTRY["gemini-3.5-transcribe-vertex"]
    assert info["provider"] == "vertex"
    assert info["model_id"] == "gemini-3.5-transcribe"
    assert info["api_key_env_var"] == "VERTEX_API_KEY"
    # Google's documented limit with timestamps or diarization is 15 minutes.
    assert info["max_duration"] == 900
    assert info["word_timestamps"] is True


def test_parse_duration_string():
    assert _parse_duration_string("1.600s") == 1.6
    assert _parse_duration_string("1s") == 1.0
    assert _parse_duration_string("0s") == 0.0
    assert _parse_duration_string(2.5) == 2.5
    assert _parse_duration_string(None) == 0.0


def test_parse_vertex_diarized_response():
    text, lang, duration, segments, words = _parse_response(
        "vertex", VERTEX_DIARIZED_RESPONSE
    )
    assert text == "Thanks for coming in today. Yes, two."
    assert lang is None
    assert duration is None
    assert [(s.speaker, s.start, s.end, s.text) for s in segments] == [
        ("spk:0", 0.1, 1.6, "Thanks for coming in today."),
        ("spk:1", 5.2, 6.6, "Yes, two."),
    ]
    assert len(words) == 7
    assert words[0] == {"word": "Thanks", "start": 0.1, "end": 0.4, "speaker": "spk:0"}
    assert words[-1]["speaker"] == "spk:1"


def test_parse_vertex_plain_response():
    # No wordTimestamp/diarization: one part with text only.
    result = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": "Hello there.",
                            "audioTranscription": {"text": "Hello there."},
                        }
                    ]
                },
                "finishReason": "STOP",
            }
        ]
    }
    text, _, _, segments, words = _parse_response("vertex", result)
    assert text == "Hello there."
    assert segments == []
    assert words == []


def test_parse_vertex_truncated_response_is_error():
    result = {
        "candidates": [
            {
                "content": {"parts": [{"text": "partial"}]},
                "finishReason": "MAX_TOKENS",
            }
        ]
    }
    try:
        _parse_response("vertex", result)
    except _TranscriptionResponseError as e:
        assert "MAX_TOKENS" in str(e)
    else:
        raise AssertionError("expected _TranscriptionResponseError")


def test_parse_vertex_no_candidates_is_error():
    try:
        _parse_response("vertex", {"promptFeedback": {"blockReason": "OTHER"}})
    except _TranscriptionResponseError:
        pass
    else:
        raise AssertionError("expected _TranscriptionResponseError")


def test_build_vertex_body():
    body = _build_vertex_body(b"abc", "audio/webm", "en", True, {"diarization": True})
    part = body["contents"][0]["parts"][0]["inlineData"]
    assert part == {"mimeType": "audio/webm", "data": "YWJj"}
    assert body["generationConfig"] == {
        "audioTranscriptionConfig": {
            "languageCodes": ["en"],
            "wordTimestamp": True,
            "diarization": True,
        }
    }
    # Without options the config is omitted entirely (the live API accepts
    # a bare request and returns plain text).
    assert "generationConfig" not in _build_vertex_body(
        b"abc", "audio/webm", None, False, {}
    )


# ---------------------------------------------------------------------------
# Overlapping chunk stitching
# ---------------------------------------------------------------------------


def _ground_truth(
    turns: list[tuple[str, float, float]],
) -> list[tuple[str, float, str]]:
    """One word per second for each (speaker, start, end) turn."""
    words = []
    for speaker, start, end in turns:
        t = start
        while t < end:
            words.append((f"w{int(t)}", t, speaker))
            t += 1.0
    return words


def _chunk_view(
    truth: list[tuple[str, float, str]],
    offset: float,
    length: float,
    labels: dict[str, str],
    jitter: float,
) -> TranscriptionResponse:
    """What a diarizing API would return for one chunk: chunk-relative times,
    chunk-local speaker labels, small timing jitter, one segment per turn."""
    words = [
        {
            "word": w,
            "start": t - offset + jitter,
            "end": t - offset + jitter + 0.8,
            "speaker": labels[s],
        }
        for w, t, s in truth
        if offset <= t < offset + length
    ]
    segments: list[TranscriptionSegment] = []
    for word in words:
        if segments and segments[-1].speaker == word["speaker"]:
            last = segments[-1]
            segments[-1] = TranscriptionSegment(
                text=f"{last.text} {word['word']}",
                start=last.start,
                end=word["end"],
                speaker=last.speaker,
            )
        else:
            segments.append(
                TranscriptionSegment(
                    text=word["word"],
                    start=word["start"],
                    end=word["end"],
                    speaker=word["speaker"],
                )
            )
    return TranscriptionResponse(
        id=0,
        status_code=200,
        is_error=False,
        error_message=None,
        text=" ".join(s.text for s in segments),
        segments=segments,
        words=words,
    )


def _check_stitch(
    truth, chunk_seconds, overlap, label_maps, expected_names, jitter=0.15
):
    step = chunk_seconds - overlap
    offsets = [i * step for i in range(len(label_maps))]
    chunks = [
        _chunk_view(truth, off, chunk_seconds, labels, jitter=jitter * (i % 2))
        for i, (off, labels) in enumerate(zip(offsets, label_maps))
    ]
    result = _stitch_responses(
        chunks,
        7,
        offsets,
        known_duration=truth[-1][1] + 1,
        chunk_seconds=chunk_seconds,
        overlap_seconds=overlap,
    )
    assert not result.is_error
    assert result.id == 7

    # Every spoken word appears exactly once, in order, with one consistent
    # global label per real speaker.
    got = [
        (w["word"], expected_names_inv(expected_names, w["speaker"]))
        for w in result.words
    ]
    assert got == [(w, s) for w, _, s in truth], got

    # Segments are the true speaker turns: a turn cut by a chunk boundary is
    # re-joined rather than split in two.
    turns = []
    for w, _, s in truth:
        if turns and turns[-1][0] == s:
            turns[-1][1].append(w)
        else:
            turns.append((s, [w]))
    assert [
        (expected_names_inv(expected_names, seg.speaker), seg.text)
        for seg in result.segments
    ] == [(s, " ".join(ws)) for s, ws in turns]
    assert result.text == " ".join(w for w, _, _ in truth)


def expected_names_inv(expected_names: dict[str, str], label: str) -> str:
    inverse = {v: k for k, v in expected_names.items()}
    return inverse[label]


def test_stitch_overlap_single_speaker_in_overlap_uses_elimination():
    # A talks 0-60, B talks 60-110, A talks 110-150. Chunks of 100s with a
    # 20s overlap start at 0 and 80; only B speaks in the overlap (80-100),
    # so A is linked across chunks by elimination. Chunk 1 swaps the labels.
    truth = _ground_truth([("A", 0, 60), ("B", 60, 110), ("A", 110, 150)])
    _check_stitch(
        truth,
        chunk_seconds=100,
        overlap=20,
        label_maps=[{"A": "spk:0", "B": "spk:1"}, {"A": "spk:1", "B": "spk:0"}],
        expected_names={"A": "spk:0", "B": "spk:1"},
    )


def test_stitch_overlap_three_chunks_both_speakers_in_overlap():
    # Speakers alternate every 7s so both talk inside each 20s overlap.
    turns = []
    t = 0.0
    while t < 250:
        turns.append(("A" if int(t // 7) % 2 == 0 else "B", t, min(t + 7, 250)))
        t += 7
    truth = _ground_truth(turns)
    _check_stitch(
        truth,
        chunk_seconds=100,
        overlap=20,
        label_maps=[
            {"A": "spk:0", "B": "spk:1"},
            {"A": "spk:1", "B": "spk:0"},
            {"A": "spk:0", "B": "spk:1"},
        ],
        expected_names={"A": "spk:0", "B": "spk:1"},
    )


def test_stitch_overlap_new_speaker_gets_new_label():
    # C first speaks in the second chunk, after the overlap; A and B both
    # speak in the overlap, so C must not be merged into either.
    truth = _ground_truth([("A", 0, 85), ("B", 85, 100), ("C", 100, 150)])
    _check_stitch(
        truth,
        chunk_seconds=100,
        overlap=20,
        label_maps=[
            {"A": "spk:0", "B": "spk:1"},
            {"A": "spk:2", "B": "spk:0", "C": "spk:1"},
        ],
        expected_names={"A": "spk:0", "B": "spk:1", "C": "spk:2"},
    )


def test_stitch_word_near_cut_is_not_duplicated():
    # Regression: the same word gets slightly different timestamps in two
    # chunks. With 100s chunks and a 20s overlap the old fixed cut was at
    # 90s; "w89" at 89.9s in chunk 0 and 90.05s in chunk 1 was kept twice.
    truth = _ground_truth([("A", 0.9, 150)])
    _check_stitch(
        truth,
        chunk_seconds=100,
        overlap=20,
        label_maps=[{"A": "spk:0"}, {"A": "spk:0"}],
        expected_names={"A": "spk:0"},
        jitter=0.15,
    )


def test_stitch_word_near_cut_is_not_dropped():
    # Mirror case: "w90" at 90.1s in chunk 0 and 89.95s in chunk 1 fell on
    # the wrong side of a fixed 90s cut in both chunks and was lost.
    truth = _ground_truth([("A", 0.1, 150)])
    _check_stitch(
        truth,
        chunk_seconds=100,
        overlap=20,
        label_maps=[{"A": "spk:0"}, {"A": "spk:0"}],
        expected_names={"A": "spk:0"},
        jitter=-0.15,
    )


def test_stitch_without_overlap_is_unchanged():
    a = TranscriptionResponse(
        id=0,
        status_code=200,
        is_error=False,
        error_message=None,
        text="one",
        words=[{"word": "one", "start": 1.0, "end": 1.5}],
    )
    b = TranscriptionResponse(
        id=1,
        status_code=200,
        is_error=False,
        error_message=None,
        text="two",
        words=[{"word": "two", "start": 2.0, "end": 2.5}],
    )
    result = _stitch_responses([a, b], 0, [0.0, 10.0])
    assert result.text == "one two"
    assert [w["start"] for w in result.words] == [1.0, 12.0]


# ---------------------------------------------------------------------------
# Splitting and duration (need ffmpeg)
# ---------------------------------------------------------------------------


def _make_tone(path: Path, seconds: int, extra: list[str] | None = None) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "quiet",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds}",
            "-c:a",
            "libopus",
            "-b:a",
            "32k",
            *(extra or []),
            str(path),
        ],
        check=True,
    )


def test_split_audio_with_overlap():
    if not shutil.which("ffmpeg"):
        print("skipping test_split_audio_with_overlap: no ffmpeg")
        return
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "tone.webm"
        _make_tone(src, 250)
        chunks = _split_audio(src, 100, overlap_seconds=20)
        try:
            assert [offset for _, offset in chunks] == [0.0, 80.0, 160.0]
            durations = [_get_audio_duration(p) for p, _ in chunks]
            assert all(dur is not None for dur in durations)
            assert [round(dur or 0) for dur in durations] == [100, 100, 90]
        finally:
            shutil.rmtree(chunks[0][0].parent, ignore_errors=True)


def test_duration_without_container_header():
    # Browser MediaRecorder WebM has no duration header; ffmpeg's live WebM
    # muxer produces the same shape.
    if not shutil.which("ffmpeg"):
        print("skipping test_duration_without_container_header: no ffmpeg")
        return
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "tone.webm"
        _make_tone(src, 30)
        piped = Path(d) / "piped.webm"
        with piped.open("wb") as out:
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "quiet",
                    "-i",
                    str(src),
                    "-c",
                    "copy",
                    "-f",
                    "webm",
                    "-live",
                    "1",
                    "pipe:1",
                ],
                stdout=out,
                check=True,
            )
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                str(piped),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert probe.stdout.strip() == "N/A", "fixture should lack a duration header"
        duration = _get_audio_duration(piped)
        assert duration is not None and abs(duration - 30) < 1


def _sent_chunk_durations(model: str, seconds: int, **kwargs) -> list[float] | str:
    """Run transcribe_async on in-memory audio with the API call stubbed out.

    Returns the duration of each chunk that would have been sent, or the
    error message.
    """
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "tone.webm"
        _make_tone(src, seconds)
        audio = src.read_bytes()

    sent: list[float] = []
    original = transcribe_module._transcribe_chunk

    async def fake_chunk(**call):
        with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
            f.write(call["audio_bytes"])
            f.flush()
            sent.append(round(_get_audio_duration(f.name) or 0))
        return TranscriptionResponse(
            id=call["task_id"],
            status_code=200,
            is_error=False,
            error_message=None,
            text="",
            segments=[],
            words=[],
        )

    transcribe_module._transcribe_chunk = fake_chunk
    try:
        results = asyncio.run(
            transcribe_async(
                [(audio, "meeting.webm")], model=model, show_progress=False, **kwargs
            )
        )
    finally:
        transcribe_module._transcribe_chunk = original
    if results[0].is_error:
        return results[0].error_message or "error"
    return sent


def test_bytes_source_is_split_into_overlapping_chunks():
    # Long in-memory audio (a (bytes, filename) tuple) must be split instead
    # of being sent whole past the model's duration limit. Default chunks are
    # 80% of the 900s limit (720s), overlapping by 60s: starts 0, 660, 1320.
    if not shutil.which("ffmpeg"):
        print("skipping: no ffmpeg")
        return
    assert _sent_chunk_durations(
        "gemini-3.5-transcribe-vertex", 1500, timestamps=True
    ) == [720, 720, 180]


def test_caller_chunk_size():
    # chunk_seconds=900 (Google's documented limit): starts 0, 840.
    if not shutil.which("ffmpeg"):
        print("skipping: no ffmpeg")
        return
    assert _sent_chunk_durations(
        "gemini-3.5-transcribe-vertex", 1500, timestamps=True, chunk_seconds=900
    ) == [900, 660]


def test_caller_chunk_size_above_model_limit_is_rejected():
    try:
        asyncio.run(
            transcribe_async(
                [(b"x", "a.webm")],
                model="gemini-3.5-transcribe-vertex",
                chunk_seconds=901,
                show_progress=False,
            )
        )
    except ValueError as e:
        assert "max_duration" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_overlap_applies_to_other_word_timestamp_models():
    # whisper-1 has no duration limit, so splitting comes from chunk_seconds.
    # With timestamps it returns word times, so chunks overlap by 60s
    # (starts 0, 540, 1080); without timestamps they don't (0, 600, 1200).
    if not shutil.which("ffmpeg"):
        print("skipping: no ffmpeg")
        return
    assert _sent_chunk_durations(
        "whisper-1", 1500, timestamps=True, chunk_seconds=600
    ) == [600, 600, 420]
    assert _sent_chunk_durations(
        "whisper-1", 1500, timestamps=False, chunk_seconds=600
    ) == [600, 600, 300]


def test_no_overlap_for_models_without_word_timestamps():
    # gpt-4o-transcribe returns text only, so overlapping audio couldn't be
    # de-duplicated; chunks are back to back even with timestamps=True.
    if not shutil.which("ffmpeg"):
        print("skipping: no ffmpeg")
        return
    assert _sent_chunk_durations(
        "gpt-4o-transcribe", 1500, timestamps=True, chunk_seconds=600
    ) == [600, 600, 300]


def test_overlap_stitch_rejects_chunks_without_words():
    chunk = TranscriptionResponse(
        id=1, status_code=200, is_error=False, error_message=None, text="hello"
    )
    result = _stitch_responses(
        [chunk, chunk], 0, [0.0, 80.0], chunk_seconds=100, overlap_seconds=20
    )
    assert result.is_error
    assert "word timestamps" in (result.error_message or "")


def test_overlap_stitch_deepgram_int_speakers():
    # Deepgram labels words with ints and segments with strings.
    truth = _ground_truth([("A", 0, 60), ("B", 60, 150)])
    chunks = []
    for offset, labels in ((0.0, {"A": 0, "B": 1}), (80.0, {"A": 1, "B": 0})):
        view = _chunk_view(
            truth, offset, 100, {k: str(v) for k, v in labels.items()}, jitter=0.0
        )
        for w in view.words:
            w["speaker"] = int(w["speaker"])
        chunks.append(view)
    result = _stitch_responses(
        chunks, 0, [0.0, 80.0], chunk_seconds=100, overlap_seconds=20
    )
    assert not result.is_error, result.error_message
    assert [s.speaker for s in result.segments] == ["spk:0", "spk:1"]
    assert result.text == " ".join(w for w, _, _ in truth)


if __name__ == "__main__":
    test_registry_entry()
    test_parse_duration_string()
    test_parse_vertex_diarized_response()
    test_parse_vertex_plain_response()
    test_parse_vertex_truncated_response_is_error()
    test_parse_vertex_no_candidates_is_error()
    test_build_vertex_body()
    test_stitch_overlap_single_speaker_in_overlap_uses_elimination()
    test_stitch_overlap_three_chunks_both_speakers_in_overlap()
    test_stitch_overlap_new_speaker_gets_new_label()
    test_stitch_word_near_cut_is_not_duplicated()
    test_stitch_word_near_cut_is_not_dropped()
    test_stitch_without_overlap_is_unchanged()
    test_split_audio_with_overlap()
    test_duration_without_container_header()
    test_bytes_source_is_split_into_overlapping_chunks()
    test_caller_chunk_size()
    test_caller_chunk_size_above_model_limit_is_rejected()
    test_overlap_applies_to_other_word_timestamp_models()
    test_no_overlap_for_models_without_word_timestamps()
    test_overlap_stitch_rejects_chunks_without_words()
    test_overlap_stitch_deepgram_int_speakers()
    print("All tests passed!")
