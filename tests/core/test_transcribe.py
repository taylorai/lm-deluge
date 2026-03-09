"""Basic tests for the transcribe module."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from lm_deluge.transcribe import (
    REGISTRY,
    TranscriptionResponse,
    TranscriptionSegment,
    _guess_mime_type,
    _parse_response,
    _parse_segments,
    _read_audio,
)


def test_registry_has_expected_models():
    expected = [
        "whisper-1",
        "gpt-4o-transcribe",
        "gpt-4o-mini-transcribe",
        "voxtral-mini-latest",
        "whisper-v3",
        "whisper-v3-turbo",
        "nova-3",
        "nova-2",
    ]
    for model in expected:
        assert model in REGISTRY, f"Missing model: {model}"
        info = REGISTRY[model]
        assert "provider" in info
        assert "api_base" in info
        assert "api_key_env_var" in info
        assert "cost_per_minute" in info


def test_guess_mime_type():
    assert _guess_mime_type("audio.mp3") == "audio/mpeg"
    assert _guess_mime_type("audio.wav") == "audio/wav"
    assert _guess_mime_type("audio.flac") == "audio/flac"
    assert _guess_mime_type("audio.m4a") == "audio/mp4"
    assert _guess_mime_type("audio.ogg") == "audio/ogg"
    assert _guess_mime_type("audio.webm") == "audio/webm"


def test_read_audio_bytes():
    raw = b"fake audio data"
    data, mime, name = _read_audio(raw)
    assert data == raw
    assert mime == "audio/wav"
    assert name == "audio.wav"


def test_read_audio_file():
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(b"fake mp3 data")
        f.flush()
        data, mime, name = _read_audio(f.name)
        assert data == b"fake mp3 data"
        assert mime == "audio/mpeg"
        assert name.endswith(".mp3")


def test_read_audio_missing_file():
    try:
        _read_audio("/nonexistent/audio.wav")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass


def test_parse_segments():
    raw = [
        {"text": "hello", "start": 0.0, "end": 1.0},
        {"text": "world", "start": 1.0, "end": 2.0, "speaker": "A"},
    ]
    segments = _parse_segments(raw)
    assert len(segments) == 2
    assert segments[0].text == "hello"
    assert segments[0].speaker is None
    assert segments[1].speaker == "A"


def test_parse_response():
    result = {
        "text": "hello world",
        "language": "en",
        "duration": 2.5,
        "segments": [{"text": "hello world", "start": 0.0, "end": 2.5}],
        "words": [{"word": "hello", "start": 0.0, "end": 1.0}],
    }
    text, lang, dur, segs, words = _parse_response("openai", result)
    assert text == "hello world"
    assert lang == "en"
    assert dur == 2.5
    assert len(segs) == 1
    assert len(words) == 1


def test_transcription_response_dataclass():
    r = TranscriptionResponse(
        id=0,
        status_code=200,
        is_error=False,
        error_message=None,
        text="test",
        language="en",
        duration=1.0,
        segments=[TranscriptionSegment(text="test", start=0.0, end=1.0)],
    )
    assert r.text == "test"
    assert len(r.segments) == 1
    assert r.segments[0].start == 0.0


def test_parse_deepgram_response():
    result = {
        "metadata": {
            "request_id": "abc123",
            "duration": 25.93,
            "channels": 1,
        },
        "results": {
            "channels": [
                {
                    "detected_language": "en",
                    "alternatives": [
                        {
                            "transcript": "Yeah as much as it's worth celebrating.",
                            "confidence": 0.999,
                            "words": [
                                {
                                    "word": "yeah",
                                    "start": 0.08,
                                    "end": 0.32,
                                    "confidence": 0.99,
                                    "punctuated_word": "Yeah.",
                                },
                                {
                                    "word": "as",
                                    "start": 0.32,
                                    "end": 0.8,
                                    "confidence": 0.99,
                                    "punctuated_word": "As",
                                },
                            ],
                            "paragraphs": {
                                "transcript": "Yeah. As much as...",
                                "paragraphs": [
                                    {
                                        "speaker": 0,
                                        "num_words": 8,
                                        "start": 0.08,
                                        "end": 5.0,
                                        "sentences": [
                                            {
                                                "text": "Yeah.",
                                                "start": 0.08,
                                                "end": 0.32,
                                            },
                                            {
                                                "text": "As much as it's worth celebrating.",
                                                "start": 0.32,
                                                "end": 5.0,
                                            },
                                        ],
                                    }
                                ],
                            },
                        }
                    ],
                }
            ]
        },
    }
    text, lang, dur, segs, words = _parse_response("deepgram", result)
    assert text == "Yeah as much as it's worth celebrating."
    assert lang == "en"
    assert dur == 25.93
    assert len(words) == 2
    assert words[0]["word"] == "yeah"
    assert len(segs) == 2
    assert segs[0].text == "Yeah."
    assert segs[0].speaker == "0"
    assert segs[1].start == 0.32


def test_parse_deepgram_empty_response():
    result = {"metadata": {"duration": 0.0}, "results": {"channels": []}}
    text, lang, dur, segs, words = _parse_response("deepgram", result)
    assert text == ""
    assert lang is None
    assert dur == 0.0
    assert segs == []
    assert words == []


if __name__ == "__main__":
    test_registry_has_expected_models()
    test_guess_mime_type()
    test_read_audio_bytes()
    test_read_audio_file()
    test_read_audio_missing_file()
    test_parse_segments()
    test_parse_response()
    test_parse_deepgram_response()
    test_parse_deepgram_empty_response()
    test_transcription_response_dataclass()
    print("All tests passed!")
