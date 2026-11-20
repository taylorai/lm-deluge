"""Parallel audio transcription API for OpenAI, Mistral, Fireworks, Deepgram, and
Gemini on Vertex AI (Gemini Enterprise Agent Platform)."""

import asyncio
import base64
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp
import filetype
from tqdm.auto import tqdm

from .api_requests.base import parse_retry_after
from .api_requests.gemini import vertex_generate_content_url
from .tracker import StatusTracker

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, dict[str, Any]] = {
    # OpenAI
    "whisper-1": {
        "provider": "openai",
        "api_base": "https://api.openai.com/v1",
        "api_key_env_var": "OPENAI_API_KEY",
        "cost_per_minute": 0.006,
        "max_duration": None,  # no hard duration limit
        "max_file_size": 25 * 1024 * 1024,  # 25 MB
        "word_timestamps": True,
    },
    "gpt-4o-transcribe": {
        "provider": "openai",
        "api_base": "https://api.openai.com/v1",
        "api_key_env_var": "OPENAI_API_KEY",
        "cost_per_minute": 0.006,
        "max_duration": 1500,  # 25 minutes
        "max_file_size": 25 * 1024 * 1024,
    },
    "gpt-4o-mini-transcribe": {
        "provider": "openai",
        "api_base": "https://api.openai.com/v1",
        "api_key_env_var": "OPENAI_API_KEY",
        "cost_per_minute": 0.003,
        "max_duration": 1500,
        "max_file_size": 25 * 1024 * 1024,
    },
    # Mistral
    "voxtral-mini-latest": {
        "provider": "mistral",
        "api_base": "https://api.mistral.ai/v1",
        "api_key_env_var": "MISTRAL_API_KEY",
        "cost_per_minute": 0.003,
        "max_duration": 10800,  # 3 hours
        "max_file_size": None,
    },
    # Fireworks (OpenAI-compatible)
    "whisper-v3": {
        "provider": "fireworks",
        "api_base": "https://audio-prod.api.fireworks.ai/v1",
        "api_key_env_var": "FIREWORKS_API_KEY",
        "cost_per_minute": 0.004,
        "max_duration": None,
        "max_file_size": 1024 * 1024 * 1024,  # 1 GB
        "word_timestamps": True,
    },
    "whisper-v3-turbo": {
        "provider": "fireworks",
        "api_base": "https://audio-turbo.api.fireworks.ai/v1",
        "api_key_env_var": "FIREWORKS_API_KEY",
        "cost_per_minute": 0.002,
        "max_duration": None,
        "max_file_size": 1024 * 1024 * 1024,
        "word_timestamps": True,
    },
    # Deepgram
    "nova-3": {
        "provider": "deepgram",
        "api_base": "https://api.deepgram.com/v1",
        "api_key_env_var": "DEEPGRAM_API_KEY",
        "cost_per_minute": 0.0043,
        "max_duration": None,
        "max_file_size": 2 * 1024 * 1024 * 1024,  # 2 GB
        "word_timestamps": True,
    },
    "nova-2": {
        "provider": "deepgram",
        "api_base": "https://api.deepgram.com/v1",
        "api_key_env_var": "DEEPGRAM_API_KEY",
        "cost_per_minute": 0.0043,
        "max_duration": None,
        "max_file_size": 2 * 1024 * 1024 * 1024,
        "word_timestamps": True,
    },
    # Gemini on Vertex AI. Uses generateContent with inline audio; the URL
    # follows VERTEX_LOCATION (see vertex_generate_content_url), so api_base
    # is informational only.
    "gemini-3.5-transcribe-vertex": {
        "provider": "vertex",
        "model_id": "gemini-3.5-transcribe",
        "api_base": "https://aiplatform.googleapis.com/v1beta1",
        "api_key_env_var": "VERTEX_API_KEY",
        # No published per-minute price; ~1,640 audio tokens/min at $2/M input
        # plus transcript output tokens.
        "cost_per_minute": 0.006,
        # Google documents a 15-minute limit when word timestamps or
        # diarization are enabled. Longer files are split with ffmpeg.
        "max_duration": 900,
        # Inline request bodies over ~20 MB are rejected; base64 adds 33%.
        "max_file_size": 14 * 1024 * 1024,
        "word_timestamps": True,
    },
}

# Models marked "word_timestamps" return per-word times when timestamps=True.
# Long audio for those models is split into chunks that overlap by this much,
# so a word at a cut isn't lost and diarization labels can be matched across
# chunks (see _stitch_overlapping). Capped at a quarter of the chunk length.
DEFAULT_CHUNK_OVERLAP_SECONDS = 60.0

# Supported audio MIME types
AUDIO_EXTENSIONS: dict[str, str] = {
    ".flac": "audio/flac",
    ".mp3": "audio/mpeg",
    ".mp4": "audio/mp4",
    ".mpeg": "audio/mpeg",
    ".mpga": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".wav": "audio/wav",
    ".webm": "audio/webm",
}

# An audio source can be a file path, raw bytes (defaults to audio/wav),
# or a (bytes, filename) tuple to preserve the original format.
AudioSource = str | bytes | Path | tuple[bytes, str]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TranscriptionSegment:
    """A segment of transcribed audio with timing information."""

    text: str
    start: float
    end: float
    speaker: str | None = None


@dataclass
class TranscriptionResponse:
    """Result of a single transcription request."""

    id: int
    status_code: int | None
    is_error: bool
    error_message: str | None
    text: str
    language: str | None = None
    duration: float | None = None
    segments: list[TranscriptionSegment] = field(default_factory=list)
    words: list[dict] = field(default_factory=list)


@dataclass
class _CostTracker:
    """Tracks cost based on audio duration."""

    cost_per_minute: float
    total_duration: float = 0.0
    total_cost: float = 0.0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def record(self, duration_seconds: float):
        async with self._lock:
            self.total_duration += duration_seconds
            self.total_cost += (duration_seconds / 60.0) * self.cost_per_minute

    def summary(self) -> str:
        parts = []
        if self.total_cost > 0:
            parts.append(f"${self.total_cost:.4f}")
        if self.total_duration > 0:
            parts.append(f"{self.total_duration:.1f}s audio")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_model_info(model: str) -> dict[str, Any]:
    if model not in REGISTRY:
        raise ValueError(
            f"Unknown transcription model '{model}'. "
            f"Available: {', '.join(REGISTRY.keys())}"
        )
    return REGISTRY[model]


def _guess_mime_type(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in AUDIO_EXTENSIONS:
        return AUDIO_EXTENSIONS[ext]
    mime, _ = mimetypes.guess_type(path)
    return mime or "application/octet-stream"


def _read_audio(source: AudioSource) -> tuple[bytes, str, str]:
    """Read audio from a file path, raw bytes, or (bytes, filename) tuple.

    Returns (audio_bytes, mime_type, filename).
    """
    if isinstance(source, tuple):
        audio_bytes, filename = source
        mime_type = _guess_mime_type(filename)
        return audio_bytes, mime_type, filename
    if isinstance(source, bytes):
        kind = filetype.guess(source)
        if kind:
            return source, kind.mime, f"audio.{kind.extension}"
        return source, "application/octet-stream", "audio"
    source = str(source)
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {source}")
    audio_bytes = path.read_bytes()
    mime_type = _guess_mime_type(source)
    filename = path.name
    return audio_bytes, mime_type, filename


# ---------------------------------------------------------------------------
# Audio duration probing and splitting (requires ffmpeg)
# ---------------------------------------------------------------------------


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _has_ffprobe() -> bool:
    return shutil.which("ffprobe") is not None


def _get_audio_duration(path: str | Path) -> float | None:
    """Get audio duration in seconds using ffprobe. Returns None if unavailable."""
    if not _has_ffprobe():
        return None
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError):
        pass
    return _decode_audio_duration(path)


_FFMPEG_TIME_RE = re.compile(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)")


def _decode_audio_duration(path: str | Path) -> float | None:
    """Measure duration by decoding the whole file.

    Fallback for containers without a duration header, e.g. WebM written by
    a browser MediaRecorder, where ffprobe reports "N/A".
    """
    if not _has_ffmpeg():
        return None
    try:
        result = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "quiet", "-stats", "-i", str(path)]
            + ["-vn", "-f", "null", "-"],
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return None
    matches = _FFMPEG_TIME_RE.findall(result.stderr)
    if result.returncode != 0 or not matches:
        return None
    hours, minutes, seconds = matches[-1]
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _split_audio(
    path: str | Path,
    max_chunk_seconds: float,
    overlap_seconds: float = 0.0,
    duration: float | None = None,
) -> list[tuple[Path, float]]:
    """Split an audio file into chunks using ffmpeg.

    Each chunk is up to max_chunk_seconds long and starts overlap_seconds
    before the previous chunk ends. Output is mono mp3 for small, universally
    accepted chunks.

    Returns (temp file path, start offset in seconds) pairs. Caller is
    responsible for cleanup.
    """
    path = Path(path)
    if duration is None:
        duration = _get_audio_duration(path)
    if duration is None:
        raise RuntimeError(
            "Could not determine audio duration. Install ffmpeg/ffprobe to "
            "enable automatic audio splitting."
        )

    if duration <= max_chunk_seconds:
        return [(path, 0.0)]
    if not 0 <= overlap_seconds < max_chunk_seconds / 2:
        raise ValueError("overlap_seconds must be less than half the chunk length")

    tmp_dir = Path(tempfile.mkdtemp(prefix="lm_deluge_audio_"))
    chunks: list[tuple[Path, float]] = []
    start = 0.0
    chunk_idx = 0
    step = max_chunk_seconds - overlap_seconds

    while start < duration:
        chunk_path = tmp_dir / f"chunk_{chunk_idx:04d}.mp3"
        cmd = [
            "ffmpeg",
            "-v",
            "quiet",
            "-y",
            "-ss",
            str(start),
            "-t",
            str(max_chunk_seconds),
            "-i",
            str(path),
            "-vn",
            "-ac",
            "1",
            "-acodec",
            "libmp3lame",
            "-b:a",
            "64k",
            str(chunk_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            # Clean up on failure
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise RuntimeError(
                f"ffmpeg failed to split audio at {start}s: {result.stderr[:500]}"
            )
        chunks.append((chunk_path, start))
        if start + max_chunk_seconds >= duration:
            break
        chunk_idx += 1
        start += step

    return chunks


def _stitch_responses(
    chunk_responses: list[TranscriptionResponse],
    original_id: int,
    chunk_offsets: list[float],
    known_duration: float | None = None,
    known_language: str | None = None,
    chunk_seconds: float | None = None,
    overlap_seconds: float = 0.0,
) -> TranscriptionResponse:
    """Stitch chunk transcription responses back into a single response.

    Adjusts segment/word timestamps by adding chunk offsets and concatenates text.
    known_duration/known_language are used as fallbacks when the API doesn't
    return them (e.g. gpt-4o-transcribe models only return {"text": "..."}).

    With overlap_seconds > 0 (chunks overlap and carry word timestamps), the
    overlap is de-duplicated at its midpoint and diarization labels are
    mapped onto one consistent set of speakers; see _stitch_overlapping.
    """
    errors = [r for r in chunk_responses if r.is_error]
    if errors:
        error_msgs = [f"Chunk {r.id}: {r.error_message}" for r in errors]
        return TranscriptionResponse(
            id=original_id,
            status_code=errors[0].status_code,
            is_error=True,
            error_message=f"{len(errors)}/{len(chunk_responses)} chunks failed:\n"
            + "\n".join(error_msgs),
            text="",
        )

    if overlap_seconds > 0:
        assert chunk_seconds is not None, "chunk_seconds is required with overlap"
        return _stitch_overlapping(
            chunk_responses,
            original_id,
            chunk_offsets,
            chunk_seconds,
            known_duration=known_duration,
            known_language=known_language,
        )

    all_text_parts: list[str] = []
    all_segments: list[TranscriptionSegment] = []
    all_words: list[dict] = []
    total_duration = 0.0
    language = known_language

    for resp, offset in zip(chunk_responses, chunk_offsets):
        all_text_parts.append(resp.text.strip())

        if resp.language and not language:
            language = resp.language

        if resp.duration:
            # The last chunk's offset + its duration gives total
            total_duration = max(total_duration, offset + resp.duration)

        for seg in resp.segments:
            all_segments.append(
                TranscriptionSegment(
                    text=seg.text,
                    start=seg.start + offset,
                    end=seg.end + offset,
                    speaker=seg.speaker,
                )
            )

        for word in resp.words:
            adjusted = dict(word)
            if "start" in adjusted:
                adjusted["start"] = adjusted["start"] + offset
            if "end" in adjusted:
                adjusted["end"] = adjusted["end"] + offset
            all_words.append(adjusted)

    final_duration = total_duration if total_duration > 0 else known_duration

    return TranscriptionResponse(
        id=original_id,
        status_code=200,
        is_error=False,
        error_message=None,
        text=" ".join(all_text_parts),
        language=language,
        duration=final_duration,
        segments=all_segments,
        words=all_words,
    )


# Words in two overlapping chunks count as the same utterance when their text
# matches and their absolute start times are within this many seconds.
_WORD_MATCH_TOLERANCE = 0.5
# Minimum matched words before two chunks' speaker labels are linked.
_MIN_SPEAKER_VOTES = 2


def _normalize_word(word: str) -> str:
    return re.sub(r"[^\w']", "", word.lower())


def _shift_words(words: list[dict], offset: float) -> list[dict]:
    shifted = []
    for word in words:
        adjusted = dict(word)
        adjusted["start"] = adjusted.get("start", 0.0) + offset
        adjusted["end"] = adjusted.get("end", 0.0) + offset
        # Deepgram labels words with ints but segments with strings.
        if adjusted.get("speaker") is not None:
            adjusted["speaker"] = str(adjusted["speaker"])
        shifted.append(adjusted)
    return shifted


def _match_words(
    prev_words: list[dict],
    cur_words: list[dict],
    overlap_start: float,
    overlap_end: float,
) -> list[tuple[int, int]]:
    """Find words transcribed in both of two overlapping chunks.

    Returns (prev index, cur index) pairs: same normalized text, absolute
    start times within _WORD_MATCH_TOLERANCE, nearest candidate wins.
    """
    lo = overlap_start - _WORD_MATCH_TOLERANCE
    hi = overlap_end + _WORD_MATCH_TOLERANCE
    prev_in = [
        (j, _normalize_word(w.get("word", "")))
        for j, w in enumerate(prev_words)
        if lo <= w["start"] <= hi
    ]
    pairs: list[tuple[int, int]] = []
    for k, cur in enumerate(cur_words):
        if not lo <= cur["start"] <= hi:
            continue
        text = _normalize_word(cur.get("word", ""))
        if not text:
            continue
        best: int | None = None
        best_delta = _WORD_MATCH_TOLERANCE
        for j, prev_text in prev_in:
            if prev_text != text:
                continue
            delta = abs(prev_words[j]["start"] - cur["start"])
            if delta <= best_delta:
                best, best_delta = j, delta
        if best is not None:
            pairs.append((best, k))
    return pairs


def _cut_points(
    prev_words: list[dict],
    cur_words: list[dict],
    pairs: list[tuple[int, int]],
    midpoint: float,
) -> tuple[float, float]:
    """Choose where to switch from the previous chunk to the current one.

    Cuts at a word found in both chunks, nearest the overlap midpoint,
    preferring one whose following word also matches. The previous chunk
    keeps words starting before that word's time in its own timeline; the
    current chunk keeps the word and everything after. Per-chunk timestamps
    of the same word differ slightly, so cutting both chunks at one fixed
    time can keep a word twice or drop it.
    """
    pair_set = set(pairs)
    anchors = [(j, k) for j, k in pairs if (j + 1, k + 1) in pair_set]
    candidates = anchors or pairs
    if not candidates:
        return midpoint, midpoint
    j, k = min(candidates, key=lambda p: abs(prev_words[p[0]]["start"] - midpoint))
    return prev_words[j]["start"], cur_words[k]["start"]


def _match_speakers(
    prev_words: list[dict],
    prev_mapping: dict[str, str],
    cur_words: list[dict],
    pairs: list[tuple[int, int]],
) -> dict[str, str]:
    """Map the current chunk's local speaker labels onto global labels.

    Each word found in both chunks (see _match_words) is a vote that its
    current label and the previous chunk's (already global) label are the
    same person; labels are paired greedily by vote count, one-to-one. If
    exactly one label is left unpaired on each side, they are paired by
    elimination (the usual two-person case where only one person spoke
    during the overlap).
    """
    votes: Counter[tuple[str, str]] = Counter()
    for j, k in pairs:
        prev_label = prev_words[j].get("speaker")
        cur_label = cur_words[k].get("speaker")
        if prev_label is not None and cur_label is not None:
            votes[(cur_label, prev_mapping[prev_label])] += 1

    mapping: dict[str, str] = {}
    used: set[str] = set()
    for (cur_label, global_label), count in votes.most_common():
        if count < _MIN_SPEAKER_VOTES:
            break
        if cur_label in mapping or global_label in used:
            continue
        mapping[cur_label] = global_label
        used.add(global_label)

    cur_labels = {w["speaker"] for w in cur_words if w.get("speaker") is not None}
    prev_globals = {
        prev_mapping[w["speaker"]] for w in prev_words if w.get("speaker") is not None
    }
    unmatched_cur = cur_labels - mapping.keys()
    unmatched_prev = prev_globals - used
    if len(unmatched_cur) == 1 and len(unmatched_prev) == 1:
        mapping[unmatched_cur.pop()] = unmatched_prev.pop()
    return mapping


def _segment_words(
    seg: TranscriptionSegment, words: list[dict], assigned: set[int]
) -> list[dict]:
    """Words inside a segment's time span with its speaker, each used once."""
    found = [
        w
        for w in words
        if id(w) not in assigned
        and seg.start <= w["start"] <= seg.end
        and w.get("speaker") == (None if seg.speaker is None else str(seg.speaker))
    ]
    assigned.update(id(w) for w in found)
    return found


def _stitch_overlapping(
    chunk_responses: list[TranscriptionResponse],
    original_id: int,
    chunk_offsets: list[float],
    chunk_seconds: float,
    known_duration: float | None = None,
    known_language: str | None = None,
) -> TranscriptionResponse:
    """Stitch overlapping chunks that carry word timestamps.

    Chunk i keeps the words between its cut points with chunk i-1 and chunk
    i+1 (see _cut_points). Segments are trimmed to the kept
    words, and a segment split by a cut is re-joined when the same speaker
    continues across it. Speaker labels become "spk:0", "spk:1", ... in order
    of first appearance.
    """
    for resp in chunk_responses:
        if resp.text.strip() and not resp.words:
            # Without word times the overlap can't be de-duplicated.
            return TranscriptionResponse(
                id=original_id,
                status_code=None,
                is_error=True,
                error_message=(
                    f"Chunk {resp.id} returned text without word timestamps; "
                    "overlapping chunks can't be stitched"
                ),
                text="",
            )

    language = known_language
    next_label = 0
    prev_mapping: dict[str, str] = {}
    segments: list[TranscriptionSegment] = []
    words: list[dict] = []

    shifted = [
        _shift_words(resp.words, offset)
        for resp, offset in zip(chunk_responses, chunk_offsets)
    ]
    # boundary_pairs[i] / cuts[i] describe the overlap between chunk i-1 and i.
    boundary_pairs: list[list[tuple[int, int]]] = [[]]
    cuts: list[tuple[float, float]] = [(float("-inf"), float("-inf"))]
    for i in range(1, len(shifted)):
        overlap_start = chunk_offsets[i]
        overlap_end = chunk_offsets[i - 1] + chunk_seconds
        pairs = _match_words(shifted[i - 1], shifted[i], overlap_start, overlap_end)
        boundary_pairs.append(pairs)
        cuts.append(
            _cut_points(
                shifted[i - 1],
                shifted[i],
                pairs,
                (overlap_start + overlap_end) / 2,
            )
        )

    for i, (resp, offset) in enumerate(zip(chunk_responses, chunk_offsets)):
        if resp.language and not language:
            language = resp.language
        chunk_words = shifted[i]

        mapping: dict[str, str] = {}
        if i > 0:
            mapping = _match_speakers(
                shifted[i - 1], prev_mapping, chunk_words, boundary_pairs[i]
            )
        for w in chunk_words:
            label = w.get("speaker")
            if label is not None and label not in mapping:
                mapping[label] = f"spk:{next_label}"
                next_label += 1

        lo = cuts[i][1]
        hi = cuts[i + 1][0] if i + 1 < len(cuts) else float("inf")

        assigned: set[int] = set()
        for seg in resp.segments:
            abs_seg = TranscriptionSegment(
                text=seg.text,
                start=seg.start + offset,
                end=seg.end + offset,
                speaker=seg.speaker,
            )
            seg_words = _segment_words(abs_seg, chunk_words, assigned)
            kept = [w for w in seg_words if lo <= w["start"] < hi]
            if not kept:
                continue
            speaker = mapping.get(str(seg.speaker)) if seg.speaker is not None else None
            text = (
                seg.text
                if len(kept) == len(seg_words)
                else " ".join(w.get("word", "") for w in kept)
            )
            if (
                segments
                and segments[-1].speaker == speaker
                and len(kept) < len(seg_words)
                and kept[0]["start"] - segments[-1].end < 1.0
            ):
                prev = segments[-1]
                segments[-1] = TranscriptionSegment(
                    text=f"{prev.text} {text}",
                    start=prev.start,
                    end=kept[-1]["end"],
                    speaker=speaker,
                )
            else:
                segments.append(
                    TranscriptionSegment(
                        text=text,
                        start=kept[0]["start"],
                        end=kept[-1]["end"],
                        speaker=speaker,
                    )
                )

        for w in chunk_words:
            if lo <= w["start"] < hi:
                kept_word = dict(w)
                if w.get("speaker") is not None:
                    kept_word["speaker"] = mapping[w["speaker"]]
                words.append(kept_word)

        prev_mapping = mapping

    text = (
        " ".join(seg.text for seg in segments)
        if segments
        else " ".join(w.get("word", "") for w in words)
    )
    return TranscriptionResponse(
        id=original_id,
        status_code=200,
        is_error=False,
        error_message=None,
        text=text,
        language=language,
        duration=known_duration,
        segments=segments,
        words=words,
    )


# ---------------------------------------------------------------------------
# Request building (per provider)
# ---------------------------------------------------------------------------


def _build_openai_form(
    model: str,
    audio_bytes: bytes,
    mime_type: str,
    filename: str,
    language: str | None,
    timestamps: bool,
    extra_params: dict[str, Any],
) -> aiohttp.FormData:
    """Build multipart form data for OpenAI-compatible transcription endpoints."""
    form = aiohttp.FormData()
    form.add_field("file", audio_bytes, filename=filename, content_type=mime_type)
    form.add_field("model", model)

    if language:
        form.add_field("language", language)

    # whisper-1 supports verbose_json which returns duration, language, segments,
    # and words. Always use it for whisper to get full metadata.
    # gpt-4o-transcribe models only support "json" and "text".
    is_whisper = model.startswith("whisper")
    if is_whisper:
        form.add_field("response_format", "verbose_json")
        if timestamps:
            form.add_field("timestamp_granularities[]", "segment")
            form.add_field("timestamp_granularities[]", "word")
    else:
        form.add_field("response_format", "json")

    for key, value in extra_params.items():
        form.add_field(key, str(value))

    return form


def _build_mistral_form(
    model: str,
    audio_bytes: bytes,
    mime_type: str,
    filename: str,
    language: str | None,
    timestamps: bool,
    extra_params: dict[str, Any],
) -> aiohttp.FormData:
    """Build multipart form data for Mistral transcription endpoint."""
    form = aiohttp.FormData()
    form.add_field("file", audio_bytes, filename=filename, content_type=mime_type)
    form.add_field("model", model)

    if language:
        form.add_field("language", language)

    if timestamps:
        form.add_field("timestamp_granularities[]", "segment")
        form.add_field("timestamp_granularities[]", "word")

    for key, value in extra_params.items():
        form.add_field(key, str(value))

    return form


def _build_fireworks_form(
    model: str,
    audio_bytes: bytes,
    mime_type: str,
    filename: str,
    language: str | None,
    timestamps: bool,
    extra_params: dict[str, Any],
) -> aiohttp.FormData:
    """Build multipart form data for Fireworks transcription endpoint."""
    form = aiohttp.FormData()
    form.add_field("file", audio_bytes, filename=filename, content_type=mime_type)
    form.add_field("model", model)

    if language:
        form.add_field("language", language)

    # Always use verbose_json for whisper models to get duration/language
    form.add_field("response_format", "verbose_json")
    if timestamps:
        form.add_field("timestamp_granularities[]", "segment")
        form.add_field("timestamp_granularities[]", "word")

    for key, value in extra_params.items():
        form.add_field(key, str(value))

    return form


def _build_vertex_body(
    audio_bytes: bytes,
    mime_type: str,
    language: str | None,
    timestamps: bool,
    extra_params: dict[str, Any],
) -> dict[str, Any]:
    """Build a generateContent body for a Gemini transcription model.

    extra_params are merged into audioTranscriptionConfig, using the REST
    field names (e.g. diarization=True, customVocabulary=[...]).
    """
    config: dict[str, Any] = {}
    if language:
        config["languageCodes"] = [language]
    if timestamps:
        config["wordTimestamp"] = True
    config.update(extra_params)
    body: dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": base64.b64encode(audio_bytes).decode("ascii"),
                        }
                    }
                ],
            }
        ]
    }
    if config:
        body["generationConfig"] = {"audioTranscriptionConfig": config}
    return body


_FORM_BUILDERS = {
    "openai": _build_openai_form,
    "mistral": _build_mistral_form,
    "fireworks": _build_fireworks_form,
}


# ---------------------------------------------------------------------------
# Response parsing (per provider)
# ---------------------------------------------------------------------------


def _parse_segments(raw_segments: list[dict]) -> list[TranscriptionSegment]:
    segments = []
    for seg in raw_segments:
        segments.append(
            TranscriptionSegment(
                text=seg.get("text", ""),
                start=seg.get("start", 0.0),
                end=seg.get("end", 0.0),
                speaker=seg.get("speaker") or seg.get("speaker_id"),
            )
        )
    return segments


class _TranscriptionResponseError(Exception):
    """A 200 response that is unusable and should not be retried."""


def _parse_duration_string(value: Any) -> float:
    """Parse a protobuf Duration as JSON ("1.600s") or a plain number."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value.removesuffix("s") or 0)
    return 0.0


def _parse_vertex_response(
    result: dict,
) -> tuple[str, str | None, float | None, list[TranscriptionSegment], list[dict]]:
    """Parse a Gemini transcription generateContent response.

    Each content part is one transcribed passage; with diarization, one part
    per speaker turn, labeled by audioTranscription.speakerLabel ("spk:0").
    """
    candidates = result.get("candidates") or []
    if not candidates:
        feedback = result.get("promptFeedback")
        raise _TranscriptionResponseError(f"No candidates in response: {feedback}")
    candidate = candidates[0]
    finish_reason = candidate.get("finishReason")
    if finish_reason and finish_reason != "STOP":
        raise _TranscriptionResponseError(
            f"Transcription stopped early: finishReason={finish_reason}"
        )

    texts: list[str] = []
    segments: list[TranscriptionSegment] = []
    words: list[dict] = []
    for part in (candidate.get("content") or {}).get("parts") or []:
        transcription = part.get("audioTranscription") or {}
        text = (transcription.get("text") or part.get("text") or "").strip()
        if not text:
            continue
        texts.append(text)
        speaker = transcription.get("speakerLabel")
        part_words: list[dict] = []
        for raw in transcription.get("words") or []:
            word: dict[str, Any] = {
                "word": raw.get("word", ""),
                "start": _parse_duration_string(raw.get("startOffset")),
                "end": _parse_duration_string(raw.get("endOffset")),
            }
            if speaker is not None:
                word["speaker"] = speaker
            part_words.append(word)
        words.extend(part_words)
        if part_words:
            segments.append(
                TranscriptionSegment(
                    text=text,
                    start=part_words[0]["start"],
                    end=part_words[-1]["end"],
                    speaker=speaker,
                )
            )
    return " ".join(texts), None, None, segments, words


def _parse_response(
    provider: str,
    result: dict,
) -> tuple[str, str | None, float | None, list[TranscriptionSegment], list[dict]]:
    """Extract transcription data from provider response.

    Returns (text, language, duration, segments, words).
    """
    if provider == "deepgram":
        return _parse_deepgram_response(result)
    if provider == "vertex":
        return _parse_vertex_response(result)

    text = result.get("text", "")
    language = result.get("language")
    duration = result.get("duration")

    segments: list[TranscriptionSegment] = []
    words: list[dict] = []

    raw_segments = result.get("segments", [])
    if raw_segments:
        segments = _parse_segments(raw_segments)

    raw_words = result.get("words", [])
    if raw_words:
        words = raw_words

    return text, language, duration, segments, words


def _parse_deepgram_response(
    result: dict,
) -> tuple[str, str | None, float | None, list[TranscriptionSegment], list[dict]]:
    """Parse Deepgram's nested response format."""
    metadata = result.get("metadata", {})
    duration = metadata.get("duration")

    channels = result.get("results", {}).get("channels", [])
    if not channels:
        return "", None, duration, [], []

    alt = channels[0].get("alternatives", [{}])[0]
    text = alt.get("transcript", "")

    # Deepgram detected language is on the channel or alternative
    language = channels[0].get("detected_language") or alt.get("detected_language")

    words: list[dict] = alt.get("words", [])

    # Build segments from paragraphs if available
    segments: list[TranscriptionSegment] = []
    paragraphs_obj = alt.get("paragraphs", {})
    raw_paragraphs = paragraphs_obj.get("paragraphs", [])
    for para in raw_paragraphs:
        for sentence in para.get("sentences", []):
            segments.append(
                TranscriptionSegment(
                    text=sentence.get("text", ""),
                    start=sentence.get("start", 0.0),
                    end=sentence.get("end", 0.0),
                    speaker=str(para["speaker"]) if "speaker" in para else None,
                )
            )

    return text, language, duration, segments, words


# ---------------------------------------------------------------------------
# Core async transcription
# ---------------------------------------------------------------------------


async def _wait_for_capacity(
    status_tracker: StatusTracker,
    capacity_lock: asyncio.Lock,
    max_requests_per_minute: int,
    retry: bool = False,
):
    """Wait until the StatusTracker has enough RPM/concurrency capacity."""
    while True:
        cooldown = status_tracker.seconds_to_pause
        if cooldown > 0:
            await asyncio.sleep(cooldown)
            continue
        async with capacity_lock:
            # Use 1 token as placeholder since transcription isn't token-based
            if status_tracker.check_capacity(1, retry=retry):
                return
        await asyncio.sleep(max(60.0 / max_requests_per_minute, 0.01))


async def _transcribe_chunk(
    task_id: int,
    audio_bytes: bytes,
    mime_type: str,
    filename: str,
    model: str,
    model_info: dict[str, Any],
    language: str | None,
    timestamps: bool,
    extra_params: dict[str, Any],
    status_tracker: StatusTracker,
    capacity_lock: asyncio.Lock,
    max_requests_per_minute: int,
    max_attempts: int,
    request_timeout: int,
    cost_tracker: _CostTracker,
    pbar: tqdm | None,
) -> TranscriptionResponse:
    """Transcribe a single audio chunk with retries and rate limiting."""
    provider = model_info["provider"]
    api_base = model_info["api_base"]
    api_key_env_var = model_info["api_key_env_var"]
    api_key = os.environ.get(api_key_env_var)
    if not api_key:
        return TranscriptionResponse(
            id=task_id,
            status_code=None,
            is_error=True,
            error_message=f"Missing API key: set {api_key_env_var} environment variable",
            text="",
        )

    is_deepgram = provider == "deepgram"
    is_vertex = provider == "vertex"

    query_params: dict[str, str] = {}
    if is_deepgram:
        # Deepgram: raw binary body, query params, Token auth
        query_params = {"model": model, "smart_format": "true"}
        if language:
            query_params["language"] = language
        if timestamps:
            query_params["utterances"] = "true"
            query_params["paragraphs"] = "true"
        for key, value in extra_params.items():
            query_params[key] = str(value)
        url = f"{api_base}/listen"
        headers = {
            "Authorization": f"Token {api_key}",
            "Content-Type": mime_type,
        }
    elif is_vertex:
        url = vertex_generate_content_url(model_info.get("model_id", model))
        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    else:
        url = f"{api_base}/audio/transcriptions"
        headers = {"Authorization": f"Bearer {api_key}"}

    build_form = _FORM_BUILDERS.get(provider)

    for attempt in range(max_attempts):
        retry = attempt > 0
        await _wait_for_capacity(
            status_tracker, capacity_lock, max_requests_per_minute, retry=retry
        )
        if retry:
            status_tracker.num_tasks_in_progress += 1

        try:
            if is_deepgram:
                request_kwargs: dict[str, Any] = {
                    "data": audio_bytes,
                    "headers": headers,
                    "params": query_params,
                }
            elif is_vertex:
                request_kwargs = {
                    "json": _build_vertex_body(
                        audio_bytes, mime_type, language, timestamps, extra_params
                    ),
                    "headers": headers,
                }
            else:
                assert build_form is not None
                form = build_form(
                    model,
                    audio_bytes,
                    mime_type,
                    filename,
                    language,
                    timestamps,
                    extra_params.copy(),
                )
                request_kwargs = {"data": form, "headers": headers}
            timeout = aiohttp.ClientTimeout(total=request_timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    url, **request_kwargs, allow_redirects=False
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        try:
                            text, lang, duration, segments, words = _parse_response(
                                provider, result
                            )
                        except _TranscriptionResponseError as e:
                            status_tracker.num_tasks_in_progress -= 1
                            status_tracker.num_tasks_failed += 1
                            return TranscriptionResponse(
                                id=task_id,
                                status_code=200,
                                is_error=True,
                                error_message=str(e),
                                text="",
                            )
                        if duration:
                            await cost_tracker.record(duration)
                        status_tracker.task_succeeded(task_id)
                        if pbar:
                            pbar.update(1)
                            pbar.set_postfix_str(cost_tracker.summary())
                        return TranscriptionResponse(
                            id=task_id,
                            status_code=200,
                            is_error=False,
                            error_message=None,
                            text=text,
                            language=lang,
                            duration=duration,
                            segments=segments,
                            words=words,
                        )
                    elif response.status == 429:
                        error_msg = await response.text()
                        retry_after = parse_retry_after(response)
                        status_tracker.rate_limit_exceeded(retry_after)
                        status_tracker.num_tasks_in_progress -= 1
                        if attempt < max_attempts - 1:
                            continue
                        status_tracker.num_tasks_failed += 1
                        return TranscriptionResponse(
                            id=task_id,
                            status_code=429,
                            is_error=True,
                            error_message=error_msg,
                            text="",
                        )
                    else:
                        error_msg = await response.text()
                        status_tracker.num_tasks_in_progress -= 1
                        if attempt < max_attempts - 1:
                            await asyncio.sleep(min(2**attempt, 16))
                            continue
                        status_tracker.num_tasks_failed += 1
                        return TranscriptionResponse(
                            id=task_id,
                            status_code=response.status,
                            is_error=True,
                            error_message=error_msg,
                            text="",
                        )
        except asyncio.TimeoutError:
            status_tracker.num_tasks_in_progress -= 1
            if attempt < max_attempts - 1:
                await asyncio.sleep(min(2**attempt, 16))
                continue
            status_tracker.num_tasks_failed += 1
            return TranscriptionResponse(
                id=task_id,
                status_code=None,
                is_error=True,
                error_message="Request timed out",
                text="",
            )
        except Exception as e:
            status_tracker.num_tasks_in_progress -= 1
            if attempt < max_attempts - 1:
                await asyncio.sleep(min(2**attempt, 16))
                continue
            status_tracker.num_tasks_failed += 1
            return TranscriptionResponse(
                id=task_id,
                status_code=None,
                is_error=True,
                error_message=f"{type(e).__name__}: {e}",
                text="",
            )

    # unreachable
    status_tracker.num_tasks_failed += 1
    return TranscriptionResponse(
        id=task_id,
        status_code=None,
        is_error=True,
        error_message="Exhausted all attempts",
        text="",
    )


async def _transcribe_file(
    task_id: int,
    source: AudioSource,
    model: str,
    model_info: dict[str, Any],
    language: str | None,
    timestamps: bool,
    extra_params: dict[str, Any],
    status_tracker: StatusTracker,
    capacity_lock: asyncio.Lock,
    max_requests_per_minute: int,
    max_attempts: int,
    request_timeout: int,
    cost_tracker: _CostTracker,
    pbar: tqdm | None,
    requested_chunk_seconds: float | None = None,
) -> TranscriptionResponse:
    """Transcribe a single audio source, auto-splitting if needed.

    Audio is split when it exceeds the model's limits or, if given,
    requested_chunk_seconds (which must not exceed the model's max_duration).
    """
    # Read audio
    try:
        audio_bytes, mime_type, filename = _read_audio(source)
    except Exception as e:
        return TranscriptionResponse(
            id=task_id,
            status_code=None,
            is_error=True,
            error_message=f"Failed to read audio: {e}",
            text="",
        )

    max_duration = model_info.get("max_duration")
    max_file_size = model_info.get("max_file_size")
    may_split = bool(max_duration or max_file_size or requested_chunk_seconds)

    # In-memory audio is written to a temp file when it may need splitting, so
    # it can be probed and split like a path.
    if (
        not isinstance(source, (str, Path))
        and may_split
        and (_has_ffprobe() or _has_ffmpeg())
    ):
        input_dir = Path(tempfile.mkdtemp(prefix="lm_deluge_input_"))
        input_path = input_dir / (Path(filename).name or "audio")
        input_path.write_bytes(audio_bytes)
        try:
            return await _transcribe_file(
                task_id=task_id,
                source=input_path,
                model=model,
                model_info=model_info,
                language=language,
                timestamps=timestamps,
                extra_params=extra_params,
                status_tracker=status_tracker,
                capacity_lock=capacity_lock,
                max_requests_per_minute=max_requests_per_minute,
                max_attempts=max_attempts,
                request_timeout=request_timeout,
                cost_tracker=cost_tracker,
                pbar=pbar,
                requested_chunk_seconds=requested_chunk_seconds,
            )
        finally:
            shutil.rmtree(input_dir, ignore_errors=True)

    # Check if splitting is needed
    needs_split = False
    source_path: Path | None = None

    if isinstance(source, (str, Path)):
        source_path = Path(source)

    # Check file size limit
    if max_file_size and len(audio_bytes) > max_file_size:
        needs_split = True

    # Check duration limit (only if we can probe and there is a limit)
    split_threshold = requested_chunk_seconds or max_duration
    audio_duration: float | None = None
    if source_path and may_split:
        audio_duration = _get_audio_duration(source_path)
        if audio_duration and split_threshold and audio_duration > split_threshold:
            needs_split = True

    if not needs_split:
        # Simple case: send directly
        result = await _transcribe_chunk(
            task_id=task_id,
            audio_bytes=audio_bytes,
            mime_type=mime_type,
            filename=filename,
            model=model,
            model_info=model_info,
            language=language,
            timestamps=timestamps,
            extra_params=extra_params,
            status_tracker=status_tracker,
            capacity_lock=capacity_lock,
            max_requests_per_minute=max_requests_per_minute,
            max_attempts=max_attempts,
            request_timeout=request_timeout,
            cost_tracker=cost_tracker,
            pbar=pbar,
        )
        # Fill in duration/language from ffprobe if the API didn't return them
        if not result.is_error:
            if result.duration is None and audio_duration:
                result.duration = audio_duration
            if result.language is None and language:
                result.language = language
        return result

    # Need to split — requires a file on disk and ffmpeg
    if not source_path:
        return TranscriptionResponse(
            id=task_id,
            status_code=None,
            is_error=True,
            error_message=(
                "Audio exceeds model limits and cannot be auto-split from raw bytes. "
                "Pass a file path instead, or use a model with higher limits."
            ),
            text="",
        )

    if not _has_ffmpeg():
        limit_info = []
        if max_duration:
            limit_info.append(f"max {max_duration}s")
        if max_file_size:
            limit_info.append(f"max {max_file_size // (1024 * 1024)}MB")
        return TranscriptionResponse(
            id=task_id,
            status_code=None,
            is_error=True,
            error_message=(
                f"Audio exceeds model limits ({', '.join(limit_info)}) and "
                f"ffmpeg is not installed. Install ffmpeg to enable automatic "
                f"audio splitting, or use a model with higher limits (e.g. whisper-1)."
            ),
            text="",
        )

    # Determine chunk size: the caller's, else 80% of max_duration to leave
    # margin, or 20 minutes if only file size is the constraint
    if requested_chunk_seconds:
        chunk_seconds = requested_chunk_seconds
    elif max_duration:
        chunk_seconds = max_duration * 0.8
    else:
        chunk_seconds = 1200.0  # 20 minutes

    # Overlap needs word timestamps to de-duplicate the overlapping audio.
    overlap_seconds = (
        min(DEFAULT_CHUNK_OVERLAP_SECONDS, chunk_seconds / 4)
        if timestamps and model_info.get("word_timestamps")
        else 0.0
    )

    # Split and transcribe chunks
    chunks: list[tuple[Path, float]] = []
    tmp_dir: Path | None = None
    try:
        chunks = _split_audio(
            source_path, chunk_seconds, overlap_seconds, duration=audio_duration
        )
        chunk_paths = [chunk_path for chunk_path, _ in chunks]
        # If split returned the original file, no temp dir was created
        if len(chunk_paths) == 1 and chunk_paths[0] == source_path:
            # Shouldn't happen since we checked needs_split, but handle it
            return await _transcribe_chunk(
                task_id=task_id,
                audio_bytes=audio_bytes,
                mime_type=mime_type,
                filename=filename,
                model=model,
                model_info=model_info,
                language=language,
                timestamps=timestamps,
                extra_params=extra_params,
                status_tracker=status_tracker,
                capacity_lock=capacity_lock,
                max_requests_per_minute=max_requests_per_minute,
                max_attempts=max_attempts,
                request_timeout=request_timeout,
                cost_tracker=cost_tracker,
                pbar=pbar,
            )

        tmp_dir = chunk_paths[0].parent
        n_chunks = len(chunk_paths)

        # Update progress bar to reflect chunk count
        if pbar:
            # Remove the 1 unit we'll never complete as a single file,
            # add n_chunks units instead
            pbar.total = (pbar.total or 0) - 1 + n_chunks
            pbar.refresh()

        chunk_offsets = [offset for _, offset in chunks]

        # Transcribe all chunks (they share the same rate limiter)
        chunk_tasks = []
        for i, chunk_path in enumerate(chunk_paths):
            chunk_bytes = chunk_path.read_bytes()
            chunk_tasks.append(
                _transcribe_chunk(
                    task_id=i,
                    audio_bytes=chunk_bytes,
                    mime_type="audio/mpeg",  # chunks are mp3
                    filename=f"chunk_{i:04d}.mp3",
                    model=model,
                    model_info=model_info,
                    language=language,
                    timestamps=timestamps,
                    extra_params=extra_params,
                    status_tracker=status_tracker,
                    capacity_lock=capacity_lock,
                    max_requests_per_minute=max_requests_per_minute,
                    max_attempts=max_attempts,
                    request_timeout=request_timeout,
                    cost_tracker=cost_tracker,
                    pbar=pbar,
                )
            )

        chunk_responses = list(await asyncio.gather(*chunk_tasks))
        return _stitch_responses(
            chunk_responses,
            task_id,
            chunk_offsets,
            known_duration=audio_duration,
            known_language=language,
            chunk_seconds=chunk_seconds,
            overlap_seconds=overlap_seconds,
        )

    except (RuntimeError, ValueError) as e:
        return TranscriptionResponse(
            id=task_id,
            status_code=None,
            is_error=True,
            error_message=str(e),
            text="",
        )
    finally:
        if tmp_dir and tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def transcribe_async(
    audio_files: list[AudioSource] | AudioSource,
    model: str = "whisper-1",
    language: str | None = None,
    timestamps: bool = False,
    max_attempts: int = 5,
    max_requests_per_minute: int = 50,
    max_concurrent_requests: int = 10,
    request_timeout: int = 300,
    show_progress: bool = True,
    chunk_seconds: float | None = None,
    **kwargs,
) -> list[TranscriptionResponse]:
    """Transcribe audio files in parallel.

    Args:
        audio_files: A single audio source or list of sources. Each source can
            be a file path (str or Path), raw bytes (assumed WAV), or a
            (bytes, filename) tuple to preserve the original format/extension.
        model: Transcription model name (see REGISTRY for options).
        language: ISO-639-1 language code to help the model.
        timestamps: Request segment/word-level timestamps (when supported).
        max_attempts: Max retries per file on failure.
        max_requests_per_minute: RPM limit for throttling.
        max_concurrent_requests: Max simultaneous API requests.
        request_timeout: Timeout per request in seconds.
        show_progress: Show a tqdm progress bar.
        chunk_seconds: Split audio longer than this into chunks of this
            length (requires ffmpeg). Defaults to 80% of the model's
            max_duration. Must not exceed max_duration. With timestamps=True
            on a model that returns word timestamps, chunks overlap by up to
            DEFAULT_CHUNK_OVERLAP_SECONDS and are stitched at a shared word.
        **kwargs: Extra parameters passed to the transcription API
            (e.g. prompt, temperature).

    Returns:
        List of TranscriptionResponse objects, one per audio file, sorted by ID.
    """
    # Normalize single input to list
    if isinstance(audio_files, (str, bytes, Path, tuple)):
        audio_files = [audio_files]

    if not audio_files:
        return []

    model_info = _get_model_info(model)
    if chunk_seconds is not None:
        if chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be positive")
        max_duration = model_info.get("max_duration")
        if max_duration and chunk_seconds > max_duration:
            raise ValueError(
                f"chunk_seconds={chunk_seconds} exceeds {model}'s "
                f"max_duration of {max_duration}s"
            )
    cost_tracker = _CostTracker(cost_per_minute=model_info["cost_per_minute"])

    status_tracker = StatusTracker(
        max_requests_per_minute=max_requests_per_minute,
        max_tokens_per_minute=10_000_000,  # not relevant for audio
        max_concurrent_requests=max_concurrent_requests,
        use_progress_bar=False,
    )
    capacity_lock = asyncio.Lock()

    pbar = (
        tqdm(total=len(audio_files), desc=f"Transcribing [{model}]")
        if show_progress
        else None
    )

    tasks = [
        _transcribe_file(
            task_id=i,
            source=source,
            model=model,
            model_info=model_info,
            language=language,
            timestamps=timestamps,
            extra_params=kwargs,
            status_tracker=status_tracker,
            capacity_lock=capacity_lock,
            max_requests_per_minute=max_requests_per_minute,
            max_attempts=max_attempts,
            request_timeout=request_timeout,
            cost_tracker=cost_tracker,
            pbar=pbar,
            requested_chunk_seconds=chunk_seconds,
        )
        for i, source in enumerate(audio_files)
    ]

    results = await asyncio.gather(*tasks)

    if pbar:
        pbar.close()

    results = sorted(results, key=lambda r: r.id)

    # Final summary
    parts = [f"Transcribed {len(audio_files)} file(s)"]
    if cost_tracker.total_duration > 0:
        parts.append(f"{cost_tracker.total_duration:.1f}s audio")
    if cost_tracker.total_cost > 0:
        parts.append(f"${cost_tracker.total_cost:.4f}")
    if status_tracker.num_tasks_failed > 0:
        parts.append(f"{status_tracker.num_tasks_failed} failed")
    if status_tracker.num_rate_limit_errors > 0:
        parts.append(f"{status_tracker.num_rate_limit_errors} rate limited")
    print("  " + " | ".join(parts))

    return list(results)


def transcribe_sync(
    audio_files: list[AudioSource] | AudioSource,
    model: str = "whisper-1",
    **kwargs,
) -> TranscriptionResponse | list[TranscriptionResponse]:
    """Synchronous convenience wrapper.

    If a single audio source is passed, returns a single TranscriptionResponse.
    If a list is passed, returns a list of TranscriptionResponse objects.
    """
    single = isinstance(audio_files, (str, bytes, Path, tuple))
    results = asyncio.run(transcribe_async(audio_files, model=model, **kwargs))
    if single:
        return results[0]
    return results
