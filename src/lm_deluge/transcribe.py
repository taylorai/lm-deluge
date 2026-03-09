"""Parallel audio transcription API for OpenAI, Mistral, Fireworks, and Deepgram."""

import asyncio
import mimetypes
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp
from tqdm.auto import tqdm

from .api_requests.base import parse_retry_after
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
    },
    "whisper-v3-turbo": {
        "provider": "fireworks",
        "api_base": "https://audio-turbo.api.fireworks.ai/v1",
        "api_key_env_var": "FIREWORKS_API_KEY",
        "cost_per_minute": 0.002,
        "max_duration": None,
        "max_file_size": 1024 * 1024 * 1024,
    },
    # Deepgram
    "nova-3": {
        "provider": "deepgram",
        "api_base": "https://api.deepgram.com/v1",
        "api_key_env_var": "DEEPGRAM_API_KEY",
        "cost_per_minute": 0.0043,
        "max_duration": None,
        "max_file_size": 2 * 1024 * 1024 * 1024,  # 2 GB
    },
    "nova-2": {
        "provider": "deepgram",
        "api_base": "https://api.deepgram.com/v1",
        "api_key_env_var": "DEEPGRAM_API_KEY",
        "cost_per_minute": 0.0043,
        "max_duration": None,
        "max_file_size": 2 * 1024 * 1024 * 1024,
    },
}

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


def _read_audio(source: str | bytes | Path) -> tuple[bytes, str, str]:
    """Read audio from a file path or bytes.

    Returns (audio_bytes, mime_type, filename).
    """
    if isinstance(source, bytes):
        return source, "audio/wav", "audio.wav"
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
    return None


def _split_audio(
    path: str | Path,
    max_chunk_seconds: float,
) -> list[Path]:
    """Split an audio file into chunks using ffmpeg.

    Each chunk is up to max_chunk_seconds long. Output format is mp3 for
    good compression and universal support.

    Returns list of temp file paths. Caller is responsible for cleanup.
    """
    path = Path(path)
    duration = _get_audio_duration(path)
    if duration is None:
        raise RuntimeError(
            "Could not determine audio duration. Install ffmpeg/ffprobe to "
            "enable automatic audio splitting."
        )

    if duration <= max_chunk_seconds:
        return [path]

    tmp_dir = Path(tempfile.mkdtemp(prefix="lm_deluge_audio_"))
    chunks: list[Path] = []
    start = 0.0
    chunk_idx = 0

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
            "-acodec",
            "libmp3lame",
            "-q:a",
            "2",
            str(chunk_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            # Clean up on failure
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise RuntimeError(
                f"ffmpeg failed to split audio at {start}s: {result.stderr[:500]}"
            )
        chunks.append(chunk_path)
        chunk_idx += 1
        start += max_chunk_seconds

    return chunks


def _stitch_responses(
    chunk_responses: list[TranscriptionResponse],
    original_id: int,
    chunk_offsets: list[float],
    known_duration: float | None = None,
    known_language: str | None = None,
) -> TranscriptionResponse:
    """Stitch chunk transcription responses back into a single response.

    Adjusts segment/word timestamps by adding chunk offsets and concatenates text.
    known_duration/known_language are used as fallbacks when the API doesn't
    return them (e.g. gpt-4o-transcribe models only return {"text": "..."}).
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


def _parse_response(
    provider: str,
    result: dict,
) -> tuple[str, str | None, float | None, list[TranscriptionSegment], list[dict]]:
    """Extract transcription data from provider response.

    Returns (text, language, duration, segments, words).
    """
    if provider == "deepgram":
        return _parse_deepgram_response(result)

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
                async with session.post(url, **request_kwargs) as response:
                    if response.status == 200:
                        result = await response.json()
                        text, lang, duration, segments, words = _parse_response(
                            provider, result
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
    source: str | bytes | Path,
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
    """Transcribe a single audio source, auto-splitting if needed."""
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

    # Check if splitting is needed
    needs_split = False
    source_path: Path | None = None

    if isinstance(source, (str, Path)):
        source_path = Path(source)

    # Check file size limit
    if max_file_size and len(audio_bytes) > max_file_size:
        needs_split = True

    # Check duration limit (only if we can probe and there is a limit)
    audio_duration: float | None = None
    if source_path and (max_duration or max_file_size):
        audio_duration = _get_audio_duration(source_path)
        if audio_duration and max_duration and audio_duration > max_duration:
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

    # Determine chunk size: use 80% of max_duration to leave margin,
    # or 20 minutes if only file size is the constraint
    if max_duration:
        chunk_seconds = max_duration * 0.8
    else:
        chunk_seconds = 1200.0  # 20 minutes

    # Split and transcribe chunks
    chunk_paths: list[Path] = []
    tmp_dir: Path | None = None
    try:
        chunk_paths = _split_audio(source_path, chunk_seconds)
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

        # Calculate time offsets for each chunk
        chunk_offsets = [i * chunk_seconds for i in range(n_chunks)]

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
        )

    except RuntimeError as e:
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
    audio_files: list[str | bytes | Path] | str | bytes | Path,
    model: str = "whisper-1",
    language: str | None = None,
    timestamps: bool = False,
    max_attempts: int = 5,
    max_requests_per_minute: int = 50,
    max_concurrent_requests: int = 10,
    request_timeout: int = 300,
    show_progress: bool = True,
    **kwargs,
) -> list[TranscriptionResponse]:
    """Transcribe audio files in parallel.

    Args:
        audio_files: A single audio source or list of sources. Each source can
            be a file path (str or Path) or raw bytes.
        model: Transcription model name (see REGISTRY for options).
        language: ISO-639-1 language code to help the model.
        timestamps: Request segment/word-level timestamps (when supported).
        max_attempts: Max retries per file on failure.
        max_requests_per_minute: RPM limit for throttling.
        max_concurrent_requests: Max simultaneous API requests.
        request_timeout: Timeout per request in seconds.
        show_progress: Show a tqdm progress bar.
        **kwargs: Extra parameters passed to the transcription API
            (e.g. prompt, temperature).

    Returns:
        List of TranscriptionResponse objects, one per audio file, sorted by ID.
    """
    # Normalize single input to list
    if isinstance(audio_files, (str, bytes, Path)):
        audio_files = [audio_files]

    if not audio_files:
        return []

    model_info = _get_model_info(model)
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
    audio_files: list[str | bytes | Path] | str | bytes | Path,
    model: str = "whisper-1",
    **kwargs,
) -> TranscriptionResponse | list[TranscriptionResponse]:
    """Synchronous convenience wrapper.

    If a single audio source is passed, returns a single TranscriptionResponse.
    If a list is passed, returns a list of TranscriptionResponse objects.
    """
    single = isinstance(audio_files, (str, bytes, Path))
    results = asyncio.run(transcribe_async(audio_files, model=model, **kwargs))
    if single:
        return results[0]
    return results
