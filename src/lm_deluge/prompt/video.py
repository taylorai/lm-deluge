import base64
import io
import mimetypes
import os
from dataclasses import dataclass, field
from pathlib import Path

import xxhash

from lm_deluge.prompt.url_fetch import read_url_bytes


@dataclass(slots=True)
class Video:
    data: bytes | io.BytesIO | Path | str
    media_type: str | None = None
    type: str = field(init=False, default="video")

    def _bytes(self) -> bytes:
        if isinstance(self.data, bytes):
            return self.data
        if isinstance(self.data, io.BytesIO):
            return self.data.getvalue()
        if isinstance(self.data, str) and self._is_url():
            return read_url_bytes(self.data)
        if isinstance(self.data, str) and os.path.exists(self.data):
            with open(self.data, "rb") as f:
                return f.read()
        if isinstance(self.data, Path) and self.data.exists():
            return Path(self.data).read_bytes()
        if isinstance(self.data, str) and self.data.startswith("data:"):
            _, encoded = self.data.split(",", 1)
            return base64.b64decode(encoded)
        raise ValueError(f"unreadable video format. type: {type(self.data)}")

    def _is_url(self) -> bool:
        return isinstance(self.data, str) and self.data.startswith(
            ("http://", "https://")
        )

    def _mime(self) -> str:
        if self.media_type:
            return self.media_type
        if isinstance(self.data, (Path, str)):
            guess = mimetypes.guess_type(str(self.data))[0]
            if guess:
                return guess
        return "video/mp4"

    def _base64(self) -> str:
        encoded = base64.b64encode(self._bytes()).decode("utf-8")
        return f"data:{self._mime()};base64,{encoded}"

    @property
    def fingerprint(self) -> str:
        if self._is_url():
            return xxhash.xxh64(str(self.data).encode()).hexdigest()
        return xxhash.xxh64(self._bytes()).hexdigest()

    def oa_chat(self) -> dict:
        url = self.data if self._is_url() else self._base64()
        return {
            "type": "video_url",
            "video_url": {
                "url": url,
            },
        }

    def oa_resp(self) -> dict:
        raise NotImplementedError("Video is only supported by Chat Completions")

    def anthropic(self) -> dict:
        raise NotImplementedError("Video is not supported by Anthropic")

    def gemini(self) -> dict:
        raise NotImplementedError("Video is not supported by native Gemini requests")

    def mistral(self) -> dict:
        raise NotImplementedError("Video is not supported by Mistral")

    def nova(self) -> dict:
        raise NotImplementedError("Video is not supported by Nova")
