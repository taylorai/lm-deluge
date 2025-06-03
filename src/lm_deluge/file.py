import os
import io
import requests
import base64
import mimetypes
import xxhash
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class File:
    # raw bytes, pathlike, http url, base64 data url, or file_id
    data: bytes | io.BytesIO | Path | str
    media_type: str | None = None  # inferred if None
    filename: str | None = None  # optional filename for uploads
    file_id: str | None = None  # for OpenAI file uploads or Anthropic file API
    type: str = field(init=False, default="file")

    # helpers -----------------------------------------------------------------
    def _bytes(self) -> bytes:
        if isinstance(self.data, bytes):
            return self.data
        elif isinstance(self.data, io.BytesIO):
            return self.data.getvalue()
        elif isinstance(self.data, str) and self.data.startswith("http"):
            res = requests.get(self.data)
            res.raise_for_status()
            return res.content
        elif isinstance(self.data, str) and os.path.exists(self.data):
            with open(self.data, "rb") as f:
                return f.read()
        elif isinstance(self.data, Path) and self.data.exists():
            return Path(self.data).read_bytes()
        elif isinstance(self.data, str) and self.data.startswith("data:"):
            header, encoded = self.data.split(",", 1)
            return base64.b64decode(encoded)
        else:
            raise ValueError("unreadable file format")

    def _mime(self) -> str:
        if self.media_type:
            return self.media_type
        if isinstance(self.data, (Path, str)):
            # For URL or path, try to guess from the string
            path_str = str(self.data)
            guess = mimetypes.guess_type(path_str)[0]
            if guess:
                return guess
        return "application/pdf"  # default to PDF

    def _filename(self) -> str:
        if self.filename:
            return self.filename
        if isinstance(self.data, (Path, str)):
            path_str = str(self.data)
            if path_str.startswith("http"):
                # Extract filename from URL
                return path_str.split("/")[-1].split("?")[0] or "document.pdf"
            else:
                # Extract from local path
                return os.path.basename(path_str) or "document.pdf"
        return "document.pdf"

    def _base64(self, include_header: bool = True) -> str:
        encoded = base64.b64encode(self._bytes()).decode("utf-8")
        if not include_header:
            return encoded
        return f"data:{self._mime()};base64,{encoded}"

    @property
    def fingerprint(self) -> str:
        # Hash the file contents for fingerprinting
        file_bytes = self._bytes()
        return xxhash.xxh64(file_bytes).hexdigest()

    @property
    def size(self) -> int:
        """Return file size in bytes."""
        return len(self._bytes())

    # ── provider-specific emission ────────────────────────────────────────────
    def oa_chat(self) -> dict:
        """For OpenAI Chat Completions - file content as base64 or file_id."""
        if self.file_id:
            return {
                "type": "file",
                "file": {
                    "file_id": self.file_id,
                },
            }
        else:
            return {
                "type": "file",
                "file": {
                    "filename": self._filename(),
                    "file_data": self._base64(),
                },
            }

    def oa_resp(self) -> dict:
        """For OpenAI Responses API - file content as base64 or file_id."""
        if self.file_id:
            return {
                "type": "input_file",
                "file_id": self.file_id,
            }
        else:
            return {
                "type": "input_file",
                "filename": self._filename(),
                "file_data": self._base64(),
            }

    def anthropic(self) -> dict:
        """For Anthropic Messages API - file content as base64 or file_id."""
        if self.file_id:
            return {
                "type": "document",
                "source": {
                    "type": "file",
                    "file_id": self.file_id,
                },
            }
        else:
            b64 = base64.b64encode(self._bytes()).decode()
            return {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": self._mime(),
                    "data": b64,
                },
            }

    def anthropic_file_upload(self) -> tuple[str, bytes, str]:
        """For Anthropic Files API - return tuple for file upload."""
        filename = self._filename()
        content = self._bytes()
        media_type = self._mime()
        return filename, content, media_type

    def gemini(self) -> dict:
        """For Gemini API - files are provided as inline data."""
        return {
            "inlineData": {
                "mimeType": self._mime(),
                "data": self._base64(include_header=False),
            }
        }

    def mistral(self) -> dict:
        """For Mistral API - not yet supported."""
        raise NotImplementedError("File support for Mistral is not yet implemented")
