import base64
import io
import mimetypes
import os
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Literal

import requests
import xxhash


@dataclass
class File:
    # raw bytes, pathlike, http url, base64 data url, or file_id
    data: bytes | io.BytesIO | Path | str | None
    media_type: str | None = None  # inferred if None
    type: str = field(init=False, default="file")
    is_remote: bool = False
    remote_provider: Literal["openai", "anthropic", "google"] | None = None
    filename: str | None = None  # optional filename for uploads
    file_id: str | None = None  # for OpenAI file uploads or Anthropic file API

    def __post_init__(self):
        if self.is_remote:
            if self.remote_provider is None:
                raise ValueError("remote_provider must be specified")
            if self.file_id is None:
                raise ValueError("file_id must be specified for remote files")
        if self.file_id and not self.is_remote:
            print("Warning: File ID specified by file not labeled as remote.")

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
            err = f"unreadable file. self.data type: {type(self.data)}"
            if isinstance(self.data, str) and len(self.data) < 1_000:
                err += f". self.data: {len(self.data)}"
            raise ValueError(err)

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

    @cached_property
    def fingerprint(self) -> str:
        # Hash the file contents for fingerprinting
        if self.is_remote:
            # For remote files, use provider:file_id for interpretability
            return f"{self.remote_provider}:{self.file_id}"
        file_bytes = self._bytes()
        return xxhash.xxh64(file_bytes).hexdigest()

    @cached_property
    def size(self) -> int:
        """Return file size in bytes."""
        if self.is_remote:
            # For remote files, we don't have the bytes available
            return 0
        return len(self._bytes())

    async def as_remote(
        self, provider: Literal["openai", "anthropic", "google"]
    ) -> "File":
        """Upload file to provider's file API and return new File with file_id.

        Args:
            provider: The provider to upload to ("openai", "anthropic", or "google")

        Returns:
            A new File object with file_id set and is_remote=True

        Raises:
            ValueError: If provider is unsupported or API key is missing
            RuntimeError: If upload fails
        """
        if self.is_remote:
            # If already remote with same provider, return self
            if self.remote_provider == provider:
                return self
            # Otherwise raise error about cross-provider incompatibility
            raise ValueError(
                f"File is already uploaded to {self.remote_provider}. "
                f"Cannot re-upload to {provider}."
            )

        if provider == "openai":
            return await self._upload_to_openai()
        elif provider == "anthropic":
            return await self._upload_to_anthropic()
        elif provider == "google":
            return await self._upload_to_google()
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    async def _upload_to_openai(self) -> "File":
        """Upload file to OpenAI's Files API."""
        import aiohttp

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable must be set")

        url = "https://api.openai.com/v1/files"
        headers = {"Authorization": f"Bearer {api_key}"}

        # Get file bytes and metadata
        file_bytes = self._bytes()
        filename = self._filename()

        # Create multipart form data
        data = aiohttp.FormData()
        data.add_field("purpose", "assistants")
        data.add_field(
            "file",
            file_bytes,
            filename=filename,
            content_type=self._mime(),
        )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, data=data) as response:
                    if response.status != 200:
                        text = await response.text()
                        raise RuntimeError(f"Failed to upload file to OpenAI: {text}")

                    response_data = await response.json()
                    file_id = response_data["id"]

            # Return new File object with file_id
            return File(
                data=None,
                media_type=self.media_type,
                is_remote=True,
                remote_provider="openai",
                filename=filename,
                file_id=file_id,
            )
        except aiohttp.ClientError as e:
            raise RuntimeError(f"Failed to upload file to OpenAI: {e}")

    async def _upload_to_anthropic(self) -> "File":
        """Upload file to Anthropic's Files API."""
        import aiohttp

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable must be set")

        url = "https://api.anthropic.com/v1/files"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "files-api-2025-04-14",
        }

        # Get file bytes and metadata
        file_bytes = self._bytes()
        filename = self._filename()

        # Create multipart form data
        data = aiohttp.FormData()
        data.add_field(
            "file",
            file_bytes,
            filename=filename,
            content_type=self._mime(),
        )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, data=data) as response:
                    if response.status != 200:
                        text = await response.text()
                        raise RuntimeError(
                            f"Failed to upload file to Anthropic: {text}"
                        )

                    response_data = await response.json()
                    file_id = response_data["id"]

            # Return new File object with file_id
            return File(
                data=None,
                media_type=self.media_type,
                is_remote=True,
                remote_provider="anthropic",
                filename=filename,
                file_id=file_id,
            )
        except aiohttp.ClientError as e:
            raise RuntimeError(f"Failed to upload file to Anthropic: {e}")

    async def _upload_to_google(self) -> "File":
        """Upload file to Google Gemini Files API."""
        import json

        import aiohttp

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable must be set")

        # Google uses a different URL structure with the API key as a parameter
        url = f"https://generativelanguage.googleapis.com/upload/v1beta/files?key={api_key}"

        # Get file bytes and metadata
        file_bytes = self._bytes()
        filename = self._filename()
        mime_type = self._mime()

        # Google expects a multipart request with metadata and file data
        # Using the resumable upload protocol
        headers = {
            "X-Goog-Upload-Protocol": "multipart",
        }

        # Create multipart form data with metadata and file
        data = aiohttp.FormData()

        # Add metadata part as JSON
        metadata = {"file": {"display_name": filename}}
        data.add_field(
            "metadata",
            json.dumps(metadata),
            content_type="application/json",
        )

        # Add file data part
        data.add_field(
            "file",
            file_bytes,
            filename=filename,
            content_type=mime_type,
        )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, data=data) as response:
                    if response.status not in [200, 201]:
                        text = await response.text()
                        raise RuntimeError(f"Failed to upload file to Google: {text}")

                    response_data = await response.json()
                    # Google returns a file object with a 'name' field like 'files/abc123'
                    file_uri = response_data.get("file", {}).get(
                        "uri"
                    ) or response_data.get("name")
                    if not file_uri:
                        raise RuntimeError(
                            f"No file URI in Google response: {response_data}"
                        )

            # Return new File object with file_id (using the file URI)
            return File(
                data=None,
                media_type=self.media_type,
                is_remote=True,
                remote_provider="google",
                filename=filename,
                file_id=file_uri,
            )
        except aiohttp.ClientError as e:
            raise RuntimeError(f"Failed to upload file to Google: {e}")

    async def delete(self) -> bool:
        """Delete the uploaded file from the remote provider.

        Returns:
            True if deletion was successful, False otherwise

        Raises:
            ValueError: If file is not a remote file or provider is unsupported
            RuntimeError: If deletion fails
        """
        if not self.is_remote:
            raise ValueError(
                "Cannot delete a non-remote file. Only remote files can be deleted."
            )

        if not self.file_id:
            raise ValueError("Cannot delete file without file_id")

        if self.remote_provider == "openai":
            return await self._delete_from_openai()
        elif self.remote_provider == "anthropic":
            return await self._delete_from_anthropic()
        elif self.remote_provider == "google":
            return await self._delete_from_google()
        else:
            raise ValueError(f"Unsupported provider: {self.remote_provider}")

    async def _delete_from_openai(self) -> bool:
        """Delete file from OpenAI's Files API."""
        import aiohttp

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable must be set")

        url = f"https://api.openai.com/v1/files/{self.file_id}"
        headers = {"Authorization": f"Bearer {api_key}"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.delete(url, headers=headers) as response:
                    if response.status == 200:
                        return True
                    else:
                        text = await response.text()
                        raise RuntimeError(f"Failed to delete file from OpenAI: {text}")
        except aiohttp.ClientError as e:
            raise RuntimeError(f"Failed to delete file from OpenAI: {e}")

    async def _delete_from_anthropic(self) -> bool:
        """Delete file from Anthropic's Files API."""
        import aiohttp

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable must be set")

        url = f"https://api.anthropic.com/v1/files/{self.file_id}"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "files-api-2025-04-14",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.delete(url, headers=headers) as response:
                    if response.status == 200:
                        return True
                    else:
                        text = await response.text()
                        raise RuntimeError(
                            f"Failed to delete file from Anthropic: {text}"
                        )
        except aiohttp.ClientError as e:
            raise RuntimeError(f"Failed to delete file from Anthropic: {e}")

    async def _delete_from_google(self) -> bool:
        """Delete file from Google Gemini Files API."""
        import aiohttp

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable must be set")

        # Google file_id is the full URI like "https://generativelanguage.googleapis.com/v1beta/files/abc123"
        # We need to extract just the file name part for the delete endpoint
        assert self.file_id, "can't delete file with no file id"
        if self.file_id.startswith("https://"):
            # Extract the path after the domain
            file_name = self.file_id.split("/v1beta/")[-1]
        else:
            file_name = self.file_id

        url = f"https://generativelanguage.googleapis.com/v1beta/{file_name}?key={api_key}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.delete(url) as response:
                    if response.status in [200, 204]:
                        return True
                    else:
                        text = await response.text()
                        raise RuntimeError(f"Failed to delete file from Google: {text}")
        except aiohttp.ClientError as e:
            raise RuntimeError(f"Failed to delete file from Google: {e}")

    # ── provider-specific emission ────────────────────────────────────────────
    def oa_chat(self) -> dict:
        """For OpenAI Chat Completions - file content as base64 or file_id."""
        # Validate provider compatibility
        if self.is_remote and self.remote_provider != "openai":
            raise ValueError(
                f"Cannot emit file uploaded to {self.remote_provider} as OpenAI format. "
                f"File must be uploaded to OpenAI or provided as raw data."
            )

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
        # Validate provider compatibility
        if self.is_remote and self.remote_provider != "openai":
            raise ValueError(
                f"Cannot emit file uploaded to {self.remote_provider} as OpenAI format. "
                f"File must be uploaded to OpenAI or provided as raw data."
            )

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
        # Validate provider compatibility
        if self.is_remote and self.remote_provider != "anthropic":
            raise ValueError(
                f"Cannot emit file uploaded to {self.remote_provider} as Anthropic format. "
                f"File must be uploaded to Anthropic or provided as raw data."
            )

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
        """For Gemini API - files are provided as inline data or file URI."""
        # Validate provider compatibility
        if self.is_remote and self.remote_provider != "google":
            raise ValueError(
                f"Cannot emit file uploaded to {self.remote_provider} as Google format. "
                f"File must be uploaded to Google or provided as raw data."
            )

        if self.file_id:
            # Use file URI for uploaded files
            return {
                "fileData": {
                    "mimeType": self._mime(),
                    "fileUri": self.file_id,
                }
            }
        else:
            # Use inline data for non-uploaded files
            return {
                "inlineData": {
                    "mimeType": self._mime(),
                    "data": self._base64(include_header=False),
                }
            }

    def mistral(self) -> dict:
        """For Mistral API - not yet supported."""
        raise NotImplementedError("File support for Mistral is not yet implemented")
