"""Utilities for working with Anthropic Files API."""

import os
from pathlib import Path

import aiohttp

from lm_deluge.prompt import ToolResult


async def download_anthropic_file(
    file_id: str,
    api_key: str | None = None,
) -> bytes:
    """
    Download a file from the Anthropic Files API.

    Args:
        file_id: The file ID returned from code execution / skills
        api_key: Anthropic API key. If not provided, uses ANTHROPIC_API_KEY env var.

    Returns:
        The file content as bytes.

    Example:
        # After getting a response with tool results containing files
        for part in response.content.parts:
            if isinstance(part, ToolResult) and part.files:
                for file_info in part.files:
                    content = await download_anthropic_file(file_info["file_id"])
                    with open(file_info["filename"], "wb") as f:
                        f.write(content)
    """
    if api_key is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable not set and no api_key provided"
            )

    url = f"https://api.anthropic.com/v1/files/{file_id}/content"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "files-api-2025-04-14",
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(
                    f"Failed to download file {file_id}: {response.status} - {error_text}"
                )
            return await response.read()


async def get_anthropic_file_metadata(
    file_id: str,
    api_key: str | None = None,
) -> dict:
    """
    Get metadata for a file from the Anthropic Files API.

    Args:
        file_id: The file ID returned from code execution / skills
        api_key: Anthropic API key. If not provided, uses ANTHROPIC_API_KEY env var.

    Returns:
        Dict with file metadata including filename, size_bytes, created_at, etc.
    """
    if api_key is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable not set and no api_key provided"
            )

    url = f"https://api.anthropic.com/v1/files/{file_id}"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "files-api-2025-04-14",
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(
                    f"Failed to get file metadata {file_id}: {response.status} - {error_text}"
                )
            return await response.json()


async def save_anthropic_file(
    file_id: str,
    output_path: str | Path,
    api_key: str | None = None,
) -> Path:
    """
    Download and save a file from the Anthropic Files API.

    Args:
        file_id: The file ID returned from code execution / skills
        output_path: Path to save the file to
        api_key: Anthropic API key. If not provided, uses ANTHROPIC_API_KEY env var.

    Returns:
        The path to the saved file.
    """
    content = await download_anthropic_file(file_id, api_key)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(content)
    return output_path


async def get_response_files(
    response,
    api_key: str | None = None,
    fetch_metadata: bool = True,
) -> dict[str, bytes]:
    """
    Get all files from ToolResult parts in an API response as a dict.

    This function finds all ToolResult parts that have files (from code execution
    or skills) and downloads them, returning a dict mapping filenames to content.

    Args:
        response: An APIResponse object that may contain ToolResult parts with files
        api_key: Anthropic API key. If not provided, uses ANTHROPIC_API_KEY env var.
        fetch_metadata: If True, fetch file metadata from API to get real filename.
            Set to False if you want to use the filename from the response (may be generic).

    Returns:
        Dict mapping filename to file content as bytes.

    Example:
        response = await client.start(conv, skills=[skill])
        files = await get_response_files(response)
        for filename, content in files.items():
            print(f"Got file: {filename} ({len(content)} bytes)")
    """
    if response.content is None:
        return {}

    files: dict[str, bytes] = {}

    for part in response.content.parts:
        if isinstance(part, ToolResult) and part.files:
            for file_info in part.files:
                file_id = file_info["file_id"]
                if file_id is None:
                    continue
                filename = file_info["filename"]

                # Try to get the real filename from metadata
                if fetch_metadata and filename in ["output", "unknown"]:
                    try:
                        metadata = await get_anthropic_file_metadata(file_id, api_key)
                        if metadata.get("filename"):
                            filename = metadata["filename"]
                    except Exception:
                        pass  # Fall back to provided filename

                content = await download_anthropic_file(file_id, api_key)
                files[filename] = content

    return files


async def save_response_files(
    response,
    output_dir: str | Path = ".",
    api_key: str | None = None,
    fetch_metadata: bool = True,
) -> list[Path]:
    """
    Save all files from ToolResult parts in an API response.

    This function finds all ToolResult parts that have files (from code execution
    or skills) and downloads them to the specified directory.

    Args:
        response: An APIResponse object that may contain ToolResult parts with files
        output_dir: Directory to save files to (default: current directory)
        api_key: Anthropic API key. If not provided, uses ANTHROPIC_API_KEY env var.
        fetch_metadata: If True, fetch file metadata from API to get real filename.
            Set to False if you want to use the filename from the response (may be generic).

    Returns:
        List of paths to saved files.

    Example:
        response = await client.start(conv, skills=[skill])
        saved_files = await save_response_files(response, output_dir="./output")
        for path in saved_files:
            print(f"Saved: {path}")
    """
    if response.content is None:
        return []

    output_dir = Path(output_dir)
    saved_paths = []

    for part in response.content.parts:
        if isinstance(part, ToolResult) and part.files:
            for file_info in part.files:
                file_id = file_info["file_id"]
                if file_id is None:
                    continue
                filename = file_info["filename"]

                # Try to get the real filename from metadata
                if fetch_metadata and filename in ["output", "unknown"]:
                    try:
                        metadata = await get_anthropic_file_metadata(file_id, api_key)
                        if metadata.get("filename"):
                            filename = metadata["filename"]
                    except Exception:
                        pass  # Fall back to provided filename

                output_path = output_dir / filename
                await save_anthropic_file(file_id, output_path, api_key)
                saved_paths.append(output_path)

    return saved_paths
