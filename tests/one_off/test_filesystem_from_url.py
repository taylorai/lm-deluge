"""Test FilesystemManager.from_zip with URL support.

This test requires network access to fetch a zip file from the internet.
"""

import io
import zipfile

from lm_deluge.tool.prefab.filesystem import FilesystemManager


def test_from_zip_with_url():
    # Use a small, stable public zip file - GitHub release artifact
    # This is the pyproject.toml from a small project zipped up
    url = "https://github.com/psf/requests/archive/refs/tags/v2.32.0.zip"

    manager = FilesystemManager.from_zip(url, max_files=500)

    # The zip should contain files - list root to verify it loaded
    tool = manager.get_tools()[0]
    import json

    result = json.loads(tool.run(command="list_dir", path=".", recursive=False))
    assert result["ok"], f"list_dir failed: {result}"
    # GitHub zips have a top-level folder like "requests-2.32.0"
    entries = result["result"]["entries"]
    assert len(entries) > 0, "Expected at least one entry in the zip"


def test_from_zip_with_bytesio():
    # Create an in-memory zip file
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("hello.txt", "Hello, World!")
        zf.writestr("subdir/nested.txt", "Nested content")
    buffer.seek(0)

    manager = FilesystemManager.from_zip(buffer)

    # Verify contents
    assert manager.backend.read_file("hello.txt") == "Hello, World!"
    assert manager.backend.read_file("subdir/nested.txt") == "Nested content"


if __name__ == "__main__":
    print("Testing from_zip with BytesIO...")
    test_from_zip_with_bytesio()
    print("OK")

    print("Testing from_zip with URL...")
    test_from_zip_with_url()
    print("OK")

    print("All tests passed!")
