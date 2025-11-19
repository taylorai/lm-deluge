import json

import pytest

from lm_deluge.llm_tools.filesystem import (
    FilesystemManager,
    InMemoryWorkspaceBackend,
)


def test_inmemory_backend_read_write_append():
    backend = InMemoryWorkspaceBackend()
    backend.write_file("foo.txt", "hello", overwrite=True)
    backend.append_file("foo.txt", "\nworld")

    assert backend.read_file("foo.txt") == "hello\nworld"


def test_inmemory_backend_prevents_path_escape():
    backend = InMemoryWorkspaceBackend()

    with pytest.raises(ValueError):
        backend.write_file("../etc/passwd", "nope", overwrite=True)


def test_list_and_delete_paths():
    backend = InMemoryWorkspaceBackend()
    backend.write_file("notes/todo.txt", "one", overwrite=True)
    backend.write_file("notes/archive/done.txt", "two", overwrite=True)

    top_level = backend.list_dir(".", recursive=False)
    assert {"path": "notes", "type": "directory", "size": None} in top_level

    recursive = backend.list_dir("notes", recursive=True)
    assert any(entry["path"] == "notes/todo.txt" for entry in recursive)
    assert any(entry["path"] == "notes/archive/done.txt" for entry in recursive)

    backend.delete_path("notes")
    with pytest.raises(FileNotFoundError):
        backend.read_file("notes/todo.txt")


def test_grep_limits_results():
    backend = InMemoryWorkspaceBackend()
    backend.write_file("code/app.py", "alpha\nbeta\nalpha", overwrite=True)

    matches = backend.grep("alpha", path="code", limit=1)
    assert len(matches) == 1

    with pytest.raises(FileNotFoundError):
        backend.grep("anything", path="missing", limit=5)


def test_filesystem_manager_tool_roundtrip():
    backend = InMemoryWorkspaceBackend({"main.py": "print('hi')\nprint('bye')"})
    manager = FilesystemManager(backend=backend, tool_name="fs")
    tool = manager.get_tools()[0]

    read_payload = json.loads(
        tool.run(command="read_file", path="main.py", start_line=2, end_line=2)
    )
    assert read_payload["ok"]
    assert read_payload["result"]["content"] == "print('bye')"

    write_payload = json.loads(
        tool.run(command="write_file", path="new.txt", content="hello world")
    )
    assert write_payload["ok"]
    assert backend.read_file("new.txt") == "hello world"

    error_payload = json.loads(tool.run(command="read_file", path="missing.txt"))
    assert not error_payload["ok"]
    assert error_payload["error"] == "FileNotFoundError"


def test_filesystem_manager_reads_empty_files():
    backend = InMemoryWorkspaceBackend({"empty.txt": ""})
    manager = FilesystemManager(backend=backend, tool_name="fs")
    tool = manager.get_tools()[0]

    payload = json.loads(tool.run(command="read_file", path="empty.txt"))
    assert payload["ok"]
    result = payload["result"]
    assert result["content"] == ""
    assert result["total_lines"] == 0
    assert result["start_line"] == 1
    assert result["end_line"] == 0
    assert result["character_count"] == 0


def test_filesystem_manager_dump(tmp_path):
    backend = InMemoryWorkspaceBackend(
        {"src/main.py": "print('hi')", "README.md": "# title"}
    )
    manager = FilesystemManager(backend=backend)

    written = manager.dump(tmp_path / "export")
    assert sorted(written) == ["README.md", "src/main.py"]

    assert (tmp_path / "export" / "README.md").read_text() == "# title"
    assert (tmp_path / "export" / "src" / "main.py").read_text() == "print('hi')"


def test_get_tools_with_exclusions():
    backend = InMemoryWorkspaceBackend({"only.txt": "data"})
    manager = FilesystemManager(backend=backend)
    tool = manager.get_tools(exclude={"write_file"})[0]

    command_schema = tool.parameters.get("command", {})
    assert "write_file" not in command_schema.get("enum", [])

    ok_payload = json.loads(tool.run(command="read_file", path="only.txt"))
    assert ok_payload["ok"]

    blocked = json.loads(
        tool.run(command="write_file", path="extra.txt", content="nope")
    )
    assert blocked["ok"] is False
    assert "disabled" in blocked["message"]
