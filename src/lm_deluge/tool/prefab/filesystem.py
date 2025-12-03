from __future__ import annotations

import io
import json
import os
import random
import re
import time
import zipfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Protocol

from pydantic import BaseModel, Field

from .. import Tool

FS_DESCRIPTION = """Interact with an isolated virtual filesystem that belongs to this session.

Paths are always relative to the workspace root and use forward slashes. Use this tool to:
- inspect files with optional line ranges
- create, overwrite, or append to files
- delete files or folders from the workspace
- list directory contents
- search for text across the workspace using regular expressions
- apply OpenAI-style apply_patch operations (create/update/delete)

This virtual filesystem is safe-by-construction. Paths that try to escape the workspace
or reference missing files will raise clear errors."""


class WorkspaceBackend(Protocol):
    """Abstract filesystem operations used by FilesystemManager."""

    def read_file(self, path: str) -> str: ...

    def write_file(self, path: str, content: str, *, overwrite: bool) -> None: ...

    def append_file(self, path: str, content: str) -> None: ...

    def delete_path(self, path: str) -> None: ...

    def list_dir(self, path: str, recursive: bool) -> list[dict[str, Any]]: ...

    def grep(
        self, pattern: str, path: str | None, limit: int
    ) -> list[dict[str, Any]]: ...


def _normalize_path(path: str | None, *, allow_root: bool = False) -> str:
    """Normalize user-provided paths and prevent directory traversal."""
    if path is None or path.strip() == "":
        if allow_root:
            return "."
        raise ValueError("Path is required")

    raw = path.strip()
    if raw.startswith("/"):
        raw = raw.lstrip("/")

    parts: list[str] = []
    for part in raw.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if parts:
                parts.pop()
            else:
                raise ValueError("Path traversal outside the workspace is not allowed")
            continue
        parts.append(part)

    normalized = "/".join(parts)
    if normalized:
        return normalized
    if allow_root:
        return "."
    raise ValueError("Path must reference a file inside the workspace")


class InMemoryWorkspaceBackend:
    """Simple backend that stores files in memory."""

    def __init__(self, files: dict[str, str] | None = None):
        self._files: Dict[str, str] = {}
        if files:
            for path, content in files.items():
                key = _normalize_path(path)
                self._files[key] = content

    def read_file(self, path: str) -> str:
        key = _normalize_path(path)
        if key not in self._files:
            raise FileNotFoundError(f"{key} does not exist")
        return self._files[key]

    def write_file(self, path: str, content: str, *, overwrite: bool) -> None:
        key = _normalize_path(path)
        if not overwrite and key in self._files:
            raise FileExistsError(f"{key} already exists")
        self._files[key] = content

    def append_file(self, path: str, content: str) -> None:
        key = _normalize_path(path)
        if key in self._files:
            self._files[key] = f"{self._files[key]}{content}"
        else:
            self._files[key] = content

    def delete_path(self, path: str) -> None:
        key = _normalize_path(path, allow_root=True)
        if key == ".":
            self._files.clear()
            return
        if key in self._files:
            del self._files[key]
            return

        prefix = f"{key}/"
        targets = [p for p in self._files if p.startswith(prefix)]
        if not targets:
            raise FileNotFoundError(f"{key} does not exist")
        for target in targets:
            del self._files[target]

    def list_dir(self, path: str, recursive: bool) -> list[dict[str, Any]]:
        key = _normalize_path(path, allow_root=True)
        if key != "." and key in self._files and not recursive:
            # Listing a file path shows metadata for that file.
            return [self._format_file_entry(key)]

        prefix = "" if key == "." else f"{key}/"
        entries: list[dict[str, Any]] = []

        if key != "." and not any(
            p == key or p.startswith(prefix) for p in self._files
        ):
            raise FileNotFoundError(f"{key} does not exist")

        if recursive:
            for file_path in sorted(self._files):
                if not (file_path == key or file_path.startswith(prefix)):
                    continue
                entries.append(self._format_file_entry(file_path))
            return entries

        seen_dirs: set[str] = set()
        for file_path in sorted(self._files):
            if not (file_path == key or file_path.startswith(prefix)):
                continue
            remainder = file_path[len(prefix) :]
            if remainder == "":
                entries.append(self._format_file_entry(file_path))
                continue
            head, _, tail = remainder.partition("/")
            if tail:
                dir_path = head if key == "." else f"{key}/{head}"
                if dir_path not in seen_dirs:
                    entries.append(
                        {"path": dir_path, "type": "directory", "size": None}
                    )
                    seen_dirs.add(dir_path)
            else:
                entries.append(self._format_file_entry(file_path))
        return entries

    def grep(self, pattern: str, path: str | None, limit: int) -> list[dict[str, Any]]:
        regex = re.compile(pattern)
        key = _normalize_path(path, allow_root=True) if path is not None else "."
        prefix = "" if key == "." else f"{key}/"
        matches: list[dict[str, Any]] = []

        for file_path in sorted(self._files):
            if not (file_path == key or file_path.startswith(prefix)):
                continue
            content = self._files[file_path]
            for line_no, line in enumerate(content.splitlines(), start=1):
                if regex.search(line):
                    matches.append(
                        {"path": file_path, "line": line_no, "text": line.strip()}
                    )
                    if len(matches) >= limit:
                        return matches
        if (
            key != "."
            and key not in self._files
            and not any(p.startswith(prefix) for p in self._files)
        ):
            raise FileNotFoundError(f"{key} does not exist")
        return matches

    def _format_file_entry(self, path: str) -> dict[str, Any]:
        content = self._files[path]
        if content == "":
            line_count = 0
        else:
            line_count = content.count("\n") + (0 if content.endswith("\n") else 1)
        return {
            "path": path,
            "type": "file",
            "size": len(content),
            "line_count": max(line_count, 0),
        }


FsCommand = Literal[
    "read_file",
    "write_file",
    "delete_path",
    "list_dir",
    "grep",
    "apply_patch",
]
ALL_COMMANDS: tuple[FsCommand, ...] = (
    "read_file",
    "write_file",
    "delete_path",
    "list_dir",
    "grep",
    "apply_patch",
)


class ApplyPatchOperation(BaseModel):
    """Subset of OpenAI apply_patch operation payload."""

    type: Literal["create_file", "update_file", "delete_file"] = Field(
        description="Type of patch operation to perform."
    )
    path: str = Field(description="Path to the file being modified.")
    diff: str | None = Field(
        default=None,
        description="V4A diff to apply for create/update operations.",
    )

    @property
    def requires_diff(self) -> bool:
        return self.type in {"create_file", "update_file"}


class FilesystemParams(BaseModel):
    """Schema describing filesystem tool calls."""

    command: FsCommand = Field(
        description="Filesystem operation to perform (read_file, write_file, delete_path, list_dir, grep)"
    )
    path: Optional[str] = Field(
        default=None,
        description="Path to operate on, relative to workspace root. Use '.' for the root directory.",
    )
    start_line: Optional[int] = Field(
        default=None,
        description="1-based inclusive start line when reading a file. Leave unset to read from the beginning.",
        ge=1,
    )
    end_line: Optional[int] = Field(
        default=None,
        description="1-based inclusive end line when reading a file. Leave unset to read through the end.",
        ge=1,
    )
    content: Optional[str] = Field(
        default=None,
        description="Content to write when using write_file.",
    )
    mode: Optional[Literal["overwrite", "append", "create_if_missing"]] = Field(
        default="overwrite",
        description="How to write content. Overwrite replaces the file, append adds to the end, create_if_missing leaves existing files untouched.",
    )
    recursive: Optional[bool] = Field(
        default=None,
        description="When listing directories, set to true to recurse.",
    )
    pattern: Optional[str] = Field(
        default=None,
        description="Regular expression pattern to search for when using grep.",
    )
    max_results: Optional[int] = Field(
        default=50,
        description="Maximum number of grep matches to return.",
        ge=1,
    )
    operation: ApplyPatchOperation | None = Field(
        default=None,
        description=(
            "When using command='apply_patch', include an operation matching the "
            "OpenAI apply_patch_call payload (type, path, diff)."
        ),
    )


class FilesystemManager:
    """Expose a TodoManager-style tool for interacting with a workspace."""

    def __init__(
        self,
        backend: WorkspaceBackend | None = None,
        *,
        tool_name: str = "filesystem",
    ):
        self.backend = backend or InMemoryWorkspaceBackend()
        self.tool_name = tool_name
        self._tool_cache: dict[tuple[str, ...], list[Tool]] = {}

    def _handle_read(
        self, path: str, start_line: Optional[int], end_line: Optional[int]
    ) -> dict[str, Any]:
        content = self.backend.read_file(path)
        total_lines = len(content.splitlines()) or (0 if content == "" else 1)
        start = start_line or 1
        end = end_line or total_lines
        if end < start:
            if not (total_lines == 0 and end_line is None and start == 1):
                raise ValueError("end_line must be greater than or equal to start_line")

        if start == 1 and end >= total_lines:
            snippet = content
        else:
            lines = content.splitlines()
            snippet = "\n".join(lines[start - 1 : end])

        return {
            "path": path,
            "start_line": start,
            "end_line": end,
            "content": snippet,
            "total_lines": total_lines,
            "character_count": len(content),
        }

    def _handle_write(
        self, path: str, content: str, mode: Optional[str]
    ) -> dict[str, Any]:
        write_mode = mode or "overwrite"
        if write_mode == "overwrite":
            self.backend.write_file(path, content, overwrite=True)
        elif write_mode == "append":
            self.backend.append_file(path, content)
        elif write_mode == "create_if_missing":
            try:
                self.backend.write_file(path, content, overwrite=False)
            except FileExistsError:
                pass
        else:
            raise ValueError(f"Unsupported write mode: {write_mode}")
        return {"path": path, "status": "ok", "mode": write_mode}

    def _handle_delete(self, path: str) -> dict[str, Any]:
        self.backend.delete_path(path)
        return {"path": path, "status": "ok"}

    def _handle_list(
        self, path: Optional[str], recursive: Optional[bool]
    ) -> dict[str, Any]:
        listing = self.backend.list_dir(path or ".", recursive=bool(recursive))
        return {"path": path or ".", "recursive": bool(recursive), "entries": listing}

    def _handle_grep(
        self, pattern: str, path: Optional[str], limit: Optional[int]
    ) -> dict[str, Any]:
        max_results = limit or 50
        matches = self.backend.grep(pattern, path=path, limit=max_results)
        return {
            "pattern": pattern,
            "path": path,
            "max_results": max_results,
            "matches": matches,
        }

    def _handle_apply_patch(self, operation: ApplyPatchOperation) -> dict[str, Any]:
        if operation.requires_diff and not operation.diff:
            raise ValueError("diff is required for create_file and update_file")

        if operation.type == "delete_file":
            self.backend.delete_path(operation.path)
            return {"path": operation.path, "operation": "delete_file"}

        assert operation.diff is not None  # for type checkers
        if operation.type == "create_file":
            new_content = apply_diff("", operation.diff, mode="create")
            self.backend.write_file(operation.path, new_content, overwrite=False)
            return {"path": operation.path, "operation": "create_file"}

        if operation.type == "update_file":
            current = self.backend.read_file(operation.path)
            new_content = apply_diff(current, operation.diff, mode="default")
            self.backend.write_file(operation.path, new_content, overwrite=True)
            return {"path": operation.path, "operation": "update_file"}

        raise ValueError(f"Unsupported patch operation: {operation.type}")

    def _filesystem_tool(self, allowed_commands: set[FsCommand], **kwargs: Any) -> str:
        params = FilesystemParams.model_validate(kwargs)

        try:
            if params.command not in allowed_commands:
                raise ValueError(
                    f"The '{params.command}' command is disabled for this tool instance"
                )
            if params.command == "read_file":
                if not params.path:
                    raise ValueError("path is required for read_file")
                result = self._handle_read(
                    params.path, params.start_line, params.end_line
                )
            elif params.command == "write_file":
                if params.path is None or params.content is None:
                    raise ValueError("path and content are required for write_file")
                result = self._handle_write(params.path, params.content, params.mode)
            elif params.command == "delete_path":
                if not params.path:
                    raise ValueError("path is required for delete_path")
                result = self._handle_delete(params.path)
            elif params.command == "list_dir":
                result = self._handle_list(params.path, params.recursive)
            elif params.command == "grep":
                if not params.pattern:
                    raise ValueError("pattern is required for grep")
                result = self._handle_grep(
                    params.pattern, params.path, params.max_results
                )
            elif params.command == "apply_patch":
                if params.operation is None:
                    raise ValueError("operation is required for apply_patch")
                result = self._handle_apply_patch(params.operation)
            else:
                raise ValueError(f"Unknown command: {params.command}")
            return json.dumps({"ok": True, "result": result}, indent=2)
        except Exception as exc:
            return json.dumps(
                {"ok": False, "error": type(exc).__name__, "message": str(exc)},
                indent=2,
            )

    def get_tools(self, *, exclude: Iterable[FsCommand] | None = None) -> list[Tool]:
        exclude_set = set(exclude or [])
        unknown = exclude_set.difference(ALL_COMMANDS)
        if unknown:
            raise ValueError(f"Unknown commands in exclude list: {sorted(unknown)}")

        allowed = tuple(cmd for cmd in ALL_COMMANDS if cmd not in exclude_set)
        if not allowed:
            raise ValueError("Cannot exclude every filesystem command")

        cache_key = allowed
        if cache_key in self._tool_cache:
            return self._tool_cache[cache_key]

        allowed_set = set(allowed)
        schema = FilesystemParams.model_json_schema(ref_template="#/$defs/{model}")
        if (
            "properties" in schema
            and "command" in schema["properties"]
            and isinstance(schema["properties"]["command"], dict)
        ):
            schema["properties"]["command"]["enum"] = list(allowed)

        tool = Tool(
            name=self.tool_name,
            description=FS_DESCRIPTION,
            parameters=schema.get("properties", {}),
            required=schema.get("required", []),
            definitions=schema.get("$defs"),
            run=partial(self._filesystem_tool, allowed_set),  # type: ignore
        )

        self._tool_cache[cache_key] = [tool]
        return [tool]

    def dump(
        self,
        destination: str | os.PathLike[str],
        *,
        as_zip: bool = False,
    ) -> list[str]:
        """
        Copy the virtual workspace to the given filesystem path.

        Args:
            destination: Path to write to. If as_zip=True, this should be a .zip file path.
            as_zip: If True, write as a zip archive instead of a directory.

        Returns:
            List of file paths that were written.
        """
        entries = self.backend.list_dir(".", recursive=True)
        written: list[str] = []

        if as_zip:
            target_path = Path(destination)
            target_path.parent.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(target_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for entry in entries:
                    if entry.get("type") != "file":
                        continue
                    rel_path = entry["path"]
                    content = self.backend.read_file(rel_path)
                    zf.writestr(rel_path, content)
                    written.append(rel_path)
        else:
            target_root = Path(destination)
            if target_root.exists() and not target_root.is_dir():
                raise NotADirectoryError(f"{target_root} exists and is not a directory")
            target_root.mkdir(parents=True, exist_ok=True)

            for entry in entries:
                if entry.get("type") != "file":
                    continue
                rel_path = entry["path"]
                destination_path = target_root.joinpath(*rel_path.split("/"))
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                destination_path.write_text(self.backend.read_file(rel_path))
                written.append(rel_path)

        return sorted(written)

    @classmethod
    def from_dir(
        cls,
        source: str | os.PathLike[str],
        *,
        max_files: int = 100,
        tool_name: str = "filesystem",
    ) -> "FilesystemManager":
        """
        Create a FilesystemManager pre-populated with files from a directory.

        Args:
            source: Path to the directory to load files from.
            max_files: Maximum number of files to load (default 100).
            tool_name: Name for the filesystem tool.

        Returns:
            A new FilesystemManager with the files loaded into memory.

        Raises:
            ValueError: If more than max_files files are found.
            NotADirectoryError: If source is not a directory.
        """
        source_path = Path(source)
        if not source_path.is_dir():
            raise NotADirectoryError(f"{source_path} is not a directory")

        files: dict[str, str] = {}
        file_count = 0

        for file_path in source_path.rglob("*"):
            if not file_path.is_file():
                continue

            file_count += 1
            if file_count > max_files:
                raise ValueError(
                    f"Directory contains more than {max_files} files. "
                    f"Increase max_files or use a smaller directory."
                )

            rel_path = file_path.relative_to(source_path).as_posix()

            # Try to read as text, skip binary files
            try:
                content = file_path.read_text()
                files[rel_path] = content
            except UnicodeDecodeError:
                # Skip binary files
                continue

        backend = InMemoryWorkspaceBackend(files)
        return cls(backend=backend, tool_name=tool_name)

    @classmethod
    def from_zip(
        cls,
        source: str | os.PathLike[str] | io.BytesIO,
        *,
        max_files: int = 100,
        tool_name: str = "filesystem",
    ) -> "FilesystemManager":
        """
        Create a FilesystemManager pre-populated with files from a zip archive.

        Args:
            source: Path to the zip file, or a BytesIO containing zip data.
            max_files: Maximum number of files to load (default 100).
            tool_name: Name for the filesystem tool.

        Returns:
            A new FilesystemManager with the files loaded into memory.

        Raises:
            ValueError: If more than max_files files are found.
            zipfile.BadZipFile: If the source is not a valid zip file.
        """
        files: dict[str, str] = {}
        file_count = 0

        with zipfile.ZipFile(source, "r") as zf:
            for info in zf.infolist():
                # Skip directories
                if info.is_dir():
                    continue

                file_count += 1
                if file_count > max_files:
                    raise ValueError(
                        f"Zip archive contains more than {max_files} files. "
                        f"Increase max_files or use a smaller archive."
                    )

                # Normalize path (remove leading slashes, handle Windows paths)
                rel_path = info.filename.lstrip("/").replace("\\", "/")
                if not rel_path:
                    continue

                # Try to read as text, skip binary files
                try:
                    content = zf.read(info.filename).decode("utf-8")
                    files[rel_path] = content
                except UnicodeDecodeError:
                    # Skip binary files
                    continue

        backend = InMemoryWorkspaceBackend(files)
        return cls(backend=backend, tool_name=tool_name)


ApplyDiffMode = Literal["default", "create"]


@dataclass
class Chunk:
    orig_index: int
    del_lines: list[str]
    ins_lines: list[str]


@dataclass
class ParserState:
    lines: list[str]
    index: int = 0
    fuzz: int = 0


@dataclass
class ParsedUpdateDiff:
    chunks: list[Chunk]
    fuzz: int


@dataclass
class ReadSectionResult:
    next_context: list[str]
    section_chunks: list[Chunk]
    end_index: int
    eof: bool


END_PATCH = "*** End Patch"
END_FILE = "*** End of File"
SECTION_TERMINATORS = [
    END_PATCH,
    "*** Update File:",
    "*** Delete File:",
    "*** Add File:",
]
END_SECTION_MARKERS = [*SECTION_TERMINATORS, END_FILE]


def apply_diff(input_text: str, diff: str, mode: ApplyDiffMode = "default") -> str:
    """Apply a V4A diff to the provided text."""

    diff_lines = _normalize_diff_lines(diff)
    if mode == "create":
        return _parse_create_diff(diff_lines)

    parsed = _parse_update_diff(diff_lines, input_text)
    return _apply_chunks(input_text, parsed.chunks)


def _normalize_diff_lines(diff: str) -> list[str]:
    lines = [line.rstrip("\r") for line in re.split(r"\r?\n", diff)]
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _is_done(state: ParserState, prefixes: Sequence[str]) -> bool:
    if state.index >= len(state.lines):
        return True
    if any(state.lines[state.index].startswith(prefix) for prefix in prefixes):
        return True
    return False


def _read_str(state: ParserState, prefix: str) -> str:
    if state.index >= len(state.lines):
        return ""
    current = state.lines[state.index]
    if current.startswith(prefix):
        state.index += 1
        return current[len(prefix) :]
    return ""


def _parse_create_diff(lines: list[str]) -> str:
    parser = ParserState(lines=[*lines, END_PATCH])
    output: list[str] = []

    while not _is_done(parser, SECTION_TERMINATORS):
        if parser.index >= len(parser.lines):
            break
        line = parser.lines[parser.index]
        parser.index += 1
        if not line.startswith("+"):
            raise ValueError(f"Invalid Add File Line: {line}")
        output.append(line[1:])

    return "\n".join(output)


def _parse_update_diff(lines: list[str], input_text: str) -> ParsedUpdateDiff:
    parser = ParserState(lines=[*lines, END_PATCH])
    input_lines = input_text.split("\n")
    chunks: list[Chunk] = []
    cursor = 0

    while not _is_done(parser, END_SECTION_MARKERS):
        anchor = _read_str(parser, "@@ ")
        has_bare_anchor = (
            anchor == ""
            and parser.index < len(parser.lines)
            and parser.lines[parser.index] == "@@"
        )
        if has_bare_anchor:
            parser.index += 1

        if not (anchor or has_bare_anchor or cursor == 0):
            current_line = (
                parser.lines[parser.index] if parser.index < len(parser.lines) else ""
            )
            raise ValueError(f"Invalid Line:\n{current_line}")

        if anchor.strip():
            cursor = _advance_cursor_to_anchor(anchor, input_lines, cursor, parser)

        section = _read_section(parser.lines, parser.index)
        find_result = _find_context(
            input_lines, section.next_context, cursor, section.eof
        )
        if find_result.new_index == -1:
            ctx_text = "\n".join(section.next_context)
            if section.eof:
                raise ValueError(f"Invalid EOF Context {cursor}:\n{ctx_text}")
            raise ValueError(f"Invalid Context {cursor}:\n{ctx_text}")

        cursor = find_result.new_index + len(section.next_context)
        parser.fuzz += find_result.fuzz
        parser.index = section.end_index

        for ch in section.section_chunks:
            chunks.append(
                Chunk(
                    orig_index=ch.orig_index + find_result.new_index,
                    del_lines=list(ch.del_lines),
                    ins_lines=list(ch.ins_lines),
                )
            )

    return ParsedUpdateDiff(chunks=chunks, fuzz=parser.fuzz)


def _advance_cursor_to_anchor(
    anchor: str,
    input_lines: list[str],
    cursor: int,
    parser: ParserState,
) -> int:
    found = False

    if not any(line == anchor for line in input_lines[:cursor]):
        for i in range(cursor, len(input_lines)):
            if input_lines[i] == anchor:
                cursor = i + 1
                found = True
                break

    if not found and not any(
        line.strip() == anchor.strip() for line in input_lines[:cursor]
    ):
        for i in range(cursor, len(input_lines)):
            if input_lines[i].strip() == anchor.strip():
                cursor = i + 1
                parser.fuzz += 1
                found = True
                break

    return cursor


def _read_section(lines: list[str], start_index: int) -> ReadSectionResult:
    context: list[str] = []
    del_lines: list[str] = []
    ins_lines: list[str] = []
    section_chunks: list[Chunk] = []
    mode: Literal["keep", "add", "delete"] = "keep"
    index = start_index
    orig_index = index

    while index < len(lines):
        raw = lines[index]
        if (
            raw.startswith("@@")
            or raw.startswith(END_PATCH)
            or raw.startswith("*** Update File:")
            or raw.startswith("*** Delete File:")
            or raw.startswith("*** Add File:")
            or raw.startswith(END_FILE)
        ):
            break
        if raw == "***":
            break
        if raw.startswith("***"):
            raise ValueError(f"Invalid Line: {raw}")

        index += 1
        last_mode = mode
        line = raw if raw else " "
        prefix = line[0]
        if prefix == "+":
            mode = "add"
        elif prefix == "-":
            mode = "delete"
        elif prefix == " ":
            mode = "keep"
        else:
            raise ValueError(f"Invalid Line: {line}")

        line_content = line[1:]
        switching_to_context = mode == "keep" and last_mode != mode
        if switching_to_context and (del_lines or ins_lines):
            section_chunks.append(
                Chunk(
                    orig_index=len(context) - len(del_lines),
                    del_lines=list(del_lines),
                    ins_lines=list(ins_lines),
                )
            )
            del_lines = []
            ins_lines = []

        if mode == "delete":
            del_lines.append(line_content)
            context.append(line_content)
        elif mode == "add":
            ins_lines.append(line_content)
        else:
            context.append(line_content)

    if del_lines or ins_lines:
        section_chunks.append(
            Chunk(
                orig_index=len(context) - len(del_lines),
                del_lines=list(del_lines),
                ins_lines=list(ins_lines),
            )
        )

    if index < len(lines) and lines[index] == END_FILE:
        return ReadSectionResult(context, section_chunks, index + 1, True)

    if index == orig_index:
        next_line = lines[index] if index < len(lines) else ""
        raise ValueError(f"Nothing in this section - index={index} {next_line}")

    return ReadSectionResult(context, section_chunks, index, False)


@dataclass
class ContextMatch:
    new_index: int
    fuzz: int


def _find_context(
    lines: list[str], context: list[str], start: int, eof: bool
) -> ContextMatch:
    if eof:
        end_start = max(0, len(lines) - len(context))
        end_match = _find_context_core(lines, context, end_start)
        if end_match.new_index != -1:
            return end_match
        fallback = _find_context_core(lines, context, start)
        return ContextMatch(new_index=fallback.new_index, fuzz=fallback.fuzz + 10000)
    return _find_context_core(lines, context, start)


def _find_context_core(
    lines: list[str], context: list[str], start: int
) -> ContextMatch:
    if not context:
        return ContextMatch(new_index=start, fuzz=0)

    for i in range(start, len(lines)):
        if _equals_slice(lines, context, i, lambda value: value):
            return ContextMatch(new_index=i, fuzz=0)
    for i in range(start, len(lines)):
        if _equals_slice(lines, context, i, lambda value: value.rstrip()):
            return ContextMatch(new_index=i, fuzz=1)
    for i in range(start, len(lines)):
        if _equals_slice(lines, context, i, lambda value: value.strip()):
            return ContextMatch(new_index=i, fuzz=100)

    return ContextMatch(new_index=-1, fuzz=0)


def _equals_slice(
    source: list[str], target: list[str], start: int, map_fn: Callable[[str], str]
) -> bool:
    if start + len(target) > len(source):
        return False
    for offset, target_value in enumerate(target):
        if map_fn(source[start + offset]) != map_fn(target_value):
            return False
    return True


def _apply_chunks(input_text: str, chunks: list[Chunk]) -> str:
    orig_lines = input_text.split("\n")
    dest_lines: list[str] = []
    cursor = 0

    for chunk in chunks:
        if chunk.orig_index > len(orig_lines):
            raise ValueError(
                f"apply_diff: chunk.origIndex {chunk.orig_index} > input length {len(orig_lines)}"
            )
        if cursor > chunk.orig_index:
            raise ValueError(
                f"apply_diff: overlapping chunk at {chunk.orig_index} (cursor {cursor})"
            )

        dest_lines.extend(orig_lines[cursor : chunk.orig_index])
        cursor = chunk.orig_index

        if chunk.ins_lines:
            dest_lines.extend(chunk.ins_lines)

        cursor += len(chunk.del_lines)

    dest_lines.extend(orig_lines[cursor:])
    return "\n".join(dest_lines)


# S3 filesystem description
S3_FS_DESCRIPTION = """Interact with a remote S3-backed filesystem that supports safe concurrent access.

This filesystem is backed by Amazon S3 with optimistic concurrency control, meaning multiple
agents can safely read and write to the same workspace without conflicts. If a write conflict
occurs (another agent modified the file), the operation will automatically retry.

Paths are always relative to the workspace root and use forward slashes. Use this tool to:
- inspect files with optional line ranges
- create, overwrite, or append to files (with automatic conflict resolution)
- delete files or folders from the workspace
- list directory contents
- search for text across the workspace using regular expressions

This filesystem is safe for distributed use - conflicts are automatically detected and resolved."""


@dataclass
class S3FileMetadata:
    """Metadata for a file stored in S3."""

    key: str
    etag: str
    size: int
    last_modified: float


@dataclass
class ConflictError(Exception):
    """Raised when a write conflict occurs due to concurrent modification."""

    key: str
    expected_etag: str | None
    actual_etag: str | None
    message: str = ""

    def __str__(self) -> str:
        if self.message:
            return self.message
        return f"Conflict writing {self.key}: expected ETag {self.expected_etag}, got {self.actual_etag}"


@dataclass
class RetryConfig:
    """Configuration for retry behavior on conflicts."""

    max_retries: int = 5
    base_delay: float = 0.1  # seconds
    max_delay: float = 5.0  # seconds
    jitter: float = 0.1  # random jitter factor


class S3WorkspaceBackend:
    """
    S3 backend with optimistic concurrency control using conditional writes.

    Uses:
    - If-None-Match: * for create-if-not-exists operations
    - If-Match: <etag> for update-only-if-unchanged operations
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        s3_client: Any | None = None,
        retry_config: RetryConfig | None = None,
    ):
        self.bucket = bucket
        self.prefix = prefix.rstrip("/") + "/" if prefix else ""
        self._client = s3_client
        self.retry_config = retry_config or RetryConfig()
        # Local cache of ETags for optimistic locking
        self._etag_cache: dict[str, str] = {}

    @property
    def client(self) -> Any:
        """Lazy initialization of S3 client."""
        if self._client is None:
            import boto3

            self._client = boto3.client("s3")
        return self._client

    def _full_key(self, path: str) -> str:
        """Convert a normalized path to a full S3 key."""
        if path == ".":
            return self.prefix.rstrip("/") if self.prefix else ""
        return f"{self.prefix}{path}"

    def _strip_prefix(self, key: str) -> str:
        """Strip the prefix from an S3 key to get the relative path."""
        if self.prefix and key.startswith(self.prefix):
            return key[len(self.prefix) :]
        return key

    def _get_with_etag(self, path: str) -> tuple[str, str]:
        """Read a file and return (content, etag)."""
        key = self._full_key(path)
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            content = response["Body"].read().decode("utf-8")
            etag = response["ETag"].strip('"')
            self._etag_cache[key] = etag
            return content, etag
        except self.client.exceptions.NoSuchKey:
            raise FileNotFoundError(f"{path} does not exist")

    def _put_with_condition(
        self,
        path: str,
        content: str,
        *,
        if_none_match: bool = False,
        if_match: str | None = None,
    ) -> str:
        """
        Write a file with conditional headers.

        Args:
            path: The file path
            content: The content to write
            if_none_match: If True, only write if file doesn't exist
            if_match: If provided, only write if ETag matches

        Returns:
            The new ETag of the written object

        Raises:
            ConflictError: If the condition fails
            FileExistsError: If if_none_match=True and file exists
        """
        key = self._full_key(path)
        kwargs: dict[str, Any] = {
            "Bucket": self.bucket,
            "Key": key,
            "Body": content.encode("utf-8"),
            "ContentType": "text/plain; charset=utf-8",
        }

        if if_none_match:
            kwargs["IfNoneMatch"] = "*"
        elif if_match:
            kwargs["IfMatch"] = (
                f'"{if_match}"' if not if_match.startswith('"') else if_match
            )

        try:
            response = self.client.put_object(**kwargs)
            new_etag = response["ETag"].strip('"')
            self._etag_cache[key] = new_etag
            return new_etag
        except self.client.exceptions.ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "PreconditionFailed":
                if if_none_match:
                    raise FileExistsError(f"{path} already exists")
                raise ConflictError(
                    key=key,
                    expected_etag=if_match,
                    actual_etag=None,
                    message=f"File {path} was modified by another process",
                )
            raise

    def _retry_with_backoff(
        self,
        operation: str,
        path: str,
        func: Any,
    ) -> Any:
        """Execute a function with retry and exponential backoff on conflicts."""
        config = self.retry_config
        last_error: Exception | None = None

        for attempt in range(config.max_retries + 1):
            try:
                return func()
            except ConflictError as e:
                last_error = e
                if attempt >= config.max_retries:
                    break
                # Calculate backoff with jitter
                delay = min(
                    config.base_delay * (2**attempt),
                    config.max_delay,
                )
                jitter = delay * config.jitter * random.random()
                time.sleep(delay + jitter)
                # Clear cached ETag to force fresh read
                key = self._full_key(path)
                self._etag_cache.pop(key, None)

        raise last_error or RuntimeError(f"Retry failed for {operation} on {path}")

    def read_file(self, path: str) -> str:
        """Read a file from S3."""
        key = _normalize_path(path)
        content, _ = self._get_with_etag(key)
        return content

    def write_file(self, path: str, content: str, *, overwrite: bool) -> None:
        """
        Write a file to S3 with optimistic locking.

        If overwrite=False, uses If-None-Match: * to ensure create-only.
        If overwrite=True, uses If-Match with cached ETag if available,
        otherwise does unconditional write.
        """
        key = _normalize_path(path)
        s3_key = self._full_key(key)

        if not overwrite:
            # Create-if-not-exists
            self._put_with_condition(key, content, if_none_match=True)
        else:
            # Check if we have a cached ETag for optimistic locking
            cached_etag = self._etag_cache.get(s3_key)
            if cached_etag:
                # Use optimistic locking with retry
                def _do_write():
                    nonlocal cached_etag
                    try:
                        self._put_with_condition(key, content, if_match=cached_etag)
                    except ConflictError:
                        # Refresh ETag and retry
                        _, cached_etag = self._get_with_etag(key)
                        raise

                self._retry_with_backoff("write_file", key, _do_write)
            else:
                # No cached ETag - try to get current state first for safety
                try:
                    _, etag = self._get_with_etag(key)
                    self._put_with_condition(key, content, if_match=etag)
                except FileNotFoundError:
                    # File doesn't exist, create it
                    self._put_with_condition(key, content, if_none_match=True)

    def append_file(self, path: str, content: str) -> None:
        """
        Append to a file with optimistic locking.

        Uses read-modify-write with If-Match for consistency.
        """
        key = _normalize_path(path)

        def _do_append():
            try:
                current, etag = self._get_with_etag(key)
                new_content = current + content
            except FileNotFoundError:
                # File doesn't exist, create it
                self._put_with_condition(key, content, if_none_match=True)
                return

            self._put_with_condition(key, new_content, if_match=etag)

        self._retry_with_backoff("append_file", key, _do_append)

    def delete_path(self, path: str) -> None:
        """Delete a file or directory from S3."""
        key = _normalize_path(path, allow_root=True)

        if key == ".":
            # Delete all files under the prefix
            paginator = self.client.get_paginator("list_objects_v2")
            prefix = self.prefix if self.prefix else ""
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    self.client.delete_object(Bucket=self.bucket, Key=obj["Key"])
                    self._etag_cache.pop(obj["Key"], None)
            return

        s3_key = self._full_key(key)

        # Try to delete as a single file first
        try:
            self.client.head_object(Bucket=self.bucket, Key=s3_key)
            self.client.delete_object(Bucket=self.bucket, Key=s3_key)
            self._etag_cache.pop(s3_key, None)
            return
        except self.client.exceptions.ClientError:
            pass

        # Try as a directory prefix
        prefix = f"{s3_key}/"
        paginator = self.client.get_paginator("list_objects_v2")
        deleted_any = False

        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                self.client.delete_object(Bucket=self.bucket, Key=obj["Key"])
                self._etag_cache.pop(obj["Key"], None)
                deleted_any = True

        if not deleted_any:
            raise FileNotFoundError(f"{key} does not exist")

    def list_dir(self, path: str, recursive: bool) -> list[dict[str, Any]]:
        """List directory contents from S3."""
        key = _normalize_path(path, allow_root=True)

        if key == ".":
            prefix = self.prefix
        else:
            prefix = f"{self._full_key(key)}/"

        entries: list[dict[str, Any]] = []
        seen_dirs: set[str] = set()

        paginator = self.client.get_paginator("list_objects_v2")
        kwargs: dict[str, Any] = {"Bucket": self.bucket, "Prefix": prefix}

        if not recursive:
            kwargs["Delimiter"] = "/"

        for page in paginator.paginate(**kwargs):
            # Handle common prefixes (directories) in non-recursive mode
            for common_prefix in page.get("CommonPrefixes", []):
                dir_key = common_prefix["Prefix"].rstrip("/")
                rel_path = self._strip_prefix(dir_key)
                if rel_path and rel_path not in seen_dirs:
                    entries.append(
                        {"path": rel_path, "type": "directory", "size": None}
                    )
                    seen_dirs.add(rel_path)

            # Handle files
            for obj in page.get("Contents", []):
                obj_key = obj["Key"]
                rel_path = self._strip_prefix(obj_key)

                # Skip the prefix itself if it's an empty marker
                if not rel_path or rel_path.endswith("/"):
                    continue

                if recursive:
                    entries.append(self._format_file_entry(rel_path, obj))
                else:
                    # Check if this is a direct child
                    remainder = rel_path
                    if key != ".":
                        # Remove the directory prefix to get relative path
                        dir_prefix = key + "/"
                        if rel_path.startswith(dir_prefix):
                            remainder = rel_path[len(dir_prefix) :]
                        else:
                            continue

                    if "/" not in remainder:
                        entries.append(self._format_file_entry(rel_path, obj))

        # Sort by path for consistent ordering
        entries.sort(key=lambda e: e["path"])

        if not entries and key != ".":
            # Check if the path itself exists as a file
            try:
                s3_key = self._full_key(key)
                response = self.client.head_object(Bucket=self.bucket, Key=s3_key)
                return [
                    {
                        "path": key,
                        "type": "file",
                        "size": response["ContentLength"],
                        "etag": response["ETag"].strip('"'),
                    }
                ]
            except self.client.exceptions.ClientError:
                raise FileNotFoundError(f"{key} does not exist")

        return entries

    def grep(self, pattern: str, path: str | None, limit: int) -> list[dict[str, Any]]:
        """Search for pattern in files."""
        regex = re.compile(pattern)
        key = _normalize_path(path, allow_root=True) if path is not None else "."

        if key == ".":
            prefix = self.prefix
        else:
            prefix = f"{self._full_key(key)}/"

        matches: list[dict[str, Any]] = []
        paginator = self.client.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                if len(matches) >= limit:
                    return matches

                obj_key = obj["Key"]
                rel_path = self._strip_prefix(obj_key)

                if not rel_path or rel_path.endswith("/"):
                    continue

                try:
                    response = self.client.get_object(Bucket=self.bucket, Key=obj_key)
                    content = response["Body"].read().decode("utf-8")

                    for line_no, line in enumerate(content.splitlines(), start=1):
                        if regex.search(line):
                            matches.append(
                                {
                                    "path": rel_path,
                                    "line": line_no,
                                    "text": line.strip(),
                                }
                            )
                            if len(matches) >= limit:
                                return matches
                except Exception:
                    # Skip files that can't be read as text
                    continue

        return matches

    def _format_file_entry(self, path: str, obj: dict[str, Any]) -> dict[str, Any]:
        """Format a file entry from S3 object metadata."""
        return {
            "path": path,
            "type": "file",
            "size": obj["Size"],
            "etag": obj["ETag"].strip('"'),
        }

    def get_file_etag(self, path: str) -> str | None:
        """Get the cached ETag for a file, or fetch it if not cached."""
        key = _normalize_path(path)
        s3_key = self._full_key(key)

        if s3_key in self._etag_cache:
            return self._etag_cache[s3_key]

        try:
            response = self.client.head_object(Bucket=self.bucket, Key=s3_key)
            etag = response["ETag"].strip('"')
            self._etag_cache[s3_key] = etag
            return etag
        except self.client.exceptions.ClientError:
            return None


# Command types for the S3 filesystem tool
S3FsCommand = Literal[
    "read_file",
    "write_file",
    "delete_path",
    "list_dir",
    "grep",
]

ALL_S3_COMMANDS: tuple[S3FsCommand, ...] = (
    "read_file",
    "write_file",
    "delete_path",
    "list_dir",
    "grep",
)


class S3FilesystemParams(BaseModel):
    """Schema describing S3 filesystem tool calls."""

    command: S3FsCommand = Field(
        description="Filesystem operation to perform (read_file, write_file, delete_path, list_dir, grep)"
    )
    path: Optional[str] = Field(
        default=None,
        description="Path to operate on, relative to workspace root. Use '.' for the root directory.",
    )
    start_line: Optional[int] = Field(
        default=None,
        description="1-based inclusive start line when reading a file. Leave unset to read from the beginning.",
        ge=1,
    )
    end_line: Optional[int] = Field(
        default=None,
        description="1-based inclusive end line when reading a file. Leave unset to read through the end.",
        ge=1,
    )
    content: Optional[str] = Field(
        default=None,
        description="Content to write when using write_file.",
    )
    mode: Optional[Literal["overwrite", "append", "create_if_missing"]] = Field(
        default="overwrite",
        description="How to write content. Overwrite replaces the file, append adds to the end, create_if_missing leaves existing files untouched.",
    )
    recursive: Optional[bool] = Field(
        default=None,
        description="When listing directories, set to true to recurse.",
    )
    pattern: Optional[str] = Field(
        default=None,
        description="Regular expression pattern to search for when using grep.",
    )
    max_results: Optional[int] = Field(
        default=50,
        description="Maximum number of grep matches to return.",
        ge=1,
    )


class S3FilesystemManager:
    """
    S3-backed filesystem manager with optimistic concurrency control.

    Uses S3 conditional writes (If-None-Match and If-Match) for safe distributed
    operations, allowing multiple AI agents to share filesystem state.

    Example:
        manager = S3FilesystemManager(
            bucket="my-ai-workspace",
            prefix="agent-123/",
        )
        tools = manager.get_tools()
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        s3_client: Any | None = None,
        retry_config: RetryConfig | None = None,
        tool_name: str = "s3_filesystem",
    ):
        """
        Initialize the S3 filesystem manager.

        Args:
            bucket: The S3 bucket name
            prefix: Optional prefix for all keys (like a workspace directory)
            s3_client: Optional pre-configured S3 client
            retry_config: Configuration for retry behavior on conflicts
            tool_name: Name for the tool (default: "s3_filesystem")
        """
        self.backend = S3WorkspaceBackend(
            bucket=bucket,
            prefix=prefix,
            s3_client=s3_client,
            retry_config=retry_config,
        )
        self.tool_name = tool_name
        self._tool_cache: dict[tuple[str, ...], list[Tool]] = {}

    def _handle_read(
        self, path: str, start_line: Optional[int], end_line: Optional[int]
    ) -> dict[str, Any]:
        content = self.backend.read_file(path)
        total_lines = len(content.splitlines()) or (0 if content == "" else 1)
        start = start_line or 1
        end = end_line or total_lines
        if end < start:
            if not (total_lines == 0 and end_line is None and start == 1):
                raise ValueError("end_line must be greater than or equal to start_line")

        if start == 1 and end >= total_lines:
            snippet = content
        else:
            lines = content.splitlines()
            snippet = "\n".join(lines[start - 1 : end])

        # Include ETag in response for transparency
        etag = self.backend.get_file_etag(path)

        return {
            "path": path,
            "start_line": start,
            "end_line": end,
            "content": snippet,
            "total_lines": total_lines,
            "character_count": len(content),
            "etag": etag,
        }

    def _handle_write(
        self, path: str, content: str, mode: Optional[str]
    ) -> dict[str, Any]:
        write_mode = mode or "overwrite"
        if write_mode == "overwrite":
            self.backend.write_file(path, content, overwrite=True)
        elif write_mode == "append":
            self.backend.append_file(path, content)
        elif write_mode == "create_if_missing":
            try:
                self.backend.write_file(path, content, overwrite=False)
            except FileExistsError:
                pass
        else:
            raise ValueError(f"Unsupported write mode: {write_mode}")

        # Get new ETag after write
        etag = self.backend.get_file_etag(path)

        return {"path": path, "status": "ok", "mode": write_mode, "etag": etag}

    def _handle_delete(self, path: str) -> dict[str, Any]:
        self.backend.delete_path(path)
        return {"path": path, "status": "ok"}

    def _handle_list(
        self, path: Optional[str], recursive: Optional[bool]
    ) -> dict[str, Any]:
        listing = self.backend.list_dir(path or ".", recursive=bool(recursive))
        return {"path": path or ".", "recursive": bool(recursive), "entries": listing}

    def _handle_grep(
        self, pattern: str, path: Optional[str], limit: Optional[int]
    ) -> dict[str, Any]:
        max_results = limit or 50
        matches = self.backend.grep(pattern, path=path, limit=max_results)
        return {
            "pattern": pattern,
            "path": path,
            "max_results": max_results,
            "matches": matches,
        }

    def _filesystem_tool(self, allowed_commands: set[str], **kwargs: Any) -> str:
        params = S3FilesystemParams.model_validate(kwargs)

        try:
            if params.command not in allowed_commands:
                raise ValueError(
                    f"The '{params.command}' command is disabled for this tool instance"
                )
            if params.command == "read_file":
                if not params.path:
                    raise ValueError("path is required for read_file")
                result = self._handle_read(
                    params.path, params.start_line, params.end_line
                )
            elif params.command == "write_file":
                if params.path is None or params.content is None:
                    raise ValueError("path and content are required for write_file")
                result = self._handle_write(params.path, params.content, params.mode)
            elif params.command == "delete_path":
                if not params.path:
                    raise ValueError("path is required for delete_path")
                result = self._handle_delete(params.path)
            elif params.command == "list_dir":
                result = self._handle_list(params.path, params.recursive)
            elif params.command == "grep":
                if not params.pattern:
                    raise ValueError("pattern is required for grep")
                result = self._handle_grep(
                    params.pattern, params.path, params.max_results
                )
            else:
                raise ValueError(f"Unknown command: {params.command}")
            return json.dumps({"ok": True, "result": result}, indent=2)
        except Exception as exc:
            return json.dumps(
                {"ok": False, "error": type(exc).__name__, "message": str(exc)},
                indent=2,
            )

    def get_tools(self, *, exclude: Iterable[S3FsCommand] | None = None) -> list[Tool]:
        """
        Get the filesystem tools.

        Args:
            exclude: Optional list of commands to exclude from the tool

        Returns:
            List containing the S3 filesystem tool
        """
        exclude_set = set(exclude or [])
        unknown = exclude_set.difference(ALL_S3_COMMANDS)
        if unknown:
            raise ValueError(f"Unknown commands in exclude list: {sorted(unknown)}")

        allowed = tuple(cmd for cmd in ALL_S3_COMMANDS if cmd not in exclude_set)
        if not allowed:
            raise ValueError("Cannot exclude every filesystem command")

        cache_key = allowed
        if cache_key in self._tool_cache:
            return self._tool_cache[cache_key]

        allowed_set = {cmd for cmd in allowed}
        schema = S3FilesystemParams.model_json_schema(ref_template="#/$defs/{model}")
        if (
            "properties" in schema
            and "command" in schema["properties"]
            and isinstance(schema["properties"]["command"], dict)
        ):
            schema["properties"]["command"]["enum"] = list(allowed)

        tool = Tool(
            name=self.tool_name,
            description=S3_FS_DESCRIPTION,
            parameters=schema.get("properties", {}),
            required=schema.get("required", []),
            definitions=schema.get("$defs"),
            run=partial(self._filesystem_tool, allowed_set),
        )

        self._tool_cache[cache_key] = [tool]
        return [tool]


__all__ = [
    "FilesystemManager",
    "FilesystemParams",
    "InMemoryWorkspaceBackend",
    "WorkspaceBackend",
    "S3FilesystemManager",
    "S3FilesystemParams",
    "S3WorkspaceBackend",
    "ConflictError",
    "RetryConfig",
]


description = """
S3-backed remote filesystem tool with optimistic concurrency control.

Uses S3 conditional writes (If-None-Match and If-Match) for safe distributed
operations, allowing multiple AI agents to share filesystem state without conflicts.

Features:
- If-None-Match: * -> Create-if-not-exists (distributed locks, idempotent writes)
- If-Match: <etag> -> Update-only-if-unchanged (optimistic locking)
- Automatic retry with exponential backoff on conflicts
- ETag tracking for all file operations

Example:
    from lm_deluge.tool.prefab.s3_filesystem import S3FilesystemManager

    manager = S3FilesystemManager(
        bucket="my-ai-workspace",
        prefix="agent-123/",  # Optional: isolate agent's workspace
    )

    # Get tools for the agent
    tools = manager.get_tools()

    # The filesystem operations are now safe for concurrent access
"""
