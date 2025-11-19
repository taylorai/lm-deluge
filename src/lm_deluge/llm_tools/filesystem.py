from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Protocol

from pydantic import BaseModel, Field

from ..tool import Tool

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

    def dump(self, destination: str | os.PathLike[str]) -> list[str]:
        """Copy the virtual workspace to the given filesystem directory."""
        target_root = Path(destination)
        if target_root.exists() and not target_root.is_dir():
            raise NotADirectoryError(f"{target_root} exists and is not a directory")
        target_root.mkdir(parents=True, exist_ok=True)

        entries = self.backend.list_dir(".", recursive=True)
        written: list[str] = []

        for entry in entries:
            if entry.get("type") != "file":
                continue
            rel_path = entry["path"]
            destination_path = target_root.joinpath(*rel_path.split("/"))
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            destination_path.write_text(self.backend.read_file(rel_path))
            written.append(rel_path)

        return sorted(written)


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


__all__ = [
    "FilesystemManager",
    "FilesystemParams",
    "InMemoryWorkspaceBackend",
    "WorkspaceBackend",
]
