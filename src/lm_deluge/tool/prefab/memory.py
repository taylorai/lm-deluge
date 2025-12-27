from __future__ import annotations

import json
import random
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence, TypeAlias

import yaml
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# from rapidfuzz import fuzz, process
from .. import Tool

MEMORY_DESCRIPTION = """Use this tool to search, read, and update a long-term "memory" that can be used across sessions, when previous messages are cleared. Whether and when to use memory depends on the situation—for complex tasks, it can store information about work so far, what needs to be done next, why you're doing what you're doing, etc. For personal conversations, it can be used to save "memories" that can be referenced later."""

MEMORY_WRITE = """

"""

MEMORY_READ = """

"""


class MemoryItem(BaseModel):
    """Structured representation of a single memory."""

    id: int
    description: str = Field(
        description='Short description ("preview") of the memory (1 sentence)'
    )
    content: str = Field(
        description="Full content of the memory. May use Markdown for formatting."
    )


class MemoryItemDict(TypedDict):
    id: int
    description: str
    content: str


MemoryLike: TypeAlias = MemoryItem | MemoryItemDict


class MemoryManager:
    """Stateful todo scratchpad that exposes read/write tools."""

    def __init__(
        self,
        memories: Sequence[MemoryLike] | dict[int, MemoryLike] | None = None,
        *,
        write_tool_name: str = "memwrite",
        read_tool_name: str = "memread",
        search_tool_name: str = "memsearch",
        update_tool_name: str = "memupdate",
        delete_tool_name: str = "memdelete",
    ):
        self.write_tool_name = write_tool_name
        self.read_tool_name = read_tool_name
        self.search_tool_name = search_tool_name
        self.update_tool_name = update_tool_name
        self.delete_tool_name = delete_tool_name
        self._memories: dict[int, MemoryItem] = {}
        self._tools: list[Tool] | None = None

        if memories:
            if isinstance(memories, dict):
                # Cast is needed because isinstance(memories, dict) narrows to dict[object, object]
                mem_dict: dict[int, MemoryLike] = memories  # type: ignore[assignment]
                self._memories = {k: self._coerce(v) for k, v in mem_dict.items()}
            else:
                coerced = [self._coerce(mem) for mem in memories]
                self._memories = {x.id: x for x in coerced}

    @classmethod
    def from_file(
        cls,
        file: str,
    ) -> MemoryManager:
        # file should be a json file
        with open(file) as f:
            memories = json.load(f)
        return cls(memories)

    def _coerce(self, mem: MemoryLike) -> MemoryItem:
        if isinstance(mem, MemoryItem):
            return mem
        if isinstance(mem, dict):
            return MemoryItem(**mem)
        raise TypeError("Memories must be MemoryItem instances or dicts")

    def _serialize(self) -> list[dict[str, Any]]:
        return [mem.model_dump() for mem in self._memories.values()]

    def to_file(self, file: str):
        mems = self._serialize()
        with open(file, "w") as f:
            f.write(json.dumps(mems))

    @staticmethod
    def _format_memory(mem: MemoryItem, include_content: bool = True) -> str:
        dumped = mem.model_dump()
        if not include_content:
            dumped.pop("content")
        return yaml.safe_dump(dumped)

    # helpers
    def _search(self, queries: list[str], limit: int = 5) -> list[MemoryItem]:
        hits = Counter()
        for q in queries:
            keywords = q.lower().split()
            for k in keywords:
                for mem_id, mem in self._memories.items():
                    if k in mem.description.lower() or k in mem.content.lower():
                        hits[mem_id] += 1

        top_k = hits.most_common(limit)

        return self._read([hit[0] for hit in top_k if hit[1] > 0])

    def _read(self, memory_ids: list[int]) -> list[MemoryItem]:
        return [
            mem
            for mem_id in memory_ids
            if mem_id is not None and (mem := self._memories.get(mem_id)) is not None
        ]

    def _add(self, description: str, content: str):
        new_id = max(self._memories) + 1 if self._memories else 1
        self._memories[new_id] = self._coerce(
            {"id": new_id, "description": description, "content": content}
        )
        return new_id

    def _update(self, mem_id: int, description: str, content: str):
        self._memories[mem_id].description = description
        self._memories[mem_id].content = content

    def _delete(self, mem_id: int):
        self._memories.pop(mem_id)

    def get_tools(self) -> list[Tool]:
        """Return Tool instances bound to this manager's state."""
        if self._tools is not None:
            return self._tools

        def search_tool(queries: list[str], limit: int = 5) -> str:
            """Search for memories using keyword search. Use as many queries as you want, the top results will be fused into one list. Search results include just id and description."""
            mems = self._search(queries, limit=limit)
            return "\n---\n".join(
                [self._format_memory(mem, include_content=False) for mem in mems]
            )

        def read_tool(mem_ids: list[int]) -> str:
            """Read the full contents of one or more memories."""
            mems = self._read(mem_ids)
            return "\n---\n".join(
                [self._format_memory(mem, include_content=True) for mem in mems]
            )

        def add_tool(description: str, content: str):
            """Add a new memory."""
            return self._add(description, content)

        def update_tool(mem_id: int, description: str, content: str) -> str:
            """Update a memory by ID. Must provide content and description, even if only changing one of them."""
            self._update(mem_id, description, content)

            return f"Memory {mem_id} updated successfully."

        def delete_tool(mem_id: int) -> str:
            """Delete a memory by ID."""
            self._delete(mem_id)
            return f"Memory {mem_id} deleted successfully."

        def _rename(tool: Tool, name: str) -> Tool:
            if tool.name == name:
                return tool
            return tool.model_copy(update={"name": name})

        self._tools = [
            _rename(Tool.from_function(search_tool), self.search_tool_name),
            _rename(Tool.from_function(read_tool), self.read_tool_name),
            _rename(Tool.from_function(add_tool), self.write_tool_name),
            _rename(Tool.from_function(update_tool), self.update_tool_name),
            _rename(Tool.from_function(delete_tool), self.delete_tool_name),
        ]
        return self._tools


@dataclass
class S3RetryConfig:
    """Configuration for retry behavior on conflicts."""

    max_retries: int = 5
    base_delay: float = 0.1
    max_delay: float = 5.0
    jitter: float = 0.1


class S3MemoryManager:
    """
    S3-backed memory manager with optimistic concurrency control.

    Same API as MemoryManager but persists to S3 with safe concurrent access
    using S3 conditional writes (If-Match with ETags).

    Example:
        manager = S3MemoryManager(
            bucket="my-ai-memories",
            key="agent-123/memories.json",
        )
        tools = manager.get_tools()
    """

    def __init__(
        self,
        bucket: str,
        key: str = "memories.json",
        s3_client: Any | None = None,
        retry_config: S3RetryConfig | None = None,
        *,
        write_tool_name: str = "memwrite",
        read_tool_name: str = "memread",
        search_tool_name: str = "memsearch",
        update_tool_name: str = "memupdate",
        delete_tool_name: str = "memdelete",
    ):
        self.bucket = bucket
        self.key = key
        self._client = s3_client
        self.retry_config = retry_config or S3RetryConfig()

        self.write_tool_name = write_tool_name
        self.read_tool_name = read_tool_name
        self.search_tool_name = search_tool_name
        self.update_tool_name = update_tool_name
        self.delete_tool_name = delete_tool_name

        self._tools: list[Tool] | None = None
        self._cached_etag: str | None = None

    @property
    def client(self):
        """Lazy initialization of S3 client."""
        if self._client is None:
            import boto3

            self._client = boto3.client("s3")
        return self._client

    def _load_memories(self) -> tuple[dict[int, MemoryItem], str | None]:
        """Load memories from S3, returning (memories_dict, etag)."""
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=self.key)
            etag = response["ETag"].strip('"')
            data = json.loads(response["Body"].read().decode("utf-8"))
            memories = {item["id"]: MemoryItem(**item) for item in data}
            return memories, etag
        except self.client.exceptions.NoSuchKey:
            return {}, None

    def _save_memories(
        self,
        memories: dict[int, MemoryItem],
        expected_etag: str | None,
    ) -> str:
        """
        Save memories to S3 with optimistic locking.

        Args:
            memories: The memories dict to save
            expected_etag: The ETag we expect (None for new file)

        Returns:
            The new ETag after saving

        Raises:
            ConflictError: If the file was modified by another process
        """
        data = [mem.model_dump() for mem in memories.values()]
        body = json.dumps(data, indent=2).encode("utf-8")

        kwargs: dict[str, Any] = {
            "Bucket": self.bucket,
            "Key": self.key,
            "Body": body,
            "ContentType": "application/json",
        }

        if expected_etag is None:
            # Creating new file - use If-None-Match
            kwargs["IfNoneMatch"] = "*"
        else:
            # Updating existing file - use If-Match
            kwargs["IfMatch"] = f'"{expected_etag}"'

        try:
            response = self.client.put_object(**kwargs)
            return response["ETag"].strip('"')
        except self.client.exceptions.ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            # Handle both PreconditionFailed and ConditionalRequestConflict
            if error_code in ("PreconditionFailed", "ConditionalRequestConflict"):
                raise S3MemoryConflictError(
                    "Memory file was modified by another process"
                ) from e
            raise

    def _retry_operation(self, operation_name: str, func):
        """Execute a function with retry on conflicts."""
        config = self.retry_config
        last_error: Exception | None = None

        for attempt in range(config.max_retries + 1):
            try:
                return func()
            except S3MemoryConflictError as e:
                last_error = e
                if attempt >= config.max_retries:
                    break
                delay = min(config.base_delay * (2**attempt), config.max_delay)
                jitter = delay * config.jitter * random.random()
                time.sleep(delay + jitter)

        raise last_error or RuntimeError(f"Retry failed for {operation_name}")

    @staticmethod
    def _format_memory(mem: MemoryItem, include_content: bool = True) -> str:
        dumped = mem.model_dump()
        if not include_content:
            dumped.pop("content")
        return yaml.safe_dump(dumped)

    def _search(self, queries: list[str], limit: int = 5) -> list[MemoryItem]:
        memories, _ = self._load_memories()
        hits: Counter[int] = Counter()

        for q in queries:
            keywords = q.lower().split()
            for k in keywords:
                for mem_id, mem in memories.items():
                    if k in mem.description.lower() or k in mem.content.lower():
                        hits[mem_id] += 1

        top_k = hits.most_common(limit)
        return [memories[hit[0]] for hit in top_k if hit[1] > 0]

    def _read(self, memory_ids: list[int]) -> list[MemoryItem]:
        memories, _ = self._load_memories()
        return [
            mem
            for mem_id in memory_ids
            if mem_id is not None and (mem := memories.get(mem_id)) is not None
        ]

    def _add(self, description: str, content: str) -> int:
        def do_add():
            memories, etag = self._load_memories()
            new_id = max(memories.keys()) + 1 if memories else 1
            memories[new_id] = MemoryItem(
                id=new_id, description=description, content=content
            )
            self._save_memories(memories, etag)
            return new_id

        return self._retry_operation("add", do_add)

    def _update(self, mem_id: int, description: str, content: str):
        def do_update():
            memories, etag = self._load_memories()
            if mem_id not in memories:
                raise KeyError(f"Memory {mem_id} not found")
            memories[mem_id].description = description
            memories[mem_id].content = content
            self._save_memories(memories, etag)

        self._retry_operation("update", do_update)

    def _delete(self, mem_id: int):
        def do_delete():
            memories, etag = self._load_memories()
            if mem_id not in memories:
                raise KeyError(f"Memory {mem_id} not found")
            memories.pop(mem_id)
            self._save_memories(memories, etag)

        self._retry_operation("delete", do_delete)

    def get_tools(self) -> list[Tool]:
        """Return Tool instances bound to this manager's state."""
        if self._tools is not None:
            return self._tools

        def search_tool(queries: list[str], limit: int = 5) -> str:
            """Search for memories using keyword search. Use as many queries as you want, the top results will be fused into one list. Search results include just id and description."""
            mems = self._search(queries, limit=limit)
            return "\n---\n".join(
                [self._format_memory(mem, include_content=False) for mem in mems]
            )

        def read_tool(mem_ids: list[int]) -> str:
            """Read the full contents of one or more memories."""
            mems = self._read(mem_ids)
            return "\n---\n".join(
                [self._format_memory(mem, include_content=True) for mem in mems]
            )

        def add_tool(description: str, content: str):
            """Add a new memory."""
            return self._add(description, content)

        def update_tool(mem_id: int, description: str, content: str) -> str:
            """Update a memory by ID. Must provide content and description, even if only changing one of them."""
            self._update(mem_id, description, content)
            return f"Memory {mem_id} updated successfully."

        def delete_tool(mem_id: int) -> str:
            """Delete a memory by ID."""
            self._delete(mem_id)
            return f"Memory {mem_id} deleted successfully."

        def _rename(tool: Tool, name: str) -> Tool:
            if tool.name == name:
                return tool
            return tool.model_copy(update={"name": name})

        self._tools = [
            _rename(Tool.from_function(search_tool), self.search_tool_name),
            _rename(Tool.from_function(read_tool), self.read_tool_name),
            _rename(Tool.from_function(add_tool), self.write_tool_name),
            _rename(Tool.from_function(update_tool), self.update_tool_name),
            _rename(Tool.from_function(delete_tool), self.delete_tool_name),
        ]
        return self._tools

    def get_all_memories(self) -> list[MemoryItem]:
        """Get all memories (useful for debugging/inspection)."""
        memories, _ = self._load_memories()
        return list(memories.values())

    def clear_all(self):
        """Delete all memories (useful for testing)."""
        try:
            self.client.delete_object(Bucket=self.bucket, Key=self.key)
        except self.client.exceptions.ClientError:
            pass


class S3MemoryConflictError(Exception):
    """Raised when a write conflict occurs due to concurrent modification."""

    pass


__all__ = ["MemoryManager", "S3MemoryManager", "S3MemoryConflictError", "S3RetryConfig"]
