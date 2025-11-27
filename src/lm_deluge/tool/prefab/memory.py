from __future__ import annotations

import json
from collections import Counter
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
    ):
        self.write_tool_name = write_tool_name
        self.read_tool_name = read_tool_name
        self._memories: dict[int, MemoryItem] = {}
        self._tools: list[Tool] | None = None

        if memories:
            if isinstance(memories, dict):
                self._memories = {k: self._coerce(v) for k, v in memories.items()}
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
        return [self._memories.get(x) for x in memory_ids if x is not None]  # type: ignore

    def _add(self, description: str, content: str):
        new_id = max(self._memories) + 1
        self._memories[new_id] = self._coerce(
            {"id": new_id, "description": description, "content": content}
        )

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
            self._update(mem_id, content, description)

            return f"Memory {mem_id} updated successfully."

        def delete_tool(mem_id: int) -> str:
            """Delete a memory by ID."""
            self._delete(mem_id)
            return f"Memory {mem_id} deleted successfully."

        self._tools = [
            Tool.from_function(search_tool),
            Tool.from_function(read_tool),
            Tool.from_function(add_tool),
            Tool.from_function(update_tool),
            Tool.from_function(delete_tool),
        ]
        return self._tools


__all__ = ["MemoryManager"]
