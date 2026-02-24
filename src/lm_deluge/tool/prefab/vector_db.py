from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Literal, Optional, Sequence

from pydantic import BaseModel, Field

from .. import Tool

try:
    import numpy as np  # type: ignore
except ImportError:
    np = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Abstract backend
# ---------------------------------------------------------------------------


class VectorDBRecord(BaseModel):
    """A single record in the vector DB."""

    id: str
    text: str
    vector: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorDBSearchResult(BaseModel):
    """A search result with similarity score."""

    id: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorDBBackend(ABC):
    """
    Abstract backend for vector storage and retrieval.

    Implementations must handle storing (id, text, vector, metadata) tuples and
    returning nearest-neighbour results by cosine similarity. This abstraction
    lets callers swap InProcessVectorDB for heavier backends (USearch,
    turbopuffer, S3-backed stores, etc.) without changing tool wiring.
    """

    @abstractmethod
    def insert(
        self,
        records: Sequence[VectorDBRecord],
    ) -> list[str]:
        """Insert records, returning their IDs."""
        ...

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
    ) -> list[VectorDBSearchResult]:
        """Return the top-k most similar records to *query_vector*."""
        ...

    @abstractmethod
    def get(self, ids: Sequence[str]) -> list[VectorDBRecord | None]:
        """Fetch records by ID. Returns None for missing IDs."""
        ...

    @abstractmethod
    def delete(self, ids: Sequence[str]) -> int:
        """Delete records by ID, returning the count actually removed."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Return the total number of stored records."""
        ...

    @abstractmethod
    def list_ids(self, limit: int = 100, offset: int = 0) -> list[str]:
        """Return a page of stored IDs."""
        ...


# ---------------------------------------------------------------------------
# In-process numpy backend
# ---------------------------------------------------------------------------


class InProcessVectorDB(VectorDBBackend):
    """
    Lightweight in-process vector DB backed by numpy arrays.

    Good for small collections (up to ~100k vectors). All data lives in memory.
    Supports cosine similarity search via brute-force dot product on
    L2-normalised vectors.
    """

    def __init__(self, dimension: int | None = None) -> None:
        if np is None:
            raise ImportError(
                "numpy is required for InProcessVectorDB. "
                "Install it with: pip install numpy"
            )
        self._dimension = dimension
        # Parallel lists – kept in insertion order
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._metadata: list[dict[str, Any]] = []
        # Normalised vectors stored as rows in a contiguous matrix.
        # None when empty; rebuilt on insert.
        self._matrix: np.ndarray | None = None  # type: ignore
        # id -> index for O(1) lookup
        self._id_to_idx: dict[str, int] = {}

    @property
    def dimension(self) -> int | None:
        return self._dimension

    # -- helpers ------------------------------------------------------------

    def _normalise(self, vecs: np.ndarray) -> np.ndarray:  # type: ignore
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)  # type: ignore
        norms = np.where(norms == 0, 1.0, norms)  # type: ignore
        return vecs / norms

    def _rebuild_matrix(self) -> None:
        """Rebuild the contiguous matrix from scratch (after deletes)."""
        if not self._ids:
            self._matrix = None
            return
        # Matrix is already maintained during inserts; this is for compaction
        # after deletes leave gaps in logical ordering.
        # Since we keep parallel lists in order, just re-stack.
        pass  # matrix is already consistent after _compact_after_delete

    # -- public API ---------------------------------------------------------

    def insert(self, records: Sequence[VectorDBRecord]) -> list[str]:
        if not records:
            return []

        new_ids: list[str] = []
        new_texts: list[str] = []
        new_meta: list[dict[str, Any]] = []
        raw_vectors: list[list[float]] = []

        for rec in records:
            rid = rec.id or str(uuid.uuid4())
            if rid in self._id_to_idx:
                raise ValueError(f"Duplicate ID: {rid}")

            vec = rec.vector
            if self._dimension is None:
                self._dimension = len(vec)
            elif len(vec) != self._dimension:
                raise ValueError(
                    f"Vector dimension mismatch: expected {self._dimension}, "
                    f"got {len(vec)}"
                )

            new_ids.append(rid)
            new_texts.append(rec.text)
            new_meta.append(rec.metadata)
            raw_vectors.append(vec)

        new_matrix = self._normalise(np.array(raw_vectors, dtype=np.float32))  # type: ignore

        base_idx = len(self._ids)
        self._ids.extend(new_ids)
        self._texts.extend(new_texts)
        self._metadata.extend(new_meta)
        for i, rid in enumerate(new_ids):
            self._id_to_idx[rid] = base_idx + i

        if self._matrix is None:
            self._matrix = new_matrix
        else:
            self._matrix = np.vstack([self._matrix, new_matrix])  # type: ignore

        return new_ids

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
    ) -> list[VectorDBSearchResult]:
        if self._matrix is None or len(self._ids) == 0:
            return []

        qvec = np.array(query_vector, dtype=np.float32).reshape(1, -1)  # type: ignore
        qvec = self._normalise(qvec)

        # Cosine similarity = dot product of normalised vectors
        scores = (self._matrix @ qvec.T).flatten()
        k = min(top_k, len(self._ids))
        # argpartition is O(n) vs O(n log n) for full sort
        if k < len(scores):
            top_indices = np.argpartition(scores, -k)[-k:]  # type: ignore
        else:
            top_indices = np.arange(len(scores))  # type: ignore
        # Sort the top-k by descending score
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]  # type: ignore

        results: list[VectorDBSearchResult] = []
        for idx in top_indices:
            idx_int = int(idx)
            results.append(
                VectorDBSearchResult(
                    id=self._ids[idx_int],
                    text=self._texts[idx_int],
                    score=float(scores[idx_int]),
                    metadata=self._metadata[idx_int],
                )
            )
        return results

    def get(self, ids: Sequence[str]) -> list[VectorDBRecord | None]:
        results: list[VectorDBRecord | None] = []
        for rid in ids:
            idx = self._id_to_idx.get(rid)
            if idx is None:
                results.append(None)
            else:
                results.append(
                    VectorDBRecord(
                        id=self._ids[idx],
                        text=self._texts[idx],
                        vector=self._matrix[idx].tolist()
                        if self._matrix is not None
                        else [],
                        metadata=self._metadata[idx],
                    )
                )
        return results

    def delete(self, ids: Sequence[str]) -> int:
        indices_to_remove: list[int] = []
        for rid in ids:
            idx = self._id_to_idx.get(rid)
            if idx is not None:
                indices_to_remove.append(idx)

        if not indices_to_remove:
            return 0

        removed = len(indices_to_remove)
        keep_mask = np.ones(len(self._ids), dtype=bool)  # type: ignore
        for idx in indices_to_remove:
            keep_mask[idx] = False

        # Compact parallel lists
        new_ids: list[str] = []
        new_texts: list[str] = []
        new_meta: list[dict[str, Any]] = []
        for i, keep in enumerate(keep_mask):
            if keep:
                new_ids.append(self._ids[i])
                new_texts.append(self._texts[i])
                new_meta.append(self._metadata[i])

        self._ids = new_ids
        self._texts = new_texts
        self._metadata = new_meta
        self._id_to_idx = {rid: i for i, rid in enumerate(self._ids)}

        if self._matrix is not None:
            self._matrix = self._matrix[keep_mask]
            if len(self._ids) == 0:
                self._matrix = None

        return removed

    def count(self) -> int:
        return len(self._ids)

    def list_ids(self, limit: int = 100, offset: int = 0) -> list[str]:
        return self._ids[offset : offset + limit]


# ---------------------------------------------------------------------------
# Tool wiring — VectorDBManager
# ---------------------------------------------------------------------------

VECTOR_DB_DESCRIPTION = """Interact with a vector database for semantic search over text.

Use this tool to:
- insert text entries (with pre-computed embedding vectors)
- search for the most similar entries to a query vector
- fetch entries by ID
- delete entries by ID
- list stored entry IDs or get a count
"""

VectorDBCommand = Literal["insert", "search", "get", "delete", "count", "list_ids"]

ALL_COMMANDS: tuple[VectorDBCommand, ...] = (
    "insert",
    "search",
    "get",
    "delete",
    "count",
    "list_ids",
)


class VectorDBParams(BaseModel):
    """Schema for vector DB tool calls."""

    command: VectorDBCommand = Field(description="Operation to run.")

    # -- insert params --
    entries: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description=(
            "List of entries to insert. Each entry must have 'text' (string) "
            "and 'vector' (list of floats). Optionally include 'id' (string) "
            "and 'metadata' (object). Required for command='insert'."
        ),
    )

    # -- search params --
    query_vector: Optional[list[float]] = Field(
        default=None,
        description="Query embedding vector. Required for command='search'.",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Number of results to return for command='search'.",
    )

    # -- get / delete params --
    ids: Optional[list[str]] = Field(
        default=None,
        description="List of entry IDs. Required for command='get' or 'delete'.",
    )

    # -- list_ids params --
    limit: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Max IDs to return for command='list_ids'.",
    )
    offset: int = Field(
        default=0,
        ge=0,
        description="Offset for pagination in command='list_ids'.",
    )


class VectorDBManager:
    """
    Wraps a VectorDBBackend and exposes it as lm-deluge Tools.

    Similar in spirit to SqliteManager: you create the manager with a backend,
    then call ``get_tools()`` to get Tool objects you can pass into an agent
    loop.

    Example::

        from lm_deluge.tool.prefab.vector_db import InProcessVectorDB, VectorDBManager

        db = InProcessVectorDB(dimension=1536)
        manager = VectorDBManager(db)
        tools = manager.get_tools()
    """

    def __init__(
        self,
        backend: VectorDBBackend,
        *,
        tool_name: str = "vector_db",
    ) -> None:
        self.backend = backend
        self.tool_name = tool_name
        self._tool_cache: dict[tuple[str, ...], list[Tool]] = {}

    def _handle(self, allowed_commands: set[VectorDBCommand], **kwargs: Any) -> str:
        params = VectorDBParams.model_validate(kwargs)

        try:
            if params.command not in allowed_commands:
                raise ValueError(
                    f"The '{params.command}' command is disabled for this tool instance"
                )

            if params.command == "insert":
                if not params.entries:
                    raise ValueError("entries is required for command='insert'")
                records = []
                for entry in params.entries:
                    if "text" not in entry or "vector" not in entry:
                        raise ValueError(
                            "Each entry must have 'text' and 'vector' fields"
                        )
                    records.append(
                        VectorDBRecord(
                            id=entry.get("id", str(uuid.uuid4())),
                            text=entry["text"],
                            vector=entry["vector"],
                            metadata=entry.get("metadata", {}),
                        )
                    )
                ids = self.backend.insert(records)
                result = {"command": "insert", "inserted_count": len(ids), "ids": ids}

            elif params.command == "search":
                if params.query_vector is None:
                    raise ValueError("query_vector is required for command='search'")
                hits = self.backend.search(params.query_vector, top_k=params.top_k)
                result = {
                    "command": "search",
                    "result_count": len(hits),
                    "results": [h.model_dump() for h in hits],
                }

            elif params.command == "get":
                if not params.ids:
                    raise ValueError("ids is required for command='get'")
                records = self.backend.get(params.ids)
                result = {
                    "command": "get",
                    "records": [
                        r.model_dump() if r is not None else None for r in records
                    ],
                }

            elif params.command == "delete":
                if not params.ids:
                    raise ValueError("ids is required for command='delete'")
                removed = self.backend.delete(params.ids)
                result = {"command": "delete", "deleted_count": removed}

            elif params.command == "count":
                result = {"command": "count", "total": self.backend.count()}

            elif params.command == "list_ids":
                ids = self.backend.list_ids(limit=params.limit, offset=params.offset)
                result = {
                    "command": "list_ids",
                    "ids": ids,
                    "returned": len(ids),
                }

            else:
                raise ValueError(f"Unknown command: {params.command}")

            return json.dumps({"ok": True, "result": result}, indent=2)
        except Exception as exc:
            return json.dumps(
                {"ok": False, "error": type(exc).__name__, "message": str(exc)},
                indent=2,
            )

    def get_tools(
        self,
        *,
        exclude: Sequence[VectorDBCommand] | None = None,
    ) -> list[Tool]:
        exclude_set = set(exclude or [])
        unknown = exclude_set.difference(ALL_COMMANDS)
        if unknown:
            raise ValueError(f"Unknown commands in exclude list: {sorted(unknown)}")

        allowed = tuple(cmd for cmd in ALL_COMMANDS if cmd not in exclude_set)
        if not allowed:
            raise ValueError("Cannot exclude every vector_db command")

        cache_key = allowed
        if cache_key in self._tool_cache:
            return self._tool_cache[cache_key]

        allowed_set = set(allowed)
        schema = VectorDBParams.model_json_schema(ref_template="#/$defs/{model}")
        if (
            "properties" in schema
            and "command" in schema["properties"]
            and isinstance(schema["properties"]["command"], dict)
        ):
            schema["properties"]["command"]["enum"] = list(allowed)

        tool = Tool(
            name=self.tool_name,
            description=VECTOR_DB_DESCRIPTION,
            parameters=schema.get("properties", {}),
            required=schema.get("required", []),
            definitions=schema.get("$defs"),
            run=partial(self._handle, allowed_set),  # type: ignore[arg-type]
        )

        self._tool_cache[cache_key] = [tool]
        return [tool]
