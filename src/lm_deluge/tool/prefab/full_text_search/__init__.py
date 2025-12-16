"""Full text search prefab tool using Tantivy."""

import json
import tempfile
from pathlib import Path
from typing import Annotated, Any

from lm_deluge.tool import Tool

from .tantivy_index import SearchResult, TantivySearch


class FullTextSearchManager:
    """
    Full-text search tools using Tantivy.

    Provides two tools:
    - search: Search the corpus and get document IDs + previews
    - fetch: Get the full contents of specific documents by ID

    Args:
        corpus: List of document dicts to index. Each dict must have an "id" field.
        search_fields: List of field names to search. If None, searches all fields.
        preview_fields: Fields to include in search result previews.
        index_path: Path to store the Tantivy index. If None, uses a temp directory.
        search_tool_name: Name for the search tool (default: "search")
        fetch_tool_name: Name for the fetch tool (default: "fetch")
        max_results: Maximum number of search results to return (default: 10)
        include_fields: Fields to include in the index (searchable). If None, includes all.
        exclude_fields: Fields to exclude from the index (not searchable).

    Example:
        ```python
        corpus = [
            {"id": "1", "title": "Hello World", "content": "This is a test document."},
            {"id": "2", "title": "Another Doc", "content": "More content here."},
        ]
        manager = FullTextSearchManager(
            corpus=corpus,
            search_fields=["title", "content"],
            preview_fields=["title"],
        )
        tools = manager.get_tools()
        ```
    """

    def __init__(
        self,
        corpus: list[dict[str, Any]],
        *,
        search_fields: list[str] | None = None,
        preview_fields: list[str] | None = None,
        index_path: str | Path | None = None,
        search_tool_name: str = "search",
        fetch_tool_name: str = "fetch",
        max_results: int = 10,
        include_fields: list[str] | None = None,
        exclude_fields: list[str] | None = None,
        deduplicate_by: str | None = None,
    ):
        # Initialize _temp_dir early to avoid __del__ issues
        self._temp_dir: str | None = None

        self.corpus = corpus
        self.search_fields = search_fields
        self.preview_fields = preview_fields
        self.search_tool_name = search_tool_name
        self.fetch_tool_name = fetch_tool_name
        self.max_results = max_results
        self._tools: list[Tool] | None = None

        # Validate corpus
        if not corpus:
            raise ValueError("Corpus cannot be empty")

        # Ensure all documents have an id field
        for i, doc in enumerate(corpus):
            if "id" not in doc:
                raise ValueError(f"Document at index {i} is missing 'id' field")

        # Set up index path
        if index_path is None:
            self._temp_dir = tempfile.mkdtemp(prefix="tantivy_")
            self._index_path = Path(self._temp_dir)
        else:
            self._temp_dir = None
            self._index_path = Path(index_path)

        # Determine search fields from corpus if not provided
        if search_fields is None:
            # Use all string fields except 'id'
            sample = corpus[0]
            self.search_fields = [
                k for k, v in sample.items() if k != "id" and isinstance(v, str)
            ]
        else:
            self.search_fields = search_fields

        # Initialize Tantivy index
        self._index = TantivySearch(
            index_path=str(self._index_path),
            include_fields=include_fields,
            exclude_fields=exclude_fields,
        )

        # Build the index
        self._index.build_index(
            records=corpus,
            deduplicate_by=deduplicate_by,
        )

        # Cache documents for efficient fetch
        self._index.cache_documents_for_fetch(corpus, id_column="id")

    def _format_preview(self, result: SearchResult) -> dict[str, Any]:
        """Format a search result for preview."""
        preview: dict[str, Any] = {
            "id": result.id,
            "score": round(result.score, 4),
        }

        # Add preview fields
        if self.preview_fields:
            for field in self.preview_fields:
                if field in result.content:
                    value = result.content[field]
                    # Truncate long values
                    if isinstance(value, str) and len(value) > 200:
                        value = value[:200] + "..."
                    preview[field] = value
        else:
            # Include all fields with truncation
            for field, value in result.content.items():
                if field == "id":
                    continue
                if isinstance(value, str) and len(value) > 200:
                    value = value[:200] + "..."
                preview[field] = value

        return preview

    def _search(
        self,
        query: Annotated[str, "Search query to find relevant documents"],
        limit: Annotated[int, "Maximum number of results to return"] = 10,
    ) -> str:
        """
        Search the corpus for documents matching the query.

        Returns a list of document previews with IDs and scores.
        Use the fetch tool to get full document contents.
        """
        try:
            # Use the search fields
            assert self.search_fields is not None
            results = self._index.search(
                queries=[query],
                fields=self.search_fields,
                limit=min(limit, self.max_results),
                escape=True,
            )

            previews = [self._format_preview(r) for r in results]

            return json.dumps(
                {
                    "status": "success",
                    "query": query,
                    "num_results": len(previews),
                    "results": previews,
                },
                indent=2,
            )

        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

    def _fetch(
        self,
        document_ids: Annotated[
            list[str], "List of document IDs to fetch full contents for"
        ],
    ) -> str:
        """
        Fetch the full contents of documents by their IDs.

        Use search first to find relevant document IDs.
        """
        try:
            documents, found_ids, missing_ids = self._index.fetch(document_ids)

            result: dict[str, Any] = {
                "status": "success",
                "found": len(found_ids),
                "missing": len(missing_ids),
                "documents": documents,
            }

            if missing_ids:
                result["missing_ids"] = missing_ids

            return json.dumps(result, indent=2)

        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

    def get_tools(self) -> list[Tool]:
        """Return the search and fetch tools."""
        if self._tools is not None:
            return self._tools

        search_tool = Tool.from_function(self._search, name=self.search_tool_name)
        fetch_tool = Tool.from_function(self._fetch, name=self.fetch_tool_name)

        # Update descriptions for clarity
        search_tool = search_tool.model_copy(
            update={
                "description": (
                    "Search the document corpus for relevant results. "
                    "Returns document IDs, relevance scores, and previews. "
                    "Use the fetch tool to get full document contents."
                )
            }
        )

        fetch_tool = fetch_tool.model_copy(
            update={
                "description": (
                    "Fetch the full contents of documents by their IDs. "
                    "Use after searching to get complete document text."
                )
            }
        )

        self._tools = [search_tool, fetch_tool]
        return self._tools

    def search(
        self, query: str, limit: int = 10, fields: list[str] | None = None
    ) -> list[SearchResult]:
        """
        Direct search method for programmatic use.

        Args:
            query: Search query string
            limit: Maximum number of results
            fields: Fields to search (defaults to self.search_fields)

        Returns:
            List of SearchResult objects
        """
        search_fields = fields or self.search_fields
        assert search_fields is not None
        return self._index.search(
            queries=[query],
            fields=search_fields,
            limit=min(limit, self.max_results),
            escape=True,
        )

    def fetch(self, document_ids: list[str]) -> list[dict[str, Any]]:
        """
        Direct fetch method for programmatic use.

        Args:
            document_ids: List of document IDs to fetch

        Returns:
            List of document dicts
        """
        documents, _, _ = self._index.fetch(document_ids)
        return documents

    def __del__(self):
        """Clean up temp directory if used."""
        if self._temp_dir is not None:
            import shutil

            try:
                shutil.rmtree(self._temp_dir, ignore_errors=True)
            except Exception:
                pass


__all__ = ["FullTextSearchManager", "SearchResult"]
