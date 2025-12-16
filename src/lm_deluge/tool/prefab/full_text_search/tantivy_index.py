import os
import re
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Set, TypedDict

from lenlp import normalizer
from tantivy import Document, Index, SchemaBuilder
from tqdm.auto import tqdm

# Pattern to match field-like constructs (X:Y)
FIELDLIKE = re.compile(r'(?<!\\)\b([^\s:"]+)\s*:\s*([^\s")]+)')


class SearchResultContent(TypedDict):
    title: str
    url: str
    description: str
    keywords: str  # comma-delimited :)
    content: str


@dataclass
class SearchResult:
    id: str
    score: float
    content: SearchResultContent | dict[str, Any]  # for our purposes should be SRC


class SearchIndex(ABC):
    @abstractmethod
    def build_index(
        self,
        records: list[dict],
        deduplicate_by: str | None = None,
    ):
        raise NotImplementedError

    @abstractmethod
    def search(
        self, queries: list[str], fields: list[str], limit: int = 8, escape: bool = True
    ) -> list[SearchResult]:
        raise NotImplementedError


def deduplicate_exact(records: list[dict], deduplicate_by: str) -> list[dict]:
    seen = set()
    deduplicated = []
    for record in tqdm(records):
        key = normalizer.normalize(record[deduplicate_by])
        if key not in seen:
            deduplicated.append(record)
            seen.add(key)
    print(
        f"Deduplication removed {len(records) - len(deduplicated)} of {len(records)} records."
    )
    return deduplicated


def _fix_unbalanced_quotes(s: str) -> str:
    """Keep balanced quotes; if an odd number of quotes, drop the strays.

    Handles both double quotes (") and single quotes (') since Tantivy
    treats both as phrase delimiters.
    """
    # Handle double quotes
    if s.count('"') % 2 != 0:
        s = s.replace('"', " ")

    # Handle single quotes
    if s.count("'") % 2 != 0:
        s = s.replace("'", " ")

    return s


def _quote_nonfield_colons(
    q: str, known_fields: Set[str], json_prefixes: Set[str]
) -> str:
    """Quote X:Y patterns when X is not a known field name.

    This prevents strings like "3:12" from being interpreted as field queries.
    Preserves legitimate field queries like "title:roof".
    """

    def repl(m: re.Match) -> str:
        raw_prefix = m.group(1)
        # Handle escaped dots (a\.b)
        norm = raw_prefix.replace(r"\.", ".")
        top = norm.split(".", 1)[0]

        if norm in known_fields or top in json_prefixes:
            return m.group(0)  # Genuine field query; keep as-is
        return f'"{m.group(0)}"'  # Protect accidental field-like token

    return FIELDLIKE.sub(repl, q)


def _strip_syntax_to_words(q: str) -> str:
    """Last-resort sanitizer: remove operators but keep words/numbers.

    This is used as a fallback when all else fails to ensure we never crash.
    """
    q = re.sub(r'([+\-!(){}\[\]^"~*?:\\\/]|&&|\|\|)', " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q or "*"  # Never return empty (match all if nothing left)


def sanitize_query_for_tantivy(
    query: str, known_fields: Iterable[str], json_field_prefixes: Iterable[str] = ()
) -> str:
    """Context-aware query sanitization for Tantivy.

    Uses heuristics to detect whether special characters are intended as
    Tantivy syntax or just regular text:
    - Parentheses: only syntax if AND/OR/&&/|| present
    - Square brackets: only syntax if TO or IN present
    - Curly braces: only syntax if TO present

    Always quotes X:Y patterns when X is not a known field, and fixes
    unbalanced quotes.
    """
    # Step 1: Detect if special syntax is actually being used
    has_boolean_ops = bool(re.search(r"\b(AND|OR)\b|&&|\|\|", query))
    has_range_keyword = bool(re.search(r"\bTO\b", query))
    has_set_keyword = bool(re.search(r"\bIN\b", query))

    # Step 2: Remove syntax chars that are likely just regular text
    if not has_boolean_ops:
        # No boolean operators, so parentheses aren't for grouping
        query = query.replace("(", " ").replace(")", " ")

    if not has_range_keyword:
        # No ranges, so curly braces aren't for exclusive bounds
        query = query.replace("{", " ").replace("}", " ")

    if not (has_range_keyword or has_set_keyword):
        # No ranges or sets, so square brackets aren't query syntax
        query = query.replace("[", " ").replace("]", " ")

    # Step 3: Fix unbalanced quotes (both " and ')
    query = _fix_unbalanced_quotes(query)

    # Step 4: Quote non-field colons (e.g., "3:12")
    query = _quote_nonfield_colons(query, set(known_fields), set(json_field_prefixes))

    return query


class TantivySearch(SearchIndex):
    def __init__(
        self,
        index_path: str,
        include_fields: list[str] | None = None,
        exclude_fields: list[str] | None = None,
    ):
        self.index_path = Path(index_path)
        self.schema = None
        self.index = None
        self.include_fields = include_fields
        self.exclude_fields = exclude_fields
        self.searchable_fields: set[str] | None = (
            None  # Will be set during schema creation
        )
        self.id_to_record: dict[str, dict] = {}  # Cache for efficient document fetching

    def create_schema(self, sample_record: dict):
        schema_builder = SchemaBuilder()

        # Determine which fields should be searchable
        all_fields = set(sample_record.keys()) - {"id"}

        if self.include_fields is not None and self.exclude_fields is not None:
            # Both provided: searchable = include - exclude
            self.searchable_fields = set(self.include_fields) - set(self.exclude_fields)
        elif self.include_fields is not None:
            # Only include provided: use only those
            self.searchable_fields = set(self.include_fields)
        elif self.exclude_fields is not None:
            # Only exclude provided: use all except those
            self.searchable_fields = all_fields - set(self.exclude_fields)
        else:
            # Neither provided: all fields searchable
            # Set explicitly so we have field names for query sanitization
            self.searchable_fields = all_fields

        # Add all fields as text fields (they're all indexed, but we'll control search access)
        for key in sample_record.keys():
            if key == "id":
                continue
            schema_builder.add_text_field(key, stored=True, tokenizer_name="en_stem")
        schema_builder.add_text_field(
            "id", stored=True, tokenizer_name="raw", index_option="basic"
        )
        self.schema = schema_builder.build()

    @classmethod
    def from_index(cls, index_path: str):
        index = cls(index_path)
        index.load_index()
        return index

    def load_index(self):
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index not found at {self.index_path}")
        self.index = Index.open(str(self.index_path))

    def _get_schema_field_names(self) -> set[str]:
        """Get field names for query sanitization."""
        # searchable_fields is always set after create_schema() is called
        return self.searchable_fields if self.searchable_fields is not None else set()

    def build_index(
        self,
        records: list[dict],
        deduplicate_by: str | None = None,
    ):
        if not records:
            raise ValueError("No records to index")

        if not self.schema:
            self.create_schema(records[0])
        assert self.schema is not None, "Schema not created"

        # Create index
        if self.index_path.exists():
            shutil.rmtree(self.index_path)
        os.makedirs(self.index_path, exist_ok=True)
        self.index = Index(self.schema, str(self.index_path))

        # Deduplicate if requested
        if deduplicate_by is not None:
            records = deduplicate_exact(records, deduplicate_by)

        # Index documents
        writer = self.index.writer()
        for record in tqdm(records):
            writer.add_document(Document(**{k: [str(v)] for k, v in record.items()}))
        writer.commit()
        writer.wait_merging_threads()

    def search(
        self, queries: list[str], fields: list[str], limit: int = 8, escape: bool = True
    ) -> list[SearchResult]:
        assert self.index is not None, "Index not built"

        # Validate that requested fields are searchable
        if self.searchable_fields is not None:
            non_searchable = set(fields) - self.searchable_fields
            if non_searchable:
                raise ValueError(
                    f"Cannot search non-searchable fields: {sorted(non_searchable)}. "
                    f"Searchable fields are: {sorted(self.searchable_fields)}"
                )

        self.index.reload()
        searcher = self.index.searcher()
        results = []

        # Get field names for query sanitization
        known_fields = self._get_schema_field_names()

        for query in queries:
            if escape:
                # Three-tier fallback strategy
                # Tier 1: Try sanitized query with context-aware cleaning
                sanitized = sanitize_query_for_tantivy(query, known_fields)
                try:
                    query_obj = self.index.parse_query(sanitized, fields)
                except ValueError:
                    # Tier 2: Bag-of-words fallback - strip all operators
                    bag_of_words = _strip_syntax_to_words(sanitized)
                    try:
                        query_obj = self.index.parse_query(bag_of_words, fields)
                    except ValueError:
                        # This should never happen, but if it does, match nothing
                        query_obj = self.index.parse_query("", fields)
            else:
                # User disabled escaping - use query as-is
                try:
                    query_obj = self.index.parse_query(query, fields)
                except Exception as e:
                    print(f"Error parsing query: {e}")
                    query_obj = None

            if query_obj:
                hits = searcher.search(query_obj, limit=limit).hits
            else:
                hits = []

            for score, doc_address in hits:
                doc = searcher.doc(doc_address)
                content = {k: v[0] for k, v in doc.to_dict().items()}
                results.append(
                    SearchResult(
                        id=content["id"],
                        score=score,
                        content=content,
                    )
                )

        # Rank fusion using reciprocal rank fusion
        doc_scores = {}
        for rank, result in enumerate(results, 1):
            doc_key = str(result.id)
            if doc_key not in doc_scores:
                doc_scores[doc_key] = 0
            doc_scores[doc_key] += 1 / (60 + rank)

        # Sort and deduplicate
        unique_results = {}
        for result in results:
            doc_key = str(result.id)
            if doc_key not in unique_results:
                unique_results[doc_key] = result

        sorted_results = sorted(
            unique_results.values(),
            key=lambda x: doc_scores[str(x.id)],
            reverse=True,
        )

        return sorted_results[:limit]

    def cache_documents_for_fetch(
        self, documents: list[dict], id_column: str = "id"
    ) -> None:
        """Cache documents for efficient retrieval by ID.

        This builds an in-memory lookup table for O(1) document fetching.
        Should be called after build_index() with the original document data.

        Args:
            documents: List of document records (typically original unfiltered data)
            id_column: Name of the ID field in the documents (default: "id")
        """
        self.id_to_record = {}
        for record in documents:
            doc_id = str(record.get(id_column, ""))
            self.id_to_record[doc_id] = record

    def fetch(self, document_ids: list[str]) -> tuple[list[dict], list[str], list[str]]:
        """Fetch documents by ID using cached lookup.

        Returns full original documents from the cache built by cache_documents_for_fetch().
        If cache is empty, falls back to querying the Tantivy index (slower, returns only indexed fields).

        Args:
            document_ids: List of document IDs to retrieve

        Returns:
            Tuple of (documents, found_ids, missing_ids) where:
            - documents: List of document dicts with all fields
            - found_ids: List of IDs that were successfully found
            - missing_ids: List of IDs that were not found
        """
        results = []
        found_ids = []
        missing_ids = []

        # Use cached lookup if available
        if self.id_to_record:
            for doc_id in document_ids:
                if doc_id in self.id_to_record:
                    results.append(self.id_to_record[doc_id])
                    found_ids.append(doc_id)
                else:
                    missing_ids.append(doc_id)
            return results, found_ids, missing_ids

        # Fallback: query Tantivy index (slower, returns only indexed fields)
        assert self.index is not None, "Index not built and no cached documents"

        self.index.reload()
        searcher = self.index.searcher()

        for doc_id in document_ids:
            query_str = f'id:"{doc_id}"'
            try:
                query = self.index.parse_query(query_str, ["id"])
                hits = searcher.search(query, limit=1).hits

                if hits:
                    _, doc_address = hits[0]
                    doc = searcher.doc(doc_address)
                    content = {k: v[0] for k, v in doc.to_dict().items()}
                    results.append(content)
                    found_ids.append(doc_id)
                else:
                    missing_ids.append(doc_id)
            except Exception:
                missing_ids.append(doc_id)

        return results, found_ids, missing_ids
