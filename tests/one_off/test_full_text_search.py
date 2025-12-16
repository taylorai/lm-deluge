"""Test for FullTextSearchManager prefab tool."""

import json
import tempfile

from lm_deluge.tool.prefab import FullTextSearchManager


def test_basic_search():
    """Test basic search functionality."""
    corpus = [
        {
            "id": "1",
            "title": "Introduction to Python",
            "content": "Python is a versatile programming language used for web development.",
        },
        {
            "id": "2",
            "title": "JavaScript Basics",
            "content": "JavaScript is essential for front-end web development.",
        },
        {
            "id": "3",
            "title": "Machine Learning with Python",
            "content": "Python is widely used in machine learning and data science.",
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = FullTextSearchManager(
            corpus=corpus,
            search_fields=["title", "content"],
            preview_fields=["title"],
            index_path=tmpdir,
        )

        # Test search
        results = manager.search("Python programming")
        print(f"Search results for 'Python programming': {len(results)} results")
        for r in results:
            print(f"  - {r.id}: score={r.score:.4f}")

        assert len(results) > 0
        # Python should match documents 1 and 3
        result_ids = {r.id for r in results}
        assert "1" in result_ids or "3" in result_ids

        # Test fetch
        docs = manager.fetch(["1", "2"])
        print(f"\nFetched documents: {len(docs)}")
        for doc in docs:
            print(f"  - {doc['id']}: {doc['title']}")

        assert len(docs) == 2


def test_tool_search():
    """Test using the tool interface for search."""
    corpus = [
        {"id": "doc1", "title": "Cats", "content": "Cats are fluffy pets."},
        {"id": "doc2", "title": "Dogs", "content": "Dogs are loyal companions."},
        {"id": "doc3", "title": "Pet Care", "content": "Taking care of cats and dogs."},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = FullTextSearchManager(
            corpus=corpus,
            index_path=tmpdir,
        )

        tools = manager.get_tools()
        assert len(tools) == 2

        search_tool = tools[0]
        fetch_tool = tools[1]

        print(f"\nSearch tool: {search_tool.name}")
        print(f"  Description: {search_tool.description}")
        print(f"  Parameters: {search_tool.parameters}")

        print(f"\nFetch tool: {fetch_tool.name}")
        print(f"  Description: {fetch_tool.description}")
        print(f"  Parameters: {fetch_tool.parameters}")

        # Call search tool
        result_str = search_tool.call(query="cats fluffy", limit=5)
        result = json.loads(result_str)
        print(f"\nSearch tool result: {json.dumps(result, indent=2)}")

        assert result["status"] == "success"
        assert result["num_results"] > 0

        # Get document IDs from search
        doc_ids = [r["id"] for r in result["results"]]

        # Call fetch tool
        fetch_result_str = fetch_tool.call(document_ids=doc_ids)
        fetch_result = json.loads(fetch_result_str)
        print(f"\nFetch tool result: {json.dumps(fetch_result, indent=2)}")

        assert fetch_result["status"] == "success"
        assert fetch_result["found"] > 0


def test_missing_fetch():
    """Test fetching non-existent documents."""
    corpus = [
        {"id": "1", "title": "Test", "content": "Test content"},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = FullTextSearchManager(corpus=corpus, index_path=tmpdir)
        tools = manager.get_tools()
        fetch_tool = tools[1]

        # Try to fetch non-existent document
        result_str = fetch_tool.call(document_ids=["1", "nonexistent"])
        result = json.loads(result_str)

        print(f"\nFetch with missing: {json.dumps(result, indent=2)}")

        assert result["status"] == "success"
        assert result["found"] == 1
        assert result["missing"] == 1
        assert "nonexistent" in result["missing_ids"]


def test_empty_corpus_error():
    """Test that empty corpus raises an error."""
    try:
        FullTextSearchManager(corpus=[])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"\nExpected error for empty corpus: {e}")
        assert "empty" in str(e).lower()


def test_missing_id_error():
    """Test that documents without id raise an error."""
    corpus = [{"title": "No ID", "content": "This doc has no id"}]

    try:
        FullTextSearchManager(corpus=corpus)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"\nExpected error for missing id: {e}")
        assert "id" in str(e).lower()


def test_auto_detect_search_fields():
    """Test that search fields are auto-detected from corpus."""
    corpus = [
        {"id": "1", "title": "Test", "content": "Content", "count": 42},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = FullTextSearchManager(corpus=corpus, index_path=tmpdir)

        # Should auto-detect string fields
        print(f"\nAuto-detected search fields: {manager.search_fields}")
        assert "title" in manager.search_fields
        assert "content" in manager.search_fields
        # count is int, not string, so should not be included
        assert "count" not in manager.search_fields


def test_custom_tool_names():
    """Test custom tool names."""
    corpus = [{"id": "1", "title": "Test", "content": "Content"}]

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = FullTextSearchManager(
            corpus=corpus,
            index_path=tmpdir,
            search_tool_name="find_docs",
            fetch_tool_name="get_docs",
        )

        tools = manager.get_tools()
        tool_names = {t.name for t in tools}

        print(f"\nCustom tool names: {tool_names}")
        assert "find_docs" in tool_names
        assert "get_docs" in tool_names


if __name__ == "__main__":
    print("=" * 60)
    print("Testing FullTextSearchManager")
    print("=" * 60)

    test_basic_search()
    print("\n" + "=" * 60)

    test_tool_search()
    print("\n" + "=" * 60)

    test_missing_fetch()
    print("\n" + "=" * 60)

    test_empty_corpus_error()
    print("\n" + "=" * 60)

    test_missing_id_error()
    print("\n" + "=" * 60)

    test_auto_detect_search_fields()
    print("\n" + "=" * 60)

    test_custom_tool_names()
    print("\n" + "=" * 60)

    print("\nAll tests passed!")
