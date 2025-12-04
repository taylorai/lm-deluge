"""Tests for the DocsManager prefab tool."""

import json
from unittest.mock import MagicMock, patch

from lm_deluge.tool.prefab.docs import DocsManager


def test_docs_manager_initialization_with_credentials():
    """Test that DocsManager can be initialized with credentials."""
    credentials = {
        "type": "service_account",
        "project_id": "test-project",
        "private_key_id": "key123",
        "private_key": "-----BEGIN PRIVATE KEY-----\ntest\n-----END PRIVATE KEY-----\n",
        "client_email": "test@test.iam.gserviceaccount.com",
        "client_id": "123456789",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
    }

    manager = DocsManager(document_id="test-doc-id", credentials_json=credentials)

    assert manager.document_id == "test-doc-id"
    assert manager.credentials == credentials
    assert manager.get_metadata_tool_name == "docs_get_metadata"
    assert manager.read_range_tool_name == "docs_read_range"
    assert manager.grep_tool_name == "docs_grep"
    assert manager.add_paragraph_tool_name == "docs_add_paragraph"
    assert manager.update_paragraph_tool_name == "docs_update_paragraph"
    assert manager.replace_tool_name == "docs_replace_text"
    assert manager.delete_range_tool_name == "docs_delete_range"


def test_docs_manager_initialization_without_document_id():
    """Test that DocsManager can be initialized without a document ID."""
    credentials = {"type": "service_account", "project_id": "test"}

    manager = DocsManager(document_title="Test Document", credentials_json=credentials)

    assert manager.document_id is None
    assert manager.document_title == "Test Document"


def test_docs_manager_custom_tool_names():
    """Test that DocsManager allows custom tool names."""
    credentials = {"type": "service_account", "project_id": "test"}

    manager = DocsManager(
        document_id="test-doc-id",
        credentials_json=credentials,
        get_metadata_tool_name="custom_metadata",
        read_range_tool_name="custom_read",
        grep_tool_name="custom_grep",
        add_paragraph_tool_name="custom_add",
        update_paragraph_tool_name="custom_update",
        replace_tool_name="custom_replace",
        delete_range_tool_name="custom_delete",
    )

    tools = manager.get_tools()
    assert len(tools) == 7
    assert tools[0].name == "custom_metadata"
    assert tools[1].name == "custom_read"
    assert tools[2].name == "custom_grep"
    assert tools[3].name == "custom_add"
    assert tools[4].name == "custom_update"
    assert tools[5].name == "custom_replace"
    assert tools[6].name == "custom_delete"


def _mock_docs_service() -> tuple[MagicMock, MagicMock]:
    """Create a mock Google Docs service."""
    mock_service = MagicMock()
    mock_documents = MagicMock()
    mock_service.documents.return_value = mock_documents
    return mock_service, mock_documents


def test_extract_text_with_formatting():
    """Test that _extract_text_with_formatting correctly extracts text from document structure."""
    credentials = {"type": "service_account", "project_id": "test"}
    manager = DocsManager(document_id="test-doc-id", credentials_json=credentials)

    # Mock document structure with formatting
    document = {
        "body": {
            "content": [
                {
                    "paragraph": {
                        "paragraphStyle": {"namedStyleType": "HEADING_1"},
                        "elements": [{"textRun": {"content": "Title\n"}}],
                    }
                },
                {
                    "paragraph": {
                        "elements": [
                            {
                                "textRun": {
                                    "content": "Hello ",
                                    "textStyle": {"bold": True},
                                }
                            },
                            {"textRun": {"content": "World\n"}},
                        ]
                    }
                },
            ]
        }
    }

    text, total_lines = manager._extract_text_with_formatting(document)
    assert "# Title" in text
    assert "**Hello **" in text  # The space is part of the bold run
    assert total_lines == 2


def test_get_metadata():
    """Test that _get_metadata returns document metadata."""
    credentials = {"type": "service_account", "project_id": "test"}
    manager = DocsManager(document_id="test-doc-id", credentials_json=credentials)

    with (
        patch("googleapiclient.discovery.build") as mock_build,
        patch("google.oauth2.service_account.Credentials.from_service_account_info"),
    ):
        mock_service, mock_documents = _mock_docs_service()
        mock_build.return_value = mock_service

        # Mock the get() method
        mock_get = MagicMock()
        mock_documents.get.return_value = mock_get
        mock_get.execute.return_value = {
            "documentId": "test-doc-id",
            "title": "Test Document",
            "body": {
                "content": [
                    {"paragraph": {"elements": [{"textRun": {"content": "Line 1\n"}}]}},
                    {"paragraph": {"elements": [{"textRun": {"content": "Line 2\n"}}]}},
                ]
            },
        }

        result = manager._get_metadata()
        data = json.loads(result)

        assert data["status"] == "success"
        assert data["document_id"] == "test-doc-id"
        assert data["title"] == "Test Document"
        assert data["total_lines"] == 2
        assert "https://docs.google.com/document/d/test-doc-id" in data["url"]


def test_read_range():
    """Test that _read_range returns document content."""
    credentials = {"type": "service_account", "project_id": "test"}
    manager = DocsManager(document_id="test-doc-id", credentials_json=credentials)

    with (
        patch("googleapiclient.discovery.build") as mock_build,
        patch("google.oauth2.service_account.Credentials.from_service_account_info"),
    ):
        mock_service, mock_documents = _mock_docs_service()
        mock_build.return_value = mock_service

        # Mock the get() method
        mock_get = MagicMock()
        mock_documents.get.return_value = mock_get
        mock_get.execute.return_value = {
            "documentId": "test-doc-id",
            "title": "Test Document",
            "body": {
                "content": [
                    {"paragraph": {"elements": [{"textRun": {"content": "Line 1\n"}}]}},
                    {"paragraph": {"elements": [{"textRun": {"content": "Line 2\n"}}]}},
                    {"paragraph": {"elements": [{"textRun": {"content": "Line 3\n"}}]}},
                ]
            },
        }

        result = manager._read_range(start_line=1, end_line=2)
        data = json.loads(result)

        assert data["status"] == "success"
        assert data["document_id"] == "test-doc-id"
        assert data["title"] == "Test Document"
        assert data["start_line"] == 1
        assert data["end_line"] == 2
        assert data["total_lines"] == 3
        assert "https://docs.google.com/document/d/test-doc-id" in data["url"]


def test_grep():
    """Test that _grep searches for patterns in the document."""
    credentials = {"type": "service_account", "project_id": "test"}
    manager = DocsManager(document_id="test-doc-id", credentials_json=credentials)

    with (
        patch("googleapiclient.discovery.build") as mock_build,
        patch("google.oauth2.service_account.Credentials.from_service_account_info"),
    ):
        mock_service, mock_documents = _mock_docs_service()
        mock_build.return_value = mock_service

        mock_get = MagicMock()
        mock_documents.get.return_value = mock_get
        mock_get.execute.return_value = {
            "body": {
                "content": [
                    {
                        "paragraph": {
                            "elements": [{"textRun": {"content": "Hello world\n"}}]
                        }
                    },
                    {
                        "paragraph": {
                            "elements": [{"textRun": {"content": "Goodbye world\n"}}]
                        }
                    },
                    {
                        "paragraph": {
                            "elements": [{"textRun": {"content": "Hello again\n"}}]
                        }
                    },
                ]
            }
        }

        result = manager._grep("Hello")
        data = json.loads(result)

        assert data["status"] == "success"
        assert data["match_count"] == 2
        assert len(data["matches"]) == 2
        assert data["matches"][0]["line"] == 1
        assert data["matches"][1]["line"] == 3


def test_add_paragraph():
    """Test that _add_paragraph adds text to the document."""
    credentials = {"type": "service_account", "project_id": "test"}
    manager = DocsManager(document_id="test-doc-id", credentials_json=credentials)

    with (
        patch("googleapiclient.discovery.build") as mock_build,
        patch("google.oauth2.service_account.Credentials.from_service_account_info"),
    ):
        mock_service, mock_documents = _mock_docs_service()
        mock_build.return_value = mock_service

        # Mock the get() method to return document structure
        mock_get = MagicMock()
        mock_documents.get.return_value = mock_get
        mock_get.execute.return_value = {
            "body": {
                "content": [
                    {"startIndex": 1, "endIndex": 10},
                    {"startIndex": 10, "endIndex": 11},
                ]
            }
        }

        # Mock the batchUpdate() method
        mock_batch = MagicMock()
        mock_documents.batchUpdate.return_value = mock_batch
        mock_batch.execute.return_value = {}

        result = manager._add_paragraph("New paragraph")
        data = json.loads(result)

        assert data["status"] == "success"
        assert "Successfully added paragraph" in data["message"]

        # Verify batchUpdate was called
        mock_documents.batchUpdate.assert_called_once()


def test_update_paragraph():
    """Test that _update_paragraph updates a specific line."""
    credentials = {"type": "service_account", "project_id": "test"}
    manager = DocsManager(document_id="test-doc-id", credentials_json=credentials)

    with (
        patch("googleapiclient.discovery.build") as mock_build,
        patch("google.oauth2.service_account.Credentials.from_service_account_info"),
    ):
        mock_service, mock_documents = _mock_docs_service()
        mock_build.return_value = mock_service

        # Mock the get() method
        mock_get = MagicMock()
        mock_documents.get.return_value = mock_get
        mock_get.execute.return_value = {
            "body": {
                "content": [
                    {
                        "startIndex": 1,
                        "endIndex": 10,
                        "paragraph": {
                            "elements": [{"textRun": {"content": "Old text\n"}}]
                        },
                    },
                ]
            }
        }

        # Mock the batchUpdate() method
        mock_batch = MagicMock()
        mock_documents.batchUpdate.return_value = mock_batch
        mock_batch.execute.return_value = {}

        result = manager._update_paragraph(line=1, new_text="New text")
        data = json.loads(result)

        assert data["status"] == "success"
        assert "Successfully updated line 1" in data["message"]

        # Verify batchUpdate was called
        mock_documents.batchUpdate.assert_called_once()


def test_replace_text():
    """Test that _replace_text replaces all occurrences."""
    credentials = {"type": "service_account", "project_id": "test"}
    manager = DocsManager(document_id="test-doc-id", credentials_json=credentials)

    with (
        patch("googleapiclient.discovery.build") as mock_build,
        patch("google.oauth2.service_account.Credentials.from_service_account_info"),
    ):
        mock_service, mock_documents = _mock_docs_service()
        mock_build.return_value = mock_service

        # Mock the batchUpdate() method
        mock_batch = MagicMock()
        mock_documents.batchUpdate.return_value = mock_batch
        mock_batch.execute.return_value = {
            "replies": [{"replaceAllText": {"occurrencesChanged": 3}}]
        }

        result = manager._replace_text("old", "new", match_case=True)
        data = json.loads(result)

        assert data["status"] == "success"
        assert data["replacements"] == 3
        assert "Replaced 3 occurrence(s)" in data["message"]

        # Verify batchUpdate was called correctly
        call_args = mock_documents.batchUpdate.call_args
        requests = call_args[1]["body"]["requests"]
        assert requests[0]["replaceAllText"]["containsText"]["text"] == "old"
        assert requests[0]["replaceAllText"]["replaceText"] == "new"
        assert requests[0]["replaceAllText"]["containsText"]["matchCase"] is True


def test_delete_range():
    """Test that _delete_range removes lines from the document."""
    credentials = {"type": "service_account", "project_id": "test"}
    manager = DocsManager(document_id="test-doc-id", credentials_json=credentials)

    with (
        patch("googleapiclient.discovery.build") as mock_build,
        patch("google.oauth2.service_account.Credentials.from_service_account_info"),
    ):
        mock_service, mock_documents = _mock_docs_service()
        mock_build.return_value = mock_service

        # Mock the get() method
        mock_get = MagicMock()
        mock_documents.get.return_value = mock_get
        mock_get.execute.return_value = {
            "body": {
                "content": [
                    {
                        "startIndex": 1,
                        "endIndex": 10,
                        "paragraph": {
                            "elements": [{"textRun": {"content": "Line 1\n"}}]
                        },
                    },
                    {
                        "startIndex": 10,
                        "endIndex": 20,
                        "paragraph": {
                            "elements": [{"textRun": {"content": "Line 2\n"}}]
                        },
                    },
                    {
                        "startIndex": 20,
                        "endIndex": 30,
                        "paragraph": {
                            "elements": [{"textRun": {"content": "Line 3\n"}}]
                        },
                    },
                ]
            }
        }

        # Mock the batchUpdate() method
        mock_batch = MagicMock()
        mock_documents.batchUpdate.return_value = mock_batch
        mock_batch.execute.return_value = {}

        result = manager._delete_range(start_line=1, end_line=2)
        data = json.loads(result)

        assert data["status"] == "success"
        assert "Successfully deleted 2 line(s)" in data["message"]

        # Verify deleteContentRange was called
        call_args = mock_documents.batchUpdate.call_args
        requests = call_args[1]["body"]["requests"]
        assert "deleteContentRange" in requests[0]


def test_ensure_document_creates_new():
    """Test that _ensure_document creates a new document if needed."""
    credentials = {"type": "service_account", "project_id": "test"}
    manager = DocsManager(document_title="New Test Doc", credentials_json=credentials)

    assert manager.document_id is None

    with (
        patch("googleapiclient.discovery.build") as mock_build,
        patch("google.oauth2.service_account.Credentials.from_service_account_info"),
    ):
        mock_service, mock_documents = _mock_docs_service()
        mock_build.return_value = mock_service

        # Mock document creation
        mock_create = MagicMock()
        mock_documents.create.return_value = mock_create
        mock_create.execute.return_value = {
            "documentId": "newly-created-id",
            "title": "New Test Doc",
        }

        doc_id = manager._ensure_document()

        assert doc_id == "newly-created-id"
        assert manager.document_id == "newly-created-id"

        # Verify create was called with correct title
        call_args = mock_documents.create.call_args
        assert call_args[1]["body"]["title"] == "New Test Doc"


def test_error_handling():
    """Test that methods handle errors gracefully."""
    credentials = {"type": "service_account", "project_id": "test"}
    manager = DocsManager(document_id="test-doc-id", credentials_json=credentials)

    with (
        patch("googleapiclient.discovery.build") as mock_build,
        patch("google.oauth2.service_account.Credentials.from_service_account_info"),
    ):
        mock_service, mock_documents = _mock_docs_service()
        mock_build.return_value = mock_service

        # Mock an API error
        mock_get = MagicMock()
        mock_documents.get.return_value = mock_get
        mock_get.execute.side_effect = Exception("API Error")

        result = manager._get_metadata()
        data = json.loads(result)

        assert data["status"] == "error"
        assert "API Error" in data["error"]


def test_get_tools_returns_seven_tools():
    """Test that get_tools returns all seven tools."""
    credentials = {"type": "service_account", "project_id": "test"}
    manager = DocsManager(document_id="test-doc-id", credentials_json=credentials)

    tools = manager.get_tools()

    assert len(tools) == 7
    assert tools[0].name == "docs_get_metadata"
    assert tools[1].name == "docs_read_range"
    assert tools[2].name == "docs_grep"
    assert tools[3].name == "docs_add_paragraph"
    assert tools[4].name == "docs_update_paragraph"
    assert tools[5].name == "docs_replace_text"
    assert tools[6].name == "docs_delete_range"

    # Verify get_metadata tool (no required parameters)
    assert tools[0].required == []

    # Verify read_range tool
    assert tools[1] and tools[1].parameters
    assert "start_line" in tools[1].parameters
    assert tools[1].required == ["start_line"]

    # Verify grep tool
    assert tools[2] and tools[2].parameters
    assert "pattern" in tools[2].parameters
    assert tools[2].required == ["pattern"]

    # Verify add_paragraph tool
    assert tools[3] and tools[3].parameters
    assert "text" in tools[3].parameters
    assert tools[3].required == ["text"]

    # Verify update_paragraph tool
    assert tools[4] and tools[4].parameters
    assert "line" in tools[4].parameters
    assert "new_text" in tools[4].parameters
    assert set(tools[4].required) == {"line", "new_text"}

    # Verify replace tool
    assert tools[5] and tools[5].parameters
    assert "search_text" in tools[5].parameters
    assert "replace_text" in tools[5].parameters
    assert set(tools[5].required) == {"search_text", "replace_text"}

    # Verify delete_range tool
    assert tools[6] and tools[6].parameters
    assert "start_line" in tools[6].parameters
    assert "end_line" in tools[6].parameters
    assert set(tools[6].required) == {"start_line", "end_line"}


def test_get_tools_caching():
    """Test that get_tools caches the tools."""
    credentials = {"type": "service_account", "project_id": "test"}
    manager = DocsManager(document_id="test-doc-id", credentials_json=credentials)

    tools1 = manager.get_tools()
    tools2 = manager.get_tools()

    # Should return the same list instance
    assert tools1 is tools2


def test_parse_markdown_text():
    """Test that markdown parsing works correctly."""
    credentials = {"type": "service_account", "project_id": "test"}
    manager = DocsManager(document_id="test-doc-id", credentials_json=credentials)

    # Test heading parsing
    segments = manager._parse_markdown_text("# Heading 1")
    assert any(seg.get("heading_level") == 1 for seg in segments)

    # Test bold parsing
    segments = manager._parse_markdown_text("**bold text**")
    assert any(seg.get("bold") and seg.get("text") == "bold text" for seg in segments)

    # Test italic parsing
    segments = manager._parse_markdown_text("*italic text*")
    assert any(
        seg.get("italic") and seg.get("text") == "italic text" for seg in segments
    )


if __name__ == "__main__":
    print("Running DocsManager tests...")

    test_docs_manager_initialization_with_credentials()
    print("✓ test_docs_manager_initialization_with_credentials")

    test_docs_manager_initialization_without_document_id()
    print("✓ test_docs_manager_initialization_without_document_id")

    test_docs_manager_custom_tool_names()
    print("✓ test_docs_manager_custom_tool_names")

    test_extract_text_with_formatting()
    print("✓ test_extract_text_with_formatting")

    test_get_metadata()
    print("✓ test_get_metadata")

    test_read_range()
    print("✓ test_read_range")

    test_grep()
    print("✓ test_grep")

    test_add_paragraph()
    print("✓ test_add_paragraph")

    test_update_paragraph()
    print("✓ test_update_paragraph")

    test_replace_text()
    print("✓ test_replace_text")

    test_delete_range()
    print("✓ test_delete_range")

    test_ensure_document_creates_new()
    print("✓ test_ensure_document_creates_new")

    test_error_handling()
    print("✓ test_error_handling")

    test_get_tools_returns_seven_tools()
    print("✓ test_get_tools_returns_seven_tools")

    test_get_tools_caching()
    print("✓ test_get_tools_caching")

    test_parse_markdown_text()
    print("✓ test_parse_markdown_text")

    print("\nAll tests passed! ✨")
