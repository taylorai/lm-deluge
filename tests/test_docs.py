"""Tests for the DocsManager prefab tool."""

import json
from typing import Any
from unittest.mock import MagicMock, Mock, patch

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

    manager = DocsManager(
        document_id="test-doc-id",
        credentials_json=credentials
    )

    assert manager.document_id == "test-doc-id"
    assert manager.credentials == credentials
    assert manager.read_tool_name == "docs_read"
    assert manager.append_tool_name == "docs_append"
    assert manager.insert_tool_name == "docs_insert_text"
    assert manager.replace_tool_name == "docs_replace_text"
    assert manager.clear_tool_name == "docs_clear"


def test_docs_manager_initialization_without_document_id():
    """Test that DocsManager can be initialized without a document ID."""
    credentials = {"type": "service_account", "project_id": "test"}

    manager = DocsManager(
        document_title="Test Document",
        credentials_json=credentials
    )

    assert manager.document_id is None
    assert manager.document_title == "Test Document"


def test_docs_manager_custom_tool_names():
    """Test that DocsManager allows custom tool names."""
    credentials = {"type": "service_account", "project_id": "test"}

    manager = DocsManager(
        document_id="test-doc-id",
        credentials_json=credentials,
        read_tool_name="custom_read",
        append_tool_name="custom_append",
        insert_tool_name="custom_insert",
        replace_tool_name="custom_replace",
        clear_tool_name="custom_clear"
    )

    tools = manager.get_tools()
    assert len(tools) == 5
    assert tools[0].name == "custom_read"
    assert tools[1].name == "custom_append"
    assert tools[2].name == "custom_insert"
    assert tools[3].name == "custom_replace"
    assert tools[4].name == "custom_clear"


def _mock_docs_service() -> tuple[MagicMock, MagicMock]:
    """Create a mock Google Docs service."""
    mock_service = MagicMock()
    mock_documents = MagicMock()
    mock_service.documents.return_value = mock_documents
    return mock_service, mock_documents


def test_extract_text():
    """Test that _extract_text correctly extracts text from document structure."""
    credentials = {"type": "service_account", "project_id": "test"}
    manager = DocsManager(document_id="test-doc-id", credentials_json=credentials)

    # Mock document structure
    document = {
        'body': {
            'content': [
                {
                    'paragraph': {
                        'elements': [
                            {'textRun': {'content': 'Hello '}},
                            {'textRun': {'content': 'World'}}
                        ]
                    }
                },
                {
                    'paragraph': {
                        'elements': [
                            {'textRun': {'content': '\n'}}
                        ]
                    }
                }
            ]
        }
    }

    text = manager._extract_text(document)
    assert text == 'Hello World\n'


def test_read_document():
    """Test that read returns document content."""
    credentials = {"type": "service_account", "project_id": "test"}
    manager = DocsManager(document_id="test-doc-id", credentials_json=credentials)

    with patch('googleapiclient.discovery.build') as mock_build, \
         patch('google.oauth2.service_account.Credentials.from_service_account_info'):

        mock_service, mock_documents = _mock_docs_service()
        mock_build.return_value = mock_service

        # Mock the get() method
        mock_get = MagicMock()
        mock_documents.get.return_value = mock_get
        mock_get.execute.return_value = {
            'documentId': 'test-doc-id',
            'title': 'Test Document',
            'body': {
                'content': [
                    {
                        'paragraph': {
                            'elements': [
                                {'textRun': {'content': 'Test content\n'}}
                            ]
                        }
                    }
                ]
            }
        }

        result = manager._read()
        data = json.loads(result)

        assert data['status'] == 'success'
        assert data['document_id'] == 'test-doc-id'
        assert data['title'] == 'Test Document'
        assert data['content'] == 'Test content\n'
        assert 'https://docs.google.com/document/d/test-doc-id' in data['url']


def test_append_text():
    """Test that append adds text to the end of the document."""
    credentials = {"type": "service_account", "project_id": "test"}
    manager = DocsManager(document_id="test-doc-id", credentials_json=credentials)

    with patch('googleapiclient.discovery.build') as mock_build, \
         patch('google.oauth2.service_account.Credentials.from_service_account_info'):

        mock_service, mock_documents = _mock_docs_service()
        mock_build.return_value = mock_service

        # Mock the get() method to return document structure
        mock_get = MagicMock()
        mock_documents.get.return_value = mock_get
        mock_get.execute.return_value = {
            'body': {
                'content': [
                    {'startIndex': 1, 'endIndex': 10},
                    {'startIndex': 10, 'endIndex': 11}
                ]
            }
        }

        # Mock the batchUpdate() method
        mock_batch = MagicMock()
        mock_documents.batchUpdate.return_value = mock_batch
        mock_batch.execute.return_value = {}

        result = manager._append("New text")
        data = json.loads(result)

        assert data['status'] == 'success'
        assert 'Successfully appended 8 characters' in data['message']

        # Verify batchUpdate was called with correct parameters
        mock_documents.batchUpdate.assert_called_once()
        call_args = mock_documents.batchUpdate.call_args
        requests = call_args[1]['body']['requests']
        assert len(requests) == 1
        assert requests[0]['insertText']['text'] == 'New text'
        assert requests[0]['insertText']['location']['index'] == 10  # endIndex - 1


def test_insert_text():
    """Test that insert_text adds text at a specific index."""
    credentials = {"type": "service_account", "project_id": "test"}
    manager = DocsManager(document_id="test-doc-id", credentials_json=credentials)

    with patch('googleapiclient.discovery.build') as mock_build, \
         patch('google.oauth2.service_account.Credentials.from_service_account_info'):

        mock_service, mock_documents = _mock_docs_service()
        mock_build.return_value = mock_service

        # Mock the batchUpdate() method
        mock_batch = MagicMock()
        mock_documents.batchUpdate.return_value = mock_batch
        mock_batch.execute.return_value = {}

        result = manager._insert_text("Inserted text", 5)
        data = json.loads(result)

        assert data['status'] == 'success'
        assert 'Successfully inserted 13 characters at index 5' in data['message']

        # Verify batchUpdate was called correctly
        call_args = mock_documents.batchUpdate.call_args
        requests = call_args[1]['body']['requests']
        assert requests[0]['insertText']['text'] == 'Inserted text'
        assert requests[0]['insertText']['location']['index'] == 5


def test_replace_text():
    """Test that replace_text replaces all occurrences."""
    credentials = {"type": "service_account", "project_id": "test"}
    manager = DocsManager(document_id="test-doc-id", credentials_json=credentials)

    with patch('googleapiclient.discovery.build') as mock_build, \
         patch('google.oauth2.service_account.Credentials.from_service_account_info'):

        mock_service, mock_documents = _mock_docs_service()
        mock_build.return_value = mock_service

        # Mock the batchUpdate() method
        mock_batch = MagicMock()
        mock_documents.batchUpdate.return_value = mock_batch
        mock_batch.execute.return_value = {
            'replies': [
                {'replaceAllText': {'occurrencesChanged': 3}}
            ]
        }

        result = manager._replace_text("old", "new", match_case=True)
        data = json.loads(result)

        assert data['status'] == 'success'
        assert data['replacements'] == 3
        assert "Replaced 3 occurrence(s)" in data['message']

        # Verify batchUpdate was called correctly
        call_args = mock_documents.batchUpdate.call_args
        requests = call_args[1]['body']['requests']
        assert requests[0]['replaceAllText']['containsText']['text'] == 'old'
        assert requests[0]['replaceAllText']['replaceText'] == 'new'
        assert requests[0]['replaceAllText']['containsText']['matchCase'] is True


def test_clear_document():
    """Test that clear removes all content."""
    credentials = {"type": "service_account", "project_id": "test"}
    manager = DocsManager(document_id="test-doc-id", credentials_json=credentials)

    with patch('googleapiclient.discovery.build') as mock_build, \
         patch('google.oauth2.service_account.Credentials.from_service_account_info'):

        mock_service, mock_documents = _mock_docs_service()
        mock_build.return_value = mock_service

        # Mock the get() method
        mock_get = MagicMock()
        mock_documents.get.return_value = mock_get
        mock_get.execute.return_value = {
            'body': {
                'content': [
                    {'startIndex': 1, 'endIndex': 50},
                    {'startIndex': 50, 'endIndex': 51}
                ]
            }
        }

        # Mock the batchUpdate() method
        mock_batch = MagicMock()
        mock_documents.batchUpdate.return_value = mock_batch
        mock_batch.execute.return_value = {}

        result = manager._clear()
        data = json.loads(result)

        assert data['status'] == 'success'
        assert 'Successfully cleared document content' in data['message']

        # Verify deleteContentRange was called correctly
        call_args = mock_documents.batchUpdate.call_args
        requests = call_args[1]['body']['requests']
        assert 'deleteContentRange' in requests[0]
        assert requests[0]['deleteContentRange']['range']['startIndex'] == 1
        assert requests[0]['deleteContentRange']['range']['endIndex'] == 50


def test_clear_empty_document():
    """Test that clear handles already empty documents."""
    credentials = {"type": "service_account", "project_id": "test"}
    manager = DocsManager(document_id="test-doc-id", credentials_json=credentials)

    with patch('googleapiclient.discovery.build') as mock_build, \
         patch('google.oauth2.service_account.Credentials.from_service_account_info'):

        mock_service, mock_documents = _mock_docs_service()
        mock_build.return_value = mock_service

        # Mock an empty document
        mock_get = MagicMock()
        mock_documents.get.return_value = mock_get
        mock_get.execute.return_value = {
            'body': {
                'content': [
                    {'startIndex': 1, 'endIndex': 1}
                ]
            }
        }

        result = manager._clear()
        data = json.loads(result)

        assert data['status'] == 'success'
        assert 'already empty' in data['message']


def test_ensure_document_creates_new():
    """Test that _ensure_document creates a new document if needed."""
    credentials = {"type": "service_account", "project_id": "test"}
    manager = DocsManager(
        document_title="New Test Doc",
        credentials_json=credentials
    )

    assert manager.document_id is None

    with patch('googleapiclient.discovery.build') as mock_build, \
         patch('google.oauth2.service_account.Credentials.from_service_account_info'):

        mock_service, mock_documents = _mock_docs_service()
        mock_build.return_value = mock_service

        # Mock document creation
        mock_create = MagicMock()
        mock_documents.create.return_value = mock_create
        mock_create.execute.return_value = {
            'documentId': 'newly-created-id',
            'title': 'New Test Doc'
        }

        doc_id = manager._ensure_document()

        assert doc_id == 'newly-created-id'
        assert manager.document_id == 'newly-created-id'

        # Verify create was called with correct title
        call_args = mock_documents.create.call_args
        assert call_args[1]['body']['title'] == 'New Test Doc'


def test_error_handling():
    """Test that methods handle errors gracefully."""
    credentials = {"type": "service_account", "project_id": "test"}
    manager = DocsManager(document_id="test-doc-id", credentials_json=credentials)

    with patch('googleapiclient.discovery.build') as mock_build, \
         patch('google.oauth2.service_account.Credentials.from_service_account_info'):

        mock_service, mock_documents = _mock_docs_service()
        mock_build.return_value = mock_service

        # Mock an API error
        mock_get = MagicMock()
        mock_documents.get.return_value = mock_get
        mock_get.execute.side_effect = Exception("API Error")

        result = manager._read()
        data = json.loads(result)

        assert data['status'] == 'error'
        assert 'API Error' in data['error']


def test_get_tools_returns_five_tools():
    """Test that get_tools returns all five tools."""
    credentials = {"type": "service_account", "project_id": "test"}
    manager = DocsManager(document_id="test-doc-id", credentials_json=credentials)

    tools = manager.get_tools()

    assert len(tools) == 5
    assert tools[0].name == "docs_read"
    assert tools[1].name == "docs_append"
    assert tools[2].name == "docs_insert_text"
    assert tools[3].name == "docs_replace_text"
    assert tools[4].name == "docs_clear"

    # Verify read tool (no parameters)
    assert tools[0].required == []

    # Verify append tool
    assert "text" in tools[1].parameters
    assert tools[1].required == ["text"]

    # Verify insert tool
    assert "text" in tools[2].parameters
    assert "index" in tools[2].parameters
    assert set(tools[2].required) == {"text", "index"}

    # Verify replace tool
    assert "search_text" in tools[3].parameters
    assert "replace_text" in tools[3].parameters
    assert set(tools[3].required) == {"search_text", "replace_text"}

    # Verify clear tool (no parameters)
    assert tools[4].required == []


def test_get_tools_caching():
    """Test that get_tools caches the tools."""
    credentials = {"type": "service_account", "project_id": "test"}
    manager = DocsManager(document_id="test-doc-id", credentials_json=credentials)

    tools1 = manager.get_tools()
    tools2 = manager.get_tools()

    # Should return the same list instance
    assert tools1 is tools2


if __name__ == "__main__":
    print("Running DocsManager tests...")

    test_docs_manager_initialization_with_credentials()
    print("✓ test_docs_manager_initialization_with_credentials")

    test_docs_manager_initialization_without_document_id()
    print("✓ test_docs_manager_initialization_without_document_id")

    test_docs_manager_custom_tool_names()
    print("✓ test_docs_manager_custom_tool_names")

    test_extract_text()
    print("✓ test_extract_text")

    test_read_document()
    print("✓ test_read_document")

    test_append_text()
    print("✓ test_append_text")

    test_insert_text()
    print("✓ test_insert_text")

    test_replace_text()
    print("✓ test_replace_text")

    test_clear_document()
    print("✓ test_clear_document")

    test_clear_empty_document()
    print("✓ test_clear_empty_document")

    test_ensure_document_creates_new()
    print("✓ test_ensure_document_creates_new")

    test_error_handling()
    print("✓ test_error_handling")

    test_get_tools_returns_five_tools()
    print("✓ test_get_tools_returns_five_tools")

    test_get_tools_caching()
    print("✓ test_get_tools_caching")

    print("\nAll tests passed! ✨")
