"""Tests for the SheetsManager prefab tool."""

import json
from typing import Any
from unittest.mock import MagicMock, Mock, patch

from lm_deluge.tool.prefab.sheets import SheetsManager


def test_sheets_manager_initialization_with_credentials():
    """Test that SheetsManager can be initialized with credentials."""
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

    manager = SheetsManager(
        sheet_id="test-sheet-id",
        credentials_json=credentials
    )

    assert manager.sheet_id == "test-sheet-id"
    assert manager.credentials == credentials
    assert manager.read_tool_name == "sheets_read_range"
    assert manager.update_tool_name == "sheets_update_cell"


def test_sheets_manager_custom_tool_names():
    """Test that SheetsManager allows custom tool names."""
    credentials = {"type": "service_account", "project_id": "test"}

    manager = SheetsManager(
        sheet_id="test-sheet-id",
        credentials_json=credentials,
        read_tool_name="custom_read",
        update_tool_name="custom_update"
    )

    tools = manager.get_tools()
    assert len(tools) == 2
    assert tools[0].name == "custom_read"
    assert tools[1].name == "custom_update"


def test_col_num_to_letter():
    """Test column number to letter conversion."""
    assert SheetsManager._col_num_to_letter(1) == "A"
    assert SheetsManager._col_num_to_letter(2) == "B"
    assert SheetsManager._col_num_to_letter(26) == "Z"
    assert SheetsManager._col_num_to_letter(27) == "AA"
    assert SheetsManager._col_num_to_letter(52) == "AZ"
    assert SheetsManager._col_num_to_letter(53) == "BA"
    assert SheetsManager._col_num_to_letter(702) == "ZZ"
    assert SheetsManager._col_num_to_letter(703) == "AAA"


def _mock_sheets_service() -> tuple[MagicMock, MagicMock]:
    """Create a mock Google Sheets service."""
    mock_service = MagicMock()
    mock_spreadsheets = MagicMock()
    mock_service.spreadsheets.return_value = mock_spreadsheets
    return mock_service, mock_spreadsheets


def test_read_range_returns_html_table():
    """Test that read_range returns an HTML table with row/col attributes."""
    credentials = {"type": "service_account", "project_id": "test"}
    manager = SheetsManager(sheet_id="test-sheet-id", credentials_json=credentials)

    # Mock the Google Sheets API
    with patch('googleapiclient.discovery.build') as mock_build, \
         patch('google.oauth2.service_account.Credentials.from_service_account_info'):

        mock_service, mock_spreadsheets = _mock_sheets_service()
        mock_build.return_value = mock_service

        # Mock the API response
        mock_values = MagicMock()
        mock_spreadsheets.values.return_value = mock_values
        mock_get = MagicMock()
        mock_values.get.return_value = mock_get
        mock_get.execute.return_value = {
            'values': [
                ['Name', 'Age', 'City'],
                ['Alice', '30', 'NYC'],
                ['Bob', '25', 'LA']
            ]
        }

        result = manager._read_range("A1:C3")
        data = json.loads(result)

        assert data['status'] == 'success'
        assert data['rows'] == 3
        assert '<table border="1">' in data['html']
        assert 'row="1"' in data['html']
        assert 'col="A"' in data['html']
        assert 'cell="A1"' in data['html']
        assert 'Name' in data['html']
        assert 'Alice' in data['html']


def test_read_range_empty_response():
    """Test that read_range handles empty ranges."""
    credentials = {"type": "service_account", "project_id": "test"}
    manager = SheetsManager(sheet_id="test-sheet-id", credentials_json=credentials)

    with patch('googleapiclient.discovery.build') as mock_build, \
         patch('google.oauth2.service_account.Credentials.from_service_account_info'):

        mock_service, mock_spreadsheets = _mock_sheets_service()
        mock_build.return_value = mock_service

        mock_values = MagicMock()
        mock_spreadsheets.values.return_value = mock_values
        mock_get = MagicMock()
        mock_values.get.return_value = mock_get
        mock_get.execute.return_value = {'values': []}

        result = manager._read_range("A1:A1")
        data = json.loads(result)

        assert data['status'] == 'success'
        assert 'No data found' in data['message']


def test_read_range_error_handling():
    """Test that read_range handles errors gracefully."""
    credentials = {"type": "service_account", "project_id": "test"}
    manager = SheetsManager(sheet_id="test-sheet-id", credentials_json=credentials)

    with patch('googleapiclient.discovery.build') as mock_build, \
         patch('google.oauth2.service_account.Credentials.from_service_account_info'):

        mock_service, mock_spreadsheets = _mock_sheets_service()
        mock_build.return_value = mock_service

        mock_values = MagicMock()
        mock_spreadsheets.values.return_value = mock_values
        mock_get = MagicMock()
        mock_values.get.return_value = mock_get
        mock_get.execute.side_effect = Exception("API Error")

        result = manager._read_range("A1:A1")
        data = json.loads(result)

        assert data['status'] == 'error'
        assert 'API Error' in data['error']


def test_update_cell_success():
    """Test that update_cell successfully updates a cell."""
    credentials = {"type": "service_account", "project_id": "test"}
    manager = SheetsManager(sheet_id="test-sheet-id", credentials_json=credentials)

    with patch('googleapiclient.discovery.build') as mock_build, \
         patch('google.oauth2.service_account.Credentials.from_service_account_info'):

        mock_service, mock_spreadsheets = _mock_sheets_service()
        mock_build.return_value = mock_service

        mock_values = MagicMock()
        mock_spreadsheets.values.return_value = mock_values
        mock_update = MagicMock()
        mock_values.update.return_value = mock_update
        mock_update.execute.return_value = {
            'updatedCells': 1,
            'updatedRange': 'Sheet1!A1'
        }

        result = manager._update_cell("A1", "Hello")
        data = json.loads(result)

        assert data['status'] == 'success'
        assert data['updated_cells'] == 1
        assert 'Successfully updated A1' in data['message']

        # Verify the API was called correctly
        mock_values.update.assert_called_once()
        call_kwargs = mock_values.update.call_args[1]
        assert call_kwargs['range'] == 'A1'
        assert call_kwargs['valueInputOption'] == 'USER_ENTERED'
        assert call_kwargs['body'] == {'values': [['Hello']]}


def test_update_cell_error_handling():
    """Test that update_cell handles errors gracefully."""
    credentials = {"type": "service_account", "project_id": "test"}
    manager = SheetsManager(sheet_id="test-sheet-id", credentials_json=credentials)

    with patch('googleapiclient.discovery.build') as mock_build, \
         patch('google.oauth2.service_account.Credentials.from_service_account_info'):

        mock_service, mock_spreadsheets = _mock_sheets_service()
        mock_build.return_value = mock_service

        mock_values = MagicMock()
        mock_spreadsheets.values.return_value = mock_values
        mock_update = MagicMock()
        mock_values.update.return_value = mock_update
        mock_update.execute.side_effect = Exception("Update failed")

        result = manager._update_cell("A1", "Hello")
        data = json.loads(result)

        assert data['status'] == 'error'
        assert 'Update failed' in data['error']


def test_get_tools_returns_two_tools():
    """Test that get_tools returns both read and update tools."""
    credentials = {"type": "service_account", "project_id": "test"}
    manager = SheetsManager(sheet_id="test-sheet-id", credentials_json=credentials)

    tools = manager.get_tools()

    assert len(tools) == 2
    assert tools[0].name == "sheets_read_range"
    assert tools[1].name == "sheets_update_cell"

    # Verify read tool parameters
    assert "range_spec" in tools[0].parameters
    assert tools[0].required == ["range_spec"]

    # Verify update tool parameters
    assert "cell" in tools[1].parameters
    assert "value" in tools[1].parameters
    assert set(tools[1].required) == {"cell", "value"}


def test_get_tools_caching():
    """Test that get_tools caches the tools."""
    credentials = {"type": "service_account", "project_id": "test"}
    manager = SheetsManager(sheet_id="test-sheet-id", credentials_json=credentials)

    tools1 = manager.get_tools()
    tools2 = manager.get_tools()

    # Should return the same list instance
    assert tools1 is tools2


if __name__ == "__main__":
    print("Running SheetsManager tests...")

    test_sheets_manager_initialization_with_credentials()
    print("✓ test_sheets_manager_initialization_with_credentials")

    test_sheets_manager_custom_tool_names()
    print("✓ test_sheets_manager_custom_tool_names")

    test_col_num_to_letter()
    print("✓ test_col_num_to_letter")

    test_read_range_returns_html_table()
    print("✓ test_read_range_returns_html_table")

    test_read_range_empty_response()
    print("✓ test_read_range_empty_response")

    test_read_range_error_handling()
    print("✓ test_read_range_error_handling")

    test_update_cell_success()
    print("✓ test_update_cell_success")

    test_update_cell_error_handling()
    print("✓ test_update_cell_error_handling")

    test_get_tools_returns_two_tools()
    print("✓ test_get_tools_returns_two_tools")

    test_get_tools_caching()
    print("✓ test_get_tools_caching")

    print("\nAll tests passed! ✨")
