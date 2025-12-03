"""Google Sheets manipulation prefab tool."""

import json
import os
from typing import Any

from lm_deluge.tool import Tool


class SheetsManager:
    """
    A prefab tool for manipulating Google Sheets.

    Provides tools to read ranges and update individual cells in a Google Sheet.
    Outputs are formatted as LLM-friendly HTML tables with row/col attributes.

    Args:
        sheet_id: The ID of the Google Sheet to manipulate
        credentials_json: Optional. JSON string or dict containing Google service account credentials.
                         If not provided, will look for GOOGLE_SHEETS_CREDENTIALS env variable.
        credentials_file: Optional. Path to a JSON file containing credentials.
                         Only used if credentials_json is not provided.
        read_tool_name: Name for the read range tool (default: "sheets_read_range")
        update_tool_name: Name for the update cell tool (default: "sheets_update_cell")

    Example:
        ```python
        # Using credentials from environment
        manager = SheetsManager(sheet_id="your-sheet-id-here")

        # Using credentials directly
        manager = SheetsManager(
            sheet_id="your-sheet-id-here",
            credentials_json={"type": "service_account", ...}
        )

        # Get tools
        tools = manager.get_tools()
        ```
    """

    def __init__(
        self,
        sheet_id: str,
        *,
        credentials_json: str | dict[str, Any] | None = None,
        credentials_file: str | None = None,
        list_sheets_tool_name: str = "sheets_list_sheets",
        get_used_range_tool_name: str = "sheets_get_used_range",
        read_tool_name: str = "sheets_read_range",
        update_tool_name: str = "sheets_update_cell",
    ):
        self.sheet_id = sheet_id
        self.list_sheets_tool_name = list_sheets_tool_name
        self.get_used_range_tool_name = get_used_range_tool_name
        self.read_tool_name = read_tool_name
        self.update_tool_name = update_tool_name

        # Handle credentials
        if credentials_json is not None:
            if isinstance(credentials_json, str):
                self.credentials = json.loads(credentials_json)
            else:
                self.credentials = credentials_json
        elif credentials_file is not None:
            with open(credentials_file, "r") as f:
                self.credentials = json.load(f)
        else:
            # Try to load from environment
            env_creds = os.environ.get("GOOGLE_SHEETS_CREDENTIALS")
            if env_creds:
                self.credentials = json.loads(env_creds)
            else:
                raise ValueError(
                    "No credentials provided. Please provide credentials_json, "
                    "credentials_file, or set GOOGLE_SHEETS_CREDENTIALS environment variable."
                )

        self._service = None
        self._tools: list[Tool] | None = None

    def _get_service(self):
        """Lazily initialize the Google Sheets API service."""
        if self._service is not None:
            return self._service

        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build
        except ImportError:
            raise ImportError(
                "Google Sheets API dependencies not installed. "
                "Please install with: pip install google-api-python-client google-auth"
            )

        # Create credentials from service account info
        creds = service_account.Credentials.from_service_account_info(
            self.credentials, scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )

        # Build the service
        self._service = build("sheets", "v4", credentials=creds)
        return self._service

    def _list_sheets(self) -> str:
        """
        List all sheets (tabs) in the spreadsheet.

        Returns:
            JSON string with status and list of sheet names
        """
        try:
            service = self._get_service()
            spreadsheet = (
                service.spreadsheets()
                .get(spreadsheetId=self.sheet_id, fields="sheets.properties")
                .execute()
            )

            sheets = []
            for sheet in spreadsheet.get("sheets", []):
                props = sheet.get("properties", {})
                sheets.append(
                    {
                        "name": props.get("title", ""),
                        "index": props.get("index", 0),
                        "sheetId": props.get("sheetId", 0),
                    }
                )

            return json.dumps({"status": "success", "sheets": sheets})

        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

    def _get_used_range(self, sheet_name: str | None = None) -> str:
        """
        Get the used range of a sheet (the bounding box of all non-empty cells).

        Args:
            sheet_name: Name of the sheet to check. If None, uses the first sheet.

        Returns:
            JSON string with status and the used range in A1 notation
        """
        try:
            service = self._get_service()

            # If no sheet name provided, get the first sheet's name
            if not sheet_name:
                spreadsheet = (
                    service.spreadsheets()
                    .get(spreadsheetId=self.sheet_id, fields="sheets.properties.title")
                    .execute()
                )
                sheets = spreadsheet.get("sheets", [])
                if not sheets:
                    return json.dumps(
                        {"status": "error", "error": "No sheets found in spreadsheet"}
                    )
                sheet_name = sheets[0]["properties"]["title"]

            # Get all values from the sheet to determine the used range
            result = (
                service.spreadsheets()
                .values()
                .get(spreadsheetId=self.sheet_id, range=f"'{sheet_name}'")
                .execute()
            )

            values = result.get("values", [])

            if not values:
                return json.dumps(
                    {
                        "status": "success",
                        "sheet_name": sheet_name,
                        "used_range": None,
                        "message": "Sheet is empty",
                    }
                )

            # Find the max column across all rows
            max_col = 0
            for row in values:
                if row:  # Skip empty rows
                    max_col = max(max_col, len(row))

            num_rows = len(values)
            end_col = self._col_num_to_letter(max_col)
            used_range = f"A1:{end_col}{num_rows}"

            return json.dumps(
                {
                    "status": "success",
                    "sheet_name": sheet_name,
                    "used_range": f"'{sheet_name}'!{used_range}",
                    "rows": num_rows,
                    "cols": max_col,
                }
            )

        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

    def _read_range(self, range_spec: str) -> str:
        """
        Read a range from the Google Sheet and return as HTML table.

        Args:
            range_spec: A1 notation range (e.g., "Sheet1!A1:C10" or just "A1:C10")

        Returns:
            JSON string with status and HTML table data
        """
        try:
            service = self._get_service()
            sheet = service.spreadsheets()

            result = (
                sheet.values()
                .get(spreadsheetId=self.sheet_id, range=range_spec)
                .execute()
            )

            values = result.get("values", [])

            if not values:
                return json.dumps(
                    {
                        "status": "success",
                        "message": "No data found in range",
                        "html": "<p>No data found</p>",
                    }
                )

            # Convert to HTML table with cell attribute for reference
            html_parts = ["<table>"]

            for row_idx, row in enumerate(values, start=1):
                html_parts.append("<tr>")
                for col_idx, cell_value in enumerate(row, start=1):
                    # Convert column index to letter (1=A, 2=B, etc.)
                    col_letter = self._col_num_to_letter(col_idx)
                    cell_ref = f"{col_letter}{row_idx}"

                    html_parts.append(f'<td cell="{cell_ref}">{cell_value}</td>')
                html_parts.append("</tr>")

            html_parts.append("</table>")
            html_table = "\n".join(html_parts)

            return json.dumps(
                {"status": "success", "rows": len(values), "html": html_table}
            )

        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

    def _update_cell(self, cell: str, value: str) -> str:
        """
        Update a single cell in the Google Sheet.

        Args:
            cell: Cell reference in A1 notation (e.g., "A1", "B5", "Sheet1!C3")
            value: The value to set in the cell

        Returns:
            JSON string with status and result
        """
        try:
            service = self._get_service()
            sheet = service.spreadsheets()

            body = {"values": [[value]]}

            result = (
                sheet.values()
                .update(
                    spreadsheetId=self.sheet_id,
                    range=cell,
                    valueInputOption="USER_ENTERED",  # Parse values like formulas, numbers, dates
                    body=body,
                )
                .execute()
            )

            return json.dumps(
                {
                    "status": "success",
                    "updated_cells": result.get("updatedCells", 0),
                    "updated_range": result.get("updatedRange", ""),
                    "message": f"Successfully updated {cell} to '{value}'",
                }
            )

        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

    @staticmethod
    def _col_num_to_letter(n: int) -> str:
        """Convert column number to letter (1=A, 2=B, ..., 26=Z, 27=AA, etc.)."""
        result = ""
        while n > 0:
            n -= 1
            result = chr(65 + (n % 26)) + result
            n //= 26
        return result

    def get_tools(self) -> list[Tool]:
        """Return the list of Google Sheets tools."""
        if self._tools is not None:
            return self._tools

        self._tools = [
            Tool(
                name=self.list_sheets_tool_name,
                description=(
                    "List all sheets (tabs) in the spreadsheet. Returns the name, index, "
                    "and ID of each sheet."
                ),
                run=self._list_sheets,
                parameters={},
                required=[],
            ),
            Tool(
                name=self.get_used_range_tool_name,
                description=(
                    "Get the used range of a sheet (the bounding box containing all non-empty cells). "
                    "Returns the range in A1 notation (e.g., 'Sheet1'!A1:C4)."
                ),
                run=self._get_used_range,
                parameters={
                    "sheet_name": {
                        "type": "string",
                        "description": (
                            "Name of the sheet to check. If not provided, uses the first sheet."
                        ),
                    }
                },
                required=[],
            ),
            Tool(
                name=self.read_tool_name,
                description=(
                    "Read a range of cells from the Google Sheet and return as an HTML table. "
                    "Each cell has a 'cell' attribute with its A1 reference (e.g., cell='A1'). "
                    "Use A1 notation for the range (e.g., 'A1:C10' or 'Sheet1!A1:C10')."
                ),
                run=self._read_range,
                parameters={
                    "range_spec": {
                        "type": "string",
                        "description": (
                            "The range to read in A1 notation. Examples: 'A1:C10', 'Sheet1!A1:C10', "
                            "'A:A' (entire column A), '1:1' (entire row 1)"
                        ),
                    }
                },
                required=["range_spec"],
            ),
            Tool(
                name=self.update_tool_name,
                description=(
                    "Update a single cell in the Google Sheet. The value will be parsed "
                    "automatically (formulas, numbers, dates, etc.)."
                ),
                run=self._update_cell,
                parameters={
                    "cell": {
                        "type": "string",
                        "description": (
                            "The cell to update in A1 notation. Examples: 'A1', 'B5', 'Sheet1!C3'"
                        ),
                    },
                    "value": {
                        "type": "string",
                        "description": "The value to set in the cell",
                    },
                },
                required=["cell", "value"],
            ),
        ]

        return self._tools
