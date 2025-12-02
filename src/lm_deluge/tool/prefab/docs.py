"""Google Docs manipulation prefab tool."""

import json
import os
from typing import Any

from lm_deluge.tool import Tool


class DocsManager:
    """
    A prefab tool for manipulating Google Docs.

    Provides tools to read, write, and edit Google Docs documents.
    All outputs are formatted to be LLM-friendly.

    Args:
        document_id: Optional. The ID of the Google Doc to manipulate.
                    If not provided, a new document will be created on first use.
        credentials_json: Optional. JSON string or dict containing Google service account credentials.
                         If not provided, will look for GOOGLE_DOCS_CREDENTIALS env variable.
        credentials_file: Optional. Path to a JSON file containing credentials.
                         Only used if credentials_json is not provided.
        document_title: Optional. Title for new document (only used if document_id is not provided)
        read_tool_name: Name for the read tool (default: "docs_read")
        append_tool_name: Name for the append tool (default: "docs_append")
        insert_tool_name: Name for the insert tool (default: "docs_insert_text")
        replace_tool_name: Name for the replace tool (default: "docs_replace_text")
        clear_tool_name: Name for the clear tool (default: "docs_clear")

    Example:
        ```python
        # Using existing document
        manager = DocsManager(
            document_id="your-doc-id-here",
            credentials_json={"type": "service_account", ...}
        )

        # Creating a new document
        manager = DocsManager(
            document_title="My New Document",
            credentials_json={"type": "service_account", ...}
        )

        # Get tools
        tools = manager.get_tools()
        ```
    """

    def __init__(
        self,
        document_id: str | None = None,
        *,
        credentials_json: str | dict[str, Any] | None = None,
        credentials_file: str | None = None,
        document_title: str = "New Document",
        read_tool_name: str = "docs_read",
        append_tool_name: str = "docs_append",
        insert_tool_name: str = "docs_insert_text",
        replace_tool_name: str = "docs_replace_text",
        clear_tool_name: str = "docs_clear",
    ):
        self.document_id = document_id
        self.document_title = document_title
        self.read_tool_name = read_tool_name
        self.append_tool_name = append_tool_name
        self.insert_tool_name = insert_tool_name
        self.replace_tool_name = replace_tool_name
        self.clear_tool_name = clear_tool_name

        # Handle credentials
        if credentials_json is not None:
            if isinstance(credentials_json, str):
                self.credentials = json.loads(credentials_json)
            else:
                self.credentials = credentials_json
        elif credentials_file is not None:
            with open(credentials_file, 'r') as f:
                self.credentials = json.load(f)
        else:
            # Try to load from environment
            env_creds = os.environ.get('GOOGLE_DOCS_CREDENTIALS')
            if env_creds:
                self.credentials = json.loads(env_creds)
            else:
                raise ValueError(
                    "No credentials provided. Please provide credentials_json, "
                    "credentials_file, or set GOOGLE_DOCS_CREDENTIALS environment variable."
                )

        self._service = None
        self._tools: list[Tool] | None = None

    def _get_service(self):
        """Lazily initialize the Google Docs API service."""
        if self._service is not None:
            return self._service

        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build
        except ImportError:
            raise ImportError(
                "Google Docs API dependencies not installed. "
                "Please install with: pip install google-api-python-client google-auth"
            )

        # Create credentials from service account info
        creds = service_account.Credentials.from_service_account_info(
            self.credentials,
            scopes=['https://www.googleapis.com/auth/documents']
        )

        # Build the service
        self._service = build('docs', 'v1', credentials=creds)
        return self._service

    def _ensure_document(self) -> str:
        """
        Ensure we have a document ID. If not, create a new document.

        Returns:
            The document ID
        """
        if self.document_id is not None:
            return self.document_id

        # Create a new document
        try:
            service = self._get_service()
            document = service.documents().create(
                body={'title': self.document_title}
            ).execute()
            self.document_id = document.get('documentId')
            return self.document_id
        except Exception as e:
            raise RuntimeError(f"Failed to create document: {str(e)}")

    def _extract_text(self, document: dict[str, Any]) -> str:
        """Extract all text content from a document."""
        content = document.get('body', {}).get('content', [])
        text_parts = []

        for element in content:
            if 'paragraph' in element:
                for para_element in element['paragraph'].get('elements', []):
                    text_run = para_element.get('textRun')
                    if text_run:
                        text_parts.append(text_run.get('content', ''))
            elif 'table' in element:
                # Handle tables
                table = element['table']
                for row in table.get('tableRows', []):
                    for cell in row.get('tableCells', []):
                        for cell_content in cell.get('content', []):
                            if 'paragraph' in cell_content:
                                for para_element in cell_content['paragraph'].get('elements', []):
                                    text_run = para_element.get('textRun')
                                    if text_run:
                                        text_parts.append(text_run.get('content', ''))

        return ''.join(text_parts)

    def _read(self) -> str:
        """
        Read the entire document content.

        Returns:
            JSON string with status, document info, and content
        """
        try:
            doc_id = self._ensure_document()
            service = self._get_service()

            document = service.documents().get(documentId=doc_id).execute()

            title = document.get('title', '')
            text_content = self._extract_text(document)

            return json.dumps({
                "status": "success",
                "document_id": doc_id,
                "title": title,
                "content": text_content,
                "url": f"https://docs.google.com/document/d/{doc_id}"
            })

        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": str(e)
            })

    def _append(self, text: str) -> str:
        """
        Append text to the end of the document.

        Args:
            text: The text to append

        Returns:
            JSON string with status and result
        """
        try:
            doc_id = self._ensure_document()
            service = self._get_service()

            # Get the current document to find the end index
            document = service.documents().get(documentId=doc_id).execute()
            body_content = document.get('body', {}).get('content', [])

            # The end index is at the last element's endIndex - 1
            # (documents always end with a newline that we can't modify)
            end_index = 1
            if body_content:
                end_index = body_content[-1].get('endIndex', 1) - 1

            # Insert the text
            requests = [{
                'insertText': {
                    'location': {'index': end_index},
                    'text': text
                }
            }]

            service.documents().batchUpdate(
                documentId=doc_id,
                body={'requests': requests}
            ).execute()

            return json.dumps({
                "status": "success",
                "message": f"Successfully appended {len(text)} characters",
                "document_id": doc_id,
                "url": f"https://docs.google.com/document/d/{doc_id}"
            })

        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": str(e)
            })

    def _insert_text(self, text: str, index: int) -> str:
        """
        Insert text at a specific index in the document.

        Args:
            text: The text to insert
            index: The position to insert at (1-based, use 1 for start of document)

        Returns:
            JSON string with status and result
        """
        try:
            doc_id = self._ensure_document()
            service = self._get_service()

            requests = [{
                'insertText': {
                    'location': {'index': index},
                    'text': text
                }
            }]

            service.documents().batchUpdate(
                documentId=doc_id,
                body={'requests': requests}
            ).execute()

            return json.dumps({
                "status": "success",
                "message": f"Successfully inserted {len(text)} characters at index {index}",
                "document_id": doc_id,
                "url": f"https://docs.google.com/document/d/{doc_id}"
            })

        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": str(e)
            })

    def _replace_text(self, search_text: str, replace_text: str, match_case: bool = True) -> str:
        """
        Replace all occurrences of text in the document.

        Args:
            search_text: The text to search for
            replace_text: The text to replace it with
            match_case: Whether to match case (default: True)

        Returns:
            JSON string with status and number of replacements
        """
        try:
            doc_id = self._ensure_document()
            service = self._get_service()

            requests = [{
                'replaceAllText': {
                    'containsText': {
                        'text': search_text,
                        'matchCase': match_case
                    },
                    'replaceText': replace_text
                }
            }]

            result = service.documents().batchUpdate(
                documentId=doc_id,
                body={'requests': requests}
            ).execute()

            occurrences = result.get('replies', [{}])[0].get('replaceAllText', {}).get('occurrencesChanged', 0)

            return json.dumps({
                "status": "success",
                "replacements": occurrences,
                "message": f"Replaced {occurrences} occurrence(s) of '{search_text}' with '{replace_text}'",
                "document_id": doc_id,
                "url": f"https://docs.google.com/document/d/{doc_id}"
            })

        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": str(e)
            })

    def _clear(self) -> str:
        """
        Clear all content from the document (except the mandatory ending newline).

        Returns:
            JSON string with status and result
        """
        try:
            doc_id = self._ensure_document()
            service = self._get_service()

            # Get the current document
            document = service.documents().get(documentId=doc_id).execute()
            body_content = document.get('body', {}).get('content', [])

            if not body_content or len(body_content) <= 1:
                return json.dumps({
                    "status": "success",
                    "message": "Document is already empty",
                    "document_id": doc_id
                })

            # Delete everything from index 1 to the end (excluding the final newline)
            start_index = body_content[0].get('startIndex', 1)
            end_index = body_content[-1].get('endIndex', 1) - 1

            if end_index > start_index:
                requests = [{
                    'deleteContentRange': {
                        'range': {
                            'startIndex': start_index,
                            'endIndex': end_index
                        }
                    }
                }]

                service.documents().batchUpdate(
                    documentId=doc_id,
                    body={'requests': requests}
                ).execute()

            return json.dumps({
                "status": "success",
                "message": "Successfully cleared document content",
                "document_id": doc_id,
                "url": f"https://docs.google.com/document/d/{doc_id}"
            })

        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": str(e)
            })

    def get_tools(self) -> list[Tool]:
        """Return the list of Google Docs tools."""
        if self._tools is not None:
            return self._tools

        self._tools = [
            Tool(
                name=self.read_tool_name,
                description=(
                    "Read the entire content of the Google Doc. "
                    "Returns the document title, full text content, and document URL."
                ),
                run=self._read,
                parameters={},
                required=[]
            ),
            Tool(
                name=self.append_tool_name,
                description=(
                    "Append text to the end of the Google Doc. "
                    "This is the most common way to add content to a document."
                ),
                run=self._append,
                parameters={
                    "text": {
                        "type": "string",
                        "description": "The text to append to the end of the document"
                    }
                },
                required=["text"]
            ),
            Tool(
                name=self.insert_tool_name,
                description=(
                    "Insert text at a specific position in the Google Doc. "
                    "Use index 1 to insert at the beginning. "
                    "For most use cases, use docs_append instead."
                ),
                run=self._insert_text,
                parameters={
                    "text": {
                        "type": "string",
                        "description": "The text to insert"
                    },
                    "index": {
                        "type": "integer",
                        "description": (
                            "The position to insert at (1-based). Use 1 for the start of the document."
                        )
                    }
                },
                required=["text", "index"]
            ),
            Tool(
                name=self.replace_tool_name,
                description=(
                    "Replace all occurrences of text in the Google Doc. "
                    "Useful for template-based document generation or bulk edits."
                ),
                run=self._replace_text,
                parameters={
                    "search_text": {
                        "type": "string",
                        "description": "The text to search for"
                    },
                    "replace_text": {
                        "type": "string",
                        "description": "The text to replace it with"
                    },
                    "match_case": {
                        "type": "boolean",
                        "description": "Whether to match case (default: true)"
                    }
                },
                required=["search_text", "replace_text"]
            ),
            Tool(
                name=self.clear_tool_name,
                description=(
                    "Clear all content from the Google Doc. "
                    "Use this to start fresh with an empty document."
                ),
                run=self._clear,
                parameters={},
                required=[]
            )
        ]

        return self._tools
