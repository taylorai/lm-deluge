"""Google Docs manipulation prefab tool."""

import json
import os
import re
from typing import Any

from lm_deluge.tool import Tool


# Mapping of heading levels to Google Docs named styles
HEADING_STYLES = {
    1: "HEADING_1",
    2: "HEADING_2",
    3: "HEADING_3",
    4: "HEADING_4",
    5: "HEADING_5",
    6: "HEADING_6",
}


class DocsManager:
    """
    A prefab tool for manipulating Google Docs.

    Provides tools to read, write, and edit Google Docs documents.
    Supports markdown-style formatting (bold, italic, underline, headings).

    Args:
        document_id: Optional. The ID of the Google Doc to manipulate.
                    If not provided, a new document will be created on first use.
        credentials_json: Optional. JSON string or dict containing Google service account credentials.
                         If not provided, will look for GOOGLE_DOCS_CREDENTIALS env variable.
        credentials_file: Optional. Path to a JSON file containing credentials.
                         Only used if credentials_json is not provided.
        document_title: Optional. Title for new document (only used if document_id is not provided)

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
        get_metadata_tool_name: str = "docs_get_metadata",
        read_range_tool_name: str = "docs_read_range",
        grep_tool_name: str = "docs_grep",
        add_paragraph_tool_name: str = "docs_add_paragraph",
        update_paragraph_tool_name: str = "docs_update_paragraph",
        replace_tool_name: str = "docs_replace_text",
        delete_range_tool_name: str = "docs_delete_range",
    ):
        self.document_id = document_id
        self.document_title = document_title
        self.get_metadata_tool_name = get_metadata_tool_name
        self.read_range_tool_name = read_range_tool_name
        self.grep_tool_name = grep_tool_name
        self.add_paragraph_tool_name = add_paragraph_tool_name
        self.update_paragraph_tool_name = update_paragraph_tool_name
        self.replace_tool_name = replace_tool_name
        self.delete_range_tool_name = delete_range_tool_name

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
            env_creds = os.environ.get("GOOGLE_DOCS_CREDENTIALS")
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
            self.credentials, scopes=["https://www.googleapis.com/auth/documents"]
        )

        # Build the service
        self._service = build("docs", "v1", credentials=creds)
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
            document = (
                service.documents()
                .create(body={"title": self.document_title})
                .execute()
            )
            self.document_id = document.get("documentId")
            return self.document_id
        except Exception as e:
            raise RuntimeError(f"Failed to create document: {str(e)}")

    def _get_document_end_index(self, document: dict[str, Any]) -> int:
        """Get the end index of the document content (before the final newline)."""
        body_content = document.get("body", {}).get("content", [])
        if not body_content:
            return 1
        # endIndex - 1 because docs always end with a mandatory newline
        end_index = body_content[-1].get("endIndex", 2) - 1
        # Ensure we never return less than 1
        return max(1, end_index)

    def _is_document_empty(self, document: dict[str, Any]) -> bool:
        """Check if the document is empty (only contains the mandatory newline)."""
        body_content = document.get("body", {}).get("content", [])
        if not body_content:
            return True
        # An empty doc has endIndex of 2 (index 1 is the mandatory newline)
        last_end = body_content[-1].get("endIndex", 1)
        return last_end <= 2

    def _extract_text_with_formatting(
        self,
        document: dict[str, Any],
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> tuple[str, int]:
        """
        Extract text content from a document with markdown formatting.

        Returns:
            Tuple of (formatted text, total line count)
        """
        content = document.get("body", {}).get("content", [])
        lines: list[str] = []

        for element in content:
            if "paragraph" in element:
                para = element["paragraph"]
                para_style = para.get("paragraphStyle", {}).get(
                    "namedStyleType", "NORMAL_TEXT"
                )

                # Build the line with inline formatting
                line_parts: list[str] = []
                for para_element in para.get("elements", []):
                    text_run = para_element.get("textRun")
                    if text_run:
                        text = text_run.get("content", "")
                        style = text_run.get("textStyle", {})

                        # Apply inline formatting
                        formatted = text.rstrip("\n")
                        if formatted:
                            is_bold = style.get("bold", False)
                            is_italic = style.get("italic", False)
                            is_underline = style.get("underline", False)

                            # Apply formatting in order: underline wraps innermost, then bold/italic
                            if is_bold and is_italic:
                                formatted = f"***{formatted}***"
                            elif is_bold:
                                formatted = f"**{formatted}**"
                            elif is_italic:
                                formatted = f"*{formatted}*"

                            if is_underline:
                                formatted = f"<u>{formatted}</u>"

                            line_parts.append(formatted)

                line = "".join(line_parts)

                # Apply heading prefix based on paragraph style
                if para_style == "HEADING_1":
                    line = f"# {line}"
                elif para_style == "HEADING_2":
                    line = f"## {line}"
                elif para_style == "HEADING_3":
                    line = f"### {line}"
                elif para_style == "HEADING_4":
                    line = f"#### {line}"
                elif para_style == "HEADING_5":
                    line = f"##### {line}"
                elif para_style == "HEADING_6":
                    line = f"###### {line}"
                elif para_style == "TITLE":
                    line = f"# {line}"
                elif para_style == "SUBTITLE":
                    line = f"## {line}"

                lines.append(line)

            elif "table" in element:
                # Handle tables - just extract text for now
                table = element["table"]
                for row in table.get("tableRows", []):
                    row_parts: list[str] = []
                    for cell in row.get("tableCells", []):
                        cell_text_parts: list[str] = []
                        for cell_content in cell.get("content", []):
                            if "paragraph" in cell_content:
                                for para_element in cell_content["paragraph"].get(
                                    "elements", []
                                ):
                                    text_run = para_element.get("textRun")
                                    if text_run:
                                        cell_text_parts.append(
                                            text_run.get("content", "").strip()
                                        )
                        row_parts.append(" ".join(cell_text_parts))
                    lines.append(" | ".join(row_parts))

        total_lines = len(lines)

        # Apply line range filter
        if start_line is not None or end_line is not None:
            start_idx = (start_line - 1) if start_line else 0
            end_idx = end_line if end_line else len(lines)
            lines = lines[start_idx:end_idx]

        return "\n".join(lines), total_lines

    def _parse_markdown_text(self, text: str) -> list[dict[str, Any]]:
        """
        Parse markdown-formatted text into segments with formatting info.

        Supports:
        - **bold**
        - *italic*
        - <u>underline</u>
        - ***bold italic***
        - # Heading 1 through ###### Heading 6

        Returns:
            List of dicts with 'text', 'bold', 'italic', 'underline', 'heading_level' keys
        """
        lines = text.split("\n")
        result: list[dict[str, Any]] = []

        for i, line in enumerate(lines):
            heading_level = 0

            # Check for heading
            heading_match = re.match(r"^(#{1,6})\s+(.*)$", line)
            if heading_match:
                heading_level = len(heading_match.group(1))
                line = heading_match.group(2)

            # Parse inline formatting
            segments = self._parse_inline_formatting(line)

            for seg in segments:
                seg["heading_level"] = heading_level

            result.extend(segments)

            # Add newline between lines (except for last line)
            if i < len(lines) - 1:
                result.append(
                    {
                        "text": "\n",
                        "bold": False,
                        "italic": False,
                        "underline": False,
                        "heading_level": 0,
                    }
                )

        return result

    def _parse_inline_formatting(self, text: str) -> list[dict[str, Any]]:
        """Parse inline formatting (bold, italic, underline) from text."""
        segments: list[dict[str, Any]] = []

        # Pattern to match formatting markers
        # Order matters: check *** before ** before *
        pattern = r"(\*\*\*(.+?)\*\*\*|\*\*(.+?)\*\*|\*(.+?)\*|<u>(.+?)</u>)"

        last_end = 0
        for match in re.finditer(pattern, text):
            # Add any text before this match
            if match.start() > last_end:
                segments.append(
                    {
                        "text": text[last_end : match.start()],
                        "bold": False,
                        "italic": False,
                        "underline": False,
                    }
                )

            # Determine what kind of formatting this is
            full_match = match.group(0)
            if full_match.startswith("***"):
                segments.append(
                    {
                        "text": match.group(2),
                        "bold": True,
                        "italic": True,
                        "underline": False,
                    }
                )
            elif full_match.startswith("**"):
                segments.append(
                    {
                        "text": match.group(3),
                        "bold": True,
                        "italic": False,
                        "underline": False,
                    }
                )
            elif full_match.startswith("*"):
                segments.append(
                    {
                        "text": match.group(4),
                        "bold": False,
                        "italic": True,
                        "underline": False,
                    }
                )
            elif full_match.startswith("<u>"):
                segments.append(
                    {
                        "text": match.group(5),
                        "bold": False,
                        "italic": False,
                        "underline": True,
                    }
                )

            last_end = match.end()

        # Add any remaining text
        if last_end < len(text):
            segments.append(
                {
                    "text": text[last_end:],
                    "bold": False,
                    "italic": False,
                    "underline": False,
                }
            )

        # If no formatting found, return the whole text as one segment
        if not segments:
            segments.append(
                {"text": text, "bold": False, "italic": False, "underline": False}
            )

        return segments

    def _build_insert_requests(
        self, segments: list[dict[str, Any]], start_index: int
    ) -> tuple[list[dict[str, Any]], int]:
        """
        Build Google Docs API requests to insert formatted text.

        Returns:
            Tuple of (list of requests, end index after insertion)
        """
        requests: list[dict[str, Any]] = []
        current_index = start_index

        # Group segments by paragraph (split on heading changes and newlines)
        paragraphs: list[tuple[int, list[dict[str, Any]]]] = []
        current_para: list[dict[str, Any]] = []
        current_heading = 0

        for seg in segments:
            if seg["text"] == "\n":
                if current_para:
                    paragraphs.append((current_heading, current_para))
                    current_para = []
                current_heading = 0
            else:
                if seg.get("heading_level", 0) > 0:
                    current_heading = seg["heading_level"]
                current_para.append(seg)

        if current_para:
            paragraphs.append((current_heading, current_para))

        # Build requests for each paragraph
        for para_idx, (heading_level, para_segments) in enumerate(paragraphs):
            para_start = current_index

            # Insert all text segments for this paragraph
            for seg in para_segments:
                text = seg["text"]
                if not text:
                    continue

                # Insert the text
                requests.append(
                    {"insertText": {"location": {"index": current_index}, "text": text}}
                )

                text_start = current_index
                text_end = current_index + len(text)

                # Apply text styling if needed
                style_updates = {}
                if seg.get("bold"):
                    style_updates["bold"] = True
                if seg.get("italic"):
                    style_updates["italic"] = True
                if seg.get("underline"):
                    style_updates["underline"] = True

                if style_updates:
                    requests.append(
                        {
                            "updateTextStyle": {
                                "range": {
                                    "startIndex": text_start,
                                    "endIndex": text_end,
                                },
                                "textStyle": style_updates,
                                "fields": ",".join(style_updates.keys()),
                            }
                        }
                    )

                current_index = text_end

            # Add newline after paragraph (except for last one if we're appending)
            if para_idx < len(paragraphs) - 1:
                requests.append(
                    {"insertText": {"location": {"index": current_index}, "text": "\n"}}
                )
                para_end = current_index + 1
                current_index = para_end
            else:
                para_end = current_index

            # Apply paragraph style for headings
            if heading_level > 0 and heading_level in HEADING_STYLES:
                requests.append(
                    {
                        "updateParagraphStyle": {
                            "range": {"startIndex": para_start, "endIndex": para_end},
                            "paragraphStyle": {
                                "namedStyleType": HEADING_STYLES[heading_level]
                            },
                            "fields": "namedStyleType",
                        }
                    }
                )

        return requests, current_index

    def _get_metadata(self) -> str:
        """
        Get document metadata including title and line count.

        Returns:
            JSON string with status, title, line count, and URL
        """
        try:
            doc_id = self._ensure_document()
            service = self._get_service()

            document = service.documents().get(documentId=doc_id).execute()

            title = document.get("title", "")
            _, total_lines = self._extract_text_with_formatting(document)

            return json.dumps(
                {
                    "status": "success",
                    "document_id": doc_id,
                    "title": title,
                    "total_lines": total_lines,
                    "url": f"https://docs.google.com/document/d/{doc_id}",
                }
            )

        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

    def _read_range(self, start_line: int, end_line: int | None = None) -> str:
        """
        Read a range of lines from the document with markdown formatting.

        Args:
            start_line: First line to read (1-based)
            end_line: Last line to read (inclusive). If None, reads to end of document.

        Returns:
            JSON string with status, content, and line info
        """
        try:
            doc_id = self._ensure_document()
            service = self._get_service()

            document = service.documents().get(documentId=doc_id).execute()
            title = document.get("title", "")

            # First get total lines
            _, total_lines = self._extract_text_with_formatting(document)

            # If end_line not specified, read to end
            actual_end_line = end_line if end_line is not None else total_lines

            text_content, _ = self._extract_text_with_formatting(
                document, start_line=start_line, end_line=actual_end_line
            )

            return json.dumps(
                {
                    "status": "success",
                    "document_id": doc_id,
                    "title": title,
                    "content": text_content,
                    "start_line": start_line,
                    "end_line": min(actual_end_line, total_lines),
                    "total_lines": total_lines,
                    "url": f"https://docs.google.com/document/d/{doc_id}",
                }
            )

        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

    def _grep(self, pattern: str, ignore_case: bool = False) -> str:
        """
        Search for lines matching a pattern in the document.

        Args:
            pattern: Regular expression pattern to search for
            ignore_case: If True, perform case-insensitive matching

        Returns:
            JSON string with status and matching lines with their line numbers
        """
        try:
            doc_id = self._ensure_document()
            service = self._get_service()

            document = service.documents().get(documentId=doc_id).execute()

            # Get all lines with formatting
            content = document.get("body", {}).get("content", [])
            lines: list[tuple[int, str]] = []  # (line_num, text)
            line_num = 1

            for element in content:
                if "paragraph" in element:
                    para = element["paragraph"]

                    # Build the line text (without formatting for search)
                    line_parts: list[str] = []
                    for para_element in para.get("elements", []):
                        text_run = para_element.get("textRun")
                        if text_run:
                            text = text_run.get("content", "").rstrip("\n")
                            line_parts.append(text)

                    line_text = "".join(line_parts)
                    lines.append((line_num, line_text))
                    line_num += 1

            # Compile the regex
            flags = re.IGNORECASE if ignore_case else 0
            try:
                regex = re.compile(pattern, flags)
            except re.error as e:
                return json.dumps(
                    {"status": "error", "error": f"Invalid regex pattern: {str(e)}"}
                )

            # Find matching lines
            matches: list[dict[str, Any]] = []
            for num, text in lines:
                if regex.search(text):
                    matches.append({"line": num, "content": text})

            return json.dumps(
                {
                    "status": "success",
                    "document_id": doc_id,
                    "pattern": pattern,
                    "ignore_case": ignore_case,
                    "matches": matches,
                    "match_count": len(matches),
                    "total_lines": len(lines),
                }
            )

        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

    def _add_paragraph(
        self, text: str, after_line: int | None = None, markdown: bool = True
    ) -> str:
        """
        Add a new paragraph to the document.

        Args:
            text: The text for the new paragraph
            after_line: Insert after this line number (1-based). If None, appends to end.
            markdown: If True, parse markdown formatting. If False, insert plain text.

        Returns:
            JSON string with status and result
        """
        try:
            doc_id = self._ensure_document()
            service = self._get_service()

            document = service.documents().get(documentId=doc_id).execute()

            if after_line is None:
                # Append to end
                insert_index = self._get_document_end_index(document)
            else:
                # Insert after the specified line
                paragraphs = self._get_paragraph_info(document)

                if after_line < 0 or after_line > len(paragraphs):
                    return json.dumps(
                        {
                            "status": "error",
                            "error": f"Line {after_line} out of range. Document has {len(paragraphs)} lines. Use 0 to insert at beginning.",
                        }
                    )

                if after_line == 0:
                    # Insert at the very beginning
                    insert_index = 1
                else:
                    # Insert after the specified paragraph
                    para = paragraphs[after_line - 1]
                    insert_index = para["end_index"] - 1  # Before the trailing newline

            # Ensure text starts with newline if we're not at the beginning
            if insert_index > 1 and not text.startswith("\n"):
                text = "\n" + text

            if markdown:
                segments = self._parse_markdown_text(text)
                requests, _ = self._build_insert_requests(segments, insert_index)
            else:
                requests = [
                    {"insertText": {"location": {"index": insert_index}, "text": text}}
                ]

            if requests:
                service.documents().batchUpdate(
                    documentId=doc_id, body={"requests": requests}
                ).execute()

            position = (
                f"after line {after_line}" if after_line is not None else "at end"
            )
            return json.dumps(
                {
                    "status": "success",
                    "message": f"Successfully added paragraph {position}",
                    "document_id": doc_id,
                    "url": f"https://docs.google.com/document/d/{doc_id}",
                }
            )

        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

    def _replace_text(
        self, search_text: str, replace_text: str, match_case: bool = True
    ) -> str:
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

            requests = [
                {
                    "replaceAllText": {
                        "containsText": {"text": search_text, "matchCase": match_case},
                        "replaceText": replace_text,
                    }
                }
            ]

            result = (
                service.documents()
                .batchUpdate(documentId=doc_id, body={"requests": requests})
                .execute()
            )

            occurrences = (
                result.get("replies", [{}])[0]
                .get("replaceAllText", {})
                .get("occurrencesChanged", 0)
            )

            return json.dumps(
                {
                    "status": "success",
                    "replacements": occurrences,
                    "message": f"Replaced {occurrences} occurrence(s) of '{search_text}' with '{replace_text}'",
                    "document_id": doc_id,
                    "url": f"https://docs.google.com/document/d/{doc_id}",
                }
            )

        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

    def _get_paragraph_info(self, document: dict[str, Any]) -> list[dict[str, Any]]:
        """Get info about each paragraph in the document."""
        content = document.get("body", {}).get("content", [])
        paragraphs: list[dict[str, Any]] = []
        line_num = 1

        for element in content:
            if "paragraph" in element:
                para = element["paragraph"]
                start_index = element.get("startIndex", 1)
                end_index = element.get("endIndex", start_index)

                # Extract text
                text_parts: list[str] = []
                for para_element in para.get("elements", []):
                    text_run = para_element.get("textRun")
                    if text_run:
                        text_parts.append(text_run.get("content", ""))

                text = "".join(text_parts).rstrip("\n")

                paragraphs.append(
                    {
                        "line": line_num,
                        "start_index": start_index,
                        "end_index": end_index,
                        "text": text,
                        "style": para.get("paragraphStyle", {}).get(
                            "namedStyleType", "NORMAL_TEXT"
                        ),
                    }
                )
                line_num += 1

        return paragraphs

    def _update_paragraph(self, line: int, new_text: str, markdown: bool = True) -> str:
        """
        Update the content of a specific paragraph/line.

        Args:
            line: The line number to update (1-based)
            new_text: The new text for the paragraph
            markdown: If True, parse markdown formatting.

        Returns:
            JSON string with status and result
        """
        try:
            doc_id = self._ensure_document()
            service = self._get_service()

            document = service.documents().get(documentId=doc_id).execute()
            paragraphs = self._get_paragraph_info(document)

            if line < 1 or line > len(paragraphs):
                return json.dumps(
                    {
                        "status": "error",
                        "error": f"Line {line} out of range. Document has {len(paragraphs)} lines.",
                    }
                )

            para = paragraphs[line - 1]
            start_index = para["start_index"]
            end_index = para["end_index"] - 1  # Don't delete the trailing newline

            requests: list[dict[str, Any]] = []

            # Delete the old content (if there is any)
            if end_index > start_index:
                requests.append(
                    {
                        "deleteContentRange": {
                            "range": {"startIndex": start_index, "endIndex": end_index}
                        }
                    }
                )

            # Insert new content
            if markdown:
                segments = self._parse_markdown_text(new_text)
                insert_requests, _ = self._build_insert_requests(segments, start_index)
                requests.extend(insert_requests)
            else:
                requests.append(
                    {
                        "insertText": {
                            "location": {"index": start_index},
                            "text": new_text,
                        }
                    }
                )

            if requests:
                service.documents().batchUpdate(
                    documentId=doc_id, body={"requests": requests}
                ).execute()

            return json.dumps(
                {
                    "status": "success",
                    "message": f"Successfully updated line {line}",
                    "document_id": doc_id,
                    "url": f"https://docs.google.com/document/d/{doc_id}",
                }
            )

        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

    def _delete_range(self, start_line: int, end_line: int) -> str:
        """
        Delete a range of lines from the document.

        Args:
            start_line: First line to delete (1-based)
            end_line: Last line to delete (inclusive)

        Returns:
            JSON string with status and result
        """
        try:
            doc_id = self._ensure_document()
            service = self._get_service()

            document = service.documents().get(documentId=doc_id).execute()
            paragraphs = self._get_paragraph_info(document)

            if not paragraphs:
                return json.dumps(
                    {
                        "status": "success",
                        "message": "Document is already empty",
                        "document_id": doc_id,
                    }
                )

            if start_line < 1:
                start_line = 1
            if end_line > len(paragraphs):
                end_line = len(paragraphs)

            if start_line > end_line or start_line > len(paragraphs):
                return json.dumps(
                    {
                        "status": "error",
                        "error": f"Invalid range. Document has {len(paragraphs)} lines.",
                    }
                )

            # Get the range to delete
            start_para = paragraphs[start_line - 1]
            end_para = paragraphs[end_line - 1]

            start_index = start_para["start_index"]
            end_index = end_para["end_index"]

            # Don't delete past the document's final newline
            # The API doesn't allow deleting the mandatory trailing newline
            doc_end = self._get_document_end_index(document)
            if end_index > doc_end:
                end_index = doc_end

            if end_index <= start_index:
                return json.dumps(
                    {
                        "status": "success",
                        "message": "Nothing to delete",
                        "document_id": doc_id,
                    }
                )

            requests = [
                {
                    "deleteContentRange": {
                        "range": {"startIndex": start_index, "endIndex": end_index}
                    }
                }
            ]

            service.documents().batchUpdate(
                documentId=doc_id, body={"requests": requests}
            ).execute()

            lines_deleted = end_line - start_line + 1
            return json.dumps(
                {
                    "status": "success",
                    "message": f"Successfully deleted {lines_deleted} line(s)",
                    "document_id": doc_id,
                    "url": f"https://docs.google.com/document/d/{doc_id}",
                }
            )

        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

    def get_tools(self) -> list[Tool]:
        """Return the list of Google Docs tools."""
        if self._tools is not None:
            return self._tools

        self._tools = [
            Tool(
                name=self.get_metadata_tool_name,
                description=(
                    "Get metadata about the Google Doc including its title and total line count. "
                    "Use this to check the document length before reading or modifying."
                ),
                run=self._get_metadata,
                parameters={},
                required=[],
            ),
            Tool(
                name=self.read_range_tool_name,
                description=(
                    "Read lines from the Google Doc with markdown formatting. "
                    "Returns content with formatting preserved (headings as #, bold as **, "
                    "italic as *, underline as <u>). "
                    "If end_line is omitted, reads from start_line to end of document. "
                    "Use start_line=1 without end_line to read the entire document."
                ),
                run=self._read_range,
                parameters={
                    "start_line": {
                        "type": "integer",
                        "description": "First line to read (1-based)",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Last line to read (inclusive). Omit to read to end of document.",
                    },
                },
                required=["start_line"],
            ),
            Tool(
                name=self.grep_tool_name,
                description=(
                    "Search for lines matching a pattern in the Google Doc. "
                    "Returns matching lines with their line numbers. "
                    "Supports regular expressions."
                ),
                run=self._grep,
                parameters={
                    "pattern": {
                        "type": "string",
                        "description": "Regular expression pattern to search for",
                    },
                    "ignore_case": {
                        "type": "boolean",
                        "description": "If true, perform case-insensitive matching (default: false)",
                    },
                },
                required=["pattern"],
            ),
            Tool(
                name=self.add_paragraph_tool_name,
                description=(
                    "Add a new paragraph to the Google Doc. "
                    "Supports markdown formatting: # headings, **bold**, *italic*, <u>underline</u>. "
                    "Use after_line to insert after a specific line, or omit to append at end. "
                    "Use after_line=0 to insert at the very beginning."
                ),
                run=self._add_paragraph,
                parameters={
                    "text": {
                        "type": "string",
                        "description": "The text for the new paragraph (with optional markdown formatting)",
                    },
                    "after_line": {
                        "type": "integer",
                        "description": "Insert after this line number (1-based). Use 0 to insert at beginning. Omit to append at end.",
                    },
                    "markdown": {
                        "type": "boolean",
                        "description": "If true (default), parse markdown formatting. If false, insert as plain text.",
                    },
                },
                required=["text"],
            ),
            Tool(
                name=self.update_paragraph_tool_name,
                description=(
                    "Update the content of a specific line/paragraph in the document. "
                    "Replaces the entire line with new content. "
                    "Supports markdown formatting."
                ),
                run=self._update_paragraph,
                parameters={
                    "line": {
                        "type": "integer",
                        "description": "The line number to update (1-based)",
                    },
                    "new_text": {
                        "type": "string",
                        "description": "The new text for the paragraph (with optional markdown)",
                    },
                    "markdown": {
                        "type": "boolean",
                        "description": "If true (default), parse markdown formatting.",
                    },
                },
                required=["line", "new_text"],
            ),
            Tool(
                name=self.replace_tool_name,
                description=(
                    "Replace all occurrences of text in the Google Doc. "
                    "Useful for template-based document generation or bulk edits. "
                    "Note: replacement text is inserted as plain text without formatting."
                ),
                run=self._replace_text,
                parameters={
                    "search_text": {
                        "type": "string",
                        "description": "The text to search for",
                    },
                    "replace_text": {
                        "type": "string",
                        "description": "The text to replace it with",
                    },
                    "match_case": {
                        "type": "boolean",
                        "description": "Whether to match case (default: true)",
                    },
                },
                required=["search_text", "replace_text"],
            ),
            Tool(
                name=self.delete_range_tool_name,
                description=(
                    "Delete a range of lines from the Google Doc. "
                    "Use this to remove content from the document."
                ),
                run=self._delete_range,
                parameters={
                    "start_line": {
                        "type": "integer",
                        "description": "First line to delete (1-based)",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Last line to delete (inclusive)",
                    },
                },
                required=["start_line", "end_line"],
            ),
        ]

        return self._tools
