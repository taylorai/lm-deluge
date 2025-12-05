"""Web search prefab tool using Exa API."""

import abc
import json
import os
from typing import Literal

from aiohttp import ClientSession, ClientTimeout

from .. import Tool


class AbstractWebSearchManager(abc.ABC):
    def __init__(
        self,
        search_tool_name: str = "web_search",
        fetch_tool_name: str = "web_fetch",
        timeout: int = 30,
    ):
        self.search_tool_name = search_tool_name
        self.fetch_tool_name = fetch_tool_name
        self.timeout = ClientTimeout(total=timeout)
        self._tools: list[Tool] | None = None

    @abc.abstractmethod
    async def _search(self, query: str, limit: int) -> list[dict]:
        """Search the web and get results with content."""
        pass

    @abc.abstractmethod
    async def _fetch(self, url: str) -> str:
        """Get the contents of a specific URL as markdown."""
        pass

    def get_tools(self) -> list[Tool]:
        """Return the web search tools."""
        if self._tools is not None:
            return self._tools

        self._tools = [
            Tool.from_function(self._search),
            Tool.from_function(self._fetch),
        ]

        return self._tools


class ExaWebSearchManager(AbstractWebSearchManager):
    """
    Simple web search tools using the Exa API.

    Provides two tools:
    - search: Search the web and get results with content
    - fetch: Get the contents of a specific URL as markdown

    Args:
        search_tool_name: Name for the search tool (default: "web_search")
        fetch_tool_name: Name for the fetch tool (default: "web_fetch")
        timeout: Request timeout in seconds (default: 30)

    Environment variables:
        EXA_API_KEY: Your Exa API key (required)

    Example:
        ```python
        manager = ExaWebSearchManager()
        tools = manager.get_tools()
        ```
    """

    BASE_URL = "https://api.exa.ai"

    def __init__(
        self,
        *,
        search_tool_name: str = "web_search",
        fetch_tool_name: str = "web_fetch",
        timeout: int = 30,
        max_contents_chars: int = 20_000,
    ):
        super().__init__(
            search_tool_name=search_tool_name,
            fetch_tool_name=fetch_tool_name,
            timeout=timeout,
        )
        self.max_contents_chars = max_contents_chars

    async def _search(  # type: ignore
        self,
        query: str,
        limit: int = 5,
        search_type: Literal["auto", "deep"] = "auto",
    ) -> str:
        """Search the web and return results with content."""
        try:
            key = os.getenv("EXA_API_KEY")
            if not key:
                raise ValueError("EXA_API_KEY environment variable not set")
            data = {
                "query": query,
                "numResults": limit,
                "type": search_type,
                "contents": {"text": True},
            }

            headers = {
                "Content-Type": "application/json",
                "x-api-key": key,
            }

            async with ClientSession() as session:
                async with session.post(
                    f"{self.BASE_URL}/search",
                    headers=headers,
                    json=data,
                    timeout=self.timeout,
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        return json.dumps(
                            {
                                "status": "error",
                                "error": f"API error: {response.status} - {error_text}",
                            }
                        )
                    result = await response.json()

            results = []
            for item in result.get("results", []):
                results.append(
                    {
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "text": item.get("text", ""),
                    }
                )

            return json.dumps({"status": "success", "results": results}, indent=2)

        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

    async def _fetch(self, url: str) -> str:
        """Fetch the contents of a URL as markdown."""
        try:
            key = os.getenv("EXA_API_KEY")
            if not key:
                raise ValueError("EXA_API_KEY environment variable not set")
            data = {
                "urls": [url],
                "text": {
                    "maxCharacters": self.max_contents_chars,
                },
            }

            headers = {
                "Content-Type": "application/json",
                "x-api-key": key,
            }

            async with ClientSession() as session:
                async with session.post(
                    f"{self.BASE_URL}/contents",
                    headers=headers,
                    json=data,
                    timeout=self.timeout,
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        return json.dumps(
                            {
                                "status": "error",
                                "error": f"API error: {response.status} - {error_text}",
                            }
                        )
                    result = await response.json()

            results = result.get("results", [])
            if not results:
                return json.dumps(
                    {"status": "error", "error": "No content found for URL"}
                )

            item = results[0]
            return json.dumps(
                {
                    "status": "success",
                    "title": item.get("title", ""),
                    "url": item.get("url", url),
                    "text": item.get("text", ""),
                },
                indent=2,
            )

        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})


__all__ = ["ExaWebSearchManager", "AbstractWebSearchManager"]
