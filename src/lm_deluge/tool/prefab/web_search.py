"""Web search prefab tool using Exa API."""

import json
import os
from typing import Literal

from aiohttp import ClientSession, ClientTimeout

from .. import Tool


class WebSearchManager:
    """
    Simple web search tools using the Exa API.

    Provides two tools:
    - search: Search the web and get results with content
    - fetch: Get the contents of a specific URL as markdown

    Args:
        api_key: Exa API key. If not provided, uses EXA_API_KEY env variable.
        search_tool_name: Name for the search tool (default: "web_search")
        fetch_tool_name: Name for the fetch tool (default: "web_fetch")
        timeout: Request timeout in seconds (default: 30)

    Example:
        ```python
        manager = WebSearchManager()
        tools = manager.get_tools()
        ```
    """

    BASE_URL = "https://api.exa.ai"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        search_tool_name: str = "web_search",
        fetch_tool_name: str = "web_fetch",
        timeout: int = 30,
    ):
        self.search_tool_name = search_tool_name
        self.fetch_tool_name = fetch_tool_name
        self.timeout = ClientTimeout(total=timeout)

        if api_key is not None:
            self.api_key = api_key
        else:
            env_key = os.environ.get("EXA_API_KEY")
            if env_key:
                self.api_key = env_key
            else:
                raise ValueError(
                    "No API key provided. Set api_key parameter or EXA_API_KEY env variable."
                )

        self._tools: list[Tool] | None = None

    async def _search(
        self,
        query: str,
        limit: int = 5,
        search_type: Literal["auto", "deep"] = "auto",
    ) -> str:
        """Search the web and return results with content."""
        try:
            data = {
                "query": query,
                "numResults": limit,
                "type": search_type,
                "contents": {"text": True},
            }

            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
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
            data = {
                "urls": [url],
                "text": True,
            }

            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
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

    def get_tools(self) -> list[Tool]:
        """Return the web search tools."""
        if self._tools is not None:
            return self._tools

        self._tools = [
            Tool(
                name=self.search_tool_name,
                description="Search the web and get results with their content.",
                run=self._search,
                parameters={
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of results (default: 5, max: 10)",
                    },
                    "search_type": {
                        "type": "string",
                        "enum": ["auto", "deep"],
                        "description": "Search type: 'auto' (default) or 'deep' for more thorough search",
                    },
                },
                required=["query"],
            ),
            Tool(
                name=self.fetch_tool_name,
                description="Fetch the contents of a specific URL as text.",
                run=self._fetch,
                parameters={
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch content from",
                    },
                },
                required=["url"],
            ),
        ]

        return self._tools


__all__ = ["WebSearchManager"]
