"""Web search prefab tools for Exa, Tavily, and Brave Search APIs."""

import abc
import json
import os
import re
from typing import Literal

from aiohttp import ClientSession, ClientTimeout

from .. import Tool

_HTML_TAG_RE = re.compile(r"<[a-z][^>]*>", re.IGNORECASE)


def _ensure_markdown(text: str) -> str:
    if not text:
        return text
    if _HTML_TAG_RE.search(text):
        try:
            from markdownify import markdownify as md
        except ImportError:
            raise ImportError(
                "markdownify is required to convert Tavily HTML to markdown. "
                "Install it with: pip install markdownify"
            )
        return md(text, strip=["img", "a"])
    return text


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
            Tool.from_function(self._search, name=self.search_tool_name),
            Tool.from_function(self._fetch, name=self.fetch_tool_name),
        ]

        return self._tools


class TavilyWebSearchManager(AbstractWebSearchManager):
    """
    Simple web search tools using the Tavily API.

    Provides two tools:
    - search: Search the web and get results with content
    - fetch: Get the contents of a specific URL as markdown (via extract endpoint)

    Args:
        search_tool_name: Name for the search tool (default: "web_search")
        fetch_tool_name: Name for the fetch tool (default: "web_fetch")
        timeout: Request timeout in seconds (default: 30)
        search_depth: Search depth - "basic" or "advanced" (default: "basic")

    Environment variables:
        TAVILY_API_KEY: Your Tavily API key (required)

    Example:
        ```python
        manager = TavilyWebSearchManager()
        tools = manager.get_tools()
        ```
    """

    BASE_URL = "https://api.tavily.com"

    def __init__(
        self,
        *,
        search_tool_name: str = "web_search",
        fetch_tool_name: str = "web_fetch",
        timeout: int = 30,
        search_depth: Literal["basic", "advanced"] = "basic",
    ):
        super().__init__(
            search_tool_name=search_tool_name,
            fetch_tool_name=fetch_tool_name,
            timeout=timeout,
        )
        self.search_depth = search_depth

    async def _search(  # type: ignore
        self,
        query: str,
        limit: int = 5,
    ) -> str:
        """Search the web and return results with content."""
        try:
            key = os.getenv("TAVILY_API_KEY")
            if not key:
                raise ValueError("TAVILY_API_KEY environment variable not set")

            data = {
                "query": query,
                "max_results": min(limit, 20),  # Tavily max is 20
                "search_depth": self.search_depth,
                "include_raw_content": False,
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
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
                        "text": item.get("content", ""),
                    }
                )

            return json.dumps({"status": "success", "results": results}, indent=2)

        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

    async def _fetch(self, url: str) -> str:
        """Fetch the contents of a URL as markdown using Tavily extract."""
        try:
            key = os.getenv("TAVILY_API_KEY")
            if not key:
                raise ValueError("TAVILY_API_KEY environment variable not set")

            data = {
                "urls": [url],
                "format": "markdown",
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            }

            async with ClientSession() as session:
                async with session.post(
                    f"{self.BASE_URL}/extract",
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
                # Check for failed results
                failed = result.get("failed_results", [])
                if failed:
                    return json.dumps(
                        {"status": "error", "error": f"Failed to extract: {failed}"}
                    )
                return json.dumps(
                    {"status": "error", "error": "No content found for URL"}
                )

            item = results[0]
            content = item.get("content") or item.get("raw_content") or ""
            text = _ensure_markdown(content)
            return json.dumps(
                {
                    "status": "success",
                    "title": "",  # Tavily extract doesn't return title
                    "url": item.get("url", url),
                    "text": text,
                },
                indent=2,
            )

        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})


class BraveWebSearchManager(AbstractWebSearchManager):
    """
    Simple web search tools using the Brave Search API.

    Provides two tools:
    - search: Search the web and get results
    - fetch: Get the contents of a specific URL (via direct HTTP fetch)

    Note: Brave Search API doesn't have a built-in content extraction endpoint,
    so fetch uses a basic HTTP request to retrieve page content.

    Args:
        search_tool_name: Name for the search tool (default: "web_search")
        fetch_tool_name: Name for the fetch tool (default: "web_fetch")
        timeout: Request timeout in seconds (default: 30)
        max_fetch_chars: Maximum characters to return from fetch (default: 20000)

    Environment variables:
        BRAVE_API_KEY: Your Brave Search API key (required)

    Example:
        ```python
        manager = BraveWebSearchManager()
        tools = manager.get_tools()
        ```
    """

    BASE_URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(
        self,
        *,
        search_tool_name: str = "web_search",
        fetch_tool_name: str = "web_fetch",
        timeout: int = 30,
        max_fetch_chars: int = 20_000,
    ):
        super().__init__(
            search_tool_name=search_tool_name,
            fetch_tool_name=fetch_tool_name,
            timeout=timeout,
        )
        self.max_fetch_chars = max_fetch_chars

    async def _search(  # type: ignore
        self,
        query: str,
        limit: int = 5,
    ) -> str:
        """Search the web and return results."""
        try:
            key = os.getenv("BRAVE_API_KEY")
            if not key:
                raise ValueError("BRAVE_API_KEY environment variable not set")

            params = {
                "q": query,
                "count": limit,
            }

            headers = {
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": key,
            }

            async with ClientSession() as session:
                async with session.get(
                    self.BASE_URL,
                    headers=headers,
                    params=params,
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
            web_results = result.get("web", {}).get("results", [])
            for item in web_results:
                results.append(
                    {
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "text": item.get("description", ""),
                    }
                )

            return json.dumps({"status": "success", "results": results}, indent=2)

        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

    async def _fetch(self, url: str) -> str:
        """Fetch the contents of a URL directly via HTTP and convert to markdown."""
        try:
            try:
                from markdownify import markdownify as md
            except ImportError:
                raise ImportError(
                    "markdownify is required for BraveWebSearchManager. "
                    "Install it with: pip install markdownify"
                )

            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ),
            }

            async with ClientSession() as session:
                async with session.get(
                    url,
                    headers=headers,
                    timeout=self.timeout,
                    allow_redirects=True,
                ) as response:
                    if response.status != 200:
                        return json.dumps(
                            {
                                "status": "error",
                                "error": f"HTTP error: {response.status}",
                            }
                        )
                    content_type = response.headers.get("Content-Type", "")
                    if (
                        "text/html" not in content_type
                        and "text/plain" not in content_type
                    ):
                        return json.dumps(
                            {
                                "status": "error",
                                "error": f"Unsupported content type: {content_type}",
                            }
                        )
                    html = await response.text()

            # Remove script and style elements before converting
            html = re.sub(
                r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE
            )
            html = re.sub(
                r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE
            )

            # Convert HTML to markdown
            text = md(html, strip=["img", "a"])

            # Normalize excessive whitespace while preserving markdown structure
            text = re.sub(r"\n{3,}", "\n\n", text).strip()

            # Truncate if needed
            text = text[: self.max_fetch_chars]

            return json.dumps(
                {
                    "status": "success",
                    "title": "",
                    "url": url,
                    "text": text,
                },
                indent=2,
            )

        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})


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


SearchBackend = Literal["exa", "tavily", "brave"]
FetchBackend = Literal["exa", "tavily", "aiohttp"]


async def _search_exa(
    query: str,
    limit: int,
    timeout: ClientTimeout,
    search_type: Literal["auto", "deep"] = "auto",
) -> str:
    """Search using Exa API."""
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
                "https://api.exa.ai/search",
                headers=headers,
                json=data,
                timeout=timeout,
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


async def _search_tavily(
    query: str,
    limit: int,
    timeout: ClientTimeout,
    search_depth: Literal["basic", "advanced"] = "basic",
) -> str:
    """Search using Tavily API."""
    try:
        key = os.getenv("TAVILY_API_KEY")
        if not key:
            raise ValueError("TAVILY_API_KEY environment variable not set")

        data = {
            "query": query,
            "max_results": min(limit, 20),
            "search_depth": search_depth,
            "include_raw_content": False,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }

        async with ClientSession() as session:
            async with session.post(
                "https://api.tavily.com/search",
                headers=headers,
                json=data,
                timeout=timeout,
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
                    "text": item.get("content", ""),
                }
            )

        return json.dumps({"status": "success", "results": results}, indent=2)

    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


async def _search_brave(
    query: str,
    limit: int,
    timeout: ClientTimeout,
) -> str:
    """Search using Brave Search API."""
    try:
        key = os.getenv("BRAVE_API_KEY")
        if not key:
            raise ValueError("BRAVE_API_KEY environment variable not set")

        params = {"q": query, "count": limit}

        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": key,
        }

        async with ClientSession() as session:
            async with session.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers=headers,
                params=params,
                timeout=timeout,
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
        web_results = result.get("web", {}).get("results", [])
        for item in web_results:
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "text": item.get("description", ""),
                }
            )

        return json.dumps({"status": "success", "results": results}, indent=2)

    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


async def _fetch_exa(
    url: str,
    timeout: ClientTimeout,
    max_chars: int = 20_000,
) -> str:
    """Fetch URL contents using Exa API."""
    try:
        key = os.getenv("EXA_API_KEY")
        if not key:
            raise ValueError("EXA_API_KEY environment variable not set")

        data = {
            "urls": [url],
            "text": {"maxCharacters": max_chars},
        }

        headers = {
            "Content-Type": "application/json",
            "x-api-key": key,
        }

        async with ClientSession() as session:
            async with session.post(
                "https://api.exa.ai/contents",
                headers=headers,
                json=data,
                timeout=timeout,
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
            return json.dumps({"status": "error", "error": "No content found for URL"})

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


async def _fetch_tavily(
    url: str,
    timeout: ClientTimeout,
) -> str:
    """Fetch URL contents using Tavily extract API."""
    try:
        key = os.getenv("TAVILY_API_KEY")
        if not key:
            raise ValueError("TAVILY_API_KEY environment variable not set")

        data = {
            "urls": [url],
            "format": "markdown",
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }

        async with ClientSession() as session:
            async with session.post(
                "https://api.tavily.com/extract",
                headers=headers,
                json=data,
                timeout=timeout,
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
            failed = result.get("failed_results", [])
            if failed:
                return json.dumps(
                    {"status": "error", "error": f"Failed to extract: {failed}"}
                )
            return json.dumps({"status": "error", "error": "No content found for URL"})

        item = results[0]
        content = item.get("content") or item.get("raw_content") or ""
        text = _ensure_markdown(content)
        return json.dumps(
            {
                "status": "success",
                "title": "",
                "url": item.get("url", url),
                "text": text,
            },
            indent=2,
        )

    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


async def _fetch_aiohttp(
    url: str,
    timeout: ClientTimeout,
    max_chars: int = 20_000,
) -> str:
    """Fetch URL contents directly via HTTP with markdownify."""
    try:
        try:
            from markdownify import markdownify as md
        except ImportError:
            raise ImportError(
                "markdownify is required for aiohttp fetch backend. "
                "Install it with: pip install markdownify"
            )

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
        }

        async with ClientSession() as session:
            async with session.get(
                url,
                headers=headers,
                timeout=timeout,
                allow_redirects=True,
            ) as response:
                if response.status != 200:
                    return json.dumps(
                        {"status": "error", "error": f"HTTP error: {response.status}"}
                    )
                content_type = response.headers.get("Content-Type", "")
                if "text/html" not in content_type and "text/plain" not in content_type:
                    return json.dumps(
                        {
                            "status": "error",
                            "error": f"Unsupported content type: {content_type}",
                        }
                    )
                html = await response.text()

        # Remove script and style elements before converting
        html = re.sub(
            r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE
        )
        html = re.sub(
            r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE
        )

        # Convert HTML to markdown
        text = md(html, strip=["img", "a"])

        # Normalize excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        # Truncate if needed
        text = text[:max_chars]

        return json.dumps(
            {
                "status": "success",
                "title": "",
                "url": url,
                "text": text,
            },
            indent=2,
        )

    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


class WebSearchManager(AbstractWebSearchManager):
    """
    Flexible web search manager with configurable search and fetch backends.

    Allows mixing any search backend (exa, tavily, brave) with any fetch backend
    (exa, tavily, aiohttp) for maximum flexibility.

    Args:
        search_backend: Which API to use for search ("exa", "tavily", "brave")
        fetch_backend: Which API to use for fetching URLs ("exa", "tavily", "aiohttp")
        search_tool_name: Name for the search tool (default: "web_search")
        fetch_tool_name: Name for the fetch tool (default: "web_fetch")
        timeout: Request timeout in seconds (default: 30)
        max_fetch_chars: Maximum characters for fetch results (default: 20000)
        exa_search_type: Exa search type - "auto" or "deep" (default: "auto")
        tavily_search_depth: Tavily search depth - "basic" or "advanced" (default: "basic")

    Environment variables (depending on backends used):
        EXA_API_KEY: Required if using exa backend
        TAVILY_API_KEY: Required if using tavily backend
        BRAVE_API_KEY: Required if using brave backend

    Example:
        ```python
        # Search with Brave, fetch with Tavily
        manager = WebSearchManager(search_backend="brave", fetch_backend="tavily")
        tools = manager.get_tools()

        # Search with Exa, fetch with direct HTTP
        manager = WebSearchManager(search_backend="exa", fetch_backend="aiohttp")
        tools = manager.get_tools()
        ```
    """

    def __init__(
        self,
        *,
        search_backend: SearchBackend = "exa",
        fetch_backend: FetchBackend = "exa",
        search_tool_name: str = "web_search",
        fetch_tool_name: str = "web_fetch",
        timeout: int = 30,
        max_fetch_chars: int = 20_000,
        exa_search_type: Literal["auto", "deep"] = "auto",
        tavily_search_depth: Literal["basic", "advanced"] = "basic",
    ):
        super().__init__(
            search_tool_name=search_tool_name,
            fetch_tool_name=fetch_tool_name,
            timeout=timeout,
        )
        self.search_backend = search_backend
        self.fetch_backend = fetch_backend
        self.max_fetch_chars = max_fetch_chars
        self.exa_search_type: Literal["auto", "deep"] = exa_search_type
        self.tavily_search_depth: Literal["basic", "advanced"] = tavily_search_depth

    async def _search(  # type: ignore
        self,
        query: str,
        limit: int = 5,
    ) -> str:
        """Search the web using the configured backend."""
        if self.search_backend == "exa":
            return await _search_exa(query, limit, self.timeout, self.exa_search_type)
        elif self.search_backend == "tavily":
            return await _search_tavily(
                query, limit, self.timeout, self.tavily_search_depth
            )
        elif self.search_backend == "brave":
            return await _search_brave(query, limit, self.timeout)
        else:
            return json.dumps(
                {
                    "status": "error",
                    "error": f"Unknown search backend: {self.search_backend}",
                }
            )

    async def _fetch(self, url: str) -> str:
        """Fetch URL contents using the configured backend."""
        if self.fetch_backend == "exa":
            return await _fetch_exa(url, self.timeout, self.max_fetch_chars)
        elif self.fetch_backend == "tavily":
            return await _fetch_tavily(url, self.timeout)
        elif self.fetch_backend == "aiohttp":
            return await _fetch_aiohttp(url, self.timeout, self.max_fetch_chars)
        else:
            return json.dumps(
                {
                    "status": "error",
                    "error": f"Unknown fetch backend: {self.fetch_backend}",
                }
            )


__all__ = [
    "AbstractWebSearchManager",
    "BraveWebSearchManager",
    "ExaWebSearchManager",
    "FetchBackend",
    "SearchBackend",
    "TavilyWebSearchManager",
    "WebSearchManager",
]
