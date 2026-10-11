"""Statular tool schemas (exact) and a shortened persisted MCP result."""

STATULAR_TOOLS = [
    {
        "name": "search",
        "description": "Search documents in one of the available remote search indices using one or more queries. Search uses the upstream full-text keyword search service; multiple queries are forwarded to the service for r",
        "parameters": {
            "index": {"type": "string"},
            "queries": {"items": {"type": "string"}, "type": "array"},
            "limit": {"default": 5, "type": "integer"},
            "metadata_filters": {
                "anyOf": [
                    {
                        "additionalProperties": {
                            "items": {"type": "string"},
                            "type": "array",
                        },
                        "type": "object",
                    },
                    {"type": "null"},
                ],
                "default": None,
            },
        },
        "required": ["index", "queries"],
    },
    {
        "name": "fetch",
        "description": "Retrieve full content from a remote search index for specific document IDs. Each document can be optionally filtered by line range or grep pattern.\n\nAvailable indices:\n - al_code: Full searchable text",
        "parameters": {
            "index": {"type": "string"},
            "document_ids": {"items": {"type": "string"}, "type": "array"},
            "filters": {
                "anyOf": [
                    {
                        "items": {"additionalProperties": True, "type": "object"},
                        "type": "array",
                    },
                    {"type": "string"},
                    {"type": "null"},
                ],
                "default": None,
            },
        },
        "required": ["index", "document_ids"],
    },
    {
        "name": "batch",
        "description": "Execute multiple search/fetch operations concurrently through the gateway. Each command has name ('search' or 'fetch'), index, and operation-specific fields.\n\nAvailable indices:\n - al_code: Full searc",
        "parameters": {
            "commands": {
                "items": {
                    "anyOf": [
                        {
                            "properties": {
                                "name": {"const": "search", "type": "string"},
                                "index": {"type": "string"},
                                "queries": {
                                    "items": {"type": "string"},
                                    "type": "array",
                                },
                                "limit": {"default": 5, "type": "integer"},
                                "metadata_filters": {
                                    "anyOf": [
                                        {
                                            "additionalProperties": {
                                                "items": {"type": "string"},
                                                "type": "array",
                                            },
                                            "type": "object",
                                        },
                                        {"type": "null"},
                                    ],
                                    "default": None,
                                },
                            },
                            "required": ["name", "index", "queries"],
                            "type": "object",
                        },
                        {
                            "properties": {
                                "name": {"const": "fetch", "type": "string"},
                                "index": {"type": "string"},
                                "document_ids": {
                                    "items": {"type": "string"},
                                    "type": "array",
                                },
                                "filters": {
                                    "anyOf": [
                                        {
                                            "items": {
                                                "additionalProperties": True,
                                                "type": "object",
                                            },
                                            "type": "array",
                                        },
                                        {"type": "string"},
                                        {"type": "null"},
                                    ],
                                    "default": None,
                                },
                            },
                            "required": ["name", "index", "document_ids"],
                            "type": "object",
                        },
                    ]
                },
                "type": "array",
            }
        },
        "required": ["commands"],
    },
    {
        "name": "web_search",
        "description": "Search the live web for current public webpages using Exa.",
        "parameters": {
            "query": {"type": "string"},
            "limit": {"default": 5, "type": "integer"},
            "search_type": {
                "default": "auto",
                "enum": ["auto", "deep"],
                "type": "string",
            },
        },
        "required": ["query"],
    },
    {
        "name": "web_fetch",
        "description": "Fetch readable text content for a public webpage URL using Exa.",
        "parameters": {"url": {"type": "string"}},
        "required": ["url"],
    },
]

STATULAR_RESULT_MESSAGE = {
    "role": "tool",
    "content": [
        {
            "type": "tool_result",
            "files": None,
            "result": [
                {
                    "text": "Found 8 results for queries ['distribution to children in equal shares', 'divide remaining trust estate into equal shares for each child', 'predeceased child share to issue by right of representation'] in index 'sample_clauses':",
                    "type": "text",
                }
            ],
            "built_in": False,
            "tool_call_id": "toolu_bdrk_01SzpKrNKaSqDbD1PXqaVEDR",
            "built_in_type": None,
        }
    ],
    "log_version": 2,
}
