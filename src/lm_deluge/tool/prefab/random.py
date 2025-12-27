"""Random generation prefab tools."""

import json
import random
import secrets
from typing import Any

from lm_deluge.tool import Tool


class RandomTools:
    """
    A prefab tool set for generating random values.

    Provides tools to generate random floats, pick random items from lists,
    generate random integers, and create secure random tokens.

    Args:
        float_tool_name: Name for the random float tool (default: "random_float")
        choice_tool_name: Name for the random choice tool (default: "random_choice")
        int_tool_name: Name for the random integer tool (default: "random_int")
        token_tool_name: Name for the random token tool (default: "random_token")

    Example:
        ```python
        # Create the random tools manager
        random_tools = RandomTools()

        # Get tools
        tools = random_tools.get_tools()
        ```
    """

    def __init__(
        self,
        *,
        float_tool_name: str = "random_float",
        choice_tool_name: str = "random_choice",
        int_tool_name: str = "random_int",
        token_tool_name: str = "random_token",
    ):
        self.float_tool_name = float_tool_name
        self.choice_tool_name = choice_tool_name
        self.int_tool_name = int_tool_name
        self.token_tool_name = token_tool_name
        self._tools: list[Tool] | None = None

    def _random_float(self) -> str:
        """
        Generate a random float between 0 and 1.

        Returns:
            JSON string with the random float value
        """
        try:
            value = random.random()
            return json.dumps({"status": "success", "value": value})
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

    def _random_choice(self, items: list[Any]) -> str:
        """
        Pick a random item from a list.

        Args:
            items: List of items to choose from

        Returns:
            JSON string with the randomly selected item
        """
        try:
            if not items:
                return json.dumps(
                    {"status": "error", "error": "Cannot choose from an empty list"}
                )

            choice = random.choice(items)
            return json.dumps({"status": "success", "value": choice})
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

    def _random_int(self, min_value: int, max_value: int) -> str:
        """
        Generate a random integer in a given range (inclusive).

        Args:
            min_value: Minimum value (inclusive)
            max_value: Maximum value (inclusive)

        Returns:
            JSON string with the random integer
        """
        try:
            if min_value > max_value:
                return json.dumps(
                    {
                        "status": "error",
                        "error": f"min_value ({min_value}) cannot be greater than max_value ({max_value})",
                    }
                )

            value = random.randint(min_value, max_value)
            return json.dumps({"status": "success", "value": value})
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

    def _random_token(self, length: int = 32) -> str:
        """
        Generate a secure random token using secrets.token_urlsafe.

        Args:
            length: Number of bytes for the token (default: 32)

        Returns:
            JSON string with the random token
        """
        try:
            if length <= 0:
                return json.dumps(
                    {"status": "error", "error": "length must be greater than 0"}
                )

            token = secrets.token_urlsafe(length)
            return json.dumps({"status": "success", "value": token})
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

    def get_tools(self) -> list[Tool]:
        """Return the list of random generation tools."""
        if self._tools is not None:
            return self._tools

        self._tools = [
            Tool(
                name=self.float_tool_name,
                description="Generate a random float between 0 and 1 (inclusive of 0, exclusive of 1).",
                run=self._random_float,
                parameters={},
                required=[],
            ),
            Tool(
                name=self.choice_tool_name,
                description="Pick a random item from a provided list of items.",
                run=self._random_choice,
                parameters={
                    "items": {
                        "type": "array",
                        "description": "List of items to choose from. Can contain any JSON-serializable values.",
                        "items": {},
                    }
                },
                required=["items"],
            ),
            Tool(
                name=self.int_tool_name,
                description="Generate a random integer within a specified range (inclusive on both ends).",
                run=self._random_int,
                parameters={
                    "min_value": {
                        "type": "integer",
                        "description": "Minimum value (inclusive)",
                    },
                    "max_value": {
                        "type": "integer",
                        "description": "Maximum value (inclusive)",
                    },
                },
                required=["min_value", "max_value"],
            ),
            Tool(
                name=self.token_tool_name,
                description=(
                    "Generate a cryptographically secure random URL-safe token. "
                    "Useful for generating passwords, API keys, or other secure tokens."
                ),
                run=self._random_token,
                parameters={
                    "length": {
                        "type": "integer",
                        "description": "Number of random bytes to use for the token (default: 32). The actual token will be longer due to base64 encoding.",
                        "default": 32,
                    }
                },
                required=[],
            ),
        ]

        return self._tools
