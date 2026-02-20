"""Philips Hue light control prefab tool."""

import json
import os
import urllib.error
import urllib.request
from typing import Any

from .. import Tool

# Color presets (hue values 0-65535)
COLORS = {
    "red": 0,
    "orange": 5000,
    "yellow": 10000,
    "lime": 18000,
    "green": 25500,
    "cyan": 35000,
    "blue": 46920,
    "purple": 50000,
    "magenta": 55000,
    "pink": 58000,
    "white": None,  # Use color temperature instead
}


class PhilipsHueManager:
    """
    A prefab tool for controlling Philips Hue lights via the local bridge API.

    Provides tools to list lights, turn them on/off, change colors, and adjust brightness.

    Args:
        bridge_ip: IP address of the Hue bridge.
                   If not provided, uses HUE_BRIDGE_IP env variable.
        api_key: API key for the Hue bridge.
                 If not provided, uses HUE_API_KEY env variable.
        timeout: Timeout in seconds for API requests (default: 10).

    Example:
        ```python
        manager = PhilipsHueManager(
            bridge_ip="192.168.1.100",
            api_key="your-api-key-here"
        )

        tools = manager.get_tools()
        ```

        Or using environment variables:
        ```python
        # Set HUE_BRIDGE_IP and HUE_API_KEY env vars
        manager = PhilipsHueManager()
        tools = manager.get_tools()
        ```
    """

    def __init__(
        self,
        bridge_ip: str | None = None,
        api_key: str | None = None,
        *,
        timeout: int = 10,
    ):
        # Handle bridge IP
        if bridge_ip is not None:
            self.bridge_ip = bridge_ip
        else:
            env_bridge_ip = os.environ.get("HUE_BRIDGE_IP")
            if env_bridge_ip:
                self.bridge_ip = env_bridge_ip
            else:
                raise ValueError(
                    "No bridge IP provided. Set bridge_ip parameter or "
                    "HUE_BRIDGE_IP environment variable."
                )

        # Handle API key
        if api_key is not None:
            self.api_key = api_key
        else:
            env_api_key = os.environ.get("HUE_API_KEY")
            if env_api_key:
                self.api_key = env_api_key
            else:
                raise ValueError(
                    "No API key provided. Set api_key parameter or "
                    "HUE_API_KEY environment variable."
                )

        self.timeout = timeout
        self._base_url = f"http://{self.bridge_ip}/api/{self.api_key}"
        self._tools: list[Tool] | None = None

    def _api_request(
        self, endpoint: str, method: str = "GET", data: dict[str, Any] | None = None
    ) -> dict[str, Any] | list[Any]:
        """Make a request to the Hue bridge API."""
        url = f"{self._base_url}{endpoint}"

        encoded_data = None
        if data:
            encoded_data = json.dumps(data).encode("utf-8")

        req = urllib.request.Request(url, data=encoded_data, method=method)
        req.add_header("Content-Type", "application/json")

        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    async def _list_lights(self) -> str:
        """
        List all lights connected to the Hue bridge with their current status.

        Returns:
            JSON string with light information including ID, name, on/off state,
            reachability, and brightness.
        """
        try:
            lights = self._api_request("/lights")
            if not isinstance(lights, dict):
                return json.dumps(
                    {"status": "error", "error": "Unexpected API response"}
                )

            light_list = []
            for light_id, light in sorted(lights.items(), key=lambda x: int(x[0])):
                state = light["state"]
                light_list.append(
                    {
                        "id": light_id,
                        "name": light["name"],
                        "on": state["on"],
                        "reachable": state["reachable"],
                        "brightness": state.get("bri"),
                    }
                )

            return json.dumps({"status": "success", "lights": light_list})

        except urllib.error.URLError as e:
            return json.dumps(
                {"status": "error", "error": f"Error connecting to bridge: {e}"}
            )
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

    def _set_light_state(self, light_id: str, state: dict[str, Any]) -> dict[str, Any]:
        """Set the state of a light and return the result."""
        result = self._api_request(f"/lights/{light_id}/state", "PUT", state)
        if not isinstance(result, list):
            return {"status": "error", "error": "Unexpected API response"}

        successes = []
        errors = []
        for item in result:
            if "success" in item:
                for key, value in item["success"].items():
                    successes.append(f"{key}: {value}")
            elif "error" in item:
                errors.append(item["error"]["description"])

        if errors:
            return {"status": "error", "errors": errors}
        return {"status": "success", "changes": successes}

    def _get_all_light_ids(self) -> list[str]:
        """Get all light IDs from the bridge."""
        lights = self._api_request("/lights")
        if isinstance(lights, dict):
            return list(lights.keys())
        return []

    async def _turn_on(self, light_id: str, brightness: int | None = None) -> str:
        """
        Turn on a light or all lights.

        Args:
            light_id: The ID of the light to turn on, or "all" for all lights.
            brightness: Optional brightness level (1-254).

        Returns:
            JSON string with status and result.
        """
        try:
            state: dict[str, Any] = {"on": True}
            if brightness is not None:
                state["bri"] = max(1, min(254, brightness))

            if light_id.lower() == "all":
                results = []
                for lid in self._get_all_light_ids():
                    result = self._set_light_state(lid, state)
                    results.append({"light_id": lid, **result})
                return json.dumps({"status": "success", "results": results})
            else:
                result = self._set_light_state(light_id, state)
                return json.dumps(result)

        except urllib.error.URLError as e:
            return json.dumps(
                {"status": "error", "error": f"Error connecting to bridge: {e}"}
            )
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

    async def _turn_off(self, light_id: str) -> str:
        """
        Turn off a light or all lights.

        Args:
            light_id: The ID of the light to turn off, or "all" for all lights.

        Returns:
            JSON string with status and result.
        """
        try:
            state = {"on": False}

            if light_id.lower() == "all":
                results = []
                for lid in self._get_all_light_ids():
                    result = self._set_light_state(lid, state)
                    results.append({"light_id": lid, **result})
                return json.dumps({"status": "success", "results": results})
            else:
                result = self._set_light_state(light_id, state)
                return json.dumps(result)

        except urllib.error.URLError as e:
            return json.dumps(
                {"status": "error", "error": f"Error connecting to bridge: {e}"}
            )
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

    async def _set_color(
        self,
        light_id: str,
        color: str,
        brightness: int | None = None,
        saturation: int = 254,
    ) -> str:
        """
        Set the color of a light or all lights.

        Args:
            light_id: The ID of the light, or "all" for all lights.
            color: Color name (red, orange, yellow, lime, green, cyan, blue,
                   purple, magenta, pink, white) or raw hue value (0-65535).
            brightness: Optional brightness level (1-254).
            saturation: Saturation level (0-254, default 254).

        Returns:
            JSON string with status and result.
        """
        try:
            state: dict[str, Any] = {"on": True, "sat": saturation}

            if brightness is not None:
                state["bri"] = max(1, min(254, brightness))

            # Check if it's a named color
            color_lower = color.lower()
            if color_lower in COLORS:
                if color_lower == "white":
                    state["ct"] = 300  # Neutral white color temperature
                    del state["sat"]
                else:
                    state["hue"] = COLORS[color_lower]
            else:
                # Try to parse as hue value
                try:
                    hue_value = int(color)
                    state["hue"] = max(0, min(65535, hue_value))
                except ValueError:
                    available = ", ".join(COLORS.keys())
                    return json.dumps(
                        {
                            "status": "error",
                            "error": f"Unknown color: {color}. Available colors: {available}",
                        }
                    )

            if light_id.lower() == "all":
                results = []
                for lid in self._get_all_light_ids():
                    result = self._set_light_state(lid, state)
                    results.append({"light_id": lid, **result})
                return json.dumps({"status": "success", "results": results})
            else:
                result = self._set_light_state(light_id, state)
                return json.dumps(result)

        except urllib.error.URLError as e:
            return json.dumps(
                {"status": "error", "error": f"Error connecting to bridge: {e}"}
            )
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

    async def _set_brightness(self, light_id: str, brightness: int) -> str:
        """
        Set the brightness of a light (turns it on if off).

        Args:
            light_id: The ID of the light.
            brightness: Brightness level (1-254).

        Returns:
            JSON string with status and result.
        """
        try:
            state = {"on": True, "bri": max(1, min(254, brightness))}
            result = self._set_light_state(light_id, state)
            return json.dumps(result)

        except urllib.error.URLError as e:
            return json.dumps(
                {"status": "error", "error": f"Error connecting to bridge: {e}"}
            )
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

    def get_tools(self) -> list[Tool]:
        """Return the Philips Hue control tools."""
        if self._tools is not None:
            return self._tools

        available_colors = ", ".join(COLORS.keys())

        self._tools = [
            Tool(
                name="hue_list_lights",
                description=(
                    "List all Philips Hue lights connected to the bridge with their "
                    "current status including ID, name, on/off state, reachability, "
                    "and brightness level."
                ),
                run=self._list_lights,
                parameters={},
                required=[],
            ),
            Tool(
                name="hue_turn_on",
                description=(
                    "Turn on a Philips Hue light or all lights. "
                    "Optionally set brightness at the same time."
                ),
                run=self._turn_on,
                parameters={
                    "light_id": {
                        "type": "string",
                        "description": "The ID of the light to turn on, or 'all' for all lights.",
                    },
                    "brightness": {
                        "type": "integer",
                        "description": "Optional brightness level (1-254).",
                    },
                },
                required=["light_id"],
            ),
            Tool(
                name="hue_turn_off",
                description="Turn off a Philips Hue light or all lights.",
                run=self._turn_off,
                parameters={
                    "light_id": {
                        "type": "string",
                        "description": "The ID of the light to turn off, or 'all' for all lights.",
                    },
                },
                required=["light_id"],
            ),
            Tool(
                name="hue_set_color",
                description=(
                    f"Set the color of a Philips Hue light or all lights. "
                    f"Available color names: {available_colors}. "
                    f"You can also use raw hue values (0-65535) where 0=red, "
                    f"25500=green, 46920=blue, 65535=red."
                ),
                run=self._set_color,
                parameters={
                    "light_id": {
                        "type": "string",
                        "description": "The ID of the light, or 'all' for all lights.",
                    },
                    "color": {
                        "type": "string",
                        "description": f"Color name ({available_colors}) or raw hue value (0-65535).",
                    },
                    "brightness": {
                        "type": "integer",
                        "description": "Optional brightness level (1-254).",
                    },
                    "saturation": {
                        "type": "integer",
                        "description": "Saturation level (0-254, default 254 for full saturation).",
                    },
                },
                required=["light_id", "color"],
            ),
            Tool(
                name="hue_set_brightness",
                description=(
                    "Set the brightness of a Philips Hue light. "
                    "Turns the light on if it's off."
                ),
                run=self._set_brightness,
                parameters={
                    "light_id": {
                        "type": "string",
                        "description": "The ID of the light.",
                    },
                    "brightness": {
                        "type": "integer",
                        "description": "Brightness level (1-254).",
                    },
                },
                required=["light_id", "brightness"],
            ),
        ]

        return self._tools


__all__ = ["PhilipsHueManager", "COLORS"]
