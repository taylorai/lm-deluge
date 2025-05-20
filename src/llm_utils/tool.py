from typing import Any, Dict, Literal, Callable
from pydantic import BaseModel, Field


class ToolSpec(BaseModel):
    """
    Provider‑agnostic tool definition with no extra nesting.
    """

    name: str
    description: str
    parameters: Dict[str, Any]
    required: list[str] = Field(default_factory=list)
    additionalProperties: bool | None = None  # only
    # if desired, can provide a callable to run the tool
    run: Callable | None = None

    def call(self, **kwargs):
        if self.run is None:
            raise ValueError("No run function provided")
        return self.run(**kwargs)

    def _json_schema(self, include_additional_properties=False) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": self.parameters,
            "required": self.required or [],
            **(
                {"additionalProperties": self.additionalProperties}
                if self.additionalProperties is not None
                and include_additional_properties
                else {}
            ),
        }

    # ---------- dumpers ----------
    def for_openai_responses(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self._json_schema(include_additional_properties=True),
        }

    def for_openai_completions(self, *, strict: bool = True) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._json_schema(),
                "strict": strict,
            },
        }

    def for_anthropic(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self._json_schema(),
        }

    def for_google(self) -> Dict[str, Any]:
        """
        Shape used by google.genai docs.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._json_schema(),
        }

    def dump_for(
        self,
        provider: Literal[
            "openai-responses", "openai-completions", "anthropic", "google"
        ],
        **kw,
    ) -> Dict[str, Any]:
        if provider == "openai-responses":
            return self.for_openai_responses()
        if provider == "openai-completions":
            return self.for_openai_completions(**kw)
        if provider == "anthropic":
            return self.for_anthropic()
        if provider == "google":
            return self.for_google()
        raise ValueError(provider)


# ---- computer tools (for non-CUA models) ----
_BUTTONS = ["left", "right", "wheel", "back", "forward"]

# --- helpers ----
_COORD_OBJECT = {
    "type": "object",
    "properties": {
        "x": {"type": "integer", "description": "X-coordinate in pixels"},
        "y": {"type": "integer", "description": "Y-coordinate in pixels"},
    },
    "required": ["x", "y"],
}


def _coord_field(desc: str):
    return {"type": "integer", "description": desc}
