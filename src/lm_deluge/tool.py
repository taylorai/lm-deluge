from typing import Any, Literal, Callable
from pydantic import BaseModel, Field


class ToolSpec(BaseModel):
    """
    Provider‑agnostic tool definition with no extra nesting.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    required: list[str] = Field(default_factory=list)
    additionalProperties: bool | None = None  # only
    # if desired, can provide a callable to run the tool
    run: Callable | None = None

    def call(self, **kwargs):
        if self.run is None:
            raise ValueError("No run function provided")
        return self.run(**kwargs)

    def _json_schema(self, include_additional_properties=False) -> dict[str, Any]:
        res = {
            "type": "object",
            "properties": self.parameters,
            # for openai all must be required
            "required": list(self.parameters.keys()),
        }
        if include_additional_properties:
            res["additionalProperties"] = False

        return res

    # ---------- dumpers ----------
    def for_openai_responses(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self._json_schema(include_additional_properties=True),
        }

    def for_openai_completions(self, *, strict: bool = True) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._json_schema(include_additional_properties=True),
                "strict": strict,
            },
        }

    def for_anthropic(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self._json_schema(),
        }

    def for_google(self) -> dict[str, Any]:
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
    ) -> dict[str, Any]:
        if provider == "openai-responses":
            return self.for_openai_responses()
        if provider == "openai-completions":
            return self.for_openai_completions(**kw)
        if provider == "anthropic":
            return self.for_anthropic()
        if provider == "google":
            return self.for_google()
        raise ValueError(provider)
