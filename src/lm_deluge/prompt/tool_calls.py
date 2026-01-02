import json
from dataclasses import dataclass, field
from typing import TypedDict

import xxhash
from .text import Text
from .image import Image

from .signatures import (
    ThoughtSignatureLike,
    normalize_signature,
    signature_for_provider,
)


class ContainerFile(TypedDict):
    """
    Metadata for a file created by Anthropic code execution or skills.

    These files are stored in Anthropic's Files API and can be downloaded
    using the file_id.
    """

    file_id: str | None
    filename: str
    media_type: str | None


@dataclass(slots=True)
class ToolCall:
    id: str  # unique identifier
    name: str  # function name
    arguments: dict  # parsed arguments
    type: str = field(init=False, default="tool_call")
    # built-in tool handling
    built_in: bool = False
    built_in_type: str | None = None
    extra_body: dict | None = None
    # for gemini 3 - thought signatures to maintain reasoning context
    thought_signature: ThoughtSignatureLike | None = None

    def __post_init__(self) -> None:
        self.thought_signature = normalize_signature(self.thought_signature)

    @property
    def fingerprint(self) -> str:
        return xxhash.xxh64(
            f"{self.id}:{self.name}:{json.dumps(self.arguments, sort_keys=True)}".encode()
        ).hexdigest()

    # ── provider-specific emission ────────────────────────────────────────────
    def oa_chat(self) -> dict:  # OpenAI Chat Completions
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": json.dumps(self.arguments)},
        }

    def oa_resp(self) -> dict:  # OpenAI Responses
        return {
            "type": "function_call",
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
        }

    def anthropic(self) -> dict:  # Anthropic Messages
        return {
            "type": "tool_use",
            "id": self.id,
            "name": self.name,
            "input": self.arguments,
        }

    def gemini(self) -> dict:
        result = {"functionCall": {"name": self.name, "args": self.arguments}}
        signature = signature_for_provider(self.thought_signature, "gemini")
        if signature is not None:
            result["thoughtSignature"] = signature  # type: ignore
        return result

    def mistral(self) -> dict:
        return {
            "type": "tool_call",
            "id": self.id,
            "function": {"name": self.name, "arguments": json.dumps(self.arguments)},
        }

    def nova(self) -> dict:
        return {
            "toolUse": {
                "toolUseId": self.id,
                "name": self.name,
                "input": self.arguments,
            }
        }


ToolResultPart = Text | Image


@dataclass(slots=True)
class ToolResult:
    tool_call_id: str  # references the ToolCall.id
    # tool execution result - can be string or list for images
    result: str | dict | list[ToolResultPart]
    type: str = field(init=False, default="tool_result")
    # NEW! instead of specific carve-out for computer use,
    # need to handle all built-ins for OpenAI
    built_in: bool = False
    built_in_type: str | None = None
    # Files created by code execution / skills (Anthropic only)
    files: list[ContainerFile] | None = None

    @property
    def fingerprint(self) -> str:
        if isinstance(self.result, str):
            result_str = self.result
        elif isinstance(self.result, list):
            result_str = json.dumps([part.fingerprint for part in self.result])
        else:
            raise ValueError("unsupported self.result type")
        return xxhash.xxh64(f"{self.tool_call_id}:{result_str}".encode()).hexdigest()

    # ── provider-specific emission ────────────────────────────────────────────
    def get_images(self) -> list[Image]:
        # for openai, we can't include images in tool result, so we have to
        # include them in the next user message
        if isinstance(self.result, str):
            return []
        elif isinstance(self.result, list):
            images = []
            for block in self.result:
                if isinstance(block, Image):
                    images.append(block)
            return images
        else:
            raise ValueError("unexpected tool result type")

    def oa_chat(
        self,
    ) -> dict:  # OpenAI Chat Completions - tool results are separate messages
        # print("serializing toolresult with oa_chat...")
        # print("typeof self.result:", type(self.result))
        if isinstance(self.result, str):
            return {
                "role": "tool",
                "tool_call_id": self.tool_call_id,
                "content": self.result,
            }
        elif isinstance(self.result, list):
            # OpenAI only accepts strings! this is a painful limitation
            image_idx = 0
            text_result = ""
            for block in self.result:
                if isinstance(block, Text):
                    text_result += block.text
                    text_result += "\n\n---\n\n"
                else:
                    image_idx += 1
                    text_result += f"[Image {image_idx} in following user message]"
                    text_result += "\n\n---\n\n"

            return {
                "role": "tool",
                "tool_call_id": self.tool_call_id,
                "content": text_result,
            }
        else:
            raise ValueError("result type not supported")

    def oa_resp(self) -> dict:  # OpenAI Responses
        # print("typeof self.result:", type(self.result))
        # if normal (not built-in just return the regular output
        if not self.built_in:
            result = (
                json.dumps(self.result)
                if isinstance(self.result, list)
                else self.result
            )
            return {
                "type": "function_result",
                "call_id": self.tool_call_id,
                "result": result,
            }

        # if it's a built-in, OpenAI expects special type for each
        else:
            assert isinstance(self.result, dict)
            output_data = self.result.copy()
            result = {
                "type": self.built_in_type,
                "call_id": self.tool_call_id,
            }
            if self.built_in_type == "computer_call":
                # OpenAI expects "computer_call_output" for the result type
                result["type"] = "computer_call_output"
                result["output"] = output_data.get("output", {})
                if "acknowledged_safety_checks" in output_data:
                    result["acknowledged_safety_checks"] = output_data[
                        "acknowledged_safety_checks"
                    ]
            elif self.built_in_type == "image_generation_call":
                raise NotImplementedError(
                    "implement image generation call handling in tool result"
                )
            elif self.built_in_type == "web_search_call":
                pass

            return result

    def anthropic(self) -> dict:  # Anthropic Messages
        if isinstance(self.result, str):
            return {
                "type": "tool_result",
                "tool_use_id": self.tool_call_id,
                "content": self.result,
            }
        elif isinstance(self.result, list):
            # handle list of content blocks
            return {
                "type": "tool_result",
                "tool_use_id": self.tool_call_id,
                "content": [part.anthropic() for part in self.result],
            }
        else:
            raise ValueError("unsupported self.result type")

    def gemini(self) -> dict:
        # Build the function response
        func_response: dict = {
            "name": self.tool_call_id,  # Gemini uses name field for ID
        }

        # Handle different result types
        if isinstance(self.result, str):
            func_response["response"] = {"result": self.result}
        elif isinstance(self.result, dict):
            # Check for Gemini computer use format with inline screenshot
            if self.built_in_type == "gemini_computer_use":
                # Gemini CU expects response dict with optional inline_data parts
                func_response["response"] = self.result.get("response", {})
                # Include inline data (screenshot) if present
                if "inline_data" in self.result:
                    func_response["parts"] = [
                        {
                            "inlineData": {
                                "mimeType": self.result["inline_data"].get(
                                    "mime_type", "image/png"
                                ),
                                "data": self.result["inline_data"]["data"],
                            }
                        }
                    ]
            else:
                func_response["response"] = self.result
        elif isinstance(self.result, list):
            # Handle content blocks (images, etc.) - not yet implemented
            raise ValueError("can't handle content blocks for gemini yet")
        else:
            func_response["response"] = {"result": str(self.result)}

        return {"functionResponse": func_response}

    def mistral(self) -> dict:
        return {
            "type": "tool_result",
            "tool_call_id": self.tool_call_id,
            "content": self.result,
        }

    def nova(self) -> dict:
        # Build content based on result type
        if isinstance(self.result, str):
            content = [{"text": self.result}]
        elif isinstance(self.result, list):
            content = []
            for part in self.result:
                if isinstance(part, Text):
                    content.append({"text": part.text})
                elif isinstance(part, Image):
                    content.append(part.nova())
                else:
                    content.append({"text": str(part)})
        else:
            content = [{"json": self.result}]

        return {
            "toolResult": {
                "toolUseId": self.tool_call_id,
                "content": content,
                "status": "success",
            }
        }
