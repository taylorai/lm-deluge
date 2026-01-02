import base64
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from lm_deluge.warnings import deprecated

from .file import File
from .image import Image, MediaType
from .serialization import json_safe
from .signatures import (
    deserialize_signature,
    serialize_signature,
    signature_for_provider,
)
from .text import Text
from .thinking import Thinking
from .tool_calls import ToolCall, ToolResult, ToolResultPart

Role = Literal["system", "user", "assistant", "tool"]
Part = Text | Image | File | ToolCall | ToolResult | Thinking


#####################################################
# Message: One conversational turn (role + parts)   #
#####################################################
@dataclass
class Message:
    role: Role
    parts: list[Part]

    @property
    def fingerprint(self) -> str:
        return self.role + "," + ",".join(part.fingerprint for part in self.parts)

    @property
    def completion(self) -> str | None:
        """Extract text content from the first Text part, for backward compatibility."""
        for part in self.parts:
            if isinstance(part, Text):
                return part.text
        return None

    @property
    def tool_calls(self) -> list["ToolCall"]:
        """Get all tool call parts with proper typing."""
        return [part for part in self.parts if part.type == "tool_call"]  # type: ignore

    @property
    def tool_results(self) -> list["ToolResult"]:
        """Get all tool result parts with proper typing."""
        return [part for part in self.parts if part.type == "tool_result"]  # type: ignore

    @property
    def text_parts(self) -> list["Text"]:
        """Get all text parts with proper typing."""
        return [part for part in self.parts if part.type == "text"]  # type: ignore

    @property
    def images(self) -> list[Image]:
        """Get all image parts with proper typing."""
        return [part for part in self.parts if part.type == "image"]  # type: ignore

    @property
    def files(self) -> list[File]:
        """Get all file parts with proper typing."""
        return [part for part in self.parts if part.type == "file"]  # type: ignore

    @property
    def thinking_parts(self) -> list["Thinking"]:
        """Get all thinking parts with proper typing."""
        return [part for part in self.parts if part.type == "thinking"]  # type: ignore

    def to_log(self, *, preserve_media: bool = False) -> dict:
        """
        Return a JSON-serialisable dict that fully captures the message.

        Args:
            preserve_media: If True, store full base64-encoded bytes for images and files.
                           If False (default), replace with placeholder tags.
        """
        content_blocks: list[dict] = []
        for p in self.parts:
            if isinstance(p, Text):
                text_block: dict = {"type": "text", "text": p.text}
                signature = serialize_signature(p.thought_signature)
                if signature is not None:
                    text_block["thought_signature"] = signature
                content_blocks.append(text_block)
            elif isinstance(p, Image):
                if preserve_media:
                    content_blocks.append(
                        {
                            "type": "image",
                            "data": base64.b64encode(p._bytes()).decode("ascii"),
                            "media_type": p.media_type,
                            "detail": p.detail,
                        }
                    )
                else:
                    w, h = p.size
                    content_blocks.append(
                        {"type": "image", "tag": f"<Image ({w}×{h})>"}
                    )
            elif isinstance(p, File):
                if preserve_media:
                    content_blocks.append(
                        {
                            "type": "file",
                            "data": base64.b64encode(p._bytes()).decode("ascii"),
                            "media_type": p.media_type,
                            "filename": p.filename,
                        }
                    )
                else:
                    size = p.size
                    content_blocks.append(
                        {"type": "file", "tag": f"<File ({size} bytes)>"}
                    )
            elif isinstance(p, ToolCall):
                tool_call_block = {
                    "type": "tool_call",
                    "id": p.id,
                    "name": p.name,
                    "arguments": json_safe(p.arguments),
                }
                signature = serialize_signature(p.thought_signature)
                if signature is not None:
                    tool_call_block["thought_signature"] = signature
                content_blocks.append(tool_call_block)
            elif isinstance(p, ToolResult):
                content_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_call_id": p.tool_call_id,
                        "result": json_safe(p.result),
                    }
                )
            elif isinstance(p, Thinking):
                thinking_block: dict = {"type": "thinking", "content": p.content}
                signature = serialize_signature(p.thought_signature)
                if signature is not None:
                    thinking_block["thought_signature"] = signature
                content_blocks.append(thinking_block)

        return {"role": self.role, "content": content_blocks}

    @classmethod
    def from_log(cls, data: dict) -> "Message":
        """Re-hydrate a Message previously produced by `to_log()`."""
        import base64

        role: Role = data["role"]
        parts: list[Part] = []

        for p in data["content"]:
            if p["type"] == "text":
                parts.append(
                    Text(
                        p["text"],
                        thought_signature=deserialize_signature(
                            p.get("thought_signature")
                        ),
                    )
                )
            elif p["type"] == "image":
                if "data" in p:
                    # Full image data was preserved
                    parts.append(
                        Image(
                            data=base64.b64decode(p["data"]),
                            media_type=p.get("media_type"),
                            detail=p.get("detail", "auto"),
                        )
                    )
                else:
                    # Placeholder tag only
                    parts.append(Text(p["tag"]))
            elif p["type"] == "file":
                if "data" in p:
                    # Full file data was preserved
                    parts.append(
                        File(
                            data=base64.b64decode(p["data"]),
                            media_type=p.get("media_type"),
                            filename=p.get("filename"),
                        )
                    )
                else:
                    # Placeholder tag only
                    parts.append(Text(p["tag"]))
            elif p["type"] == "tool_call":
                parts.append(
                    ToolCall(
                        id=p["id"],
                        name=p["name"],
                        arguments=p["arguments"],
                        thought_signature=deserialize_signature(
                            p.get("thought_signature")
                        ),
                    )
                )
            elif p["type"] == "tool_result":
                parts.append(
                    ToolResult(tool_call_id=p["tool_call_id"], result=p["result"])
                )
            elif p["type"] == "thinking":
                parts.append(
                    Thinking(
                        content=p["content"],
                        thought_signature=deserialize_signature(
                            p.get("thought_signature")
                        ),
                    )
                )
            else:
                raise ValueError(f"Unknown part type {p['type']!r}")

        return cls(role, parts)

    def with_text(self, content: str) -> "Message":
        """Append a text block and return self for chaining."""
        self.parts.append(Text(content))
        return self

    @deprecated("with_text")
    def add_text(self, content: str) -> "Message":
        """Append a text block and return self for chaining."""
        return self.with_text(content)

    def with_image(
        self,
        data: bytes | str | Path | io.BytesIO | Image,
        *,
        media_type: MediaType | None = None,
        detail: Literal["low", "high", "auto"] = "auto",
        max_size: int | None = None,
    ) -> "Message":
        """
        Append an image block and return self for chaining.

        If max_size is provided, the image will be resized so that its longer
        dimension equals max_size, but only if the longer dimension is currently
        larger than max_size.
        """
        if not isinstance(data, Image):
            img = Image(data, media_type=media_type, detail=detail)
        else:
            img = data
        # Resize if max_size is provided
        if max_size is not None:
            img.resize(max_size)

        self.parts.append(img)
        return self

    @deprecated("with_image")
    def add_image(
        self,
        data: bytes | str | Path | io.BytesIO | Image,
        *,
        media_type: MediaType | None = None,
        detail: Literal["low", "high", "auto"] = "auto",
        max_size: int | None = None,
    ) -> "Message":
        """
        Append an image block and return self for chaining.

        If max_size is provided, the image will be resized so that its longer
        dimension equals max_size, but only if the longer dimension is currently
        larger than max_size.
        """
        return self.with_image(
            data=data, media_type=media_type, detail=detail, max_size=max_size
        )

    def with_file(
        self,
        data: bytes | str | Path | io.BytesIO | File,
        *,
        media_type: str | None = None,
        filename: str | None = None,
        # remote: bool = False,
        # provider: Literal["openai", "anthropic", "google"] | None = None,
    ) -> "Message":
        """
        Append a file block and return self for chaining.
        """
        if not isinstance(data, File):
            file = File(data, media_type=media_type, filename=filename)
        else:
            file = data

        self.parts.append(file)
        return self

    @deprecated("with_file")
    def add_file(
        self,
        data: bytes | str | Path | io.BytesIO | File,
        *,
        media_type: str | None = None,
        filename: str | None = None,
    ) -> "Message":
        """
        Append a file block and return self for chaining.
        """
        return self.with_file(data, media_type=media_type, filename=filename)

    async def with_remote_file(
        self,
        data: bytes | str | Path | io.BytesIO | File,
        *,
        media_type: str | None = None,
        filename: str | None = None,
        provider: Literal["openai", "anthropic", "google"] = "openai",
    ):
        if not isinstance(data, File):
            file = File(data, media_type=media_type, filename=filename)
        else:
            file = data

        if not file.is_remote:
            file = await file.as_remote(provider=provider)
        else:
            if file.remote_provider != provider:
                raise ValueError(
                    f"File is already remote with provider {file.remote_provider}, cannot change provider"
                )

        self.parts.append(file)
        return self

    def with_tool_call(self, id: str, name: str, arguments: dict) -> "Message":
        """Append a tool call block and return self for chaining."""
        self.parts.append(ToolCall(id=id, name=name, arguments=arguments))
        return self

    @deprecated("with_tool_call")
    def add_tool_call(self, id: str, name: str, arguments: dict) -> "Message":
        """Append a tool call block and return self for chaining."""
        return self.with_tool_call(id, name, arguments)

    def with_tool_result(
        self, tool_call_id: str, result: str | list[ToolResultPart]
    ) -> "Message":
        """Append a tool result block and return self for chaining."""
        self.parts.append(ToolResult(tool_call_id=tool_call_id, result=result))
        return self

    @deprecated("with_tool_result")
    def add_tool_result(
        self, tool_call_id: str, result: str | list[ToolResultPart]
    ) -> "Message":
        """Append a tool result block and return self for chaining."""
        return self.with_tool_result(tool_call_id, result)

    def with_thinking(self, content: str) -> "Message":
        """Append a thinking block and return self for chaining."""
        self.parts.append(Thinking(content=content))
        return self

    @deprecated("with_thinking")
    def add_thinking(self, content: str) -> "Message":
        """Append a thinking block and return self for chaining."""
        return self.with_thinking(content)

    # -------- convenient constructors --------
    @classmethod
    def user(
        cls,
        text: str | None = None,
        *,
        image: str | bytes | Path | io.BytesIO | None = None,
        file: File | str | bytes | Path | io.BytesIO | None = None,
    ) -> "Message":
        res = cls("user", [])
        if text is not None:
            res.with_text(text)
        if image is not None:
            res.with_image(image)
        if file is not None:
            res.with_file(file)
        return res

    @classmethod
    def system(cls, text: str | None = None) -> "Message":
        res = cls("system", [])
        if text is not None:
            res.with_text(text)
        return res

    @classmethod
    def ai(cls, text: str | None = None) -> "Message":
        res = cls("assistant", [])
        if text is not None:
            res.with_text(text)
        return res

    # ──── provider-specific constructors ───
    @classmethod
    def from_oa(cls, msg: dict):
        role = (
            "system"
            if msg["role"] in ["developer", "system"]
            else ("user" if msg["role"] == "user" else "assistant")
        )
        parts: list[Part] = []
        content = msg["content"]
        if isinstance(content, str):
            parts = [Text(content)]
        else:
            part_list = []
            for item in content:
                if item["type"] == "text":
                    part_list.append(Text(item["text"]))
                elif item["type"] == "image_url":
                    part_list.append(Image(data=item["image_url"]["url"]))
                elif item["type"] == "file":
                    file_data = item["file"]
                    if "file_id" in file_data:
                        # Handle file ID reference (not implemented yet)
                        part_list.append(File(data=file_data["file_id"]))
                    elif "file_data" in file_data:
                        # Handle base64 file data
                        part_list.append(
                            File(
                                data=file_data["file_data"],
                                filename=file_data.get("filename"),
                            )
                        )
            parts = part_list

        # Handle tool calls (assistant messages)
        if "tool_calls" in msg:
            part_list = list(parts) if parts else []
            for tool_call in msg["tool_calls"]:
                part_list.append(
                    ToolCall(
                        id=tool_call["id"],
                        name=tool_call["function"]["name"],
                        arguments=json.loads(tool_call["function"]["arguments"]),
                    )
                )
            parts = part_list

        return cls(role, parts)

    @classmethod
    def from_oa_resp(cls, msg: dict):
        raise NotImplementedError("not implemented")

    @classmethod
    def from_anthropic(cls, msg: dict):
        pass
        # role = (
        #     "system"
        #     if msg["role"] in ["developer", "system"]
        #     else ("user" if msg["role"] == "user" else "assistant")
        # )
        # parts: list[Part] = []
        # content = msg["content"]
        # if isinstance(content, str):
        #     parts = [Text(content)]
        # else:
        #     part_list = []
        #     for item in content:
        #         if item["type"] == "text":
        #             part_list.append(Text(item["text"]))
        #         elif item["type"] == "image_url":
        #             part_list.append(Image(data=item["image_url"]["url"]))
        #         elif item["type"] == "file":
        #             file_data = item["file"]
        #             if "file_id" in file_data:
        #                 # Handle file ID reference (not implemented yet)
        #                 part_list.append(File(data=file_data["file_id"]))
        #             elif "file_data" in file_data:
        #                 # Handle base64 file data
        #                 part_list.append(
        #                     File(
        #                         data=file_data["file_data"],
        #                         filename=file_data.get("filename"),
        #                     )
        #                 )
        #     parts = part_list

        # # Handle tool calls (assistant messages)
        # if "tool_calls" in msg:
        #     part_list = list(parts) if parts else []
        #     for tool_call in msg["tool_calls"]:
        #         part_list.append(
        #             ToolCall(
        #                 id=tool_call["id"],
        #                 name=tool_call["function"]["name"],
        #                 arguments=json.loads(tool_call["function"]["arguments"]),
        #             )
        #         )
        #     parts = part_list

        # return cls(role, parts)

    # ───── provider-specific emission ─────
    def oa_chat(self) -> dict:
        if self.role == "tool":
            # For tool messages, we expect a single ToolResult part (after splitting in to_openai)
            tool_results = [p for p in self.parts if isinstance(p, ToolResult)]
            if len(tool_results) == 1:
                tool_result = tool_results[0]
                return tool_result.oa_chat()
            else:
                raise ValueError(
                    f"Tool role messages must contain exactly one ToolResult part for OpenAI, got {len(tool_results)}"
                )
        else:
            content = []
            tool_calls = []

            for p in self.parts:
                if isinstance(p, ToolCall):
                    tool_calls.append(p.oa_chat())
                else:
                    content.append(p.oa_chat())

            result = {"role": self.role, "content": content}
            if tool_calls:
                result["tool_calls"] = tool_calls

            return result

    def oa_resp(self) -> dict:
        content = [p.oa_resp() for p in self.parts]
        # For OpenAI Responses API, handle tool results specially
        if self.role == "tool" or (
            self.role == "user" and any(isinstance(p, ToolResult) for p in self.parts)
        ):
            # Tool results are returned directly, not wrapped in a message
            # This handles computer_call_output when stored as ToolResult
            if len(self.parts) == 1 and isinstance(self.parts[0], ToolResult):
                return self.parts[0].oa_resp()
        return {"role": self.role, "content": content}

    def anthropic(self) -> dict:
        # Anthropic: system message is *not* in the list
        if self.role == "system":
            raise ValueError("Anthropic keeps system outside message list")
        content: list[dict] = []
        for part in self.parts:
            if isinstance(part, Thinking) and part.raw_payload is None:
                signature = signature_for_provider(part.thought_signature, "anthropic")
                if signature is None:
                    continue
            content.append(part.anthropic())
        if not content:
            content = [{"type": "text", "text": ""}]
        # Shortcut: single text becomes a bare string
        if len(content) == 1 and content[0].get("type") == "text":
            content = content[0]["text"]
        return {"role": self.role, "content": content}

    def gemini(self) -> dict:
        parts = [p.gemini() for p in self.parts]
        # Shortcut: single text becomes a bare string
        role = "user" if self.role == "user" else "model"
        return {"role": role, "parts": parts}

    def mistral(self) -> dict:
        parts = [p.mistral() for p in self.parts]
        # Shortcut: single text becomes a bare string
        role = self.role
        return {"role": role, "content": parts}

    def nova(self) -> dict:
        # Nova: system message is kept outside message list (like Anthropic/Gemini)
        if self.role == "system":
            raise ValueError("Nova keeps system outside message list")
        # For tool messages, we need to emit tool results in user role
        if self.role == "tool":
            content = [p.nova() for p in self.parts if isinstance(p, ToolResult)]
            return {"role": "user", "content": content}
        # Regular user/assistant messages
        content = [p.nova() for p in self.parts]
        return {"role": self.role, "content": content}
