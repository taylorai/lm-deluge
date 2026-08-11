import base64
import io
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

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
from .video import Video

Role = Literal["system", "user", "assistant", "tool"]
Part = Text | Image | Video | File | ToolCall | ToolResult | Thinking


#####################################################
# Message: One conversational turn (role + parts)   #
#####################################################
@dataclass
class Message:
    role: Role
    parts: list[Part]
    # Provider-specific per-message metadata (e.g. OpenAI Responses `phase`).
    # Round-tripped through serialization but ignored by providers that don't use it.
    extra: dict | None = None

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
    def tool_calls_to_execute(self) -> list["ToolCall"]:
        """Get tool calls that require client-side execution."""
        tool_calls: list[ToolCall] = []
        for part in self.tool_calls:
            if not part.built_in:
                tool_calls.append(part)
                continue
            if part.built_in_type == "computer_call":
                tool_calls.append(part)
        return tool_calls

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

    def to_log(
        self,
        *,
        lossless: bool = True,
        preserve_media: bool | None = None,
    ) -> dict:
        """
        Return a JSON-serialisable v2 log representation of the message.

        Args:
            lossless: Preserve media and opaque provider payloads. This is the default.
            preserve_media: Deprecated compatibility alias. When provided, it overrides
                ``lossless`` (True selects lossless mode, False selects compact mode).
        """
        if preserve_media is not None:
            warnings.warn(
                "preserve_media is deprecated; use lossless instead",
                DeprecationWarning,
                stacklevel=2,
            )
            lossless = preserve_media

        def serialize_media_part(part: Text | Image) -> dict:
            if isinstance(part, Text):
                block: dict = {"type": "text", "text": part.text}
                signature = serialize_signature(part.thought_signature)
                if signature is not None:
                    block["thought_signature"] = signature
                return block

            if lossless:
                return {
                    "type": "image",
                    "data": base64.b64encode(part._bytes()).decode("ascii"),
                    "media_type": part.media_type,
                    "detail": part.detail,
                }
            width, height = part.size
            return {
                "type": "image",
                "omitted": "media",
                "tag": f"<Image {width}×{height}>",
            }

        content_blocks: list[dict] = []
        for p in self.parts:
            if isinstance(p, (Text, Image)):
                content_blocks.append(serialize_media_part(p))
            elif isinstance(p, File):
                if lossless:
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
                        {
                            "type": "file",
                            "omitted": "media",
                            "tag": f"<File {size} bytes>",
                        }
                    )
            elif isinstance(p, ToolCall):
                tool_call_block: dict = {
                    "type": "tool_call",
                    "id": p.id,
                    "name": p.name,
                    "arguments": json_safe(p.arguments),
                    "built_in": p.built_in,
                    "built_in_type": p.built_in_type,
                }
                if p.extra_body is not None:
                    extra_body = dict(p.extra_body)
                    if not lossless:
                        extra_body.pop("raw_item", None)
                    tool_call_block["extra_body"] = json_safe(extra_body)
                signature = serialize_signature(p.thought_signature)
                if signature is not None:
                    tool_call_block["thought_signature"] = signature
                content_blocks.append(tool_call_block)
            elif isinstance(p, ToolResult):
                if isinstance(p.result, list):
                    serialized_result: Any = [
                        serialize_media_part(part) for part in p.result
                    ]
                else:
                    serialized_result = json_safe(p.result)
                content_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_call_id": p.tool_call_id,
                        "result": serialized_result,
                        "built_in": p.built_in,
                        "built_in_type": p.built_in_type,
                        "files": json_safe(p.files),
                    }
                )
            elif isinstance(p, Thinking):
                thinking_block: dict = {
                    "type": "thinking",
                    "content": p.content,
                    "id": p.id,
                    "summary": p.summary,
                }
                if lossless and p.raw_payload is not None:
                    thinking_block["raw_payload"] = json_safe(p.raw_payload)
                signature = serialize_signature(p.thought_signature)
                if signature is not None:
                    thinking_block["thought_signature"] = signature
                content_blocks.append(thinking_block)

        result: dict = {
            "log_version": 2,
            "role": self.role,
            "content": content_blocks,
        }
        if self.extra:
            result["extra"] = json_safe(self.extra)
        return result

    @classmethod
    def from_log(cls, data: dict) -> "Message":
        """Re-hydrate a Message previously produced by `to_log()`."""
        role: Role = data["role"]
        parts: list[Part] = []

        def deserialize_tool_result_part(part: dict) -> ToolResultPart:
            if part.get("type") == "text":
                return Text(
                    part.get("text", ""),
                    thought_signature=deserialize_signature(
                        part.get("thought_signature")
                    ),
                )
            if part.get("type") == "image":
                if "data" in part:
                    return Image(
                        data=base64.b64decode(part["data"]),
                        media_type=part.get("media_type"),
                        detail=part.get("detail", "auto"),
                    )
                return Text(part.get("tag", "<Image omitted>"))
            raise ValueError(f"Unknown tool result part type {part.get('type')!r}")

        for p in data.get("content", []):
            if p["type"] == "text":
                parts.append(
                    Text(
                        p.get("text", ""),
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
                    # v1 placeholder or v2 typed omission marker.
                    parts.append(Text(p.get("tag", "<Image omitted>")))
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
                    # v1 placeholder or v2 typed omission marker.
                    parts.append(Text(p.get("tag", "<File omitted>")))
            elif p["type"] == "tool_call":
                parts.append(
                    ToolCall(
                        id=p.get("id", ""),
                        name=p.get("name", ""),
                        arguments=p.get("arguments", {}),
                        built_in=p.get("built_in", False),
                        built_in_type=p.get("built_in_type"),
                        extra_body=p.get("extra_body"),
                        thought_signature=deserialize_signature(
                            p.get("thought_signature")
                        ),
                    )
                )
            elif p["type"] == "tool_result":
                serialized_result = p.get("result", "")
                if isinstance(serialized_result, list):
                    tool_result: str | dict | list[ToolResultPart] = [
                        deserialize_tool_result_part(part)
                        for part in serialized_result
                        if isinstance(part, dict)
                    ]
                else:
                    tool_result = serialized_result
                parts.append(
                    ToolResult(
                        tool_call_id=p.get("tool_call_id", ""),
                        result=tool_result,
                        built_in=p.get("built_in", False),
                        built_in_type=p.get("built_in_type"),
                        files=p.get("files"),
                    )
                )
            elif p["type"] == "thinking":
                parts.append(
                    Thinking(
                        content=p.get("content", ""),
                        raw_payload=p.get("raw_payload"),
                        id=p.get("id"),
                        thought_signature=deserialize_signature(
                            p.get("thought_signature")
                        ),
                        summary=p.get("summary"),
                    )
                )
            else:
                raise ValueError(f"Unknown part type {p['type']!r}")

        extra = data.get("extra")
        return cls(role, parts, extra=extra if isinstance(extra, dict) else None)

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

    def with_video(
        self,
        data: bytes | str | Path | io.BytesIO | Video,
        *,
        media_type: str | None = None,
    ) -> "Message":
        if not isinstance(data, Video):
            video = Video(data, media_type=media_type)
        else:
            video = data

        self.parts.append(video)
        return self

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
        self, tool_call_id: str, result: str | dict | list[ToolResultPart]
    ) -> "Message":
        """Append a tool result block and return self for chaining."""
        self.parts.append(ToolResult(tool_call_id=tool_call_id, result=result))
        return self

    @deprecated("with_tool_result")
    def add_tool_result(
        self, tool_call_id: str, result: str | dict | list[ToolResultPart]
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
        video: Video | str | bytes | Path | io.BytesIO | None = None,
    ) -> "Message":
        res = cls("user", [])
        if text is not None:
            res.with_text(text)
        if image is not None:
            res.with_image(image)
        if file is not None:
            res.with_file(file)
        if video is not None:
            res.with_video(video)
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
                elif item["type"] == "video_url":
                    part_list.append(Video(data=item["video_url"]["url"]))
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
