import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

import tiktoken
import xxhash

from lm_deluge.file import File
from lm_deluge.image import Image
from lm_deluge.models import APIModel

CachePattern = Literal[
    "tools_only",
    "system_and_tools",
    "last_user_message",
    "last_2_user_messages",
    "last_3_user_messages",
]

###############################################################################
# 1. Low-level content blocks – either text or an image                       #
###############################################################################
Role = Literal["system", "user", "assistant", "tool"]


@dataclass(slots=True)
class Text:
    text: str
    type: str = field(init=False, default="text")

    @property
    def fingerprint(self) -> str:
        return xxhash.xxh64(self.text.encode()).hexdigest()

    # ── provider-specific emission ────────────────────────────────────────────
    def oa_chat(self) -> dict | str:  # OpenAI Chat Completions
        return {"type": "text", "text": self.text}

    def oa_resp(self) -> dict:  # OpenAI *Responses*  (new)
        return {"type": "input_text", "text": self.text}

    def anthropic(self) -> dict:  # Anthropic Messages
        return {"type": "text", "text": self.text}

    def gemini(self) -> dict:
        return {"text": self.text}

    def mistral(self) -> dict:
        return {"type": "text", "text": self.text}


@dataclass(slots=True)
class ToolCall:
    id: str  # unique identifier
    name: str  # function name
    arguments: dict  # parsed arguments
    type: str = field(init=False, default="tool_call")

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
        return {"functionCall": {"name": self.name, "args": self.arguments}}

    def mistral(self) -> dict:
        return {
            "type": "tool_call",
            "id": self.id,
            "function": {"name": self.name, "arguments": json.dumps(self.arguments)},
        }


@dataclass(slots=True)
class ToolResult:
    tool_call_id: str  # references the ToolCall.id
    result: (
        str | dict | list[dict]
    )  # tool execution result - can be string or list for images
    type: str = field(init=False, default="tool_result")

    @property
    def fingerprint(self) -> str:
        result_str = (
            json.dumps(self.result, sort_keys=True)
            if isinstance(self.result, list) or isinstance(self.result, dict)
            else str(self.result)
        )
        return xxhash.xxh64(f"{self.tool_call_id}:{result_str}".encode()).hexdigest()

    # ── provider-specific emission ────────────────────────────────────────────
    def oa_chat(
        self,
    ) -> dict:  # OpenAI Chat Completions - tool results are separate messages
        content = (
            json.dumps(self.result) if isinstance(self.result, list) else self.result
        )
        return {"tool_call_id": self.tool_call_id, "content": content}

    def oa_resp(self) -> dict:  # OpenAI Responses
        # Check if this is a computer use output (special case)
        if isinstance(self.result, dict) and self.result.get("_computer_use_output"):
            # This is a computer use output, emit it properly
            output_data = self.result.copy()
            output_data.pop("_computer_use_output")  # Remove marker

            result = {
                "type": "computer_call_output",
                "call_id": self.tool_call_id,
                "output": output_data.get("output", {}),
            }

            # Add acknowledged safety checks if present
            if "acknowledged_safety_checks" in output_data:
                result["acknowledged_safety_checks"] = output_data[
                    "acknowledged_safety_checks"
                ]

            return result

        # Regular function result
        result = (
            json.dumps(self.result) if isinstance(self.result, list) else self.result
        )
        return {
            "type": "function_result",
            "call_id": self.tool_call_id,
            "result": result,
        }

    def anthropic(self) -> dict:  # Anthropic Messages
        return {
            "type": "tool_result",
            "tool_use_id": self.tool_call_id,
            "content": self.result,
        }

    def gemini(self) -> dict:
        return {
            "functionResponse": {
                "name": self.tool_call_id,  # Gemini uses name field for ID
                "response": {"result": self.result},
            }
        }

    def mistral(self) -> dict:
        return {
            "type": "tool_result",
            "tool_call_id": self.tool_call_id,
            "content": self.result,
        }


@dataclass(slots=True)
class Thinking:
    content: str  # reasoning content (o1, Claude thinking, etc.)
    type: str = field(init=False, default="thinking")

    @property
    def fingerprint(self) -> str:
        return xxhash.xxh64(self.content.encode()).hexdigest()

    # ── provider-specific emission ────────────────────────────────────────────
    def oa_chat(self) -> dict:  # OpenAI Chat Completions
        # Thinking is typically not emitted back, but if needed:
        return {"type": "text", "text": f"[Thinking: {self.content}]"}

    def oa_resp(self) -> dict:  # OpenAI Responses
        return {"type": "reasoning", "content": self.content}

    def anthropic(self) -> dict:  # Anthropic Messages
        return {"type": "thinking", "thinking": self.content}

    def gemini(self) -> dict:
        return {"text": f"[Thinking: {self.content}]"}

    def mistral(self) -> dict:
        return {"type": "text", "text": f"[Thinking: {self.content}]"}


Part = Text | Image | File | ToolCall | ToolResult | Thinking


###############################################################################
# 2. One conversational turn (role + parts)                                   #
###############################################################################
@dataclass(slots=True)
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

    def to_log(self) -> dict:
        """
        Return a JSON-serialisable dict that fully captures the message.
        """
        content_blocks: list[dict] = []
        for p in self.parts:
            if isinstance(p, Text):
                content_blocks.append({"type": "text", "text": p.text})
            elif isinstance(p, Image):  # Image – redact the bytes, keep a hint
                w, h = p.size
                content_blocks.append({"type": "image", "tag": f"<Image ({w}×{h})>"})
            elif isinstance(p, File):  # File – redact the bytes, keep a hint
                size = p.size
                content_blocks.append({"type": "file", "tag": f"<File ({size} bytes)>"})
            elif isinstance(p, ToolCall):
                content_blocks.append(
                    {
                        "type": "tool_call",
                        "id": p.id,
                        "name": p.name,
                        "arguments": p.arguments,
                    }
                )
            elif isinstance(p, ToolResult):
                content_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_call_id": p.tool_call_id,
                        "result": p.result,
                    }
                )
            elif isinstance(p, Thinking):
                content_blocks.append({"type": "thinking", "content": p.content})

        return {"role": self.role, "content": content_blocks}

    @classmethod
    def from_log(cls, data: dict) -> "Message":
        """Re-hydrate a Message previously produced by `to_log()`."""
        role: Role = data["role"]
        parts: list[Part] = []

        for p in data["content"]:
            if p["type"] == "text":
                parts.append(Text(p["text"]))
            elif p["type"] == "image":
                # We only stored a placeholder tag, so keep that placeholder.
                parts.append(Image(p["tag"], detail="low"))
            elif p["type"] == "file":
                # We only stored a placeholder tag, so keep that placeholder.
                parts.append(File(p["tag"]))
            elif p["type"] == "tool_call":
                parts.append(
                    ToolCall(id=p["id"], name=p["name"], arguments=p["arguments"])
                )
            elif p["type"] == "tool_result":
                parts.append(
                    ToolResult(tool_call_id=p["tool_call_id"], result=p["result"])
                )
            elif p["type"] == "thinking":
                parts.append(Thinking(content=p["content"]))
            else:
                raise ValueError(f"Unknown part type {p['type']!r}")

        return cls(role, parts)

    def add_text(self, content: str) -> "Message":
        """Append a text block and return self for chaining."""
        self.parts.append(Text(content))
        return self

    def add_image(
        self,
        data: bytes | str | Path | io.BytesIO,
        *,
        media_type: str | None = None,
        detail: Literal["low", "high", "auto"] = "auto",
        max_size: int | None = None,
    ) -> "Message":
        """
        Append an image block and return self for chaining.

        If max_size is provided, the image will be resized so that its longer
        dimension equals max_size, but only if the longer dimension is currently
        larger than max_size.
        """
        img = Image(data, media_type=media_type, detail=detail)

        # Resize if max_size is provided
        if max_size is not None:
            img.resize(max_size)

        self.parts.append(img)
        return self

    def add_file(
        self,
        data: bytes | str | Path | io.BytesIO,
        *,
        media_type: str | None = None,
        filename: str | None = None,
    ) -> "Message":
        """
        Append a file block and return self for chaining.
        """
        file = File(data, media_type=media_type, filename=filename)
        self.parts.append(file)
        return self

    def add_tool_call(self, id: str, name: str, arguments: dict) -> "Message":
        """Append a tool call block and return self for chaining."""
        self.parts.append(ToolCall(id=id, name=name, arguments=arguments))
        return self

    def add_tool_result(self, tool_call_id: str, result: str) -> "Message":
        """Append a tool result block and return self for chaining."""
        self.parts.append(ToolResult(tool_call_id=tool_call_id, result=result))
        return self

    def add_thinking(self, content: str) -> "Message":
        """Append a thinking block and return self for chaining."""
        self.parts.append(Thinking(content=content))
        return self

    # -------- convenient constructors --------
    @classmethod
    def user(
        cls,
        text: str | None = None,
        *,
        image: str | bytes | Path | io.BytesIO | None = None,
        file: str | bytes | Path | io.BytesIO | None = None,
    ) -> "Message":
        res = cls("user", [])
        if text is not None:
            res.add_text(text)
        if image is not None:
            res.add_image(image)
        if file is not None:
            res.add_file(file)
        return res

    @classmethod
    def system(cls, text: str | None = None) -> "Message":
        res = cls("system", [])
        if text is not None:
            res.add_text(text)
        return res

    @classmethod
    def ai(cls, text: str | None = None) -> "Message":
        res = cls("assistant", [])
        if text is not None:
            res.add_text(text)
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

    # ───── provider-specific emission ─────
    def oa_chat(self) -> dict:
        if self.role == "tool":
            # For tool messages, we expect a single ToolResult part (after splitting in to_openai)
            tool_results = [p for p in self.parts if isinstance(p, ToolResult)]
            if len(tool_results) == 1:
                tool_result = tool_results[0]
                return {
                    "role": "tool",
                    "tool_call_id": tool_result.tool_call_id,
                    "content": tool_result.result,
                }
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
        content = [p.anthropic() for p in self.parts]
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


###############################################################################
# 3. A whole conversation (ordered list of messages)                          #
###############################################################################


@dataclass(slots=True)
class Conversation:
    messages: list[Message] = field(default_factory=list)

    # ── convenience shorthands ------------------------------------------------
    @classmethod
    def system(cls, text: str) -> "Conversation":
        return cls([Message.system(text)])

    @classmethod
    def user(
        cls,
        text: str,
        *,
        image: bytes | str | Path | None = None,
        file: bytes | str | Path | None = None,
    ) -> "Conversation":
        msg = Message.user(text)
        if image is not None:
            msg.add_image(image)
        if file is not None:
            msg.add_file(file)
        return cls([msg])

    @classmethod
    def from_openai(cls, messages: list[dict]):
        """Compatibility with openai-formatted messages"""
        pass

    @classmethod
    def from_anthropic(cls, messages: list[dict], system: str | None = None):
        """Compatibility with anthropic-formatted messages"""
        pass

    # fluent additions
    def add(self, msg: Message) -> "Conversation":
        self.messages.append(msg)
        return self

    def add_tool_result(self, tool_call_id: str, result: str) -> "Conversation":
        """Add a tool result to the conversation.

        If the conversation ends with a tool message, append to it (for parallel tool calls).
        Otherwise, create a new tool message.
        """
        if self.messages and self.messages[-1].role == "tool":
            # Append to existing tool message (parallel tool calls)
            self.messages[-1].add_tool_result(tool_call_id, result)
        else:
            # Create new tool message
            tool_msg = Message("tool", [])
            tool_msg.add_tool_result(tool_call_id, result)
            self.messages.append(tool_msg)
        return self

    # ── conversions -----------------------------------------------------------
    def to_openai(self) -> list[dict]:
        result = []
        for m in self.messages:
            if m.role == "tool" and len(m.tool_results) > 1:
                # Split tool messages with multiple results into separate messages for OpenAI
                for tool_result in m.tool_results:
                    tool_msg = Message("tool", [tool_result])
                    result.append(tool_msg.oa_chat())
            else:
                result.append(m.oa_chat())
        return result

    def to_openai_responses(self) -> dict:
        # OpenAI Responses = single “input” array, role must be user/assistant
        input_items = []

        for m in self.messages:
            if m.role == "system":
                continue
            elif m.role == "assistant":
                # For assistant messages, extract computer calls as separate items
                text_parts = []
                for p in m.parts:
                    if isinstance(p, ToolCall) and p.name.startswith("_computer_"):
                        # Computer calls become separate items in the input array
                        action_type = p.name.replace("_computer_", "")
                        input_items.append(
                            {
                                "type": "computer_call",
                                "call_id": p.id,
                                "action": {"type": action_type, **p.arguments},
                            }
                        )
                    elif isinstance(p, Text):
                        text_parts.append({"type": "output_text", "text": p.text})
                    # TODO: Handle other part types as needed

                # Add message if it has text content
                if text_parts:
                    input_items.append({"role": m.role, "content": text_parts})
            else:
                # User and tool messages use normal format
                input_items.append(m.oa_resp())

        return {"input": input_items}

    def to_anthropic(
        self, cache_pattern: CachePattern | None = None
    ) -> tuple[str | list[dict] | None, list[dict]]:
        system_msg = next(
            (
                m.parts[0].text
                for m in self.messages
                if m.role == "system" and isinstance(m.parts[0], Text)
            ),
            None,
        )
        other = []
        for m in self.messages:
            if m.role == "system":
                continue
            elif m.role == "tool":
                # Convert tool messages to user messages for Anthropic
                user_msg = Message("user", m.parts)
                other.append(user_msg.anthropic())
            else:
                other.append(m.anthropic())

        # Apply cache control if specified
        if cache_pattern is not None:
            system_msg, other = self._apply_cache_control(
                system_msg, other, cache_pattern
            )

        return system_msg, other

    def _apply_cache_control(
        self,
        system_msg: str | None | list[dict],
        messages: list[dict],
        cache_pattern: CachePattern,
    ) -> tuple[str | list[dict] | None, list[dict]]:
        """Apply cache control to system message and/or messages based on the pattern."""

        if cache_pattern == "system_and_tools" and system_msg is not None:
            # Convert system message to structured format with cache control
            # This caches tools+system prefix (since system comes after tools)
            system_msg = [
                {
                    "type": "text",
                    "text": system_msg,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        if cache_pattern == "last_user_message":
            # Cache the last user message
            user_messages = [i for i, m in enumerate(messages) if m["role"] == "user"]
            if user_messages:
                last_user_idx = user_messages[-1]
                self._add_cache_control_to_message(messages[last_user_idx])

        elif cache_pattern == "last_2_user_messages":
            # Cache the last 2 user messages
            user_messages = [i for i, m in enumerate(messages) if m["role"] == "user"]
            for idx in user_messages[-2:]:
                self._add_cache_control_to_message(messages[idx])

        elif cache_pattern == "last_3_user_messages":
            # Cache the last 3 user messages
            user_messages = [i for i, m in enumerate(messages) if m["role"] == "user"]
            for idx in user_messages[-3:]:
                self._add_cache_control_to_message(messages[idx])

        return system_msg, messages

    def lock_images_as_bytes(self) -> "Conversation":
        """
        Convert all images to bytes format to ensure they remain unchanged for caching.
        This should be called when caching is enabled to prevent cache invalidation
        from image reference changes.
        """
        for message in self.messages:
            for part in message.parts:
                if isinstance(part, Image):
                    # Force conversion to bytes if not already
                    part.data = part._bytes()
                elif isinstance(part, File):
                    # Force conversion to bytes if not already
                    part.data = part._bytes()
        return self

    def _add_cache_control_to_message(self, message: dict) -> None:
        """Add cache control to a message's content."""
        content = message["content"]
        if isinstance(content, str):
            # Convert string content to structured format with cache control
            message["content"] = [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        elif isinstance(content, list) and content:
            # Add cache control to the last content block
            content[-1]["cache_control"] = {"type": "ephemeral"}

    def to_gemini(self) -> tuple[str | None, list[dict]]:
        system_msg = next(
            (
                m.parts[0].text
                for m in self.messages
                if m.role == "system" and isinstance(m.parts[0], Text)
            ),
            None,
        )
        other = [m.gemini() for m in self.messages if m.role != "system"]
        return system_msg, other

    def to_mistral(self) -> list[dict]:
        return [m.mistral() for m in self.messages]

    # ── misc helpers ----------------------------------------------------------
    _tok = tiktoken.encoding_for_model("gpt-4")

    def count_tokens(self, max_new_tokens: int = 0, img_tokens: int = 85) -> int:
        n = max_new_tokens
        for m in self.messages:
            for p in m.parts:
                if isinstance(p, Text):
                    n += len(self._tok.encode(p.text))
                else:  # Image – crude flat cost per image
                    n += img_tokens

        # very rough BOS/EOS padding
        return n + 6 * len(self.messages)

    def dry_run(self, model_name: str, max_new_tokens: int):
        model_obj = APIModel.from_registry(model_name)
        if model_obj.api_spec == "openai":
            image_tokens = 85
        elif model_obj.api_spec == "anthropic":
            image_tokens = 1_200
        else:
            image_tokens = 0
        input_tokens = self.count_tokens(0, image_tokens)
        output_tokens = max_new_tokens

        min_cost, max_cost = None, None
        if model_obj.input_cost and model_obj.output_cost:
            min_cost = model_obj.input_cost * input_tokens / 1e6
            max_cost = min_cost + model_obj.output_cost * output_tokens / 1e6

        return input_tokens, output_tokens, min_cost, max_cost

    @property
    def fingerprint(self) -> str:
        hasher = xxhash.xxh64()
        hasher.update(json.dumps([m.fingerprint for m in self.messages]).encode())
        return hasher.hexdigest()

    def to_log(self) -> dict:
        """
        Return a JSON-serialisable dict that fully captures the conversation.
        """
        serialized: list[dict] = []

        for msg in self.messages:
            content_blocks: list[dict] = []
            for p in msg.parts:
                if isinstance(p, Text):
                    content_blocks.append({"type": "text", "text": p.text})
                elif isinstance(p, Image):  # Image – redact the bytes, keep a hint
                    w, h = p.size
                    content_blocks.append(
                        {"type": "image", "tag": f"<Image ({w}×{h})>"}
                    )
                elif isinstance(p, File):  # File – redact the bytes, keep a hint
                    size = p.size
                    content_blocks.append(
                        {"type": "file", "tag": f"<File ({size} bytes)>"}
                    )
                elif isinstance(p, ToolCall):
                    content_blocks.append(
                        {
                            "type": "tool_call",
                            "id": p.id,
                            "name": p.name,
                            "arguments": p.arguments,
                        }
                    )
                elif isinstance(p, ToolResult):
                    content_blocks.append(
                        {
                            "type": "tool_result",
                            "tool_call_id": p.tool_call_id,
                            "result": p.result,
                        }
                    )
                elif isinstance(p, Thinking):
                    content_blocks.append({"type": "thinking", "content": p.content})
            serialized.append({"role": msg.role, "content": content_blocks})

        return {"messages": serialized}

    @classmethod
    def from_log(cls, payload: dict) -> "Conversation":
        """Re-hydrate a Conversation previously produced by `to_log()`."""
        msgs: list[Message] = []

        for m in payload.get("messages", []):
            role: Role = m["role"]  # 'system' | 'user' | 'assistant'
            parts: list[Part] = []

            for p in m["content"]:
                if p["type"] == "text":
                    parts.append(Text(p["text"]))
                elif p["type"] == "image":
                    # We only stored a placeholder tag, so keep that placeholder.
                    # You could raise instead if real image bytes are required.
                    parts.append(Image(p["tag"], detail="low"))
                elif p["type"] == "file":
                    # We only stored a placeholder tag, so keep that placeholder.
                    parts.append(File(p["tag"]))
                elif p["type"] == "tool_call":
                    parts.append(
                        ToolCall(id=p["id"], name=p["name"], arguments=p["arguments"])
                    )
                elif p["type"] == "tool_result":
                    parts.append(
                        ToolResult(tool_call_id=p["tool_call_id"], result=p["result"])
                    )
                elif p["type"] == "thinking":
                    parts.append(Thinking(content=p["content"]))
                else:
                    raise ValueError(f"Unknown part type {p['type']!r}")

            msgs.append(Message(role, parts))

        return cls(msgs)


def prompts_to_conversations(prompts: Sequence[str | list[dict] | Conversation]):
    if any(isinstance(x, list) for x in prompts):
        raise ValueError("can't convert list[dict] to conversation yet")
    return [  # type: ignore
        Conversation.user(p) if isinstance(p, str) else p for p in prompts
    ]


###############################################################################
# --------------------------------------------------------------------------- #
# Basic usage examples                                                        #
# --------------------------------------------------------------------------- #

# 1️⃣  trivial single-turn (text only)  ---------------------------------------
# conv = Conversation.user("Hi Claude, who won the 2018 World Cup?")
# client.messages.create(model="claude-3-7-sonnet", **conv.to_anthropic())

# # 2️⃣  system + vision + follow-up for OpenAI Chat Completions  ---------------
# conv = (
#     Conversation.system("You are a visual assistant.")
#     .add(
#         Message.with_image(
#             "user",
#             "What's in this photo?",
#             Image("boardwalk.jpg", detail="low"),
#         )
#     )
#     .add(Message.text("assistant", "Looks like a lakeside boardwalk."))
#     .add(Message.text("user", "Great, write a haiku about it."))
# )

# openai.chat.completions.create(model="gpt-4o-mini", messages=conv.to_openai_chat())

# # 3️⃣  Same conversation sent through new Responses API -----------------------
# openai.responses.create(model="gpt-4o-mini", **conv.to_openai_responses())
