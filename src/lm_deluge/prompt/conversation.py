import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import base64
import tiktoken
import xxhash

from .file import File
from .image import Image
from .thinking import Thinking
from .tool_calls import ToolCall, ToolResult, ToolResultPart
from .text import Text
from .signatures import (
    ThoughtSignature,
    serialize_signature,
    deserialize_signature,
    normalize_signature,
)
from .message import Message

CachePattern = Literal[
    "tools_only",
    "system_and_tools",
    "last_user_message",
    "last_2_user_messages",
    "last_3_user_messages",
]
Role = Literal["system", "user", "assistant", "tool"]
Part = Text | Image | File | ToolCall | ToolResult | Thinking


###############################################################################
# 3. A whole conversation (ordered list of messages)                          #
###############################################################################


@dataclass(slots=True)
class Conversation:
    messages: list[Message] = field(default_factory=list)

    # ── convenience shorthands ------------------------------------------------
    def system(self, text: str) -> "Conversation":
        """Add a system message and return self for chaining."""
        self.messages.append(Message.system(text))
        return self

    def user(
        self,
        text: str,
        *,
        image: bytes | str | Path | None = None,
        file: bytes | str | Path | None = None,
    ) -> "Conversation":
        """Add a user message and return self for chaining."""
        msg = Message.user(text)
        if image is not None:
            msg.with_image(image)
        if file is not None:
            msg.with_file(file)
        self.messages.append(msg)
        return self

    def ai(
        self,
        text: str,
        *,
        image: bytes | str | Path | None = None,
        file: bytes | str | Path | None = None,
    ) -> "Conversation":
        """Add an assistant message and return self for chaining."""
        msg = Message.ai(text)
        if image is not None:
            msg.with_image(image)
        if file is not None:
            msg.with_file(file)
        self.messages.append(msg)
        return self

    @classmethod
    def from_openai_chat(cls, messages: list[dict]):
        """Compatibility with openai-formatted messages"""

        def _to_image_from_url(block: dict) -> Image:
            payload = block.get("image_url") or block.get("input_image") or {}
            url = payload.get("url") or payload.get("file_id")
            detail = payload.get("detail", "auto")
            media_type = payload.get("media_type")
            if url is None:
                raise ValueError("image content missing url")
            return Image(data=url, media_type=media_type, detail=detail)

        def _to_file(block: dict) -> File:
            payload = block.get("file") or block.get("input_file") or {}
            file_id = payload.get("file_id") or block.get("file_id")
            filename = payload.get("filename")
            file_data = payload.get("file_data")
            if file_id is not None:
                return File(data=b"", filename=filename, file_id=file_id)
            if file_data is not None:
                return File(data=file_data, filename=filename)
            raise ValueError("file content missing file data or id")

        def _to_audio_file(block: dict) -> File:
            payload = block.get("audio") or block.get("input_audio") or {}
            file_id = payload.get("file_id")
            audio_format = payload.get("format", "wav")
            media_type = f"audio/{audio_format}"
            data = payload.get("data")
            if file_id is not None:
                return File(data=b"", media_type=media_type, file_id=file_id)
            if data is not None:
                data_url = f"data:{media_type};base64,{data}"
                return File(data=data_url, media_type=media_type)
            raise ValueError("audio block missing data or file id")

        text_types = {"text", "input_text", "output_text", "refusal"}
        image_types = {"image_url", "input_image", "image"}
        file_types = {"file", "input_file"}
        audio_types = {"audio", "input_audio"}

        def _convert_content_blocks(content: str | list[dict] | None) -> list[Part]:
            parts: list[Part] = []
            if content is None:
                return parts
            if isinstance(content, str):
                if content.strip():
                    parts.append(Text(content))
                return parts

            for block in content:
                block_type = block.get("type")
                if block_type in text_types:
                    text_value = block.get("text") or block.get(block_type) or ""
                    if text_value.strip():
                        parts.append(Text(text_value))
                elif block_type in image_types:
                    parts.append(_to_image_from_url(block))
                elif block_type in file_types:
                    parts.append(_to_file(block))
                elif block_type in audio_types:
                    parts.append(_to_audio_file(block))
                elif block_type == "tool_result":
                    # Rare: assistant echoing tool results – convert to text
                    result = block.get("content")
                    if isinstance(result, str):
                        parts.append(Text(result))
                    else:
                        parts.append(Text(json.dumps(result)))
                elif block_type == "image_file":
                    payload = block.get("image_file", {})
                    file_id = payload.get("file_id")
                    placeholder = {"type": "image_file", "file_id": file_id}
                    parts.append(Text(json.dumps(placeholder)))
                else:
                    parts.append(Text(json.dumps(block)))
            return parts

        def _convert_tool_arguments(raw: str | dict | None) -> dict:
            if isinstance(raw, dict):
                return raw
            if raw is None:
                return {}
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {"__raw__": raw}

        def _convert_tool_result_content(
            content: str | list[dict] | None,
        ) -> str | list[ToolResultPart]:
            if content is None:
                return ""
            if isinstance(content, str):
                return content
            result_parts: list[ToolResultPart] = []
            for block in content:
                block_type = block.get("type")
                if block_type in {"text", "input_text", "output_text", "refusal"}:
                    text_value = block.get("text") or block.get(block_type) or ""
                    result_parts.append(Text(text_value))
                elif block_type in image_types:
                    result_parts.append(_to_image_from_url(block))
                else:
                    result_parts.append(Text(json.dumps(block)))
            return result_parts

        conversation_messages: list[Message] = []

        for idx, raw_message in enumerate(messages):
            role = raw_message.get("role")
            if role is None:
                raise ValueError("OpenAI message missing role")

            role_lower = role.lower()
            if role_lower in {"system", "developer"}:
                parts = _convert_content_blocks(raw_message.get("content"))
                conversation_messages.append(Message("system", parts))
                continue

            if role_lower == "tool" or role_lower == "function":
                tool_call_id = (
                    raw_message.get("tool_call_id")
                    or raw_message.get("id")
                    or raw_message.get("name")
                    or f"tool_call_{idx}"
                )
                tool_result = ToolResult(
                    tool_call_id=tool_call_id,
                    result=_convert_tool_result_content(raw_message.get("content")),
                )
                conversation_messages.append(Message("tool", [tool_result]))
                continue

            mapped_role: Role
            if role_lower == "user":
                mapped_role = "user"
            elif role_lower == "assistant":
                mapped_role = "assistant"
            else:
                raise ValueError(f"Unsupported OpenAI message role: {role}")

            parts = _convert_content_blocks(raw_message.get("content"))

            tool_calls = raw_message.get("tool_calls")
            if not tool_calls and raw_message.get("function_call") is not None:
                tool_calls = [
                    {
                        "id": raw_message.get("id"),
                        "type": "function",
                        "function": raw_message["function_call"],
                    }
                ]

            if tool_calls:
                for call_index, call in enumerate(tool_calls):
                    call_type = call.get("type", "function")
                    call_id = (
                        call.get("id")
                        or call.get("tool_call_id")
                        or call.get("call_id")
                        or f"tool_call_{idx}_{call_index}"
                    )

                    if call_type == "function":
                        function_payload = call.get("function", {})
                        name = (
                            function_payload.get("name")
                            or call.get("name")
                            or "function"
                        )
                        arguments = _convert_tool_arguments(
                            function_payload.get("arguments")
                        )
                        parts.append(
                            ToolCall(
                                id=call_id,
                                name=name,
                                arguments=arguments,
                            )
                        )
                    else:
                        payload = call.get(call_type, {})
                        if not isinstance(payload, dict):
                            payload = {"value": payload}
                        arguments = payload.get("arguments")
                        if arguments is None:
                            arguments = payload
                        parts.append(
                            ToolCall(
                                id=call_id,
                                name=call_type,
                                arguments=arguments
                                if isinstance(arguments, dict)
                                else {"value": arguments},
                                built_in=True,
                                built_in_type=call_type,
                                extra_body=payload,
                            )
                        )

            if parts:
                conversation_messages.append(Message(mapped_role, parts))

        return cls(conversation_messages)

    @classmethod
    def from_anthropic(
        cls, messages: list[dict], system: str | list[dict] | None = None
    ):
        """Compatibility with anthropic-formatted messages"""

        def _anthropic_text_part(text_value: str | None) -> Text:
            return Text(text_value or "")

        def _anthropic_image(block: dict) -> Image:
            source = block.get("source", {})
            source_type = source.get("type")
            if source_type == "base64":
                media_type = source.get("media_type", "image/png")
                data = source.get("data", "")
                return Image(
                    data=f"data:{media_type};base64,{data}",
                    media_type=media_type,
                )
            if source_type == "url":
                media_type = source.get("media_type")
                url = source.get("url")
                if url is None:
                    raise ValueError("Anthropic image source missing url")
                return Image(data=url, media_type=media_type)
            if source_type == "file":
                file_id = source.get("file_id")
                if file_id is None:
                    raise ValueError("Anthropic image file source missing file_id")
                raise ValueError(
                    "Anthropic image file references require external fetch"
                )
            raise ValueError(f"Unsupported Anthropic image source: {source_type}")

        def _anthropic_file(block: dict) -> File:
            source = block.get("source", {})
            source_type = source.get("type")
            if source_type == "file":
                file_id = source.get("file_id")
                if file_id is None:
                    raise ValueError("Anthropic file source missing file_id")
                return File(data=b"", file_id=file_id)
            if source_type == "base64":
                media_type = source.get("media_type")
                data = source.get("data", "")
                return File(
                    data=f"data:{media_type};base64,{data}",
                    media_type=media_type,
                    filename=block.get("name"),
                )
            raise ValueError(f"Unsupported Anthropic file source: {source_type}")

        def _anthropic_tool_result_content(
            content: str | list[dict] | None,
        ) -> str | list[ToolResultPart]:
            if content is None:
                return ""
            if isinstance(content, str):
                return content
            result_parts: list[ToolResultPart] = []
            for part in content:
                part_type = part.get("type")
                if part_type == "text":
                    result_parts.append(_anthropic_text_part(part.get("text")))
                elif part_type == "image":
                    try:
                        result_parts.append(_anthropic_image(part))
                    except ValueError:
                        result_parts.append(Text(json.dumps(part)))
                else:
                    result_parts.append(Text(json.dumps(part)))
            return result_parts

        def _anthropic_content_to_parts(
            role: Role,
            content: str | list[dict] | None,
            signature_state: dict[str, ThoughtSignature | None] | None = None,
        ) -> list[Part]:
            parts: list[Part] = []
            if content is None:
                return parts
            if isinstance(content, str):
                parts.append(_anthropic_text_part(content))
                return parts

            for block in content:
                block_type = block.get("type")
                if block_type == "text":
                    parts.append(_anthropic_text_part(block.get("text")))
                elif block_type == "image":
                    try:
                        parts.append(_anthropic_image(block))
                    except ValueError:
                        parts.append(Text(json.dumps(block)))
                elif block_type == "document":
                    try:
                        parts.append(_anthropic_file(block))
                    except ValueError:
                        parts.append(Text(json.dumps(block)))
                elif block_type == "tool_use":
                    tool_id = block.get("id")
                    if tool_id is None:
                        raise ValueError("Anthropic tool_use block missing id")
                    name = block.get("name") or "tool"
                    arguments = block.get("input") or {}
                    tool_call = ToolCall(
                        id=tool_id,
                        name=name,
                        arguments=arguments
                        if isinstance(arguments, dict)
                        else {"value": arguments},
                    )
                    if signature_state is not None:
                        pending_signature = signature_state.get("pending")
                        if pending_signature:
                            tool_call.thought_signature = pending_signature
                            signature_state["pending"] = None
                    parts.append(tool_call)
                elif block_type == "redacted_thinking":
                    parts.append(
                        Thinking(content=block.get("data", ""), raw_payload=block)
                    )
                elif block_type == "thinking":
                    thinking_content = block.get("thinking", "")
                    signature = normalize_signature(
                        block.get("signature"),
                        provider="anthropic",
                    )
                    parts.append(
                        Thinking(
                            content=thinking_content,
                            raw_payload=block,
                            thought_signature=signature,
                        )
                    )
                    if signature_state is not None and signature is not None:
                        signature_state["pending"] = signature
                elif block_type == "tool_result":
                    tool_use_id = block.get("tool_use_id")
                    if tool_use_id is None:
                        raise ValueError(
                            "Anthropic tool_result block missing tool_use_id"
                        )
                    result = _anthropic_tool_result_content(block.get("content"))
                    tool_result = ToolResult(tool_call_id=tool_use_id, result=result)
                    parts.append(tool_result)
                else:
                    parts.append(Text(json.dumps(block)))
            return parts

        conversation_messages: list[Message] = []

        if system is not None:
            if isinstance(system, str):
                conversation_messages.append(Message("system", [Text(system)]))
            elif isinstance(system, list):
                system_parts = _anthropic_content_to_parts("system", system)
                conversation_messages.append(Message("system", system_parts))
            else:
                raise ValueError(
                    "Anthropic system prompt must be string or list of blocks"
                )

        for message in messages:
            role = message.get("role")
            if role is None:
                raise ValueError("Anthropic message missing role")

            if role not in {"user", "assistant"}:
                raise ValueError(f"Unsupported Anthropic role: {role}")

            base_role: Role = role  # type: ignore[assignment]
            content = message.get("content")
            if isinstance(content, list):
                buffer_parts: list[Part] = []
                signature_state: None | dict[str, ThoughtSignature | None] = (
                    {"pending": None} if base_role == "assistant" else None
                )
                for block in content:
                    block_type = block.get("type")
                    if block_type == "tool_result":
                        if buffer_parts:
                            conversation_messages.append(
                                Message(base_role, buffer_parts)
                            )
                            buffer_parts = []
                        tool_use_id = block.get("tool_use_id")
                        if tool_use_id is None:
                            raise ValueError(
                                "Anthropic tool_result block missing tool_use_id"
                            )
                        result = _anthropic_tool_result_content(block.get("content"))
                        conversation_messages.append(
                            Message(
                                "tool",
                                [ToolResult(tool_call_id=tool_use_id, result=result)],
                            )
                        )
                    else:
                        block_parts = _anthropic_content_to_parts(
                            base_role,
                            [block],
                            signature_state=signature_state,
                        )
                        buffer_parts.extend(block_parts)

                if buffer_parts:
                    conversation_messages.append(Message(base_role, buffer_parts))
            else:
                parts = _anthropic_content_to_parts(base_role, content)
                conversation_messages.append(Message(base_role, parts))

        return cls(conversation_messages)

    @classmethod
    def from_unknown(
        cls, messages: list[dict] | dict, *, system: str | list[dict] | None = None
    ) -> tuple["Conversation", str]:
        """Attempt to convert provider-formatted messages without knowing the provider.

        Returns the parsed conversation together with the provider label that succeeded
        ("openai", "anthropic", or "log").
        """

        # Check if input is in log format (output from to_log())
        if isinstance(messages, dict) and "messages" in messages:
            return cls.from_log(messages), "log"

        # Ensure messages is a list for provider detection
        if not isinstance(messages, list):
            raise ValueError(
                "messages must be a list of dicts or a dict with 'messages' key"
            )

        def _detect_provider() -> str:
            has_openai_markers = False
            has_anthropic_markers = False

            for msg in messages:
                role = msg.get("role")
                if role == "tool":
                    has_openai_markers = True

                if role == "system":
                    has_openai_markers = True

                if (
                    "tool_calls" in msg
                    or "function_call" in msg
                    or "tool_call_id" in msg
                ):
                    has_openai_markers = True

                content = msg.get("content")
                if isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        block_type = block.get("type")
                        if block_type in {
                            "tool_use",
                            "tool_result",
                            "thinking",
                            "assistant_response",
                            "redacted",
                        }:
                            has_anthropic_markers = True
                        if block_type == "tool_result" and block.get("tool_use_id"):
                            has_anthropic_markers = True
                        if block_type == "tool_use":
                            has_anthropic_markers = True

            if has_openai_markers and not has_anthropic_markers:
                return "openai"
            if has_anthropic_markers and not has_openai_markers:
                return "anthropic"
            if has_openai_markers:
                return "openai"
            if has_anthropic_markers:
                return "anthropic"
            # As a fallback, default to OpenAI which is the most permissive
            return "openai"

        provider = _detect_provider()
        if provider == "openai":
            try:
                return cls.from_openai_chat(messages), "openai"
            except Exception:
                try:
                    return cls.from_anthropic(messages, system=system), "anthropic"
                except Exception as anthropic_error:
                    raise ValueError(
                        "Unable to parse messages as OpenAI or Anthropic"
                    ) from anthropic_error
        else:
            try:
                return cls.from_anthropic(messages, system=system), "anthropic"
            except Exception:
                return cls.from_openai_chat(messages), "openai"

    # fluent additions
    def with_message(self, msg: Message) -> "Conversation":
        self.messages.append(msg)
        return self

    # another way of doing the same thing
    def add(self, msg: Message) -> "Conversation":
        self.messages.append(msg)
        return self

    def with_tool_result(
        self, tool_call_id: str, result: str | list[ToolResultPart]
    ) -> "Conversation":
        """Add a tool result to the conversation.

        If the conversation ends with a tool message, append to it (for parallel tool calls).
        Otherwise, create a new tool message.
        """
        if self.messages and self.messages[-1].role == "tool":
            # Append to existing tool message (parallel tool calls)
            self.messages[-1].with_tool_result(tool_call_id, result)
        else:
            # Create new tool message
            tool_msg = Message("tool", [])
            tool_msg.with_tool_result(tool_call_id, result)
            self.messages.append(tool_msg)
        return self

    # ── conversions -----------------------------------------------------------
    def to_openai(self) -> list[dict]:
        result = []
        for m in self.messages:
            if m.role == "tool":
                # Split tool messages with multiple results into separate messages for OpenAI
                for tool_result in m.tool_results:
                    tool_msg = Message("tool", [tool_result])
                    result.append(tool_msg.oa_chat())

                # if tool response included images, add those to next user message
                user_msg = Message("user", [])
                for i, tool_result in enumerate(m.tool_results):
                    images = tool_result.get_images()
                    if len(images) > 0:
                        user_msg.with_text(
                            f"[Images for Tool Call {tool_result.tool_call_id}]"
                        )
                        for img in images:
                            user_msg.with_image(img)

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
                    if isinstance(p, ToolCall) and p.built_in_type == "computer_call":
                        # Computer calls become separate items in the input array
                        # p.arguments already contains the full action dict with "type"
                        input_items.append(
                            {
                                "type": "computer_call",
                                "call_id": p.id,
                                "action": p.arguments,
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
        from lm_deluge.models import APIModel

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

    def to_log(self, *, preserve_media: bool = False) -> dict:
        """
        Return a JSON-serialisable dict that fully captures the conversation.

        Args:
            preserve_media: If True, store full base64-encoded bytes for images and files.
                           If False (default), replace with placeholder tags.
        """

        serialized: list[dict] = []

        for msg in self.messages:
            content_blocks: list[dict] = []
            for p in msg.parts:
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
                        "arguments": p.arguments,
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
                            "result": p.result
                            if isinstance(p.result, str)
                            else f"<Tool result ({len(p.result)} blocks)>",
                        }
                    )
                elif isinstance(p, Thinking):
                    thinking_block: dict = {"type": "thinking", "content": p.content}
                    signature = serialize_signature(p.thought_signature)
                    if signature is not None:
                        thinking_block["thought_signature"] = signature
                    content_blocks.append(thinking_block)
            serialized.append({"role": msg.role, "content": content_blocks})

        return {"messages": serialized}

    def print(self, max_text_length: int = 500, indent: int = 2) -> None:
        """Pretty-print the conversation to stdout.

        Args:
            max_text_length: Truncate text content longer than this (default 500 chars)
            indent: JSON indentation for tool calls/results (default 2)
        """
        ROLE_COLORS = {
            "system": "\033[95m",  # magenta
            "user": "\033[94m",  # blue
            "assistant": "\033[92m",  # green
            "tool": "\033[93m",  # yellow
        }
        RESET = "\033[0m"
        DIM = "\033[2m"
        BOLD = "\033[1m"

        def truncate(text: str, max_len: int) -> str:
            if len(text) <= max_len:
                return text
            return (
                text[:max_len] + f"{DIM}... [{len(text) - max_len} more chars]{RESET}"
            )

        def format_json(obj: dict | list, ind: int) -> str:
            return json.dumps(obj, indent=ind, ensure_ascii=False)

        print(f"\n{BOLD}{'=' * 60}{RESET}")
        print(f"{BOLD}Conversation ({len(self.messages)} messages){RESET}")
        print(f"{BOLD}{'=' * 60}{RESET}\n")

        for i, msg in enumerate(self.messages):
            role_color = ROLE_COLORS.get(msg.role, "")
            print(f"{role_color}{BOLD}[{msg.role.upper()}]{RESET}")

            for part in msg.parts:
                if isinstance(part, Text):
                    text = truncate(part.text, max_text_length)
                    # Indent multiline text
                    lines = text.split("\n")
                    if len(lines) > 1:
                        print("  " + "\n  ".join(lines))
                    else:
                        print(f"  {text}")

                elif isinstance(part, Image):
                    w, h = part.size
                    print(f"  {DIM}<Image ({w}x{h})>{RESET}")

                elif isinstance(part, File):
                    size = part.size
                    filename = getattr(part, "filename", None)
                    if filename:
                        print(f"  {DIM}<File: {filename} ({size} bytes)>{RESET}")
                    else:
                        print(f"  {DIM}<File ({size} bytes)>{RESET}")

                elif isinstance(part, ToolCall):
                    print(
                        f"  {DIM}Tool Call:{RESET} {BOLD}{part.name}{RESET} (id: {part.id})"
                    )
                    if part.arguments:
                        args_json = format_json(part.arguments, indent)
                        # Indent the JSON
                        indented = "\n".join(
                            "    " + line for line in args_json.split("\n")
                        )
                        print(indented)

                elif isinstance(part, ToolResult):
                    print(f"  {DIM}Tool Result:{RESET} (call_id: {part.tool_call_id})")
                    if isinstance(part.result, str):
                        result_text = truncate(part.result, max_text_length)
                        lines = result_text.split("\n")
                        for line in lines:
                            print(f"    {line}")
                    elif isinstance(part.result, dict):
                        result_json = format_json(part.result, indent)
                        indented = "\n".join(
                            "    " + line for line in result_json.split("\n")
                        )
                        print(indented)
                    elif isinstance(part.result, list):
                        print(f"    {DIM}<{len(part.result)} content blocks>{RESET}")
                        for block in part.result:
                            if isinstance(block, Text):
                                block_text = truncate(block.text, max_text_length // 2)
                                print(f"      [text] {block_text}")
                            elif isinstance(block, Image):
                                bw, bh = block.size
                                print(f"      {DIM}<Image ({bw}x{bh})>{RESET}")

                elif isinstance(part, Thinking):
                    print(f"  {DIM}Thinking:{RESET}")
                    thought = truncate(part.content, max_text_length)
                    lines = thought.split("\n")
                    for line in lines:
                        print(f"    {DIM}{line}{RESET}")

            # Separator between messages
            if i < len(self.messages) - 1:
                print(f"\n{'-' * 40}\n")

        print(f"\n{BOLD}{'=' * 60}{RESET}\n")

    @classmethod
    def from_log(cls, payload: dict) -> "Conversation":
        """Re-hydrate a Conversation previously produced by `to_log()`."""

        msgs: list[Message] = []

        for m in payload.get("messages", []):
            role: Role = m["role"]  # 'system' | 'user' | 'assistant'
            parts: list[Part] = []

            for p in m["content"]:
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

            msgs.append(Message(role, parts))

        return cls(msgs)
