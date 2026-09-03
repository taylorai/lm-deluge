from lm_deluge.prompt import (
    ContainerFile,
    Message,
    Text,
    Thinking,
    ThoughtSignature,
    ToolCall,
    ToolResult,
)


def parse_anthropic_response_content(
    response_content: list[dict],
    *,
    summarized_thinking: bool,
) -> tuple[Message, str | None]:
    """Parse Anthropic Messages content without altering replayable blocks."""
    parts = []
    latest_thinking = None

    for item in response_content:
        item_type = item["type"]
        if item_type == "text":
            parts.append(Text(item["text"]))
        elif item_type == "thinking":
            thinking_content = item.get("thinking", "")
            latest_thinking = thinking_content
            signature = item.get("signature")
            parts.append(
                Thinking(
                    thinking_content,
                    summary=thinking_content if summarized_thinking else None,
                    raw_payload=dict(item),
                    thought_signature=ThoughtSignature(
                        signature,
                        provider="anthropic",
                    )
                    if signature is not None
                    else None,
                )
            )
        elif item_type == "redacted_thinking":
            parts.append(
                Thinking(
                    item.get("data", ""),
                    raw_payload=dict(item),
                )
            )
        elif item_type == "tool_use":
            parts.append(
                ToolCall(
                    id=item["id"],
                    name=item["name"],
                    arguments=item["input"],
                )
            )
        elif item_type in {
            "bash_code_execution_tool_result",
            "text_editor_code_execution_tool_result",
        }:
            inner_content = item.get("content", {})
            files: list[ContainerFile] = []
            text_content = ""

            result_type = inner_content.get("type", "")
            if result_type in {
                "bash_code_execution_result",
                "text_editor_code_execution_result",
            }:
                if inner_content.get("stdout"):
                    text_content += inner_content["stdout"]
                if inner_content.get("stderr"):
                    text_content += inner_content["stderr"]

                for content_item in inner_content.get("content", []):
                    content_type = content_item.get("type", "")
                    if content_type in {"file", "bash_code_execution_output"}:
                        if "file_id" in content_item:
                            file_info: ContainerFile = {
                                "file_id": content_item["file_id"],
                                "filename": content_item.get("filename", "output"),
                                "media_type": content_item.get("media_type"),
                            }
                            files.append(file_info)
                    elif content_type == "text":
                        text_content += content_item.get("text", "")

            parts.append(
                ToolResult(
                    tool_call_id=item.get("tool_use_id", ""),
                    result=text_content or inner_content,
                    built_in=True,
                    built_in_type="bash_code_execution",
                    files=files or None,
                )
            )

    return Message("assistant", parts), latest_thinking


def is_anthropic_thinking_prefix_mismatch(error_message: str) -> bool:
    lower_error = error_message.lower()
    return any(
        marker in lower_error
        for marker in (
            "bound to a different conversation",
            "prefix_binding_mismatch",
            "thinking block no longer matches",
        )
    )
