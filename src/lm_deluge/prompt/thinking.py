from dataclasses import dataclass, field

import xxhash


from .signatures import (
    ThoughtSignatureLike,
    normalize_signature,
    signature_for_provider,
)


@dataclass(slots=True)
class Thinking:
    content: str  # reasoning content (o1, Claude thinking, etc.)
    type: str = field(init=False, default="thinking")
    # for openai - to keep conversation chain
    raw_payload: dict | None = None
    # for gemini 3 - thought signatures to maintain reasoning context
    thought_signature: ThoughtSignatureLike | None = None
    summary: str | None = None  # to differentiate summary text from actual content

    def __post_init__(self) -> None:
        self.thought_signature = normalize_signature(self.thought_signature)

    @property
    def fingerprint(self) -> str:
        return xxhash.xxh64(self.content.encode()).hexdigest()

    # ── provider-specific emission ────────────────────────────────────────────
    def oa_chat(self) -> dict:  # OpenAI Chat Completions
        # Thinking is typically not emitted back, but if needed:
        return {"type": "text", "text": f"[Thinking: {self.content}]"}

    def oa_resp(self) -> dict:  # OpenAI Responses
        # If we have the raw payload, use it (preserves all fields including id, summary)
        if self.raw_payload:
            return dict(self.raw_payload)

        # Otherwise, construct with required fields
        # The summary field is REQUIRED by OpenAI's Responses API for reasoning items
        return {
            "type": "reasoning",
            "id": f"reasoning_{id(self)}",  # Generate an ID if needed
            "summary": [{"type": "summary_text", "text": self.summary or self.content}],
        }

    def anthropic(self) -> dict:  # Anthropic Messages
        if self.raw_payload:
            return dict(self.raw_payload)
        result = {"type": "thinking", "thinking": self.content}
        signature = signature_for_provider(self.thought_signature, "anthropic")
        if signature is not None:
            result["signature"] = signature
        return result

    def gemini(self) -> dict:
        result = {"text": f"[Thinking: {self.content}]"}
        signature = signature_for_provider(self.thought_signature, "gemini")
        if signature is not None:
            result["thoughtSignature"] = signature
        return result

    def mistral(self) -> dict:
        return {"type": "text", "text": f"[Thinking: {self.content}]"}

    def nova(self) -> dict:
        return {"text": f"[Thinking: {self.content}]"}
