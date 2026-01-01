from dataclasses import dataclass, field

import xxhash


from .signatures import (
    normalize_signature,
    signature_value,
    ThoughtSignatureLike,
    signature_for_provider,
)


@dataclass(slots=True)
class Text:
    text: str
    type: str = field(init=False, default="text")
    # for gemini 3 - thought signatures to maintain reasoning context
    thought_signature: ThoughtSignatureLike | None = None

    def __post_init__(self) -> None:
        self.thought_signature = normalize_signature(self.thought_signature)

    @property
    def fingerprint(self) -> str:
        signature = signature_value(self.thought_signature) or ""
        return xxhash.xxh64(f"{self.text}:{signature}".encode()).hexdigest()

    # ── provider-specific emission ────────────────────────────────────────────
    def oa_chat(self) -> dict | str:  # OpenAI Chat Completions
        return {"type": "text", "text": self.text}

    def oa_resp(self) -> dict:  # OpenAI *Responses*  (new)
        return {"type": "input_text", "text": self.text}

    def anthropic(self) -> dict:  # Anthropic Messages
        return {"type": "text", "text": self.text}

    def gemini(self) -> dict:
        result = {"text": self.text}
        signature = signature_for_provider(self.thought_signature, "gemini")
        if signature is not None:
            result["thoughtSignature"] = signature
        return result

    def mistral(self) -> dict:
        return {"type": "text", "text": self.text}

    def nova(self) -> dict:
        return {"text": self.text}
