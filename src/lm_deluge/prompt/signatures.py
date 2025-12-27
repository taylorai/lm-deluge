from dataclasses import dataclass
from typing import Literal, TypeAlias


SignatureProvider = Literal["anthropic", "gemini"]


@dataclass(slots=True)
class ThoughtSignature:
    value: str
    provider: SignatureProvider | None = None

    def for_provider(self, provider: SignatureProvider) -> str | None:
        if self.provider is None or self.provider == provider:
            return self.value
        return None


ThoughtSignatureLike: TypeAlias = ThoughtSignature | str


def normalize_signature(
    signature: ThoughtSignatureLike | None,
    *,
    provider: SignatureProvider | None = None,
) -> ThoughtSignature | None:
    if signature is None:
        return None
    if isinstance(signature, ThoughtSignature):
        if provider is not None and signature.provider is None:
            return ThoughtSignature(signature.value, provider)
        return signature
    return ThoughtSignature(signature, provider)


def signature_for_provider(
    signature: ThoughtSignatureLike | None, provider: SignatureProvider
) -> str | None:
    if signature is None:
        return None
    if isinstance(signature, ThoughtSignature):
        return signature.for_provider(provider)
    return signature


def signature_value(signature: ThoughtSignatureLike | None) -> str | None:
    if signature is None:
        return None
    if isinstance(signature, ThoughtSignature):
        return signature.value
    return signature


def serialize_signature(signature: ThoughtSignatureLike | None) -> str | dict | None:
    if signature is None:
        return None
    if isinstance(signature, ThoughtSignature):
        if signature.provider is None:
            return signature.value
        return {"value": signature.value, "provider": signature.provider}
    return signature


def deserialize_signature(payload: str | dict | None) -> ThoughtSignature | None:
    if payload is None:
        return None
    if isinstance(payload, dict):
        value = payload.get("value")
        provider = payload.get("provider")
        if isinstance(value, str):
            if provider in ("anthropic", "gemini"):
                return ThoughtSignature(value, provider)
            return ThoughtSignature(value)
        return None
    if isinstance(payload, str):
        return ThoughtSignature(payload)
    return None
