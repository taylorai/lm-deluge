import io
import json
import tiktoken
import xxhash
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence
from lm_deluge.models import APIModel
from lm_deluge.image import Image

###############################################################################
# 1. Low-level content blocks – either text or an image                       #
###############################################################################
Role = Literal["system", "user", "assistant"]


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


###############################################################################
# 2. One conversational turn (role + parts)                                   #
###############################################################################
@dataclass(slots=True)
class Message:
    role: Role
    parts: list[Text | Image]

    @property
    def fingerprint(self) -> str:
        return self.role + "," + ",".join(part.fingerprint for part in self.parts)

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

    # -------- convenient constructors --------
    @classmethod
    def user(
        cls,
        text: str | None = None,
        *,
        image: str | bytes | Path | io.BytesIO | None = None,
    ) -> "Message":
        res = cls("user", [])
        if text is not None:
            res.add_text(text)
        if image is not None:
            res.add_image(image)
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
        parts: Sequence[Text | Image] = []
        content = msg["content"]
        if isinstance(content, str):
            parts.append(Text(content))
        else:
            for item in content:
                if item["type"] == "text":
                    parts.append(Text(item["text"]))
                elif item["type"] == "image_url":
                    parts.append(Image(data=item["image_url"]["url"]))
        return cls(role, parts)

    @classmethod
    def from_oa_resp(cls, msg: dict):
        raise NotImplementedError("not implemented")

    @classmethod
    def from_anthropic(cls, msg: dict):
        pass

    # ───── provider-specific emission ─────
    def oa_chat(self) -> dict:
        content = []
        for p in self.parts:
            content.append(p.oa_chat())
        return {"role": self.role, "content": content}

    def oa_resp(self) -> dict:
        content = [p.oa_resp() for p in self.parts]
        return {"role": self.role, "content": content}

    def anthropic(self) -> dict:
        # Anthropic: system message is *not* in the list
        if self.role == "system":
            raise ValueError("Anthropic keeps system outside message list")
        content = [p.anthropic() for p in self.parts]
        # Shortcut: single text becomes a bare string
        if len(content) == 1 and content[0]["type"] == "text":
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
        cls, text: str, *, image: bytes | str | Path | None = None
    ) -> "Conversation":
        msg = (
            Message.user(text) if image is None else Message.user(text).add_image(image)
        )
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

    # ── conversions -----------------------------------------------------------
    def to_openai(self) -> list[dict]:
        return [m.oa_chat() for m in self.messages]

    def to_openai_responses(self) -> dict:
        # OpenAI Responses = single “input” array, role must be user/assistant
        return {"input": [m.oa_resp() for m in self.messages if m.role != "system"]}

    def to_anthropic(self) -> tuple[str | None, list[dict]]:
        system_msg = next(
            (
                m.parts[0].text
                for m in self.messages
                if m.role == "system" and isinstance(m.parts[0], Text)
            ),
            None,
        )
        other = [m.anthropic() for m in self.messages if m.role != "system"]
        return system_msg, other

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
                else:  # Image – redact the bytes, keep a hint
                    w, h = p.size
                    content_blocks.append(
                        {"type": "image", "tag": f"<Image ({w}×{h})>"}
                    )
            serialized.append({"role": msg.role, "content": content_blocks})

        return {"messages": serialized}

    @classmethod
    def from_log(cls, payload: dict) -> "Conversation":
        """Re-hydrate a Conversation previously produced by `to_log()`."""
        msgs: list[Message] = []

        for m in payload.get("messages", []):
            role: Role = m["role"]  # 'system' | 'user' | 'assistant'
            parts: list[Text | Image] = []

            for p in m["content"]:
                if p["type"] == "text":
                    parts.append(Text(p["text"]))
                elif p["type"] == "image":
                    # We only stored a placeholder tag, so keep that placeholder.
                    # You could raise instead if real image bytes are required.
                    parts.append(Image(p["tag"], detail="low"))
                else:
                    raise ValueError(f"Unknown part type {p['type']!r}")

            msgs.append(Message(role, parts))

        return cls(msgs)


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
