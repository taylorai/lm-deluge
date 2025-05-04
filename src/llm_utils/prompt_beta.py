from __future__ import annotations
import io, json, base64, mimetypes, tiktoken, xxhash
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, List, Literal, Sequence

###############################################################################
# 1. Low-level content blocks – either text or an image                       #
###############################################################################

Role = Literal["system", "user", "assistant"]

@dataclass(slots=True)
class Text:
    text: str
    type: str = field(init=False, default="text")

    # ── provider-specific emission ────────────────────────────────────────────
    def oa_chat(self) -> dict | str:               # OpenAI Chat Completions
        return self.text
    def oa_resp(self) -> dict: # OpenAI *Responses*  (new)
        return {"type": "input_text", "text": self.text}
    def anthropic(self) -> dict: # Anthropic Messages
        return {"type": "text", "text": self.text}

@dataclass(slots=True)
class Image:
    data: bytes | Path | io.BytesIO | str         # raw bytes or a path-like
    media_type: str | None = None           # inferred if None
    detail: Literal["low", "high", "auto"] = "auto"

    type: str = field(init=False, default="image")

    # helpers -----------------------------------------------------------------
    def _bytes(self) -> bytes:
        if isinstance(self.data, bytes):
            return self.data
        if isinstance(self.data, io.BytesIO):
            return self.data.getvalue()
        return Path(self.data).read_bytes()

    def _mime(self) -> str:
        if self.media_type:
            return self.media_type
        if isinstance(self.data, (Path, str)):
            guess = mimetypes.guess_type(str(self.data))[0]
            if guess:
                return guess
        return "image/png"

    # ── provider-specific emission ────────────────────────────────────────────
    def oa_chat(self) -> dict:
        b64 = base64.b64encode(self._bytes()).decode()
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{self._mime()};base64,{b64}",
                "detail": self.detail,
            },
        }

    def oa_resp(self) -> dict:
        b64 = base64.b64encode(self._bytes()).decode()
        return {"type": "input_image", "image_url": f"data:{self._mime()};base64,{b64}"}

    def anthropic(self) -> dict:
        b64 = base64.b64encode(self._bytes()).decode()
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": self._mime(),
                "data": b64,
            },
        }

###############################################################################
# 2. One conversational turn (role + parts)                                   #
###############################################################################

@dataclass(slots=True)
class Message:
    role: Role
    parts: List[Text | Image]

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
    ) -> "Message":
        """Append an image block and return self for chaining."""
        self.parts.append(Image(data, media_type=media_type, detail=detail))
        return self

    # convenient constructors ­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­
    @classmethod
    def user(cls) -> "Message":
        return cls("user", [])

    @classmethod
    def system(cls) -> "Message":
        return cls("system", [])

    @classmethod
    def ai(cls) -> "Message":
        return cls("assistant", [])

    @classmethod
    def text(cls, role: Role, content: str) -> "Message":
        return cls(role, [Text(content)])

    @classmethod
    def with_image(cls, role: Role, text: str, img: Image) -> "Message":
        return cls(role, [img, Text(text)])

    # ── provider-specific emission ────────────────────────────────────────────
    def oa_chat(self) -> dict:
        # Single-text shortcut if possible
        if len(self.parts) == 1 and isinstance(self.parts[0], Text):
            return {"role": self.role, "content": self.parts[0].oa_chat()}
        # Otherwise build an array of blocks
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

###############################################################################
# 3. A whole conversation (ordered list of messages)                          #
###############################################################################

@dataclass(slots=True)
class Conversation:
    messages: List[Message] = field(default_factory=list)

    # ── convenience shorthands ------------------------------------------------
    @classmethod
    def system(cls, text: str) -> "Conversation":
        return cls([Message.text("system", text)])

    @classmethod
    def user(cls, text: str, *, image_path: str | Path | None = None) -> "Conversation":
        msg = (
            Message.text("user", text)
            if image_path is None
            else Message.with_image("user", text, Image(image_path))
        )
        return cls([msg])

    # fluent additions
    def add(self, msg: Message) -> "Conversation":
        self.messages.append(msg)
        return self

    # ── conversions -----------------------------------------------------------
    def to_openai_chat(self) -> List[dict]:
        return [m.oa_chat() for m in self.messages]

    def to_openai_responses(self) -> dict:
        # OpenAI Responses = single “input” array, role must be user/assistant
        return {"input": [m.oa_resp() for m in self.messages if m.role != "system"]}

    def to_anthropic(self) -> tuple[str | None, List[dict]]:
        system_msg = next((m.parts[0].text for m in self.messages if m.role == "system" and isinstance(m.parts[0], Text)), None)
        other = [m.anthropic() for m in self.messages if m.role != "system"]
        return system_msg, other

    # ── misc helpers ----------------------------------------------------------
    _tok = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def token_count(self, img_tokens: int = 85) -> int:
        n = 0
        for m in self.messages:
            for p in m.parts:
                if isinstance(p, Text):
                    n += len(self._tok.encode(p.text))
                else:                       # Image – crude flat cost per image
                    n += img_tokens

        # very rough BOS/EOS padding
        return n + 6 * len(self.messages)

    @property
    def fingerprint(self) -> str:
        hasher = xxhash.xxh64()
        hasher.update(json.dumps([asdict(m) for m in self.messages]).encode())
        return hasher.hexdigest()

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
