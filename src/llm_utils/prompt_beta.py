from __future__ import annotations
import io, json, base64, mimetypes, tiktoken, xxhash
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal, Sequence

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
        return {"type": "text", "text": self.text}
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

    def resize(self, max_size: int) -> None:
        """
        Resize the image so that the longer side equals max_size,
        but only if the longer side is currently larger than max_size.
        Uses Lanczos antialiasing for high quality resizing.
        """
        # We need to convert the image data to a PIL Image
        from PIL import Image as PILImage
        import io

        # Convert bytes to PIL Image
        img = PILImage.open(io.BytesIO(self._bytes()))

        # Get current dimensions
        width, height = img.size
        longer_side = max(width, height)

        # Only resize if the image is larger than max_size
        if longer_side > max_size:
            # Calculate the new dimensions
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))

            # Resize with Lanczos antialiasing
            img = img.resize((new_width, new_height), PILImage.Resampling.LANCZOS)

            # Convert back to bytes
            buffer = io.BytesIO()
            img.save(buffer, format=self._mime().split('/')[-1].upper())

            # Update the data attribute
            self.data = buffer.getvalue()

        del img

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
    parts: list[Text | Image]

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

    # convenient constructors ­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­
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

    # ── provider-specific emission ────────────────────────────────────────────
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

###############################################################################
# 3. A whole conversation (ordered list of messages)                          #
###############################################################################

@dataclass(slots=True)
class Conversation:
    messages: list[Message] = field(default_factory=list)

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
    def to_openai(self) -> list[dict]:
        return [m.oa_chat() for m in self.messages]

    def to_openai_responses(self) -> dict:
        # OpenAI Responses = single “input” array, role must be user/assistant
        return {"input": [m.oa_resp() for m in self.messages if m.role != "system"]}

    def to_anthropic(self) -> tuple[str | None, list[dict]]:
        system_msg = next((m.parts[0].text for m in self.messages if m.role == "system" and isinstance(m.parts[0], Text)), None)
        other = [m.anthropic() for m in self.messages if m.role != "system"]
        return system_msg, other

    # ── misc helpers ----------------------------------------------------------
    _tok = tiktoken.encoding_for_model("gpt-4")

    def count_tokens(self, max_new_tokens: int = 0, img_tokens: int = 85) -> int:
        n = max_new_tokens
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
                    w, h = getattr(p, "width", "??"), getattr(p, "height", "??")
                    content_blocks.append({"type": "image", "tag": f"<Image ({w}×{h})>"})
            serialized.append({"role": msg.role, "content": content_blocks})

        return {"messages": serialized}


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
