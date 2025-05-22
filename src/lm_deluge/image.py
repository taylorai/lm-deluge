import os
from contextlib import contextmanager
from functools import cached_property
import io
import requests
from PIL import Image as PILImage  # type: ignore
import base64
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(slots=True)
class Image:
    # raw bytes, pathlike, http url, or base64 data url
    data: bytes | io.BytesIO | Path | str
    media_type: str | None = None  # inferred if None
    detail: Literal["low", "high", "auto"] = "auto"
    type: str = field(init=False, default="image")

    @classmethod
    def from_pdf(
        cls,
        pdf_path: str,
        dpi: int = 200,
        target_size: int = 1024,
        first_page: int | None = None,
        last_page: int | None = None,
    ) -> list["Image"]:
        try:
            from pdf2image import convert_from_path  # type: ignore
        except ImportError:
            raise RuntimeError("pdf2image is required for PDF conversion.")

        # Convert the first page of the PDF to an image
        pages = convert_from_path(
            pdf_path,
            dpi=dpi,
            first_page=first_page or 1,
            last_page=last_page,  # type: ignore
        )
        images = []
        for page in pages:
            buffer = io.BytesIO()
            page.save(buffer, format="JPEG")
            image = cls(buffer.getvalue(), media_type="image/jpeg")
            image.resize(target_size)
            images.append(image)
        return images

    # helpers -----------------------------------------------------------------
    def _bytes(self) -> bytes:
        if isinstance(self.data, bytes):
            return self.data
        elif isinstance(self.data, io.BytesIO):
            return self.data.getvalue()
        elif isinstance(self.data, str) and self.data.startswith("http"):
            res = requests.get(self.data)
            res.raise_for_status()
            return res.content
        elif isinstance(self.data, str) and os.path.exists(self.data):
            with open(self.data, "rb") as f:
                return f.read()
        elif isinstance(self.data, Path) and self.data.exists():
            return Path(self.data).read_bytes()
        elif isinstance(self.data, str) and self.data.startswith("data:"):
            header, encoded = self.data.split(",", 1)
            return base64.b64decode(encoded)
        else:
            raise ValueError("unreadable image format")

    def _mime(self) -> str:
        if self.media_type:
            return self.media_type
        if isinstance(self.data, (Path, str)):
            guess = mimetypes.guess_type(str(self.data))[0]
            if guess:
                return guess
        return "image/png"

    def _base64(self, include_header: bool = True) -> str:
        encoded = base64.b64encode(self._bytes()).decode("utf-8")
        if not include_header:
            return encoded
        return f"data:{self._mime()};base64,{encoded}"

    @contextmanager
    def _image(self):
        img = None
        try:
            img = PILImage.open(io.BytesIO(self._bytes()))
            yield img
        finally:
            if img:
                img.close()

    @cached_property
    def size(self) -> tuple[int, int]:
        with self._image() as img:
            return img.size

    @cached_property
    def num_pixels(self) -> int:
        return self.size[0] * self.size[1]

    def _resize(self, size: tuple[int, int]) -> bytes:
        buffer = io.BytesIO()
        new_width, new_height = size
        with self._image() as img:
            # Resize with Lanczos antialiasing
            img.resize((new_width, new_height), PILImage.Resampling.LANCZOS).save(
                buffer, format=self._mime().split("/")[-1].upper()
            )

        return buffer.getvalue()

    def _resize_longer(
        self, *, size: int | None = None, max_size: int | None = None
    ) -> bytes:
        if not max_size and not size:
            raise ValueError("Either size or max_size must be provided")
        width, height = self.size
        if width > height:
            new_width = size if size is not None else min(max_size, width)  # type: ignore
            new_height = int(new_width / width * height)
        else:
            new_height = size if size is not None else min(max_size, height)  # type: ignore
            new_width = int(new_height / height * width)
        return self._resize((new_width, new_height))

    def _resize_shorter(
        self, *, size: int | None = None, max_size: int | None = None
    ) -> bytes:
        if not max_size and not size:
            raise ValueError("Either size or max_size must be provided")
        width, height = self.size
        if width <= height:
            new_width = size if size is not None else min(max_size, width)  # type: ignore
            new_height = int(new_width / width * height)
        else:
            new_height = size if size is not None else min(max_size, height)  # type: ignore
            new_width = int(new_height / height * width)
        return self._resize((new_width, new_height))

    @cached_property
    def fingerprint(self) -> str:
        # return base64 of a very small version of the image
        small_image = self._resize_longer(max_size=48)  # longer side = 48px
        return base64.b64encode(small_image).decode("utf-8")

    def resize(self, max_size: int) -> None:
        """
        Resize the image and save to the data value.
        """
        self.data = self._resize_longer(max_size=max_size)

    # ── provider-specific emission ────────────────────────────────────────────
    def oa_chat(self) -> dict:
        # if max(self.size) > 1_568:
        #     self.resize_longer_side(1_568)
        return {
            "type": "image_url",
            "image_url": {
                "url": self._base64(),
                "detail": self.detail,
            },
        }

    def oa_resp(self) -> dict:
        # if max(self.size) > 1_568:
        #     self.resize_longer_side(1_568)
        return {"type": "input_image", "image_url": self._base64()}

    def anthropic(self) -> dict:
        # n_pixels = self.num_pixels
        # if n_pixels > 1_200_000:
        #     resize_factor = (1_200_000 / n_pixels) ** 0.5
        #     new_size = (
        #         int(self.size[0] * resize_factor),
        #         int(self.size[1] * resize_factor),
        #     )
        #     self.resize(new_size)
        b64 = base64.b64encode(self._bytes()).decode()
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": self._mime(),
                "data": b64,
            },
        }

    def mistral(self) -> dict:
        return {
            "type": "image_url",
            "image_url": self._base64(),
        }

    def gemini(self) -> dict:
        return {
            "inlineData": {
                "mimeType": self._mime(),
                "data": self._base64(include_header=False),
            }
        }
