import requests
from typing import Optional, TYPE_CHECKING
from io import BytesIO
import base64

# Type checking imports
if TYPE_CHECKING:
    from PIL.Image import Image as PILImageType


class Image:
    def __init__(self, image: "PILImageType", url: Optional[str] = None):
        self.image = image
        self.url = url

    @classmethod
    def from_url(cls, url: str) -> "Image":
        from PIL import Image as PILImage

        response = requests.get(url)
        image = PILImage.open(BytesIO(response.content))
        return cls(image, url)

    @classmethod
    def from_file(cls, file_path: str) -> "Image":
        from PIL import Image as PILImage

        with PILImage.open(file_path) as img:
            img_copy = img.copy()  # close the file
        return cls(img_copy)

    @classmethod
    def from_base64(cls, base64_str: str) -> "Image":
        from PIL import Image as PILImage

        image = PILImage.open(BytesIO(base64.b64decode(base64_str)))
        return cls(image)

    @classmethod
    def from_pdf(
        cls, pdf_path: str, dpi: int = 200, target_size: int = 1024
    ) -> "Image":
        try:
            from pdf2image import convert_from_path  # type: ignore
        except ImportError:
            raise RuntimeError(
                "pdf2image is required for PDF conversion. Install with 'pip install pdf2image'"
            )

        # Convert the first page of the PDF to an image
        pages = convert_from_path(pdf_path, dpi=dpi, first_page=1, last_page=1)
        if pages:
            image = cls(pages[0])
            image.resize_shorter_side(target_size)
            return image
        else:
            raise ValueError("Failed to extract image from PDF")

    @property
    def size(self) -> tuple[int, int]:
        return self.image.size

    @property
    def fingerprint(self) -> str:
        # return base64 of a very small version of the image
        small_image = self.image.resize((32, 32))
        buffered = BytesIO()
        small_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    @property
    def num_pixels(self) -> int:
        return self.size[0] * self.size[1]

    def resize(self, size: tuple[int, int]) -> None:
        from PIL import Image as PILImage

        self.image = self.image.resize(size, resample=PILImage.Resampling.LANCZOS)

    def resize_longer_side(self, size: int) -> None:
        width, height = self.image.size
        if width > height:
            new_width = size
            new_height = int(size * height / width)
        else:
            new_height = size
            new_width = int(size * width / height)
        self.resize((new_width, new_height))

    def resize_shorter_side(self, size: int) -> None:
        width, height = self.image.size
        if width < height:
            new_width = size
            new_height = int(size * height / width)
        else:
            new_height = size
            new_width = int(size * width / height)
        self.resize((new_width, new_height))

    def to_base64(self) -> str:
        buffered = BytesIO()
        self.image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def save(self, file_path: str) -> None:
        self.image.save(file_path)

    def to_gemini_input(self) -> dict:
        return {"inlineData": {"mimeType": "image/png", "data": self.to_base64()}}

    def to_anthropic_input(self) -> dict:
        # image needs to be resized if it's > 1_200_000 pixels
        n_pixels = self.size[0] * self.size[1]
        if n_pixels > 1_200_000:
            resize_factor = (1_200_000 / n_pixels) ** 0.5
            new_size = (
                int(self.size[0] * resize_factor),
                int(self.size[1] * resize_factor),
            )
            self.resize(new_size)

        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": self.to_base64(),
            },
        }

    def to_openai_input(self, use_url: bool = False, detail: str = "low") -> dict:
        if max(self.size) > 1_568:
            self.resize_longer_side(1_568)

        if self.url is None:
            use_url = False
        if use_url:
            url = self.url
        else:
            url = f"data:image/png;base64,{self.to_base64()}"
        return {"type": "image_url", "image_url": {"url": url, "detail": detail}}
