import base64
from lm_deluge.image import Image

PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/Ps4TAAAAAElFTkSuQmCC"
PNG_BYTES = base64.b64decode(PNG_B64)


def test_mime_inferred():
    img = Image(PNG_BYTES, media_type="image/png")
    assert img._mime() == "image/png"


def test_base64_header():
    img = Image(PNG_BYTES, media_type="image/png")
    encoded = img._base64()
    assert encoded.startswith("data:image/png;base64,")


if __name__ == "__main__":
    test_mime_inferred()
    test_base64_header()
